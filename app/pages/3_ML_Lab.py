# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 13:15:20 2026

@author: hamza khlefat
"""

"""
Machine Learning Model Training and Evaluation
This module provides experimental workflows for regression models, anomaly detection, 
and time-series forecasting using LSTM architectures.
"""

import os
import sys

# Configure environment for legacy Keras support
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Resolve and add root directory to system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from src.config import Cols, DB, Thresholds
from src.db import read_sql

from src.models_regression import train_regressors, shap_explain_regression
from src.models_anomaly import label_by_quantile, train_classifiers, shap_explain_classifier
from src.forecasting import prepare_daily_series, forecast_baseline, forecast_lstm


st.set_page_config(page_title="Machine Learning Lab", layout="wide")
st.title("Predictive Modeling Laboratory")

cols = Cols()
db = DB()
th = Thresholds()

def human_format(num, base_unit="bps"):
    """Format large numbers into human-readable strings with logical unit transitions."""
    if num is None or np.isnan(num): return "N/A"
    magnitude = 0
    if base_unit in ["bps", "Mbps", "kbps"]:
        units = ["bps", "Kbps", "Mbps", "Gbps", "Tbps"]
    else:
        units = [base_unit, f"K{base_unit}", f"M{base_unit}", f"G{base_unit}"]

    while abs(num) >= 1000 and magnitude < len(units) - 1:
        magnitude += 1
        num /= 1000.0
    
    return f"{num:.2f} {units[magnitude]}".strip()

st.markdown("""
### Model Framework Implementation
- **Supervised Learning:** Regression and Classification pipelines.
- **Explainable AI:** SHAP integration for all fitted models.
- **Deep Learning:** LSTM-based time-series forecasting.
- **Validation:** Automated cross-validation and independent test set evaluation.
""")

# ============================================================================
# TRAINING CONFIGURATION (SIDEBAR)
# ============================================================================
st.sidebar.header("Operational Configuration")

train_n = st.sidebar.slider("Training Sample Volume", 1_000, 1500_000, 1_000, 1_000)
st.sidebar.caption("Recommendation: Lower volume reduces memory overhead and processing time.")
test_size_pct = st.sidebar.slider("Test Set Allocation (%)", 10, 40, 20, 5)
test_size = test_size_pct / 100
cv_folds = st.sidebar.slider("Cross-Validation Folds", 3, 10, 5, 1)

st.sidebar.divider()
st.sidebar.subheader("Explanatory Analysis")
shap_max = st.sidebar.slider("SHAP Sampling Limit", 30, 1000, 200, 10)

# ============================================================================
# DATA INGESTION
# ============================================================================
@st.cache_data(show_spinner=False)
def _load_sample(n: int) -> pd.DataFrame:
    """Fetch randomized records for model experimentation."""
    return read_sql(db.path, f"SELECT * FROM {db.table} ORDER BY RANDOM() LIMIT ?", (int(n),))

with st.spinner(f"Ingesting {train_n:,} records from database..."):
    df = _load_sample(int(train_n))

if df is None or df.empty:
    st.error("Database Error: Unable to retrieve records. Verify database initialization status.")
    st.stop()

st.success(f"Records Successfully Loaded: {len(df):,}")

# ============================================================================
# DIMENSIONALITY & FEATURE SELECTION
# ============================================================================
features = [
    cols.prb_dl,
    cols.rrc_users_max,
    cols.ho_sr,
    cols.traffic_dl_gb,
    cols.harq_ack_64qam,
    cols.ho_exe_succ,
    cols.ho_prep_att,
]
features = [f for f in features if isinstance(f, str) and f in df.columns]

with st.expander("Operational Feature Set", expanded=False):
    st.write("Target feature vector for cross-model evaluation:")
    for i, feat in enumerate(features, 1):
        st.write(f"{i}. `{feat}`")

# ============================================================================
# PIPELINE SELECTION
# ============================================================================
st.divider()
st.header("1. Application Workflow Definition")

task = st.selectbox(
    "Select Implementation Objective:",
    [
        "Throughput Prediction (Regression)",
        "PRB Utilization Prediction (Regression)",
        "RRC Users Prediction (Regression)",
        "Handover Success Rate Prediction (Regression)",
        "Drop-Rate Anomaly Filtering (Classification)",
    ],
    index=0
)

# Persistent state management for modeling artifacts
if "ml_pack" not in st.session_state:
    st.session_state["ml_pack"] = None
if "ml_task" not in st.session_state:
    st.session_state["ml_task"] = None


# ============================================================================
# ANALYTICAL VISUALIZATION MODULES
# ============================================================================

def render_train_test_split_info(X_train, X_test, y_train, y_test):
    """
    Visual representation of the data partition strategy.
    Ensures transparent reporting of training and validation volume.
    """
    st.subheader("Data Partitioning Schema")
    
    col1, col2, col3 = st.columns(3)
    
    total = len(X_train) + len(X_test)
    train_pct = len(X_train) / total * 100
    test_pct = len(X_test) / total * 100
    
    col1.metric("Training Population", f"{len(X_train):,}", f"{train_pct:.1f}%")
    col2.metric("Verification Population", f"{len(X_test):,}", f"{test_pct:.1f}%")
    col3.metric("Aggregate Population", f"{total:,}", "100%")
    
    # Stratification Visualization
    fig = go.Figure(data=[go.Pie(
        labels=['Training Set', 'Test Set'],
        values=[len(X_train), len(X_test)],
        hole=0.4,
        marker=dict(colors=['#3498db', '#e74c3c']),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title="Sample Distribution Ratio",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_professional_model_comparison(results_df, task_type="regression"):
    """
    Comparative performance dashboard for evaluated models.
    Supports multi-metric ranking for regression and classification.
    """
    st.subheader("Model Performance Assessment")
    
    # Conditional heatmap coloring based on task metric directionality
    cmap = 'RdYlGn_r' if task_type == "regression" else 'RdYlGn'
    target_metric = 'test_MAE' if task_type == "regression" else 'test_F1'

    st.dataframe(
        results_df.style.background_gradient(cmap=cmap, subset=[target_metric]),
        use_container_width=True
    )
    
    if task_type == "regression":
        # Grouped Error Metric Comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Test MAE',
            x=results_df['model'],
            y=results_df['test_MAE'],
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            name='Test RMSE',
            x=results_df['model'],
            y=results_df['test_RMSE'],
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title="Regression Error Metrics (Lesser Magnitude Preferred)",
            xaxis_title="Model Architecture",
            yaxis_title="Error Magnitude",
            barmode='group',
            height=400,
            margin=dict(l=40, r=40, t=60, b=100),
            xaxis={'tickangle': -45}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # R2 Determination Comparison
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=results_df['model'],
            y=results_df['test_R2'],
            marker_color='#2ecc71',
            text=results_df['test_R2'].round(4),
            textposition='outside'
        ))
        
        fig2.update_layout(
            title="Coefficient of Determination (R²) Comparison",
            xaxis_title="Model Architecture",
            yaxis_title="R² Score",
            height=350,
            margin=dict(l=40, r=40, t=60, b=100),
            xaxis={'tickangle': -45}
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:  # classification
        # Comparison of Precision, Recall, and F1
        fig = go.Figure()
        
        metrics = ['test_F1', 'test_Precision', 'test_Recall']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for metric, color in zip(metrics, colors):
            if metric in results_df.columns:
                fig.add_trace(go.Bar(
                    name=metric.replace('test_', ''),
                    x=results_df['model'],
                    y=results_df[metric],
                    marker_color=color
                ))
        
        fig.update_layout(
            title="Classification Performance Metrics",
            xaxis_title="Model Architecture",
            yaxis_title="Metric Value",
            barmode='group',
            height=400,
            margin=dict(l=40, r=40, t=60, b=100),
            xaxis={'tickangle': -45}
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_predictions_plot(preds_df, target_name, model_name):
    """Correlation analysis of predicted vs actual observations."""
    st.subheader("Predicted vs. Observed Value Correlation")
    
    fig = go.Figure()
    
    # Theoretical Baseline: Identity Line
    min_val = min(preds_df['y_true'].min(), preds_df['y_pred'].min())
    max_val = max(preds_df['y_true'].max(), preds_df['y_pred'].max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Ideal Regression (y=x)',
        line=dict(color='gray', dash='dash', width=2)
    ))
    
    # Model Observations
    fig.add_trace(go.Scatter(
        x=preds_df['y_true'],
        y=preds_df['y_pred'],
        mode='markers',
        name='Model Observations',
        marker=dict(
            size=8,
            color=preds_df['y_pred'],
            colorscale='Viridis',
            showscale=False,
            opacity=0.4
        )
    ))
    
    fig.update_layout(
        title=f"Diagnostic: {model_name} Predicted vs. Observed {target_name}",
        xaxis_title="Observed Ground Truth",
        yaxis_title="Predicted Estimates",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Estimation Statistics
    col1, col2, col3 = st.columns(3)
    mae = np.abs(preds_df['y_true'] - preds_df['y_pred']).mean()
    rmse = np.sqrt(((preds_df['y_true'] - preds_df['y_pred']) ** 2).mean())
    
    col1.metric("Mean Absolute Error", f"{mae:.4f}")
    col2.metric("Root Mean Squared Error", f"{rmse:.4f}")
    col3.metric("Evaluation Sample Size", f"{len(preds_df):,}")


def render_enhanced_shap(shap_pack, X_used, title_prefix=""):
    """
    Model Interpretability Module.
    Visualizes feature attribution using SHAP kernels.
    """
    if not shap_pack["ok"]:
        st.warning(f"Interpretability Engine Failure: {shap_pack['error']}")
        return
    
    st.subheader(f"{title_prefix} - Feature Interpretation (SHAP)")
    
    try:
        import shap
        shap_values = shap_pack["shap_values"]
        X = shap_pack["X_used"]
        
        # Categorical Tabs for Interpretability Views
        tab1, tab2, tab3 = st.tabs(["Feature Attribution", "Impact Distribution", "Variable Interactions"])
        
        with tab1:
            st.markdown("Global importance ranking by mean absolute attribution.")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.bar(shap_values, max_display=12, show=False)
            st.pyplot(fig, clear_figure=True)
        
        with tab2:
            st.markdown("Distribution of impact magnitude across the feature range.")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, max_display=12, show=False)
            st.pyplot(fig, clear_figure=True)
        
        with tab3:
            st.markdown("Observation of impact magnitude relative to specific variable values.")
            
            # Extract underlying value array for index computation
            sv = shap_values.values if hasattr(shap_values, "values") else shap_values
            
            # Automatically focus on high-impact feature
            mean_abs = np.abs(sv).mean(axis=0)
            top_idx = int(np.argmax(mean_abs))
            
            feat = st.selectbox(
                "Select Variable for Interaction Analysis:",
                options=list(X.columns),
                index=top_idx,
                key=f"shap_dep_{title_prefix}"
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feat, sv, X, show=False, ax=ax)
            st.pyplot(fig, clear_figure=True)
            
    except Exception as e:
        st.error(f"Visualization Error: Unable to render interpretability plots. Detail: {e}")


# ============================================================================
# 2. MODEL INITIALIZATION AND TRAINING
# ============================================================================
if "Regression" in task:
    target_map = {
        "Throughput Prediction (Regression)": cols.throughput,
        "PRB Utilization Prediction (Regression)": cols.prb_dl,
        "RRC Users Prediction (Regression)": cols.rrc_users_max,
        "Handover Success Rate Prediction (Regression)": cols.ho_sr,
    }
    target = target_map[task]
    
    st.divider()
    st.header(f"2. Regression Framework Initialization: {target}")
    
    if target not in df.columns:
        st.error(f"Mapping Failure: Objective parameter `{target}` is absent from datasource.")
        st.stop()
    
    # Execution Trigger
    if st.button("Initialize Multi-Model Training Pipeline", type="primary", use_container_width=True):
        with st.spinner("Executing regression training sequence..."):
            try:
                pack = train_regressors(
                    df=df,
                    target=target,
                    features=features,
                    test_size=test_size,
                    cv_folds=cv_folds,
                )
                st.session_state["ml_pack"] = pack
                st.session_state["ml_task"] = task
                st.success("Regression Training Cycle Complete.")
            except Exception as e:
                st.error(f"Pipeline Failure: {e}")
    
    # Results Presentation
    pack = st.session_state.get("ml_pack")
    if pack is not None and st.session_state.get("ml_task") == task:
        st.divider()
        st.header("3. Quantitative Performance & Analysis")
        
        st.success(f"Optimal Architecture Identified: {pack['best_model_name']}")
        
        render_train_test_split_info(
            pack['X_train'], pack['X_test'],
            pack['y_train'], pack['y_test']
        )
        
        st.divider()
        render_professional_model_comparison(pack['results_df'], task_type="regression")
        
        st.divider()
        render_predictions_plot(pack['preds_df'], target, pack['best_model_name'])
        
        st.divider()
        
        # Interpretability Module (On-Demand Implementation)
        st.subheader("Diagnostic: Model Interpretability")
        st.caption(f"Context: Evaluation of {shap_max} samples. Configure sampling limit in sidebar to modulate compute latency.")
        
        if st.button("Generate SHAP Visualizations", key="run_shap_reg"):
            try:
                X_bg = pack['X_train'].sample(n=min(len(pack['X_train']), shap_max), random_state=42)
                X_exp = pack['X_test'].sample(n=min(len(pack['X_test']), shap_max), random_state=42)
                
                with st.spinner(f"Computing interpretability kernel for {shap_max} observations..."):
                    shap_pack = shap_explain_regression(pack['best_model'], X_bg, X_exp, max_samples=shap_max)
                
                if shap_pack["ok"]:
                    render_enhanced_shap(shap_pack, shap_pack.get("X_used", X_exp), f"{pack['best_model_name']}")
                else:
                    st.warning(f"Interpretability Error: {shap_pack.get('error', 'Unknown Exception')}")
                    st.info("Resolution Suggestion: Adjust sample volume in the operational configuration.")
            except Exception as e:
                st.error(f"Kernel Computation Failure: {e}")

# ============================================================================
# CLASSIFICATION PIPELINE (Anomaly Diagnostics)
# ============================================================================
else:
    st.divider()
    st.header("2. Classification Framework Initialization")
    
    if cols.drop_rate not in df.columns:
        st.error(f"Mapping Failure: Anomaly parameter `{cols.drop_rate}` is absent.")
        st.stop()
    
    q = st.slider(
        "Observation Isolation Threshold (Quantile)",
        0.80, 0.99,
        float(th.drop_anomaly_quantile),
        0.01
    )
    
    df2 = df.copy()
    df2["drop_anomaly"] = label_by_quantile(df2, cols.drop_rate, q)
    
    anomaly_rate = float(df2["drop_anomaly"].mean())
    st.info(f"Operational Anomaly Statistics: Rate={anomaly_rate:.2%} (Upper {int((1-q)*100)}% percentile identification)")
    
    # Execution Trigger
    if st.button("Initialize Classification Pipeline", type="primary", use_container_width=True):
        with st.spinner("Executing classification training sequence..."):
            try:
                pack = train_classifiers(
                    df=df2,
                    target="drop_anomaly",
                    features=features,
                    test_size=test_size,
                    cv_folds=cv_folds,
                )
                st.session_state["ml_pack"] = pack
                st.session_state["ml_task"] = task
                st.success("Classification Training Cycle Complete.")
            except Exception as e:
                st.error(f"Pipeline Failure: {e}")
    
    # Results Presentation
    pack = st.session_state.get("ml_pack")
    if pack is not None and st.session_state.get("ml_task") == task:
        st.divider()
        st.header("3. Quantitative Performance & Analysis")
        
        st.success(f"Optimal Architecture Identified: {pack['best_model_name']}")
        
        render_train_test_split_info(
            pack['X_train'], pack['X_test'],
            pack['y_train'], pack['y_test']
        )
        
        st.divider()
        render_professional_model_comparison(pack['results_df'], task_type="classification")
        
        st.divider()
        st.subheader("Confusion Matrix Observation")
        
        cm = pack['confusion_matrix']
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Baseline', 'Predicted Anomaly'],
            y=['Observed Baseline', 'Observed Anomaly'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="Model Classification Accuracy (Confusion Matrix)",
            height=400,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Metric Breakdown
        col1, col2, col3, col4 = st.columns(4)
        tn, fp, fn, tp = cm.ravel()
        col1.metric("Correct Normal Identification", int(tn))
        col2.metric("False Anomaly Alerts", int(fp))
        col3.metric("Missed Anomaly Events", int(fn))
        col4.metric("Correct Anomaly Identification", int(tp))
        
        st.divider()
        
        # Interpretability Module (On-Demand Implementation)
        st.subheader("Diagnostic: Model Interpretability")
        st.caption(f"Context: Evaluation of {shap_max} samples. Configure sampling limit in sidebar to modulate compute latency.")
        
        if st.button("Generate SHAP Visualizations", key="run_shap_clf"):
            try:
                X_bg = pack['X_train'].sample(n=min(len(pack['X_train']), shap_max), random_state=42)
                X_exp = pack['X_test'].sample(n=min(len(pack['X_test']), shap_max), random_state=42)
                
                with st.spinner(f"Computing interpretability kernel for {shap_max} observations..."):
                    shap_pack = shap_explain_classifier(pack['best_model'], X_bg, X_exp, max_samples=shap_max)
                
                if shap_pack["ok"]:
                    render_enhanced_shap(shap_pack, shap_pack.get("X_used", X_exp), f"{pack['best_model_name']}")
                else:
                    st.warning(f"Interpretability Error: {shap_pack.get('error', 'Unknown Exception')}")
                    st.info("Resolution Suggestion: Adjust sample volume in the operational configuration.")
            except Exception as e:
                st.error(f"Kernel Computation Failure: {e}")

# ============================================================================
# 4. TIME-SERIES FORECASTING (LSTM ARCHITECTURE)
# ============================================================================
st.divider()
st.header("4. Time-Series Forecasting (LSTM Architecture)")

st.markdown("""
Implementation of **Long Short-Term Memory (LSTM)** recurrent neural networks for forecasting 
temporal KPI trends. Provides comparison against deterministic baseline models.
""")

if "forecast_result" not in st.session_state:
    st.session_state["forecast_result"] = None

horizon = st.selectbox("Predictive Horizon (Days)", [7, 14, 30], index=0)

# Dataset Subset Filtering
cell_id_val = None
if "cell_id" in df.columns:
    sample_cells = df["cell_id"].dropna().astype(str).unique().tolist()[:200]
    cell_id_val = st.selectbox("Identifier Filtering (Optional)", ["Aggregate Model"] + sample_cells)
    if cell_id_val == "Aggregate Model":
        cell_id_val = None

# Multivariate Feature Selection
use_dl = st.checkbox("Enable Neural Forecasting (LSTM Architecture)", value=False)

forecast_features = []
if use_dl:
    st.info("Multivariate Training: Selecting secondary KPI drivers can improve forecast stability.")
    forecast_features = st.multiselect(
        "Explanatory Features for Multivariate LSTM:",
        options=[f for f in features if f != cols.throughput],
        default=[cols.prb_dl] if cols.prb_dl in features else []
    )
    epochs = st.slider("Neural Network Training Iterations (Epochs)", 5, 30, 10, 1)
else:
    epochs = 10

# Execution Pipeline
if st.button("Initialize Forecasting Sequence", type="primary", use_container_width=True):
    try:
        with st.spinner("Processing temporal series and training architecture..."):
            from src.forecasting import prepare_multivariate_series
            
            if use_dl and forecast_features:
                series_data, meta = prepare_multivariate_series(
                    df=df,
                    target_col=cols.throughput,
                    feature_cols=forecast_features,
                    cell_col="cell_id",
                    selected_cell=cell_id_val,
                )
                hist_df, fut_df = forecast_lstm(series_data, horizon=horizon, epochs=epochs)
                method = f"Multivariate LSTM ({len(forecast_features)} features)"
            else:
                from src.forecasting import prepare_daily_series
                series, meta = prepare_daily_series(
                    df=df,
                    target_col=cols.throughput,
                    cell_col="cell_id",
                    selected_cell=cell_id_val,
                )
                
                if use_dl:
                    hist_df, fut_df = forecast_lstm(series, horizon=horizon, epochs=epochs)
                    method = "Univariate LSTM (Single Target)"
                else:
                    hist_df, fut_df = forecast_baseline(series, horizon=horizon)
                    method = "Deterministic Baseline (Linear Trend + SMA)"
            
            st.session_state["forecast_result"] = {
                "hist_df": hist_df,
                "fut_df": fut_df,
                "method": method,
                "meta": meta
            }
            
        st.success(f"Predictive Sequence Complete: Strategy = {method}")
        
    except Exception as e:
        st.error(f"Forecasting Sequence Failure: {e}")
        st.session_state["forecast_result"] = None

# Forecast Visualization Dashboard
forecast_result = st.session_state.get("forecast_result")
if forecast_result is not None:
    st.divider()
    st.subheader("Quantitative Forecast Projection")
    
    hist_df = forecast_result["hist_df"]
    fut_df = forecast_result["fut_df"]
    method = forecast_result["method"]
    meta = forecast_result["meta"]
    
    # Determine scaling magnitude
    max_metric = max(hist_df["y"].max(), fut_df["yhat"].max())
    scale_factor = 1.0
    unit_label = ""
    
    if max_metric >= 1e9:
        scale_factor = 1e9
        unit_label = " (Gbps)"
    elif max_metric >= 1e6:
        scale_factor = 1e6
        unit_label = " (Mbps)"

    fig = go.Figure()
    
    # Historical Observations
    fig.add_trace(go.Scatter(
        x=hist_df["ds"],
        y=hist_df["y"] / scale_factor,
        mode="lines",
        name=f"Historical Observations{unit_label}",
        line=dict(color='#3498db', width=2)
    ))
    
    # Projected Estimates
    fig.add_trace(go.Scatter(
        x=fut_df["ds"],
        y=fut_df["yhat"] / scale_factor,
        mode="lines+markers",
        name=f"Forecast Estimate ({method}){unit_label}",
        line=dict(color='#e74c3c', width=3, dash='dot'),
        marker=dict(size=8)
    ))

    # Prediction Point Definition
    forecast_start_val = fut_df["ds"].iloc[0]
    forecast_start = forecast_start_val.timestamp() * 1000 if hasattr(forecast_start_val, "timestamp") else pd.to_datetime(forecast_start_val).timestamp() * 1000
        
    fig.add_vline(
        x=forecast_start,
        line_dash="dash",
        line_color="gray",
        annotation_text="Prediction Start",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f"Projection Analysis: Throughput {horizon}-Day Predictive Horizon",
        xaxis_title="Temporal Axis",
        yaxis_title=f"Throughput Magnitude{unit_label}",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictive Metadata
    st.caption(f"Context: Series Resolution = {meta['time_col']} | Population = {meta['cell_used'] or 'Global Aggregation'}")
    
    # Summary Metrics
    col1, col2 = st.columns(2)
    col1.metric("Observations (Days)", len(hist_df))
    col2.metric("Projections (Days)", len(fut_df))
    
    st.info(f"**Analytical Engine:** {method}")
