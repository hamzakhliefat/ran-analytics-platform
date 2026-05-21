# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 17:20:10 2026

@author: hamza khlefat
"""

"""
Data Profiling and Quality Assessment
This module provides automated data exploration, statistical summaries, and quality metrics
for the network KPI dataset.
"""

import os
import sys

# System path configuration for module resolution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import Cols, DB, RawCols, rename_map
from src.db import read_sql

st.set_page_config(page_title="Data Quality Explorer", layout="wide")
st.title("Data Profiling & Quality Assessment")

cols = Cols()
db = DB()

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
Interface for evaluating dataset integrity, statistical distribution, and pre-processing transformations.
""")

# ============================================================================
# DATA INGESTION
# ============================================================================
@st.cache_data
def load_sample_data(n=50000):
    """Fetch randomized records for exploratory data analysis."""
    return read_sql(db.path, f"SELECT * FROM {db.table} ORDER BY RANDOM() LIMIT ?", (n,))

sample_size = st.sidebar.slider("Sampling Volume", 1000, 1500000, 50000, 5000)
df = load_sample_data(sample_size)

if df is None or df.empty:
    st.error("Data Source Error: Unable to retrieve records. Verify database status.")
    st.stop()

st.success(f"Records Successfully Ingested: {len(df):,}")

# ============================================================================
# DATASET ARCHITECTURE OVERVIEW
# ============================================================================
st.header("1. Dataset Architecture Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Record Count", f"{len(df):,}")
col2.metric("Feature Count", len(df.columns))
col3.metric("Resident Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
col4.metric("Redundant Observations (Sample)", df.duplicated().sum())

st.divider()

# ============================================================================
# PIPELINE EFFICIENCY & METADATA
# ============================================================================
st.header("2. Pipeline Efficiency & ETL Metadata")

@st.cache_data
def get_etl_metadata():
    """Retrieve pre-processing statistics from the database."""
    try:
        return read_sql(db.path, "SELECT * FROM etl_metadata LIMIT 1")
    except:
        return None

etl_meta = get_etl_metadata()

if etl_meta is not None and not etl_meta.empty:
    m = etl_meta.iloc[0]
    c1, c2, c3 = st.columns(3)
    
    # Visualizing Data Retention
    raw = m['raw_total']
    clean = m['clean_total']
    retention = (clean / raw * 100) if raw > 0 else 0
    
    c1.metric("Raw Record Population", f"{raw:,}")
    c2.metric("Cleaned Production Set", f"{clean:,}")
    c3.metric("Dataset Retention Rate", f"{retention:.2f}%")
    
    st.info(f"**ETL Insight:** The pre-processing pipeline identified and removed **{m['redundant_removed']:,}** redundant observations. "
            f"Additionally, **{m['imputations_performed']:,}** missing data points were resolved via forward-fill and median imputation.")
    
    st.caption(f"Last Pipeline Execution: {m['process_date']} | Data Flow: Raw CSV → Chunked ETL → SQLite Production Table")
else:
    st.warning("ETL Metadata Unavailable: Execute `scripts/build_db.py` to generate pipeline statistics.")

st.divider()

# ============================================================================
# SCHEMA CHARACTERISTICS
# ============================================================================
st.header("3. Schema Characteristics")

col_info = []
for col in df.columns:
    col_info.append({
        "Feature": col,
        "Dtype": str(df[col].dtype),
        "Non-Null Observations": df[col].count(),
        "Null Observations": df[col].isnull().sum(),
        "Null Percentage": f"{df[col].isnull().sum() / len(df) * 100:.2f}%",
        "Cardinality": df[col].nunique()
    })

col_df = pd.DataFrame(col_info)

st.dataframe(col_df, use_container_width=True)

st.divider()

# ============================================================================
# QUALITY METRICS & PRE-PROCESSING PIPELINE
# ============================================================================
st.header("4. Data Quality & ETL Metrics")

st.markdown("""
### Implemented Pre-processing Pipeline:
1. **Deduplication:** Removal of redundant observations across the feature space.
2. **Missing Value Treatment:** Implementation of forward-fill and median-based imputation strategies.
3. **Outlier Mitigation:** Identification and treatment of statistical outliers using the Interquartile Range (IQR) method.
4. **Type Enforcement:** Systematic casting to ensure numerical compatibility for modeling.
5. **Feature Engineering:** Computation of derived operational KPIs (e.g., Handover Success Rate).
""")

# Missing Value Visualization
st.subheader("Missing Value Density Analysis")

missing_data = df.isnull().sum()
missing_pct = (missing_data / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Feature': missing_data.index,
    'Missing Count': missing_data.values,
    'Missing Percentage': missing_pct.values
}).sort_values('Missing Count', ascending=False)

# Filter parameters for visualization clarity
missing_df_filtered = missing_df[missing_df['Missing Count'] > 0]

if len(missing_df_filtered) > 0:
    fig = px.bar(
        missing_df_filtered,
        x='Feature',
        y='Missing Percentage',
        title='Missing Data Distribution by Feature',
        color='Missing Percentage',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=120)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(missing_df_filtered, use_container_width=True)
else:
    st.success("Integrity Verified: No missing values detected across the feature space.")

st.divider()

# ============================================================================
# STATISTICAL OUTLIER DETECTION
# ============================================================================
st.header("5. Statistical Outlier Detection")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_col = st.selectbox("Select Feature for Outlier Analysis:", numeric_cols)

if selected_col:
    # Statistical bounds computation (IQR)
    Q1 = df[selected_col].quantile(0.25)
    Q3 = df[selected_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df) * 100)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Anomalous Observations", f"{outlier_count:,}")
    col2.metric("Anomalous Density", f"{outlier_pct:.2f}%")
    col3.metric("Theoretical Lower Bound", f"{lower_bound:.2f}")
    col4.metric("Theoretical Upper Bound", f"{upper_bound:.2f}")
    
    # Univariate Distribution (Box Plot)
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df[selected_col],
        name=selected_col,
        marker_color='#3498db',
        boxmean='sd'
    ))
    
    fig.update_layout(
        title=f"Box Plot: {selected_col}",
        yaxis_title=selected_col,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Review anomalous subset
    if outlier_count > 0:
        with st.expander(f"Review Anomalous Subset (Displaying top {min(100, outlier_count)} records)"):
            st.dataframe(outliers.head(100), use_container_width=True)

st.divider()

# ============================================================================
# KEY KPI STATISTICAL SUMMARY
# ============================================================================
st.header("6. Feature Descriptive Statistics")

st.markdown("Statistical summary of core operational KPIs (post-processing).")

kpi_cols = [cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max]
kpi_cols = [c for c in kpi_cols if c in df.columns]

stats_df = df[kpi_cols].describe().T
stats_df = stats_df.round(4)

st.dataframe(
    stats_df.style.background_gradient(cmap='YlOrRd', subset=['mean', 'std']),
    use_container_width=True
)

st.divider()

# ============================================================================
# DISTRIBUTION VISUALIZATIONS
# ============================================================================
st.header("7. Distributional Characteristics")

selected_kpi = st.selectbox("Select Objective KPI for Visualization:", kpi_cols)

if selected_kpi:
    # Kernel Density Estimation (Histogram representation)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[selected_kpi],
        nbinsx=50,
        marker_color='#3498db',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f"Frequency Distribution: {selected_kpi}",
        xaxis_title=selected_kpi,
        yaxis_title="Frequency Magnitude",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary of Central Tendency
    col1, col2, col3, col4, col5 = st.columns(5)
    
    is_tp = "Throughput" in selected_kpi
    u = "bps" if is_tp else ""
    
    col1.metric("Arithmetic Mean", human_format(df[selected_kpi].mean(), u) if is_tp else f"{df[selected_kpi].mean():.4f}")
    col2.metric("Median (Q2)", human_format(df[selected_kpi].median(), u) if is_tp else f"{df[selected_kpi].median():.4f}")
    col3.metric("Standard Deviation", human_format(df[selected_kpi].std(), u) if is_tp else f"{df[selected_kpi].std():.4f}")
    col4.metric("Observed Minimum", human_format(df[selected_kpi].min(), u) if is_tp else f"{df[selected_kpi].min():.4f}")
    col5.metric("Observed Maximum", human_format(df[selected_kpi].max(), u) if is_tp else f"{df[selected_kpi].max():.4f}")

st.divider()

# ============================================================================
# DATASET SUBSET REVIEW
# ============================================================================
st.header("8. Record-Level Visualization")

st.dataframe(df.head(100), use_container_width=True)

# Export Functionality
csv = df.sample(min(10000, len(df))).to_csv(index=False).encode('utf-8')
st.download_button(
    label="Export Analytical Dataset (CSV)",
    data=csv,
    file_name="ran_kpi_sample.csv",
    mime="text/csv",
)
