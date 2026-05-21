# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 16:10:44 2026

@author: hamza khlefat
"""

import os
import sys

# Module path resolution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import streamlit as st
import pandas as pd
import numpy as np
from src.config import Cols, DB, Thresholds
from src.db import read_sql
from src.rca import add_baselines, explain_row

st.set_page_config(page_title="Root Cause Analysis", layout="wide")
st.title("Deterministic Root Cause Diagnostic")

cols = Cols()
db = DB()
th = Thresholds()

st.markdown("""
Perform diagnostic analysis across multiple granularities. This module evaluates 
**Site-Level Aggregated SHAP** metrics to identify systemic drivers and 
**Cell-Level Heuristics** for isolated event diagnostics.
""")

# ============================================================================
# ANALYSIS MODE SELECTION
# ============================================================================
mode = st.radio("Diagnostic Depth:", ["Site-Level Aggregated SHAP", "Individual Cell Heuristics"], horizontal=True)

if mode == "Individual Cell Heuristics":
    cell = st.text_input("Enter Cell Identifier (cell_id):", "")

    if st.button("Execute Cell Diagnostic") and cell.strip():
        with st.spinner(f"Ingesting telemetry for Cell {cell.strip()}..."):
            df = read_sql(
                db.path,
                f"SELECT * FROM {db.table} WHERE {cols.cell_id} = ? ORDER BY {cols.date} DESC LIMIT 200",
                (cell.strip(),)
            )

        if df.empty:
            st.warning("Data Availability: No records found for the specified Cell Identifier.")
            st.stop()

        # Apply temporal baseline calculations
        df = add_baselines(df, cols)

        st.subheader("Recent Operational Telemetry")
        st.dataframe(df.head(20), use_container_width=True)

        st.divider()
        st.subheader("Heuristic Diagnostic Results")
        reasons = explain_row(df.iloc[0], cols, th)
        
        if reasons:
            for r in reasons:
                st.info(r)
        else:
            st.success("Analysis Complete: No critical performance degradation patterns identified in the latest record.")

else: # Site-Level Aggregated SHAP
    ml_pack = st.session_state.get("ml_pack")
    
    if not ml_pack:
        st.warning("Predictive Engine Not Loaded: Site-level SHAP aggregation requires a pre-trained regression model.")
        st.info("Resolution: Establish an optimized model in the **ML Lab** before executing site diagnostics.")
        st.stop()

    site_id = st.text_input("Enter Site Identifier (site_id):", "")
    
    if st.button("Execute Site-Level Aggregation") and site_id.strip():
        with st.spinner(f"Aggregating cluster telemetry for Site {site_id}..."):
            df_site = read_sql(
                db.path,
                f"SELECT * FROM {db.table} WHERE {cols.site_id} = ? LIMIT 1000",
                (site_id.strip(),)
            )
            
        if df_site.empty:
            st.error("Data Availability Error: Specified Site Identifier not found in operational database.")
            st.stop()
            
        st.success(f"Cluster Data Ingested: {len(df_site)} observations from Site {site_id}")
        
        # Prepare for SHAP aggregation
        model = ml_pack["best_model"]
        features = ml_pack["feature_names"]
        
        X = df_site[features].copy()
        for f in features:
            X[f] = pd.to_numeric(X[f], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
        
        with st.spinner("Computing cluster-level impact attribution (Aggregated SHAP)..."):
            try:
                import shap
                import plotly.graph_objects as go
                
                # Use TreeExplainer for ensembles, Kernel/Explainer for others
                if hasattr(model, "estimators_") or model.__class__.__name__ in ["RandomForestRegressor", "GradientBoostingRegressor"]:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                else:
                    explainer = shap.Explainer(model, X.sample(min(100, len(X))))
                    shap_values = explainer(X).values
                
                # Aggregate Impact: Mean Absolute SHAP score per feature
                if isinstance(shap_values, list): # Multi-output handling
                    shap_values = shap_values[0]
                
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                driver_df = pd.DataFrame({
                    "Feature": features,
                    "Mean_Absolute_Impact": mean_abs_shap
                }).sort_values("Mean_Absolute_Impact", ascending=True)
                
                st.subheader("Dominant Performance Drivers (Aggregated)")
                
                # Use the same unit detection logic for SHAP axis
                max_shap = driver_df["Mean_Absolute_Impact"].max()
                scale_factor = 1.0
                unit_label = ""
                if max_shap >= 1e9:
                    scale_factor = 1e9
                    unit_label = " (Gbps)"
                elif max_shap >= 1e6:
                    scale_factor = 1e6
                    unit_label = " (Mbps)"

                fig = go.Figure(go.Bar(
                    x=driver_df["Mean_Absolute_Impact"] / scale_factor,
                    y=driver_df["Feature"],
                    orientation='h',
                    marker_color='#3498db'
                ))
                
                fig.update_layout(
                    title=f"Site {site_id}: Aggregated Feature Attribution (SHAP){unit_label}",
                    xaxis_title=f"Magnitude of Impact on Prediction{unit_label}",
                    height=450,
                    margin=dict(l=30, r=30, t=60, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Narrative Interpretation
                top_driver = driver_df.iloc[-1]["Feature"]
                st.info(f"**Cluster Diagnostic Conclusion:** Across Site {site_id}, the most significant driver of performance variance is `{top_driver}`. "
                        f"Technical intervention focusing on `{top_driver}` stabilization is recommended for maximum impact on site-wide KPIs.")
                
            except Exception as e:
                st.error(f"Interpretability Kernel Failure: {e}")
                st.info("Ensure the `shap` library is correctly installed and compatible with your Python environment.")
