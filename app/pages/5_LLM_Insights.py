# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 17:20:11 2026

@author: hamza khlefat
"""

"""
LLM-Powered Insights Generation
Provides automated analysis using local LLM (Ollama) for performance diagnostic and optimization.
"""

import os
import sys

# Ensure root directory is in path for module resolution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np

from src.config import Cols, DB
from src.db import read_sql
from src.llm_insights import get_insights_generator

st.set_page_config(page_title="LLM Network Insights", layout="wide")
st.title("Network Intelligence & LLM Insights")

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
### Automated Performance Analysis
Integration with **Ollama** for local Large Language Model inference. This module provides 
technical diagnostics, root cause identification, and parameter optimization recommendations.

**Supported Model Architectures:** Llama 2, Mistral, Phi-2, Code Llama
""")

cols = Cols()
db = DB()

# ============================================================================
# Sidebar Configuration
# ============================================================================
st.sidebar.header("Ollama Configuration")

use_llm = st.sidebar.checkbox("Enable LLM Inference", value=True, 
                              help="Toggle between LLM-based analysis and deterministic rule-based logic.")

model_options = {
    "Llama 2 (Standard)": "llama2",
    "Mistral (High Performance)": "mistral",
    "Phi-2 (Resource Optimized)": "phi",
    "Code Llama (Technical)": "codellama",
}

selected_model_name = st.sidebar.selectbox(
    "Model Selection",
    options=list(model_options.keys()),
    index=0,
    disabled=not use_llm,
    help="Select the LLM model for inference. Note: Uncached models require initial download (~4GB)."
)

model_name = model_options[selected_model_name]

if use_llm:
    st.sidebar.info(f"Active Model: {selected_model_name}")
    st.sidebar.caption("System Requirement: Ollama runtime must be active (https://ollama.ai)")
    st.sidebar.caption(f"Initialization: `ollama pull {model_name}`")
else:
    st.sidebar.warning("Operational Mode: Rule-based fallback active.")

# ============================================================================
# LLM Engine Initialization
# ============================================================================
@st.cache_resource
def load_insights_generator(use_llm_flag, model):
    """
    Instance generator for insights engine.
    Caches the connection to prevent redundant initialization.
    """
    return get_insights_generator(use_ollama=use_llm_flag, model_name=model)

# Engine Loading
with st.spinner("Initializing Inference Engine..." if use_llm else "Loading Rule Engine..."):
    try:
        generator = load_insights_generator(use_llm, model_name)
        if use_llm:
            st.success("LLM Engine Initialized Successfully.")
        else:
            st.success("Deterministic Rule Engine Load Complete.")
    except Exception as e:
        st.error(f"Initialization Failure: {e}")
        st.info("Verification Required: Ensure Ollama service is active. Documentation: https://ollama.ai")
        st.stop()

# ============================================================================
# Data Ingestion
# ============================================================================
@st.cache_data
def load_data_sample(n=50000):
    """Fetch sample records for analysis context."""
    return read_sql(db.path, f"SELECT * FROM {db.table} ORDER BY RANDOM() LIMIT ?", (n,))

sample_size = st.sidebar.slider("Training Sample Volume", 10000, 1500000, 50000, 10000)
df = load_data_sample(sample_size)

if df is None or df.empty:
    st.error("Data Source Error: Dataset is empty or inaccessible.")
    st.stop()

# ============================================================================
# Analysis Operational Mode
# ============================================================================
st.divider()
st.header("Analysis Framework Selection")

mode = st.radio(
    "Select Diagnostic Mode:",
    ["Network Performance Summary", "Specific Cell Diagnostics", "KPI Anomaly Interpretation", 
     "Protocol-Level Optimization Recommendations"],
    horizontal=False
)

# ============================================================================
# MODE 1: Dataset Summary Insights
# ============================================================================
if mode == "Network Performance Summary":
    st.subheader("High-Level Network Performance Analysis")
    
    st.markdown("""
    The engine performs multi-dimensional analysis on aggregated KPI statistics to identify 
    global network health trends.
    """)
    
    kpi_cols = [cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max]
    kpi_cols = [c for c in kpi_cols if c in df.columns]
    
    if st.button("Execute Global Analysis", type="primary"):
        with st.spinner("Processing network telemetry..."):
            try:
                if hasattr(generator, 'generate_summary_insights'):
                    insights = generator.generate_summary_insights(df, kpi_cols)
                else:
                    # Deterministic fallback summary
                    stats_text = "\n".join([f"- {col}: mean={df[col].mean():.4f}" for col in kpi_cols])
                    insights = f"Aggregate Telemetry Data:\n{stats_text}"
                
                st.success("Diagnostics Complete")
                st.markdown("### Technical Findings:")
                st.info(insights)
                
            except Exception as e:
                st.error(f"Analysis Execution Fault: {e}")

# ============================================================================
# MODE 2: Individual Cell Analysis
# ============================================================================
elif mode == "Specific Cell Diagnostics":
    st.subheader("Cell-Level Performance Diagnostics")
    
    if "cell_id" in df.columns:
        cells = sorted(df["cell_id"].dropna().unique().tolist())
        selected_cell = st.selectbox("Select Target Cell ID:", cells)
        
        if st.button("Execute Diagnostics", type="primary"):
            # Compute operational metrics
            cell_df = df[df["cell_id"] == selected_cell]
            
            cell_stats = {
                'cell_id': selected_cell,
                'throughput': cell_df[cols.throughput].mean() if cols.throughput in cell_df.columns else 0,
                'prb_util': cell_df[cols.prb_dl].mean() if cols.prb_dl in cell_df.columns else 0,
                'ho_sr': cell_df[cols.ho_sr].mean() if cols.ho_sr in cell_df.columns else 0,
                'drop_rate': cell_df[cols.drop_rate].mean() if cols.drop_rate in cell_df.columns else 0,
                'rrc_users': cell_df[cols.rrc_users_max].mean() if cols.rrc_users_max in cell_df.columns else 0,
            }
            
            # Metric visualization dashboard
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Throughput", human_format(cell_stats['throughput'], "bps"))
            col2.metric("PRB Utilization", f"{cell_stats['prb_util']:.2%}")
            col3.metric("Handover SR", f"{cell_stats['ho_sr']:.2%}")
            col4.metric("Drop Rate", f"{cell_stats['drop_rate']:.4f}")
            
            st.divider()
            
            with st.spinner("Analyzing operational telemetry..."):
                try:
                    analysis = generator.analyze_cell_performance(cell_stats)
                    
                    st.markdown("### Diagnostic Report:")
                    st.info(analysis)
                    
                except Exception as e:
                    st.error(f"Diagnostic Execution Fault: {e}")
    else:
        st.warning("Data Schema Error: Cell ID column not detected.")

# ============================================================================
# MODE 3: Anomaly Explanation
# ============================================================================
elif mode == "KPI Anomaly Interpretation":
    st.subheader("KPI Anomaly Root Cause Analysis")
    
    st.markdown("""
    Contextual interpretation of identified KPI deviations and likely hardware/software causes.
    """)
    
    if "cell_id" in df.columns:
        cells = sorted(df["cell_id"].dropna().unique().tolist())
        selected_cell = st.selectbox("Select Target Cell ID:", cells)
        
        kpi_options = {
            "Drop Rate": cols.drop_rate,
            "Throughput": cols.throughput,
            "PRB Utilization": cols.prb_dl,
            "Handover Success Rate": cols.ho_sr,
        }
        
        selected_kpi_name = st.selectbox("Select Operational KPI:", list(kpi_options.keys()))
        selected_kpi = kpi_options[selected_kpi_name]
        
        # Anomaly thresholding
        threshold = st.number_input(f"Threshold for {selected_kpi_name}:", 
                                    value=0.01 if "drop" in selected_kpi_name.lower() else 0.5)
        
        if st.button("Interpret Anomaly", type="primary"):
            cell_df = df[df["cell_id"] == selected_cell]
            
            if selected_kpi in cell_df.columns:
                kpi_value = cell_df[selected_kpi].mean()
                
                context = {
                    'prb': cell_df[cols.prb_dl].mean() if cols.prb_dl in cell_df.columns else 0,
                    'users': cell_df[cols.rrc_users_max].mean() if cols.rrc_users_max in cell_df.columns else 0,
                    'traffic': cell_df[cols.traffic_dl_gb].mean() if cols.traffic_dl_gb in cell_df.columns else 0,
                }
                
                st.metric(f"Observed {selected_kpi_name}", 
                          human_format(kpi_value, "bps") if "Throughput" in selected_kpi_name else f"{kpi_value:.4f}")
                
                # Logical deviation check
                is_anomalous = False
                if "drop" in selected_kpi_name.lower() and kpi_value > threshold:
                    is_anomalous = True
                elif "throughput" in selected_kpi_name.lower() and kpi_value < threshold:
                    is_anomalous = True
                elif "success" in selected_kpi_name.lower() and kpi_value < threshold:
                    is_anomalous = True
                elif "util" in selected_kpi_name.lower() and kpi_value > threshold:
                    is_anomalous = True

                if is_anomalous:
                    with st.spinner("Correlating root causes..."):
                        try:
                            explanation = generator.explain_anomaly(
                                selected_cell, selected_kpi_name, kpi_value, threshold, context
                            )
                            
                            st.markdown("### Technical Interpretation:")
                            st.warning(explanation)
                            
                        except Exception as e:
                            st.error(f"Interpretation Failure: {e}")
                else:
                    st.success(f"Stability Verified: {selected_kpi_name} is within baseline parameters.")
            else:
                st.error(f"Parameter Mapping Error: {selected_kpi} not available in dataset.")
    else:
        st.warning("Data Schema Error: Cell identifier unavailable.")

# ============================================================================
# MODE 4: Optimization Suggestions
# ============================================================================
else:
    st.subheader("Network Parameter Optimization")
    
    st.markdown("""
    Technical recommendations for parameter tuning and infrastructure expansion 
    based on identified bottleneck classifications.
    """)
    
    bottleneck_type = st.selectbox(
        "Bottleneck Classification:",
        ["Capacity/Congestion", "RF Coverage", "Handover Configuration", "Drop Rate Mitigation", "Backhaul Optimization"]
    )
    
    severity = st.select_slider(
        "Criticality Level:",
        options=["Low", "Medium", "High", "Critical"]
    )
    
    if "cell_id" in df.columns:
        # Compute impact scale
        if bottleneck_type == "Capacity/Congestion":
            affected = (df[cols.prb_dl] > 0.85).sum() if cols.prb_dl in df.columns else 0
        elif bottleneck_type == "Handover Configuration":
            affected = (df[cols.ho_sr] < 0.95).sum() if cols.ho_sr in df.columns else 0
        else:
            affected = len(df["cell_id"].unique())
        
        st.metric("Estimated Impacted Cells", affected)
    else:
        affected = 0
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Computing optimization strategy..."):
            try:
                if hasattr(generator, 'suggest_optimization'):
                    suggestions = generator.suggest_optimization(
                        bottleneck_type, severity, affected
                    )
                else:
                    suggestions = (f"Baseline optimization strategy for {bottleneck_type} at {severity} criticality. "
                                   f"Impacted population: {affected} units. Reference standard RAN optimization procedures.")
                
                st.markdown("### Optimization Strategy:")
                st.success(suggestions)
                
            except Exception as e:
                st.error(f"Strategy Generation Fault: {e}")

# ============================================================================
# Operational Metadata
# ============================================================================
st.divider()
st.caption(f"Engine Configuration: {selected_model_name if use_llm else 'Legacy Rule Engine'} | "
           f"Analysis Context: {len(df):,} records")
