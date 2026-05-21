# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:00:00 2026

@author: hamza khlefat
"""

import os
import sys

# Configure environment for legacy Keras support to maintain compatibility with diagnostic libraries.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from src.config import DB
from src.db import read_sql

st.set_page_config(page_title="RAN Analytics", layout="wide")

st.title("Advanced RAN Analytical Intelligence Platform")

st.markdown("""
### Transforming Raw Telemetry into Strategic Intelligence
Welcome to the **RAN Intelligence Portal**. This platform serves as a centralized environment 
for large-scale network telemetry analysis, predictive modeling, and AI-driven diagnostic synthesis.
""")

col_m1, col_m2 = st.columns([2, 1])

with col_m1:
    st.header("Executive Overview")
    st.markdown("""
    In the modern telecommunications landscape, the ability to rapidly interpret multi-dimensional 
    Radio Access Network (RAN) performance data is a critical competitive advantage. 
    This system implements a "Neural-Augmented Loop" to support engineering teams in:
    *   **Proactive Failure Detection:** Identifying systemic degradation before it impacts the subscriber base.
    *   **Automated Root Cause Diagnosis:** Leveraging SHAP-based interpretability and local Large Language Models (LLMs) to isolate performance drivers.
    *   **Strategic Capacity Planning:** Utilizing "What-If" simulations and multivariate forecasting to optimize infrastructure investment.
    """)

with col_m2:
    db = DB()
    try:
        df_count = read_sql(db.path, f"SELECT COUNT(*) AS n FROM {db.table}")
        count = int(df_count['n'].iloc[0])
        st.info("### Platform Connectivity")
        st.metric("Operational Population", f"{count:,} Records")
        st.success("✅ Main Production Database Connected")
        st.caption("Last Sync: Real-time via SQLite Production Layer")
    except Exception as e:
        st.error("🚨 Database Connection Failure")
        st.warning(f"Technical Detail: {e}")
        st.info("Requirement: Execute `python scripts/build_db.py` to restore telemetry services.")

st.divider()

st.header("Core System Capabilities")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("🔍 Diagnostics")
    st.write("Multi-metric correlation analysis, outlier detection, and exception profiling for mobility and load.")

with c2:
    st.subheader("🤖 Predictive ML")
    st.write("Ensemble regression and forecasting models with built-in Explainable AI (SHAP) for technical clarity.")

with c3:
    st.subheader("🔮 Simulation")
    st.write("Dynamic 'What-If' sensitivity analysis environment to forecast the impact of parameter adjustments.")

with c4:
    st.subheader("🧠 Neural RAG")
    st.write("Llama 2-powered natural language query engine for semantic retrieval of cell performance artifacts.")

st.divider()

st.subheader("Navigation Architecture")
st.markdown("""
1.  **Data Explorer:** Assess data quality, ETL efficiency, and statistical distributions.
2.  **KPI Analysis:** Interactive visualizations and exception profiling.
3.  **ML Lab:** Supervised model training, evaluation, and SHAP explainability.
4.  **Root Cause:** Site-level cluster diagnostics and heuristic interpretation.
5.  **LLM Insights:** Multi-modal AI diagnostics and optimization recommendations.
6.  **RAG Query:** Neural semantic search and diagnostic synthesis.
7.  **Strategic Simulation:** Dynamic 'What-If' sensitivity analysis environment.
8.  **Executive Dashboard:** High-level strategic briefing and site priority scoring.
""")

st.caption("Developed by: Hamza Khlefat | [LinkedIn Profile](https://www.linkedin.com/in/hamza-khlefat/) | Contact: +962799693068 | Email: hamzakhliefat@yahoo.com")
