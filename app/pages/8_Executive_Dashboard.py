# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 15:01:39 2026

@author: hamza khlefat
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Resolve root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import Cols, DB, Thresholds
from src.db import read_sql
from src.llm_insights import get_insights_generator

st.set_page_config(page_title="Executive Analytics", layout="wide")
st.title("Executive Network Intelligence Dashboard")

cols = Cols()
db = DB()
th = Thresholds()

st.markdown("""
### Strategic Decision Support
A high-level synthesis of network performance telemetry, identifying critical intervention 
points and providing automated executive summaries driven by Large Language Models.
""")

def human_format(num, base_unit="bps"):
    """Format large numbers into human-readable strings with logical unit transitions."""
    magnitude = 0
    if base_unit in ["bps", "Mbps", "kbps"]:
        units = ["bps", "Kbps", "Mbps", "Gbps", "Tbps"]
    else:
        units = [base_unit, f"K{base_unit}", f"M{base_unit}", f"G{base_unit}"]

    while abs(num) >= 1000 and magnitude < len(units) - 1:
        magnitude += 1
        num /= 1000.0
    
    return f"{num:.2f} {units[magnitude]}".strip()

# ============================================================================
# DATA AGGREGATION & PRIORITY SCORING
# ============================================================================
@st.cache_data(ttl=600)
def get_executive_summary_data():
    """Aggregate site-level statistics and compute priority scores."""
    df = read_sql(db.path, f"SELECT * FROM {db.table}")
    
    # Site-level aggregation
    site_agg = df.groupby(cols.site_id).agg({
        cols.throughput: 'mean',
        cols.prb_dl: 'mean',
        cols.ho_sr: 'mean' if cols.ho_sr in df.columns else 'count',
        cols.drop_rate: 'mean' if cols.drop_rate in df.columns else 'count',
    }).reset_index()
    
    # Normalize metrics for scoring (0-1 scale)
    def normalize(s, reverse=False):
        if s.max() == s.min(): return s * 0
        norm = (s - s.min()) / (s.max() - s.min())
        return 1 - norm if reverse else norm

    # Score components (Higher = Worse)
    s_tp = normalize(site_agg[cols.throughput], reverse=True)
    s_prb = normalize(site_agg[cols.prb_dl])
    s_ho = normalize(site_agg[cols.ho_sr], reverse=True) if cols.ho_sr in site_agg.columns else 0
    s_dr = normalize(site_agg[cols.drop_rate]) if cols.drop_rate in site_agg.columns else 0
    
    # Weighted priority score
    site_agg['Priority_Score'] = (s_tp * 0.3) + (s_prb * 0.3) + (s_ho * 0.2) + (s_dr * 0.2)
    return site_agg.sort_values('Priority_Score', ascending=False)

with st.spinner("Synthesizing network telemetry..."):
    site_summary = get_executive_summary_data()

# ============================================================================
# KEY NETWORK INDICATORS (KNI)
# ============================================================================
st.header("1. Core Network Indicators (Aggregate)")
k1, k2, k3, k4 = st.columns(4)

avg_tp = site_summary[cols.throughput].mean()
avg_prb = site_summary[cols.prb_dl].mean()
avg_ho = site_summary[cols.ho_sr].mean() if cols.ho_sr in site_summary.columns else 0
avg_dr = site_summary[cols.drop_rate].mean() if cols.drop_rate in site_summary.columns else 0

k1.metric("Network Avg Throughput", human_format(avg_tp, "bps"))
k2.metric("Network PRB Load", f"{avg_prb:.2%}")
k3.metric("Network Mobility (HO SR)", f"{avg_ho:.2%}")
k4.metric("Network Drop Rate", f"{avg_dr:.4f}")

# ============================================================================
# TOP PRIORITY SITES
# ============================================================================
st.divider()
st.header("2. Strategic Priority Watchlist")
st.markdown("Sites identified for immediate optimization based on multi-dimensional performance degradation.")

top_5 = site_summary.head(5).copy()
st.dataframe(top_5, use_container_width=True)

# Scatter plot of Priority vs Throughput
fig_priority = px.scatter(
    site_summary, 
    x=cols.throughput, 
    y='Priority_Score',
    color=cols.prb_dl,
    hover_data=[cols.site_id],
    title="Site Priority Distribution vs. Throughput Capacity",
    color_continuous_scale="RdYlGn_r"
)
st.plotly_chart(fig_priority, use_container_width=True)

# ============================================================================
# LLM EXECUTIVE BRIEF
# ============================================================================
st.divider()
st.header("3. Automated Executive Briefing")

@st.cache_resource
def get_exec_brief_generator():
    return get_insights_generator(use_ollama=True, model_name="llama2")

if st.button("Generate Executive Brief", type="primary"):
    with st.spinner("Consulting intelligence engine..."):
        try:
            generator = get_exec_brief_generator()
            
            # Prepare context for LLM
            context = f"""
            Network-wide stats:
            - Avg Throughput: {avg_tp:.2f} Mbps
            - Avg PRB Load: {avg_prb:.2%}
            - Avg Handover Success: {avg_ho:.2%}
            - Avg Drop Rate: {avg_dr:.4f}
            
            Top 3 Problematic Sites:
            1. Site {top_5.iloc[0][cols.site_id]}: Priority Score {top_5.iloc[0]['Priority_Score']:.2f}
            2. Site {top_5.iloc[1][cols.site_id]}: Priority Score {top_5.iloc[1]['Priority_Score']:.2f}
            3. Site {top_5.iloc[2][cols.site_id]}: Priority Score {top_5.iloc[2]['Priority_Score']:.2f}
            """
            
            prompt = f"As a Senior Network Executive, provide a 3-paragraph summary of these network KPIs. Focus on overall health, specific risks at the top problematic sites, and strategic recommendations for the next quarter. Context: {context}"
            
            brief = generator.generate_text(prompt)
            
            st.info(brief)
            
        except Exception as e:
            st.error(f"Inference Failure: {e}")
            st.warning("Ensure Ollama is running and Llama 2 is pulled.")

# ============================================================================
# ACTIONABLE RECOMMENDATIONS
# ============================================================================
st.divider()
st.header("4. Strategic Action Items")
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Capacity & Infrastructure")
    if avg_prb > 0.7:
        st.warning("🚨 High Network Load: Expansion of PRB capacity or small-cell offloading recommended.")
    else:
        st.success("✅ Capacity Stable: Current resource allocation is sufficient for existing traffic.")
    
    critical_tp = (site_summary[cols.throughput] < 5).sum()
    if critical_tp > 0:
        st.error(f"🚨 Throughput Bottlenecks: {critical_tp} sites reported throughput < 5Mbps. Urgent backhaul audit required.")

with c2:
    st.markdown("### Quality & Mobility")
    if avg_ho < 0.95:
        st.warning("⚠️ Mobility Issues: Network-wide HO SR is below target. Optimization of neighbor relations recommended.")
    
    if avg_dr > 0.005:
        st.error("🚨 Critical Drops: High drop rate detected. Investigate RF interference at identified priority sites.")

st.divider()
st.caption("Intelligence Source: Heuristic Scoring Engine + Ollama LLM")
