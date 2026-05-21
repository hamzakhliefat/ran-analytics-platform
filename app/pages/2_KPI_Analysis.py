# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 09:45:30 2026

@author: hamza khlefat
"""

import os
import sys

# Standard module path resolution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.config import Cols, DB, Thresholds
from src.db import read_sql
import plotly.express as px
from src.viz_trends import interactive_trend, aggregate_by_time


st.set_page_config(page_title="KPI Performance Analysis", layout="wide")
st.title("Network Performance & KPI Analysis")

cols = Cols()
db = DB()
th = Thresholds()

# ============================================================================
# OPERATIONAL CONFIGURATION (SIDEBAR)
# ============================================================================
st.sidebar.header("Analysis Parameters")

sample_n = st.sidebar.slider("Sampling Volume (Randomized)", 5000, 1500000, 50000, 5000)
group_level = st.sidebar.selectbox("Aggregation Granularity", ["gov", "site_id"])
date_mode = st.sidebar.selectbox("Temporal Filtering", ["All Records", "Last 7 Days", "Last 30 Days", "Custom Range"])

gov_filter = st.sidebar.text_input("Filter by Governance (Optional)", "")
site_filter = st.sidebar.text_input("Filter by Site ID (Optional)", "")

custom_from = None
custom_to = None
if date_mode == "Custom Range":
    c1, c2 = st.sidebar.columns(2)
    with c1:
        custom_from = st.date_input("Start Date")
    with c2:
        custom_to = st.date_input("End Date")

st.sidebar.divider()
st.sidebar.subheader("Operational Thresholds")
ho_thr = st.sidebar.number_input("Handover SR (Low) Threshold", 0.50, 0.999, float(th.ho_sr_low), 0.01)
prb_thr = st.sidebar.number_input("PRB Utilization (High) Threshold", 0.10, 0.999, float(th.prb_high), 0.01)
drop_q = st.sidebar.number_input("Drop Rate Anomaly Quantile", 0.80, 0.999, float(th.drop_anomaly_quantile), 0.01)

kpi_cols = [c for c in [cols.throughput, cols.prb_dl, cols.drop_rate, cols.ho_sr, cols.rrc_users_max] if c]


def load_sample() -> pd.DataFrame:
    """Ingest dataset with applied spatial and temporal filters."""
    where = []
    params = []

    if gov_filter.strip():
        where.append(f"{cols.gov} = ?")
        params.append(gov_filter.strip())

    if site_filter.strip():
        where.append(f"{cols.site_id} = ?")
        params.append(site_filter.strip())

    if date_mode == "Last 7 Days":
        where.append(f"{cols.date} >= datetime('now', '-7 day')")

    if date_mode == "Last 30 Days":
        where.append(f"{cols.date} >= datetime('now', '-30 day')")

    if date_mode == "Custom Range" and custom_from and custom_to:
        where.append(f"{cols.date} >= ? AND {cols.date} <= ?")
        params.append(str(custom_from))
        params.append(str(custom_to))

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    query = f"SELECT * FROM {db.table} {where_sql} ORDER BY RANDOM() LIMIT ?"
    params.append(int(sample_n))

    df = read_sql(db.path, query, tuple(params) if params else None)
    return df


df = load_sample()

if df.empty:
    st.warning("Query Optimization Required: No records matched the specified filter criteria.")
    st.stop()

st.caption(f"Operational Sample Size: {len(df):,} records")

missing = [c for c in [cols.cell_id, cols.site_id, cols.gov, cols.date, cols.throughput, cols.prb_dl] if c not in df.columns]
if missing:
    st.error(f"Schema Integrity Error: Missing required columns in dataset - {missing}")
    st.stop()

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
# KPI DESCRIPTIVE CARDS
# ============================================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Mean Throughput", human_format(df[cols.throughput].mean(), "bps"))
with c2:
    st.metric("Mean PRB DL Util", f"{df[cols.prb_dl].mean():.3f}")
with c3:
    if cols.ho_sr in df.columns:
        st.metric("Mean Handover SR", f"{df[cols.ho_sr].mean():.3f}")
    else:
        st.metric("Mean Handover SR", "N/A")
with c4:
    if cols.drop_rate in df.columns:
        st.metric("Mean Drop Rate", f"{df[cols.drop_rate].mean():.4f}")
    else:
        st.metric("Mean Drop Rate", "N/A")

st.divider()

# ============================================================================
# STATISTICAL SUMMARY BY DIMENSION
# ============================================================================
st.subheader("Summary Statistics by Aggregation Level")

group_col = cols.gov if group_level == "gov" else cols.site_id
agg_cols = [c for c in [cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max] if c in df.columns]

summary = (
    df.groupby(group_col)[agg_cols]
      .agg(["mean", "min", "max"])
      .reset_index()
)

st.dataframe(summary, use_container_width=True)

csv_bytes = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    "Export Group Summary (CSV)",
    data=csv_bytes,
    file_name=f"summary_by_{group_level}.csv",
    mime="text/csv"
)

st.divider()

# ============================================================================
# KPI INTER-CORRELATION
# ============================================================================
st.subheader("KPI Interaction (Correlation Heatmap)")

corr_cols = [c for c in [cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max] if c in df.columns]
corr_df = df[corr_cols].corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Inter-Metric Correlation Analysis")
st.pyplot(fig)

def generate_correlation_diagnostics(corr):
    """Generate technical narratives based on significant Pearson correlations."""
    insights = []
    
    # Define significant metric interactions
    threshold = 0.35
    
    # 1. Congestion Check (Throughput vs PRB)
    if cols.throughput in corr.index and cols.prb_dl in corr.columns:
        r = corr.loc[cols.throughput, cols.prb_dl]
        if r < -threshold:
            insights.append(f"**Congestion Signal:** High negative correlation ({r:.2f}) between Throughput and PRB Utilization detected. This typically indicates that resource saturation is actively depressing user throughput.")
        elif r > threshold:
            # Unusual but possible in some scheduling scenarios
            insights.append(f"**Anomalous Scaling:** Positive correlation ({r:.2f}) between Throughput and PRB. This may indicate efficient resource allocation or high-rank MIMO conditions under load.")

    # 2. Load Check (RRC Users vs PRB)
    if cols.rrc_users_max in corr.index and cols.prb_dl in corr.columns:
        r = corr.loc[cols.rrc_users_max, cols.prb_dl]
        if r > threshold:
            insights.append(f"**User-Driven Load:** RRC User count and PRB Utilization show strong positive correlation ({r:.2f}). Current resource consumption is primarily driven by active user volume.")

    # 3. Quality vs Mobility (HO_SR vs Drop Rate)
    if cols.ho_sr in corr.index and cols.drop_rate in corr.columns:
        r = corr.loc[cols.ho_sr, cols.drop_rate]
        if r < -threshold:
            insights.append(f"**Mobility Impairment:** Negative correlation ({r:.2f}) between Handover Success and Service Drop Rate. Mobility failures (e.g., failed handovers) appear to be a primary contributor to call drops.")

    if not insights:
        return "Numerical Analysis: No statistically significant (|r| > 0.35) linear interactions detected among primary KPIs in the current sample."
    
    return "\n\n".join(insights)

st.info("**Diagnostic Interpretation:**\n\n" + generate_correlation_diagnostics(corr_df))

st.divider()

# ============================================================================
# PERFORMANCE DEVIATION DETECTION
# ============================================================================
st.subheader("Underperforming Observations")

mask = pd.Series(False, index=df.index)

if cols.ho_sr in df.columns:
    mask = mask | (df[cols.ho_sr] < ho_thr)

mask = mask | (df[cols.prb_dl] > prb_thr)

if cols.drop_rate in df.columns:
    drop_thr = df[cols.drop_rate].quantile(drop_q)
    mask = mask | (df[cols.drop_rate] >= drop_thr)
else:
    drop_thr = None

bad = df[mask].copy()

show_cols = [c for c in [cols.cell_id, cols.site_id, cols.gov, cols.date, cols.ho_sr, cols.prb_dl, cols.drop_rate, cols.throughput, cols.rrc_users_max] if c in bad.columns]
bad = bad[show_cols]

if cols.ho_sr in bad.columns:
    bad = bad.sort_values(cols.ho_sr, ascending=True)
else:
    bad = bad.sort_values(cols.prb_dl, ascending=False)

st.write(f"Observations Flagged for Review: {len(bad):,} (Sample-based)")
st.dataframe(bad.head(100), use_container_width=True)

bad_csv = bad.head(5000).to_csv(index=False).encode("utf-8")
st.download_button(
    "Export Identified Exceptions (CSV)",
    data=bad_csv,
    file_name="underperforming_cells_sample.csv",
    mime="text/csv"
)

st.divider()

# ============================================================================
# EXCEPTION BREAKDOWN
# ============================================================================
st.subheader("Reason Code Distribution (Sample)")

reason_counts = {
    "Handover Failure": int(((df[cols.ho_sr] < ho_thr) if cols.ho_sr in df.columns else pd.Series(False, index=df.index)).sum()),
    "Resource Congestion (PRB)": int((df[cols.prb_dl] > prb_thr).sum()),
}

if cols.drop_rate in df.columns:
    thr_drop = df[cols.drop_rate].quantile(drop_q)
    reason_counts[f"Anomalous Drop Rate (P{int(drop_q*100)})"] = int((df[cols.drop_rate] >= thr_drop).sum())

reason_df = pd.DataFrame({"Classification": list(reason_counts.keys()), "Frequency": list(reason_counts.values())})

fig_reasons = px.bar(
    reason_df,
    x="Classification",
    y="Frequency",
    title="Exception Frequency by Violation Rule",
    labels={"Frequency": "Observation Count", "Classification": "Failure Mode"},
    color="Frequency",
    color_continuous_scale="Reds"
)

fig_reasons.update_layout(
    height=400,
    margin=dict(l=30, r=30, t=60, b=30),
    xaxis_tickangle=-15,
)

st.plotly_chart(fig_reasons, use_container_width=True)

st.divider()

# ============================================================================
# TEMPORAL KPI ANALYSIS
# ============================================================================
st.header("Temporal KPI Trends")

view_mode = st.radio(
    "Visualization Perspective:",
    ["Individual Cell", "Multi-Cell (Comparative)", "Network Aggregate"],
    horizontal=True,
)

max_rows = st.sidebar.slider("Maximum Temporal Data Points", min_value=200, max_value=5000, value=400, step=100)

base_cols = [cols.date, cols.cell_id, cols.site_id, cols.gov, cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max]
base_cols = [c for c in base_cols if c in df.columns]

dft = df[base_cols].copy()
dft = dft.dropna(subset=[cols.date]).sort_values(cols.date)

if view_mode == "Individual Cell":
    cell_list = dft[cols.cell_id].dropna().astype(int).unique().tolist()
    cell_id = st.selectbox("Select Target Cell Identifier:", options=cell_list)
    cdf = dft[dft[cols.cell_id] == cell_id].tail(max_rows)

    fig1 = interactive_trend(
        cdf,
        date_col=cols.date,
        series=[(cols.throughput, "Observed Throughput")],
        title=f"Temporal Trend: Throughput (Cell {cell_id})",
        hlines=[],
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = interactive_trend(
        cdf,
        date_col=cols.date,
        series=[(cols.prb_dl, "PRB DL Utilization")],
        title=f"Temporal Trend: Resource Consumption (Cell {cell_id})",
        hlines=[(prb_thr, "Operational Threshold (High PRB)")],
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = interactive_trend(
        cdf,
        date_col=cols.date,
        series=[(cols.ho_sr, "Handover Success Rate")],
        title=f"Temporal Trend: Mobility Performance (Cell {cell_id})",
        hlines=[(ho_thr, "Operational Threshold (Low HO_SR)")],
    )
    st.plotly_chart(fig3, use_container_width=True)

elif view_mode == "Multi-Cell (Comparative)":
    top_n = st.slider("Comparative Sample Size (N Cells)", min_value=3, max_value=20, value=8, step=1)

    tmp = dft.dropna(subset=[cols.throughput]).copy()
    tmp["score"] = tmp[cols.throughput]
    worst_cells = (
        tmp.groupby(cols.cell_id, as_index=False)["score"]
        .median()
        .sort_values("score", ascending=True)
        .head(top_n)[cols.cell_id]
        .astype(int)
        .tolist()
    )

    mdf = dft[dft[cols.cell_id].isin(worst_cells)].copy()
    mdf = mdf.groupby([cols.cell_id], group_keys=False).tail(max_rows)

    fig = interactive_trend(
        mdf,
        date_col=cols.date,
        series=[(cols.throughput, "Observed Throughput")],
        title=f"Comparative Analysis: Throughput (Lowest {top_n} Cells by Median)",
        hlines=[],
        group_col=cols.cell_id,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_prb = interactive_trend(
        mdf,
        date_col=cols.date,
        series=[(cols.prb_dl, "PRB DL Utilization")],
        title=f"Comparative Analysis: Resource Consumption (Lowest {top_n} Cells)",
        hlines=[(prb_thr, "Utilization Threshold")],
        group_col=cols.cell_id,
    )
    st.plotly_chart(fig_prb, use_container_width=True)

else:
    agg_method = st.selectbox("Aggregation Strategy", options=["median", "mean"], index=0)

    agg_df = aggregate_by_time(
        dft,
        date_col=cols.date,
        group_cols=[cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max],
        agg=agg_method,
    ).tail(2000)

    fig = interactive_trend(
        agg_df,
        date_col=cols.date,
        series=[(cols.throughput, f"Throughput ({agg_method.capitalize()})")],
        title="Network-Wide Temporal Trend: Throughput",
        hlines=[],
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = interactive_trend(
        agg_df,
        date_col=cols.date,
        series=[(cols.prb_dl, f"PRB DL ({agg_method.capitalize()})")],
        title="Network-Wide Temporal Trend: Resource Consumption",
        hlines=[(prb_thr, "Utilization Threshold")],
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = interactive_trend(
        agg_df,
        date_col=cols.date,
        series=[(cols.ho_sr, f"HO_SR ({agg_method.capitalize()})")],
        title="Network-Wide Temporal Trend: Mobility Performance",
        hlines=[(ho_thr, "Mobility Threshold")],
    )
    st.plotly_chart(fig3, use_container_width=True)


st.divider()
st.header("Inter-KPI Relationship Explorer")

left, right = st.columns([1, 1])

with left:
    sample_n_sc = st.slider("Visualization Population", min_value=5_000, max_value=200_000, value=30_000, step=5_000)
    x_metric = st.selectbox("Abscissa Axis (X)", options=[cols.prb_dl, cols.rrc_users_max, cols.drop_rate], index=0)
    y_metric = st.selectbox("Ordinate Axis (Y)", options=[cols.throughput, cols.ho_sr, cols.drop_rate], index=0)

with right:
    color_metric = st.selectbox("Dimensional Coloring", options=[cols.drop_rate, cols.prb_dl, cols.ho_sr], index=0)
    size_metric = st.selectbox("Observation Scaling (Size)", options=[cols.rrc_users_max, cols.prb_dl, cols.drop_rate], index=0)
    show_trendline = st.checkbox("Execute Statistical Trend Overlay (OLS)", value=True)

need_cols = [cols.date, cols.cell_id, cols.site_id, cols.gov, x_metric, y_metric, color_metric, size_metric]
need_cols = [c for c in need_cols if c in df.columns]

sc = df[need_cols].copy()

for c in [x_metric, y_metric, color_metric, size_metric]:
    if c in sc.columns:
        sc[c] = pd.to_numeric(sc[c], errors="coerce")

sc = sc.dropna(subset=[x_metric, y_metric])

if cols.gov in sc.columns:
    govs = ["Aggregate"] + sorted(sc[cols.gov].dropna().unique().tolist())
    sel_gov = st.selectbox("Spatial Filtering (Governance)", options=govs, index=0)
    if sel_gov != "Aggregate":
        sc = sc[sc[cols.gov] == sel_gov]

if cols.site_id in sc.columns:
    site_text = st.text_input("Operational Site ID Filter (Optional)", value="").strip()
    if site_text:
        sc = sc[sc[cols.site_id].astype(str).str.contains(site_text)]

if len(sc) > sample_n_sc:
    sc = sc.sample(n=sample_n_sc, random_state=42)

hover_cols = []
for c in [cols.cell_id, cols.site_id, cols.gov, cols.date]:
    if c in sc.columns:
        hover_cols.append(c)

trendline_opt = "ols" if show_trendline else None

fig_scatter = px.scatter(
    sc,
    x=x_metric,
    y=y_metric,
    color=color_metric if color_metric in sc.columns else None,
    size=size_metric if size_metric in sc.columns else None,
    hover_data=hover_cols,
    trendline=trendline_opt,
)

if x_metric == cols.prb_dl:
    fig_scatter.add_vline(x=prb_thr, line_dash="dash", annotation_text="Congestion Threshold")

if y_metric == cols.ho_sr:
    fig_scatter.add_hline(y=ho_thr, line_dash="dash", annotation_text="Mobility Threshold")


fig_scatter.update_layout(
    height=620,
    margin=dict(l=30, r=30, t=40, b=30),
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.caption(
    "Operational Insight: PRB vs. Throughput correlation typically indicates resource saturation (high utilization paired with throughput degradation). "
    "Secondary coloring by drop rate facilitates the isolation of RF quality impairments from capacity-driven bottlenecks."
)




metric_options = {
    "Throughput (Median)": (cols.throughput, "median", True),
    "PRB DL Utilization (Median)": (cols.prb_dl, "median", False),
    "Handover Success Rate (Mean)": (cols.ho_sr, "mean", True),
    "Anomaly Drop Rate (Mean)": (cols.drop_rate, "mean", False),
}

col_a, col_b, col_c = st.columns([1, 1, 1])

with col_a:
    level = st.selectbox("Ranking Granularity", options=["cell_id", "site_id"], index=0)

with col_b:
    metric_label = st.selectbox("Optimization Metric", options=list(metric_options.keys()), index=0)

with col_c:
    top_k = st.slider("Subset Size (Top K)", min_value=10, max_value=100, value=20, step=10)

metric_col, agg_func, ascending = metric_options[metric_label]

rank_cols = [level, metric_col]
for c in [cols.gov, cols.site_id, cols.cell_id]:
    if c in df.columns and c not in rank_cols:
        rank_cols.append(c)

rank_df = df[rank_cols].copy()
rank_df[metric_col] = pd.to_numeric(rank_df[metric_col], errors="coerce")
rank_df = rank_df.dropna(subset=[metric_col])

if cols.gov in rank_df.columns:
    govs = ["Aggregate"] + sorted(rank_df[cols.gov].dropna().unique().tolist())
    gov_filter_rank = st.selectbox("Spatial Filter (Governance)", options=govs, index=0, key="rank_gov_filter")
    if gov_filter_rank != "Aggregate":
        rank_df = rank_df[rank_df[cols.gov] == gov_filter_rank]

group_keys = [level]
if level == "cell_id" and cols.site_id in rank_df.columns:
    group_keys.append(cols.site_id)
if cols.gov in rank_df.columns:
    group_keys.append(cols.gov)

ranked = rank_df.groupby(group_keys, as_index=False).agg({metric_col: agg_func})
ranked = ranked.sort_values(metric_col, ascending=ascending).head(top_k)

ranked = ranked.rename(columns={metric_col: f"{metric_col}_{agg_func}"})

st.dataframe(ranked, use_container_width=True)

csv_bytes = ranked.to_csv(index=False).encode("utf-8")
st.download_button(
    "Export Ranking Data (CSV)",
    data=csv_bytes,
    file_name=f"leaderboard_{level}_{metric_col}_{agg_func}.csv",
    mime="text/csv",
)



st.divider()
st.header("Comparative Performance: Temporal Shift Analysis")

entity_mode = st.selectbox(
    "Entity Target Type",
    options=["cell_id", "site_id"],
    index=0,
    key="deg_entity_mode"
)

kpi_choices = [cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max]
kpi_choices = [c for c in kpi_choices if c in df.columns]

target_kpi = st.selectbox(
    "Metric of Interest",
    options=kpi_choices,
    index=0,
    key="deg_target_kpi"
)

need_cols = [cols.date, entity_mode, target_kpi]
for c in [cols.gov, cols.site_id, cols.cell_id]:
    if c in df.columns and c not in need_cols:
        need_cols.append(c)

dd = df[need_cols].copy()
dd[cols.date] = pd.to_datetime(dd[cols.date], errors="coerce")
dd[target_kpi] = pd.to_numeric(dd[target_kpi], errors="coerce")
dd = dd.dropna(subset=[cols.date, entity_mode, target_kpi]).sort_values(cols.date)


entities = dd[entity_mode].dropna().astype(str).unique().tolist()
entities = sorted(entities)

selected_entity = st.selectbox(
    f"Select Target {entity_mode}",
    options=entities,
    key=f"deg_selected_{entity_mode}"
)

edf = dd[dd[entity_mode].astype(str) == str(selected_entity)].copy()

if edf.empty:
    st.warning("Data Availability: No records found for the selected entity.")
else:
    min_dt = edf[cols.date].min()
    max_dt = edf[cols.date].max()

    split_dt = st.date_input(
        "Observation Reference Date (Degradation Start)",
        value=max_dt.date(),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
        key="deg_split_date"
    )
    split_ts = pd.to_datetime(split_dt)

    window_days = st.slider(
        "Analysis Window (Days) Pre/Post Reference",
        min_value=3,
        max_value=60,
        value=14,
        step=1,
        key="deg_window_days"
    )

    before_start = split_ts - pd.Timedelta(days=window_days)
    after_end = split_ts + pd.Timedelta(days=window_days)

    wdf = edf[(edf[cols.date] >= before_start) & (edf[cols.date] <= after_end)].copy()

    before = wdf[wdf[cols.date] < split_ts][target_kpi]
    after = wdf[wdf[cols.date] >= split_ts][target_kpi]

    def _stats(s: pd.Series) -> dict:
        """Compute statistical primitives for shift comparison."""
        return {
            "observation_count": int(s.count()),
            "arithmetic_mean": float(s.mean()) if s.count() else np.nan,
            "median_val": float(s.median()) if s.count() else np.nan,
            "absolute_min": float(s.min()) if s.count() else np.nan,
            "absolute_max": float(s.max()) if s.count() else np.nan,
        }

    before_stats = _stats(before)
    after_stats = _stats(after)

    compare = pd.DataFrame([before_stats, after_stats], index=["Baseline (Pre)", "Observation (Post)"])\
        .reset_index().rename(columns={"index": "Phase"})

    base_mean = before_stats["arithmetic_mean"]
    new_mean = after_stats["arithmetic_mean"]
    delta_pct = np.nan
    if base_mean is not None and not np.isnan(base_mean) and base_mean != 0 and not np.isnan(new_mean):
        delta_pct = (new_mean - base_mean) / base_mean * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Mean (Pre)", f"{before_stats['arithmetic_mean']:.4g}" if not np.isnan(before_stats["arithmetic_mean"]) else "NA")
    c2.metric("Observation Mean (Post)", f"{after_stats['arithmetic_mean']:.4g}" if not np.isnan(after_stats["arithmetic_mean"]) else "NA")
    c3.metric("Magnitude Delta %", f"{delta_pct:.2f}%" if not np.isnan(delta_pct) else "NA")

    st.subheader("Statistical Summary Table")
    st.dataframe(compare, use_container_width=True)

    st.subheader("Temporal Trend Analysis: Pre vs. Post Shift")
    fig_deg = px.line(
        wdf.sort_values(cols.date),
        x=cols.date,
        y=target_kpi,
        title=f"Temporal Displacement: {target_kpi} ({entity_mode}={selected_entity})",
    )

    
    split_x = split_ts.to_pydatetime()
    
    wdf[cols.date] = pd.to_datetime(wdf[cols.date], errors="coerce")
    
    split_ts = pd.to_datetime(split_dt)    
    split_str = split_ts.strftime("%Y-%m-%d %H:%M:%S")  
    fig_deg.update_xaxes(type="date")

    fig_deg = px.line(
        wdf.sort_values(cols.date),
        x=cols.date,
        y=target_kpi,
        title=f"Temporal Displacement: {target_kpi} ({entity_mode}={selected_entity})",
    )
    
  
    fig_deg.add_shape(
        type="line",
        x0=split_str,
        x1=split_str,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(dash="dash", width=2, color="red"),
    )
    
   
    fig_deg.add_annotation(
        x=split_str,
        y=1,
        xref="x",
        yref="paper",
        text="Displacement Event Start",
        showarrow=False,
        yanchor="bottom",
    )
    
    fig_deg.update_layout(height=520, margin=dict(l=30, r=30, t=60, b=30))
    st.plotly_chart(fig_deg, use_container_width=True)

    

    out_csv = wdf[[cols.date, entity_mode, target_kpi]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Windowed Telemetry (CSV)",
        data=out_csv,
        file_name=f"degradation_window_{entity_mode}_{selected_entity}_{target_kpi}.csv",
        mime="text/csv",
    )



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, f1_score

st.divider()
st.header("Diagnostic Evidence & RCA Framework")

entity_mode_rca = st.selectbox("Entity Granularity", options=["cell_id", "site_id"], index=0, key="rca_entity")
kpi_choices_rca = [cols.throughput, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max]
kpi_choices_rca = [c for c in kpi_choices_rca if c in df.columns]
target_kpi_rca = st.selectbox("Diagnostic Target (KPI)", options=kpi_choices_rca, index=0, key="rca_target")

need_cols = [cols.date, entity_mode_rca, target_kpi_rca]
for c in [cols.gov, cols.site_id, cols.cell_id, cols.prb_dl, cols.ho_sr, cols.drop_rate, cols.rrc_users_max]:
    if c in df.columns and c not in need_cols:
        need_cols.append(c)

rca_df = df[need_cols].copy()
rca_df[cols.date] = pd.to_datetime(rca_df[cols.date], errors="coerce")
rca_df[target_kpi_rca] = pd.to_numeric(rca_df[target_kpi_rca], errors="coerce")
rca_df = rca_df.dropna(subset=[cols.date, entity_mode_rca, target_kpi_rca]).sort_values(cols.date)

entities_rca = rca_df[entity_mode_rca].dropna().unique().tolist()
selected_entity_rca = st.selectbox(f"Select Target {entity_mode_rca}", options=entities_rca, key="rca_selected")

edf = rca_df[rca_df[entity_mode_rca] == selected_entity_rca].copy()

if edf.empty:
    st.warning("Data Availability: No records found for the selected diagnostic target.")
    st.stop()

min_dt = edf[cols.date].min()
max_dt = edf[cols.date].max()

col_r1, col_r2, col_r3 = st.columns([1, 1, 1])
with col_r1:
    split_dt = st.date_input(
        "Reference Baseline Date",
        value=max_dt.date(),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
        key="rca_ref_date",
    )
with col_r2:
    window_days = st.slider("Diagnostic Window (Days)", min_value=3, max_value=60, value=14, step=1, key="rca_window")
with col_r3:
    top_drivers = st.slider("Feature Subset (Top Drivers)", min_value=5, max_value=20, value=10, step=1, key="rca_top")

split_ts = pd.to_datetime(split_dt)
start_ts = split_ts - pd.Timedelta(days=window_days)
end_ts = split_ts + pd.Timedelta(days=window_days)

wdf = edf[(edf[cols.date] >= start_ts) & (edf[cols.date] <= end_ts)].copy()
if wdf.empty:
    st.warning("Query Result: No observations found within the selected temporal window.")
    st.stop()

st.caption(f"Active Diagnostic Observations: {len(wdf):,}")

# ----- Rule-Based Heuristics (Evidence Tally) -----
rules = []

if cols.ho_sr in wdf.columns:
    wdf[cols.ho_sr] = pd.to_numeric(wdf[cols.ho_sr], errors="coerce")
    low_ho = (wdf[cols.ho_sr] < ho_thr).sum()
    rules.append(("Mobility Violation (Low HO_SR)", int(low_ho)))

if cols.prb_dl in wdf.columns:
    wdf[cols.prb_dl] = pd.to_numeric(wdf[cols.prb_dl], errors="coerce")
    high_prb = (wdf[cols.prb_dl] > prb_thr).sum()
    rules.append(("Saturation Event (High PRB)", int(high_prb)))

if cols.drop_rate in wdf.columns:
    wdf[cols.drop_rate] = pd.to_numeric(wdf[cols.drop_rate], errors="coerce")
    dr = wdf[cols.drop_rate].dropna()
    if len(dr) > 0:
        dr_thr = float(np.nanpercentile(dr.values, 95))
        high_dr = (wdf[cols.drop_rate] > dr_thr).sum()
        rules.append(("Critical Call Drop (P95 Threshold)", int(high_dr)))
    else:
        dr_thr = np.nan

rule_df = pd.DataFrame(rules, columns=["Classification Rule", "Frequency"]).sort_values("Frequency", ascending=False)

st.subheader("Heuristic-Based Evidence Dashboard")
c1, c2 = st.columns([1, 2])
with c1:
    st.dataframe(rule_df, use_container_width=True)
with c2:
    if not rule_df.empty:
        fig_rules = px.bar(rule_df, x="Classification Rule", y="Frequency", title="Heuristic Rule Trigger Frequency")
        fig_rules.update_layout(height=360, margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig_rules, use_container_width=True)

# ----- Statistical Correlation Analysis -----
st.subheader("Statistical Drivers: Correlation Analysis")

numeric_cols = []
for c in wdf.columns:
    if c in [cols.date, entity_mode_rca, cols.gov, cols.site_id, cols.cell_id]:
        continue
    try:
        wdf[c] = pd.to_numeric(wdf[c], errors="coerce")
        numeric_cols.append(c)
    except Exception:
        pass

corr_table = pd.DataFrame()
if target_kpi_rca in numeric_cols and len(numeric_cols) >= 2:
    corr = wdf[numeric_cols].corr(numeric_only=True)
    if target_kpi_rca in corr.columns:
        s = corr[target_kpi_rca].dropna().drop(labels=[target_kpi_rca], errors="ignore")
        s = s.reindex(s.abs().sort_values(ascending=False).index).head(top_drivers)
        corr_table = s.reset_index()
        corr_table.columns = ["Target Feature", "Pearson Correlation"]

if corr_table.empty:
    st.info("Analysis Limitation: Insufficient numerical features detected for correlation mapping.")
else:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(corr_table, use_container_width=True)
    with c2:
        fig_corr = px.bar(
            corr_table,
            x="Target Feature",
            y="Pearson Correlation",
            title=f"Predictive Correlation: Impact on {target_kpi_rca}",
        )
        fig_corr.update_layout(height=380, margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig_corr, use_container_width=True)

# ----- Interpretability Analysis (Feature Importance) -----
st.subheader("Predictive Drivers: Statistical Importance")

feature_candidates = [c for c in numeric_cols if c != target_kpi_rca]
X = wdf[feature_candidates].copy() if feature_candidates else pd.DataFrame()
y = wdf[target_kpi_rca].copy()

X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
y = pd.to_numeric(y, errors="coerce")

if X.empty or y.dropna().empty or len(wdf) < 200:
    st.info("Insufficient Data: Sampling volume in this window is inadequate for predictive modeling. (Minimum required: 200 records).")
else:
    is_classification = False
    if target_kpi_rca == cols.drop_rate:
        is_classification = True
        dr_vals = y.dropna().values
        y_thr = float(np.nanpercentile(dr_vals, 95)) if len(dr_vals) else np.nan
        y_bin = (y > y_thr).astype(int)
        y_use = y_bin
    else:
        y_use = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_use, test_size=0.2, random_state=42
    )

    if is_classification:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = f1_score(y_test, pred, average="binary") if len(np.unique(y_test)) > 1 else np.nan
        st.caption(f"Diagnostic Model: RandomForestClassifier | F1-Score: {score:.3f}" if not np.isnan(score) else "Diagnostic Model: RandomForestClassifier")
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        st.caption(f"Diagnostic Model: RandomForestRegressor | Mean Absolute Error: {mae:.4g}")

    imp = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).head(top_drivers)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(imp, use_container_width=True)
    with c2:
        fig_imp = px.bar(imp, x="feature", y="importance", title="Observed Feature Importance (Predictive Drivers)")
        fig_imp.update_layout(height=380, margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig_imp, use_container_width=True)
        
        
st.subheader("Automated Diagnostic Summary")

def _get_count(rule_name: str) -> int:
    """Retrieve frequency count for a specific heuristic rule."""
    if rule_df.empty:
        return 0
    x = rule_df[rule_df["Classification Rule"] == rule_name]
    return int(x["Frequency"].iloc[0]) if len(x) else 0

low_ho_cnt = _get_count("Mobility Violation (Low HO_SR)")
high_prb_cnt = _get_count("Saturation Event (High PRB)")
high_drop_cnt = _get_count("Critical Call Drop (P95 Threshold)")

window_size = len(wdf)

def _ratio(cnt: int) -> float:
    """Compute violation density ratio."""
    return (cnt / window_size) if window_size else 0.0

low_ho_ratio = _ratio(low_ho_cnt)
high_prb_ratio = _ratio(high_prb_cnt)
high_drop_ratio = _ratio(high_drop_cnt)

top_corr_features = []
if not corr_table.empty:
    top_corr_features = corr_table["Target Feature"].head(3).tolist()

top_imp_features = []
if "imp" in locals() and isinstance(imp, pd.DataFrame) and not imp.empty:
    top_imp_features = imp["feature"].head(3).tolist()

summary_lines = []

# Automated hypothesis generation based on empirical evidence
if target_kpi_rca == cols.throughput:
    if high_prb_ratio >= 0.30:
        summary_lines.append(
            f"Throughput impairment is likely correlated with capacity constraints: PRB saturation observed in {high_prb_ratio:.0%} of samples."
        )
    if high_drop_ratio >= 0.25:
        summary_lines.append(
            f"Anomalous drop events detected ({high_drop_ratio:.0%}) potentially indicate RF quality degradations affecting spectral efficiency."
        )
    if low_ho_ratio >= 0.25:
        summary_lines.append(
            f"Handover SR degradation ({low_ho_ratio:.0%}) suggests mobility-driven impairments or neighbor relation misalignment."
        )

elif target_kpi_rca == cols.ho_sr:
    if low_ho_ratio >= 0.30:
        summary_lines.append(
            f"Systematic mobility failure detected: Handover performance is consistently below threshold ({low_ho_ratio:.0%} of window)."
        )
    if high_prb_ratio >= 0.30:
        summary_lines.append(
            f"High resource utilization ({high_prb_ratio:.0%}) detected; potential signaling congestion during handover procedures."
        )
    if high_drop_ratio >= 0.25:
        summary_lines.append(
            f"Elevated call drop rates ({high_drop_ratio:.0%}) reinforce the hypothesis of RF instability or mobility failures."
        )

elif target_kpi_rca == cols.drop_rate:
    if high_drop_ratio >= 0.30:
        summary_lines.append(
            f"Anomalous drop frequency detected ({high_drop_ratio:.0%} exceeding P95). Primarily indicative of RF interference or mobility failure."
        )
    if low_ho_ratio >= 0.25:
        summary_lines.append(
            f"Handover failures correlate with drop spikes ({low_ho_ratio:.0%}), suggesting mobility-related root causes."
        )
    if high_prb_ratio >= 0.30:
        summary_lines.append(
            f"High PRB load ({high_prb_ratio:.0%}) suggests traffic-induced radio resource exhaustion contributing to drops."
        )

elif target_kpi_rca == cols.prb_dl:
    if high_prb_ratio >= 0.30:
        summary_lines.append(
            f"PRB utilization is consistently above established baseline ({high_prb_ratio:.0%}), indicating capacity saturation."
        )
    if target_kpi_rca in df.columns and cols.throughput in df.columns:
        summary_lines.append(
            "Inverse correlation between PRB and Throughput confirms active congestion state."
        )

# Append predictive supporting evidence
if top_corr_features:
    summary_lines.append(
        "Dominant Statistical Correlations: " + ", ".join(top_corr_features) + "."
    )

if top_imp_features:
    summary_lines.append(
        "Primary Predictive Drivers (Model Importance): " + ", ".join(top_imp_features) + "."
    )

if not summary_lines:
    summary_lines.append(
        "Ambiguous Pattern Detection: No definitive heuristic patterns identified. Increase analysis window or include supplemental counters."
    )

# Display prioritized summary components
for s in summary_lines[:3]:
    st.success(s)
       

# ============================================================================
# DIAGNOSTIC ARTIFACT EXPORT
# ============================================================================
st.subheader("Diagnostic Evidence Export")

evidence_pack = {
    "Heuristic Rules": rule_df,
    "Statistical Correlations": corr_table,
}

if 'imp' in locals() and isinstance(imp, pd.DataFrame):
    evidence_pack["Feature Importances"] = imp

tab_names = list(evidence_pack.keys())
selected_tab = st.selectbox("Select Evidence Layer for Export:", options=tab_names, index=0)
out_tbl = evidence_pack[selected_tab]

st.dataframe(out_tbl, use_container_width=True)

out_csv = out_tbl.to_csv(index=False).encode("utf-8")
st.download_button(
    "Export Evidence Subset (CSV)",
    data=out_csv,
    file_name=f"evidence_{selected_tab}_{entity_mode_rca}_{selected_entity_rca}_{target_kpi_rca}.csv",
    mime="text/csv",
)

st.caption(
    "Operational Guideline: Evaluate heuristic rule triggers first, cross-verify with statistical correlations, and confirm using predictive model importance. "
    "Persistent high PRB utilization coupled with reduced throughput strongly correlates with capacity congestion."
)


st.divider()
# Operational Decision Support (Rules)
st.subheader("Operational Diagnostic Rules (Quick Reference)")

rules = []
if cols.ho_sr in df.columns:
    rules.append(f"- Handover SR < {ho_thr:.2f}: Mobility Impairment / Neighbor Relation / RF Coverage.")
rules.append(f"- PRB DL > {prb_thr:.2f} with low Throughput: Capacity Saturation / Traffic Congestion.")
if drop_thr is not None:
    rules.append(f"- Drop Rate > P{int(drop_q*100)}: RF Quality / Pilot Pollution / Mobility Failure.")

for r in rules:
    st.write(r)
