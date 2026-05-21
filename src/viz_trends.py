# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 09:05:33 2026

@author: hamza khlefat
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def interactive_trend(
    df: pd.DataFrame,
    date_col: str,
    series: list[tuple[str, str]],
    title: str,
    hlines: list[tuple[float, str]] | None = None,
    group_col: str | None = None,
) -> go.Figure:
    temp = df.copy()

    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col]).sort_values(date_col)

    # Unit Detection & Scaling logic
    # Find max value across all series to determine scaling
    max_val = 0
    for col, _ in series:
        if col in temp.columns:
            v = pd.to_numeric(temp[col], errors='coerce').max()
            if not pd.isna(v): max_val = max(max_val, v)
    
    scale_factor = 1.0
    unit_suffix = ""
    
    # If values are in the range of bps (billions), scale to Gbps
    if max_val >= 1e9:
        scale_factor = 1e9
        unit_suffix = " (Gbps)"
    elif max_val >= 1e6:
        scale_factor = 1e6
        unit_suffix = " (Mbps)"

    fig = go.Figure()

    if group_col and group_col in temp.columns:
        groups = temp[group_col].dropna().unique().tolist()
        for g in groups:
            gdf = temp[temp[group_col] == g]
            for col, label in series:
                if col not in gdf.columns:
                    continue
                y = _safe_numeric(gdf[col]) / scale_factor
                fig.add_trace(
                    go.Scatter(
                        x=gdf[date_col],
                        y=y,
                        mode="lines",
                        name=f"{label} ({g})",
                        hovertemplate="date=%{x}<br>value=%{y:.2f}" + unit_suffix + "<extra></extra>",
                    )
                )
    else:
        for col, label in series:
            if col not in temp.columns:
                continue
            y = _safe_numeric(temp[col]) / scale_factor
            fig.add_trace(
                go.Scatter(
                    x=temp[date_col],
                    y=y,
                    mode="lines",
                    name=label,
                    hovertemplate="date=%{x}<br>value=%{y:.2f}" + unit_suffix + "<extra></extra>",
                )
            )

    if hlines:
        for yval, label in hlines:
            fig.add_hline(y=yval / scale_factor, line_dash="dash", annotation_text=label)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=f"Magnitude{unit_suffix}",
        hovermode="x unified",
        legend_title="Series",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def aggregate_by_time(
    df: pd.DataFrame,
    date_col: str,
    group_cols: list[str],
    agg: str = "median",
) -> pd.DataFrame:
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])

    agg_map = {}
    for c in group_cols:
        if c in temp.columns:
            agg_map[c] = agg

    out = temp.groupby(date_col, as_index=False).agg(agg_map)
    return out.sort_values(date_col)
