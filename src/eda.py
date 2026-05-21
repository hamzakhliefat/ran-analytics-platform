# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 16:20:05 2026

@author: hamza khlefat
"""

from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summary_by_group(df: pd.DataFrame, group_col: str, kpis: List[str]) -> pd.DataFrame:
    kpis = [k for k in kpis if k in df.columns]
    return df.groupby(group_col)[kpis].agg(["mean", "min", "max"]).reset_index()

def corr_heatmap_figure(df: pd.DataFrame, kpis: List[str], title: str):
    kpis = [k for k in kpis if k in df.columns]
    corr = df[kpis].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="viridis", ax=ax)
    ax.set_title(title)
    return fig
