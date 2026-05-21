# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 10:30:15 2026

@author: hamza khlefat
"""

from typing import List
import pandas as pd
from src.config import Cols, Thresholds

def add_baselines(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    df = df.copy()
    if cols.cell_id not in df.columns:
        return df

    if cols.throughput in df.columns:
        df = df.join(df.groupby(cols.cell_id)[cols.throughput].quantile(0.25).rename("tp_p25"), on=cols.cell_id)

    if cols.rrc_users_max in df.columns:
        df = df.join(df.groupby(cols.cell_id)[cols.rrc_users_max].quantile(0.75).rename("rrc_p75"), on=cols.cell_id)

    if cols.drop_rate in df.columns:
        df = df.join(df.groupby(cols.cell_id)[cols.drop_rate].quantile(0.75).rename("drop_p75"), on=cols.cell_id)

    return df

def explain_row(row: pd.Series, cols: Cols, th: Thresholds) -> List[str]:
    reasons: List[str] = []

    ho_sr = row.get(cols.ho_sr)
    prb = row.get(cols.prb_dl)
    tp = row.get(cols.throughput)
    rrc = row.get(cols.rrc_users_max)
    drop = row.get(cols.drop_rate)

    if prb is not None and tp is not None:
        if prb > th.prb_high and tp < row.get("tp_p25", tp):
            reasons.append("PRB is high while throughput is below baseline, likely congestion.")

    if rrc is not None and prb is not None:
        if rrc > row.get("rrc_p75", rrc) and prb > th.prb_high:
            reasons.append("High user load with high PRB suggests capacity pressure during busy hours.")

    if ho_sr is not None and ho_sr < th.ho_sr_low:
        reasons.append("Low handover success suggests mobility or neighbor relation issues.")

    if drop is not None and drop > row.get("drop_p75", drop):
        reasons.append("Drop rate is high compared to baseline, check radio quality, interference, or mobility failures.")

    if not reasons:
        reasons.append("No strong signal from current KPIs, consider adding more counters.")

    return reasons
