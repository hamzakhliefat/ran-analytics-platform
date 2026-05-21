# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 13:55:10 2026

@author: hamza khlefat
"""

import numpy as np
import pandas as pd
from src.config import Cols, Thresholds

def add_ho_sr(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    df = df.copy()

    if cols.ho_exe_succ not in df.columns or cols.ho_prep_att not in df.columns:
        df[cols.ho_sr] = np.nan
        return df

    exe = pd.to_numeric(df[cols.ho_exe_succ], errors="coerce")
    att = pd.to_numeric(df[cols.ho_prep_att], errors="coerce")

    att = att.replace(0, np.nan)
    ho = (exe / att)

    ho = ho.clip(lower=0, upper=1)
    df[cols.ho_sr] = ho.fillna(0)

    return df

def add_flags(df: pd.DataFrame, cols: Cols, th: Thresholds) -> pd.DataFrame:
    df = df.copy()

    if cols.ho_sr in df.columns:
        df["flag_low_ho"] = (df[cols.ho_sr] < th.ho_sr_low).astype(int)
    else:
        df["flag_low_ho"] = 0

    if cols.prb_dl in df.columns:
        df["flag_high_prb"] = (df[cols.prb_dl] > th.prb_high).astype(int)
    else:
        df["flag_high_prb"] = 0

    return df
    