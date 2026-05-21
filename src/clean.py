# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:10:44 2026

@author: hamza khlefat
"""

from typing import Dict, List, Optional
import pandas as pd

def strip_column_spaces(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def apply_rename(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    present = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=present)

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def drop_dupes(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    df = df.copy()
    return df.drop_duplicates(subset=subset) if subset else df.drop_duplicates()

def fill_missing(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> pd.DataFrame:
    df = df.copy()

    for c in numeric:
        if c in df.columns:
            med = df[c].median(skipna=True)
            df[c] = df[c].fillna(med)

    for c in categorical:
        if c in df.columns:
            mode = df[c].mode(dropna=True)
            val = mode.iloc[0] if len(mode) else "UNKNOWN"
            df[c] = df[c].fillna(val)

    return df

def clip_outliers(df: pd.DataFrame, numeric: List[str], low_q: float = 0.01, high_q: float = 0.99) -> pd.DataFrame:
    df = df.copy()
    for c in numeric:
        if c in df.columns:
            lo = df[c].quantile(low_q)
            hi = df[c].quantile(high_q)
            df[c] = df[c].clip(lo, hi)
    return df
