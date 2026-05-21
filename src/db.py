# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:20:11 2026

@author: hamza khlefat
"""

import os
import sqlite3
from typing import Optional, Tuple
import pandas as pd

def conn(path: str) -> sqlite3.Connection:
    # If path is relative, resolve relative to project root
    if not os.path.isabs(path):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        abs_path = os.path.join(root_dir, path)
        if os.path.exists(abs_path):
            path = abs_path
        elif path == "db/ran_kpis.sqlite":
            sample_path = os.path.join(root_dir, "db/ran_kpis_sample.sqlite")
            if os.path.exists(sample_path):
                path = sample_path
    return sqlite3.connect(path)

def append_sqlite(df: pd.DataFrame, path: str, table: str) -> None:
    with conn(path) as c:
        df.to_sql(table, c, if_exists="append", index=False)

def read_sql(path: str, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
    with conn(path) as c:
        return pd.read_sql_query(query, c, params=params)

def add_indexes(path: str, table: str, cols: list[str]) -> None:
    with conn(path) as c:
        cur = c.cursor()
        for col in cols:
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col});")
        c.commit()
