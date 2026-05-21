# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 11:20:11 2026

@author: hamza khlefat
"""

import sqlite3
from typing import Optional, Tuple
import pandas as pd

def conn(path: str) -> sqlite3.Connection:
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
