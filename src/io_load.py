# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:30:22 2026

@author: hamza khlefat
"""

from typing import Iterator, Optional, List
import pandas as pd

def read_csv_chunks(
    csv_path: str,
    chunksize: int = 200_000,
    parse_dates: Optional[List[str]] = None
) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        csv_path,
        chunksize=chunksize,
        low_memory=False,
        parse_dates=parse_dates
    )
