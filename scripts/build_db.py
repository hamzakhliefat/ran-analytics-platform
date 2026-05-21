# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 13:10:00 2026

@author: hamza khlefat
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RawCols, Cols, Thresholds, DB, rename_map, numeric_cols
from src.io_load import read_csv_chunks
from src.clean import strip_column_spaces, apply_rename, coerce_numeric, drop_dupes, fill_missing, clip_outliers
from src.features import add_ho_sr, add_flags
from src.db import append_sqlite, add_indexes

import pandas as pd

def main():
    raw = RawCols()
    cols = Cols()
    th = Thresholds()
    db = DB()

    csv_path = "data/raw/Data Set Ajloun 2025.csv"
    processed_dir = "data/processed"
    os.makedirs("db", exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if os.path.exists(db.path):
        os.remove(db.path)

    mapping = rename_map(raw, cols)
    num = numeric_cols(cols)
    cats = [cols.gov, cols.site_id, cols.cell_id]

    # ETL Metadata Tracking
    total_raw_rows = 0
    total_clean_rows = 0
    total_missing_processed = 0
    all_chunks = []

    print("Initializing ETL Pipeline...")

    for i, chunk in enumerate(read_csv_chunks(csv_path, chunksize=200_000, parse_dates=[raw.date])):
        raw_count = len(chunk)
        total_raw_rows += raw_count
        
        # Capture pre-cleaning missing count
        null_pre = chunk.isnull().sum().sum()

        chunk = strip_column_spaces(chunk)
        chunk = apply_rename(chunk, mapping)
        chunk = coerce_numeric(chunk, [c for c in num if c in chunk.columns])

        keys = [c for c in [cols.date, cols.cell_id] if c in chunk.columns]
        chunk = drop_dupes(chunk, subset=keys if keys else None)

        chunk = fill_missing(
            chunk,
            numeric=[c for c in num if c in chunk.columns],
            categorical=[c for c in cats if c in chunk.columns]
        )

        chunk = clip_outliers(chunk, numeric=[c for c in num if c in chunk.columns])
        chunk = add_ho_sr(chunk, cols)
        chunk = add_flags(chunk, cols, th)

        # Increment cleaning metrics
        total_clean_rows += len(chunk)
        null_post = chunk.isnull().sum().sum()
        total_missing_processed += (null_pre - null_post)

        append_sqlite(chunk, db.path, db.table)
        all_chunks.append(chunk)
        print(f"Chunk {i+1}: Processed {len(chunk)}/{raw_count} rows.")

    # Persist Processed Dataset
    print(f"Exporting processed dataset to {processed_dir}...")
    full_df = pd.concat(all_chunks, ignore_index=True)
    full_df.to_csv(os.path.join(processed_dir, "clean_kpis.csv"), index=False)

    # Save ETL Metadata to Database
    import sqlite3
    with sqlite3.connect(db.path) as conn:
        meta_df = pd.DataFrame([{
            "raw_total": total_raw_rows,
            "clean_total": total_clean_rows,
            "redundant_removed": total_raw_rows - total_clean_rows,
            "imputations_performed": int(total_missing_processed),
            "process_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        meta_df.to_sql("etl_metadata", conn, if_exists="replace", index=False)

    idx_cols = [c for c in [cols.date, cols.gov, cols.site_id, cols.cell_id] if c]
    add_indexes(db.path, db.table, idx_cols)

    print("\nETL COMPLETE")
    print(f"- Raw Records: {total_raw_rows:,}")
    print(f"- Clean Records: {total_clean_rows:,}")
    print(f"- Processed Data Saved: {os.path.join(processed_dir, 'clean_kpis.csv')}")
    print(f"- Database Ready: {db.path}")

if __name__ == "__main__":
    main()
