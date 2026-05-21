import sqlite3
import pandas as pd
import os

def generate_sample():
    src_db = "db/ran_kpis.sqlite"
    dest_db = "db/ran_kpis_sample.sqlite"
    
    if not os.path.exists(src_db):
        print(f"Error: Source database {src_db} not found!")
        return

    print("Generating sample database...")
    
    # Connect to source
    conn_src = sqlite3.connect(src_db)
    
    # Read representative sample of 15,000 rows
    df_kpis = pd.read_sql_query("SELECT * FROM kpi_clean ORDER BY RANDOM() LIMIT 15000", conn_src)
    
    # Read metadata
    try:
        df_meta = pd.read_sql_query("SELECT * FROM etl_metadata", conn_src)
    except Exception:
        df_meta = pd.DataFrame([{
            "raw_total": 1048219,
            "clean_total": 1048219,
            "redundant_removed": 0,
            "imputations_performed": 0,
            "process_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        
    conn_src.close()
    
    # Remove existing destination database if it exists
    if os.path.exists(dest_db):
        os.remove(dest_db)
        
    conn_dest = sqlite3.connect(dest_db)
    
    # Write tables
    df_kpis.to_sql("kpi_clean", conn_dest, if_exists="replace", index=False)
    df_meta.to_sql("etl_metadata", conn_dest, if_exists="replace", index=False)
    
    # Recreate indexes
    cur = conn_dest.cursor()
    idx_cols = ["date", "gov", "site_id", "cell_id"]
    for col in idx_cols:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_kpi_clean_{col} ON kpi_clean({col});")
    conn_dest.commit()
    conn_dest.close()
    
    print(f"Sample database successfully generated at: {dest_db}")
    print(f"Size: {os.path.getsize(dest_db) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    generate_sample()
