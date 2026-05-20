"""
Dataset to SQLite Loader
Downloads a specified Kaggle dataset and loads it into a SQLite database.

Usage:
    uv run python src/setup_database.py <kaggle_path>
    Example: uv run python src/setup_database.py uciml/adult-census-income
"""

from __future__ import annotations

import sys
import sqlite3
import re
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

def download_dataset(kaggle_path: str) -> tuple[pd.DataFrame, str]:
    """Downloads dataset via kagglehub and determines table name."""
    try:
        import kagglehub
    except ImportError:
        import subprocess
        subprocess.run(["uv", "add", "kagglehub"], check=True)
        import kagglehub

    print(f"Downloading from Kaggle: {kaggle_path} ...")
    path = kagglehub.dataset_download(kaggle_path)
    print(f"Downloaded to: {path}")

    import glob
    csv_files = glob.glob(f"{path}/**/*.csv", recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"CSV bulunamadı: {path}")

    # Read the first found CSV file
    csv_file = csv_files[0]
    df = pd.read_csv(csv_file)
    print(f"Read: {Path(csv_file).name} | {len(df)} rows, {len(df.columns)} columns")
    
    # Derive table and DB name from Kaggle path
    table_name = kaggle_path.split("/")[-1].replace("-", "_").lower()
    # Clean up special characters
    table_name = re.sub(r'[^a-z0-9_]', '', table_name)
    
    return df, table_name

def clean_dataset(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Cleans dataset and prepares it for SQLite."""
    
    # Clean column names
    df.columns = [str(c).strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]

    # Convert common missing value indicators to None (NULL in SQLite)
    df = df.replace(["?", " ?", "NA", "N/A", "null", ""], None)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip() if df[col].dtype == object else df[col]

    # Apply special transformation if dataset is adult_income
    if "income" in df.columns and "adult" in table_name:
        df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
        df["income_binary"] = (df["income"].str.contains(">50K", na=False)).astype(int)

    print(f"Cleaned: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    return df


def load_to_sqlite(df: pd.DataFrame, table_name: str) -> Path:
    """Loads DataFrame into SQLite."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    db_path = DATA_DIR / f"{table_name}.db"
    conn = sqlite3.connect(db_path)

    # Create or overwrite table
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"\nLoaded to SQLite: {db_path}")
    print(f"Table: {table_name} | {len(df)} rows")

    cursor = conn.cursor()
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}'")
    schema = cursor.fetchone()
    if schema:
        print(f"\nSchema:\n{schema[0]}")

    conn.close()
    return db_path


def verify(db_path: Path, table_name: str) -> None:
    """Verifies the database load."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]

    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]

    conn.close()

    print(f"\nVerification: {count} rows loaded")
    print("First 3 rows:")
    for row in rows:
        print(dict(zip(cols, row)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Kaggle dataset path not provided!")
        print("Usage: uv run python src/setup_database.py <kaggle_path>")
        print("Example: uv run python src/setup_database.py uciml/adult-census-income")
        sys.exit(1)

    kaggle_path = sys.argv[1]
    
    print(f"=== Dataset Setup: {kaggle_path} ===\n")

    df, table_name = download_dataset(kaggle_path)
    df = clean_dataset(df, table_name)
    db_path = load_to_sqlite(df, table_name)
    verify(db_path, table_name)

    print(f"\n✓ Done!")
    print(f"  DB_PATH = {db_path}")
    print(f"\nUpdate mcp_server.py with the following to connect to this database:")
    print(f'  DB_PATH = PROJECT_DIR / "data" / "{table_name}.db"')