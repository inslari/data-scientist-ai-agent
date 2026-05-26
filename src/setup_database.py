"""
Dataset to Database Loader
Downloads a specified Kaggle dataset and loads it into the configured database.
Supports SQLite, PostgreSQL, MySQL, and SQL Server via SQLAlchemy.

Usage:
    uv run python src/setup_database.py <kaggle_path>
    Example: uv run python src/setup_database.py uciml/adult-census-income
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

# Add project root to sys.path so we can import src.* when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import text, inspect

from core.db_provider import db_manager

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

    if "kaggle.com/datasets/" in kaggle_path:
        kaggle_path = kaggle_path.split("kaggle.com/datasets/")[-1].strip("/")

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
    """Cleans dataset and prepares it for database loading."""
    
    # Clean column names
    df.columns = [str(c).strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]

    # Convert common missing value indicators to None (NULL in database)
    df = df.replace(["?", " ?", "NA", "N/A", "null", ""], None)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip() if df[col].dtype == object else df[col]

    # Apply special transformation if dataset is adult_income
    if "income" in df.columns and "adult" in table_name:
        df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
        df["income_binary"] = (df["income"].str.contains(">50K", na=False)).astype(int)

    print(f"Cleaned: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    return df


def load_to_database(df: pd.DataFrame, table_name: str, provider) -> None:
    """Loads DataFrame into the configured database using SQLAlchemy."""
    engine = provider.engine

    # For SQLite, ensure data directory exists
    if provider.provider == "sqlite":
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    # pandas to_sql works with SQLAlchemy engines for all providers
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"\nLoaded to {provider.provider_display_name}: {table_name}")
    print(f"Table: {table_name} | {len(df)} rows")

    # Show schema info
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        col_defs = ", ".join(f"{c['name']} {c['type']}" for c in columns)
        print(f"\nSchema:\nCREATE TABLE {table_name} ({col_defs})")
    except Exception as e:
        print(f"\nSchema bilgisi alınamadı: {e}")


def verify(table_name: str, provider) -> None:
    """Verifies the database load."""
    with provider.connect() as conn:
        count_result = conn.execute(
            text(provider.get_count_query(table_name))
        )
        count = count_result.fetchone()[0]

        sample_result = conn.execute(
            text(provider.get_limit_query(table_name, 3))
        )
        cols = list(sample_result.keys())
        rows = sample_result.fetchall()

    print(f"\nVerification: {count} rows loaded")
    print("First 3 rows:")
    for row in rows:
        print(dict(zip(cols, row)))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset to Database Loader")
    parser.add_argument("kaggle_path", help="Kaggle dataset path (e.g. uciml/adult-census-income)")
    parser.add_argument("--db", type=str, default="default", help="Target database name (default: 'default')")
    args = parser.parse_args()

    kaggle_path = args.kaggle_path
    db_name = args.db
    
    try:
        provider = db_manager.get_provider(db_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"=== Dataset Setup: {kaggle_path} ===\n")
    print(f"Database Provider: {provider.provider_display_name} ({db_name})")

    df, table_name = download_dataset(kaggle_path)
    df = clean_dataset(df, table_name)
    load_to_database(df, table_name, provider)
    verify(table_name, provider)

    print(f"\n✓ Done!")
    print(f"  Provider: {provider.provider_display_name} ({db_name})")
    print(f"  Table   : {table_name}")