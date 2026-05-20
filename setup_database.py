"""
Adult Income Dataset → SQLite
------------------------------
Bu script UCI Adult Income veri setini indirip
SQLite veritabanına yükler.

Kullanım:
    uv run python setup_database.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "data" / "adult_income.db"


def download_dataset() -> pd.DataFrame:
    """kagglehub ile indirir."""
    try:
        import kagglehub
    except ImportError:
        import subprocess
        subprocess.run(["uv", "add", "kagglehub"], check=True)
        import kagglehub

    print("Kaggle'dan indiriliyor...")
    path = kagglehub.dataset_download("uciml/adult-census-income")
    print(f"İndirildi: {path}")

    import glob
    csv_files = glob.glob(f"{path}/**/*.csv", recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"CSV bulunamadı: {path}")

    df = pd.read_csv(csv_files[0])
    print(f"Okundu: {len(df)} satır, {len(df.columns)} kolon")
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Veri setini temizler ve SQLite için hazırlar."""

    df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_")
                  for c in df.columns]

    df = df.replace("?", None)
    df = df.replace(" ?", None)

    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip() if df[col].dtype == object else df[col]

    if "income" in df.columns:
        df["income"] = df["income"].str.strip().str.replace(".", "", regex=False)
        df["income_binary"] = (df["income"].str.contains(">50K")).astype(int)

    print(f"Temizlendi: {len(df)} satır, {len(df.columns)} kolon")
    print(f"Kolonlar: {df.columns.tolist()}")
    return df


def load_to_sqlite(df: pd.DataFrame) -> None:
    """DataFrame'i SQLite'a yükler."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    df.to_sql("adult_income", conn, if_exists="replace", index=False)
    print(f"\nSQLite'a yüklendi: {DB_PATH}")
    print(f"Tablo: adult_income | {len(df)} satır")

    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE name='adult_income'")
    schema = cursor.fetchone()
    if schema:
        print(f"\nŞema:\n{schema[0]}")

    conn.close()


def verify(db_path: Path) -> None:
    """Yüklemeyi doğrular."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM adult_income")
    count = cursor.fetchone()[0]

    cursor.execute("SELECT * FROM adult_income LIMIT 3")
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]

    conn.close()

    print(f"\nDoğrulama: {count} satır yüklendi")
    print(f"İlk 3 satır:")
    for row in rows:
        print(dict(zip(cols, row)))


if __name__ == "__main__":
    print("=== Adult Income Dataset Setup ===\n")

    df = download_dataset()
    df = clean_dataset(df)
    load_to_sqlite(df)
    verify(DB_PATH)

    print(f"\n✓ Tamamlandı!")
    print(f"  DB_PATH = {DB_PATH}")
    print(f"\nmcp_server.py'de şu satırı güncelle:")
    print(f'  DB_PATH = PROJECT_DIR / "data" / "adult_income.db"')