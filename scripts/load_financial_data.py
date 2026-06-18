"""
Financial Data Loader
=====================
Loads the 5 financial CSV files from /output_csv/ into data/financial.db (SQLite).

Tables created:
  transactions  — 13.3M rows (id, date, client_id, card_id, amount, use_chip,
                               merchant_id, merchant_city, merchant_state, zip, mcc, errors)
  users         — 2K rows    (id, current_age, gender, yearly_income, credit_score, ...)
  cards         — 6.1K rows  (id, client_id, card_brand, card_type, credit_limit, ...)
  fraud_labels  — 8.9M rows  (transaction_id, is_fraud)
  mcc_codes     — 109 rows   (mcc, description)

Indexes:
  transactions: merchant_id, client_id, card_id, date
  fraud_labels: transaction_id, is_fraud

Run from project root:
    python scripts/load_financial_data.py
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT.parent / "output_csv"
DB_PATH = PROJECT_ROOT / "data" / "financial.db"

CHUNK_SIZE = 500_000


def load_table(conn: sqlite3.Connection, csv_path: Path, table_name: str, dtype_map: dict = None):
    print(f"\n[{table_name}] Yükleniyor: {csv_path.name}")
    t0 = time.time()
    total = 0

    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()

    first_chunk = True
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, dtype=dtype_map, low_memory=False):
        chunk.to_sql(table_name, conn, if_exists="replace" if first_chunk else "append", index=False)
        first_chunk = False
        total += len(chunk)
        print(f"  {total:,} satır yüklendi...", end="\r")

    elapsed = time.time() - t0
    print(f"  ✓ {total:,} satır, {elapsed:.1f}s")
    return total


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Eski financial.db silindi.")

    print(f"{'='*60}")
    print(f"  Financial Data Loader")
    print(f"  Kaynak: {CSV_DIR}")
    print(f"  Hedef : {DB_PATH}")
    print(f"{'='*60}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-128000")

    # Load tables
    fraud_dtype = {"transaction_id": str, "is_fraud": int, "label": str}
    load_table(conn, CSV_DIR / "transactions_data.csv", "transactions")
    load_table(conn, CSV_DIR / "users_data.csv",        "users")
    load_table(conn, CSV_DIR / "cards_data.csv",        "cards")
    load_table(conn, CSV_DIR / "fraud_labels.csv",      "fraud_labels", dtype_map=fraud_dtype)
    load_table(conn, CSV_DIR / "mcc_codes.csv",         "mcc_codes")

    # Indexes
    print("\nIndex'ler oluşturuluyor...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_tx_merchant ON transactions(merchant_id)",
        "CREATE INDEX IF NOT EXISTS idx_tx_client   ON transactions(client_id)",
        "CREATE INDEX IF NOT EXISTS idx_tx_card     ON transactions(card_id)",
        "CREATE INDEX IF NOT EXISTS idx_tx_date     ON transactions(date)",
        "CREATE INDEX IF NOT EXISTS idx_fl_txid     ON fraud_labels(transaction_id)",
        "CREATE INDEX IF NOT EXISTS idx_fl_fraud    ON fraud_labels(is_fraud)",
        "CREATE INDEX IF NOT EXISTS idx_cards_client ON cards(client_id)",
    ]
    for idx in indexes:
        conn.execute(idx)
    conn.commit()
    print("  ✓ Index'ler tamamlandı")

    # Stats
    print(f"\n{'='*60}")
    print("  Final istatistikler:")
    for table in ["transactions", "users", "cards", "fraud_labels", "mcc_codes"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:20s}: {count:>12,} satır")

    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\n  Dosya boyutu : {size_mb:.1f} MB")
    print(f"  Konum        : {DB_PATH}")
    print(f"{'='*60}")
    conn.close()
    print("\nTamamlandı!")


if __name__ == "__main__":
    main()
