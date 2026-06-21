"""
Financial Dataset Generator
============================
Generates a realistic financial transactions database.

Tables:
  accounts     →   500K rows  (müşteri hesapları)
  merchants    →    50K rows  (satıcılar)
  transactions →  10M  rows  (işlemler — fraud dahil)

Schema:
  accounts:
    account_id, customer_name, age, gender, country, city,
    account_type, balance, credit_limit, open_date,
    risk_score, is_active

  merchants:
    merchant_id, merchant_name, category, country, city,
    avg_transaction_amount, is_blacklisted

  transactions:
    transaction_id, account_id, merchant_id,
    amount, currency, transaction_type, channel,
    transaction_date, hour_of_day,
    is_foreign, is_weekend, is_fraud,
    fraud_reason

Run:
    python generate_financial_db.py
    → financial.db (SQLite, ~1.5GB)
"""

import random
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# ── Config ───────────────────────────────────────────────────
N_ACCOUNTS     =    500_000
N_MERCHANTS    =     50_000
N_TRANSACTIONS = 10_000_000
FRAUD_RATE     = 0.015       # %1.5 fraud
BATCH_SIZE     = 100_000
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "financial.db"
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Lookup tables ─────────────────────────────────────────────
COUNTRIES = [
    ("TR","Turkey",["Istanbul","Ankara","Izmir","Bursa","Antalya"]),
    ("US","United States",["New York","Los Angeles","Chicago","Houston","Phoenix"]),
    ("DE","Germany",["Berlin","Munich","Hamburg","Frankfurt","Cologne"]),
    ("GB","United Kingdom",["London","Manchester","Birmingham","Leeds","Glasgow"]),
    ("FR","France",["Paris","Lyon","Marseille","Toulouse","Nice"]),
    ("AE","UAE",["Dubai","Abu Dhabi","Sharjah"]),
    ("RU","Russia",["Moscow","Saint Petersburg","Novosibirsk"]),
    ("CN","China",["Beijing","Shanghai","Shenzhen","Guangzhou"]),
    ("IN","India",["Mumbai","Delhi","Bangalore","Chennai"]),
    ("BR","Brazil",["Sao Paulo","Rio de Janeiro","Brasilia"]),
]
COUNTRY_CODES   = [c[0] for c in COUNTRIES]
COUNTRY_WEIGHTS = [0.25,0.20,0.10,0.10,0.08,0.07,0.05,0.05,0.05,0.05]

ACCOUNT_TYPES   = ["checking","savings","credit","premium","business"]
MERCHANT_CATS   = [
    "grocery","restaurant","fuel","hotel","airline",
    "electronics","clothing","pharmacy","entertainment",
    "subscription","atm","transfer","online_shopping",
    "utilities","insurance","education","healthcare",
]
TRANSACTION_TYPES  = ["purchase","withdrawal","transfer","refund","payment"]
CHANNELS           = ["pos","online","mobile","atm","bank_branch"]
CURRENCIES         = ["USD","EUR","TRY","GBP","AED","RUB","CNY","INR","BRL"]
FRAUD_REASONS      = [
    "card_cloning","stolen_card","account_takeover",
    "synthetic_identity","unusual_location",
    "velocity_abuse","card_not_present","money_laundering",
]

START_DATE = datetime(2020, 1, 1)
END_DATE   = datetime(2024, 12, 31)
DATE_RANGE = (END_DATE - START_DATE).days


def random_date() -> str:
    return (START_DATE + timedelta(days=random.randint(0, DATE_RANGE))).strftime("%Y-%m-%d")


def pick_country():
    return random.choices(COUNTRIES, weights=COUNTRY_WEIGHTS, k=1)[0]


# ── Schema ────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id        INTEGER PRIMARY KEY,
    customer_name     TEXT,
    age               INTEGER,
    gender            TEXT,
    country           TEXT,
    city              TEXT,
    account_type      TEXT,
    balance           REAL,
    credit_limit      REAL,
    open_date         TEXT,
    risk_score        REAL,
    is_active         INTEGER
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id              INTEGER PRIMARY KEY,
    merchant_name            TEXT,
    category                 TEXT,
    country                  TEXT,
    city                     TEXT,
    avg_transaction_amount   REAL,
    is_blacklisted           INTEGER
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id    INTEGER PRIMARY KEY,
    account_id        INTEGER,
    merchant_id       INTEGER,
    amount            REAL,
    currency          TEXT,
    transaction_type  TEXT,
    channel           TEXT,
    transaction_date  TEXT,
    hour_of_day       INTEGER,
    is_foreign        INTEGER,
    is_weekend        INTEGER,
    is_fraud          INTEGER,
    fraud_reason      TEXT
);

CREATE INDEX IF NOT EXISTS idx_txn_account  ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_txn_merchant ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_txn_date     ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_txn_fraud    ON transactions(is_fraud);
"""

# ── Generators ────────────────────────────────────────────────

def gen_accounts(n: int) -> list[tuple]:
    rows = []
    first_names = ["Ali","Ayse","Mehmet","Fatma","Ahmet","Emine","James","Emma",
                   "Luca","Sophie","Hans","Anna","Wei","Mei","Raj","Priya"]
    last_names  = ["Yilmaz","Demir","Smith","Johnson","Mueller","Rossi","Wang","Patel"]

    for i in range(1, n+1):
        country_info = pick_country()
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        age   = int(np.random.normal(38, 12))
        age   = max(18, min(80, age))
        bal   = round(np.random.lognormal(8, 1.5), 2)  # log-normal dağılım
        bal   = max(0, min(500_000, bal))
        limit = round(bal * random.uniform(0.5, 3.0), 2)
        risk  = round(random.betavariate(2, 5), 4)  # çoğunluk düşük risk

        rows.append((
            i,
            f"{fname} {lname}",
            age,
            random.choice(["M","F","M","F","M","F","Other"]),
            country_info[0],
            random.choice(country_info[2]),
            random.choices(ACCOUNT_TYPES, weights=[0.35,0.30,0.20,0.10,0.05])[0],
            bal,
            limit,
            random_date(),
            risk,
            1 if random.random() > 0.05 else 0,   # %95 aktif
        ))
    return rows


def gen_merchants(n: int) -> list[tuple]:
    rows = []
    prefixes = ["Global","Premium","Super","Fast","Smart","Easy","Pro","Digital","City","Metro"]
    suffixes = ["Shop","Store","Market","Hub","Point","Zone","Center","Plus","Go","Now"]

    for i in range(1, n+1):
        country_info = pick_country()
        cat = random.choices(MERCHANT_CATS,
              weights=[0.15,0.12,0.08,0.07,0.05,0.08,0.08,0.06,0.05,
                       0.04,0.04,0.04,0.07,0.03,0.02,0.01,0.01])[0]
        avg_amt = round(np.random.lognormal(4, 0.8), 2)

        rows.append((
            i,
            f"{random.choice(prefixes)} {random.choice(suffixes)} {i}",
            cat,
            country_info[0],
            random.choice(country_info[2]),
            avg_amt,
            1 if random.random() < 0.002 else 0,  # %0.2 kara listede
        ))
    return rows


def gen_transactions_batch(
    start_id: int,
    batch_n: int,
    n_accounts: int,
    n_merchants: int,
    fraud_rate: float,
) -> list[tuple]:
    rows = []

    # Batch için vektörize üretim (hızlı)
    account_ids  = np.random.randint(1, n_accounts+1,  batch_n)
    merchant_ids = np.random.randint(1, n_merchants+1, batch_n)
    amounts_base = np.random.lognormal(4.5, 1.2, batch_n)
    amounts      = np.clip(amounts_base, 0.5, 50_000).round(2)
    currencies   = np.random.choice(CURRENCIES,  batch_n,
                       p=[0.30,0.20,0.15,0.10,0.08,0.05,0.04,0.04,0.04])
    txn_types    = np.random.choice(TRANSACTION_TYPES, batch_n,
                       p=[0.65,0.10,0.12,0.05,0.08])
    channels     = np.random.choice(CHANNELS, batch_n,
                       p=[0.35,0.30,0.20,0.10,0.05])
    days_offset  = np.random.randint(0, DATE_RANGE+1, batch_n)
    hours        = np.random.randint(0, 24, batch_n)
    is_foreign   = np.random.binomial(1, 0.12, batch_n)
    is_weekend   = np.zeros(batch_n, dtype=int)

    # Fraud assignment
    is_fraud     = np.random.binomial(1, fraud_rate, batch_n)

    for j in range(batch_n):
        txn_date = (START_DATE + timedelta(days=int(days_offset[j]))).strftime("%Y-%m-%d")
        dow      = (START_DATE + timedelta(days=int(days_offset[j]))).weekday()
        is_weekend[j] = 1 if dow >= 5 else 0

        # Fraud pattern'ları daha gerçekçi yap
        fraud = int(is_fraud[j])
        fraud_reason = ""
        if fraud:
            # Fraud: gece saatleri, yüksek tutar, foreign
            amount_mult = random.uniform(2, 8)
            amt = round(min(float(amounts[j]) * amount_mult, 50_000), 2)
            hour = random.choice([0,1,2,3,22,23])
            is_for = 1 if random.random() < 0.6 else int(is_foreign[j])
            fraud_reason = random.choice(FRAUD_REASONS)
        else:
            amt   = float(amounts[j])
            hour  = int(hours[j])
            is_for = int(is_foreign[j])

        rows.append((
            start_id + j,
            int(account_ids[j]),
            int(merchant_ids[j]),
            amt,
            str(currencies[j]),
            str(txn_types[j]),
            str(channels[j]),
            txn_date,
            hour,
            is_for,
            int(is_weekend[j]),
            fraud,
            fraud_reason,
        ))
    return rows


# ── Main ──────────────────────────────────────────────────────
def main():
    logger.info(f"{'='*60}")
    logger.info(f"  Financial Dataset Generator")
    logger.info(f"  Accounts    : {N_ACCOUNTS:,}")
    logger.info(f"  Merchants   : {N_MERCHANTS:,}")
    logger.info(f"  Transactions: {N_TRANSACTIONS:,}")
    logger.info(f"  Fraud rate  : %{FRAUD_RATE*100:.1f}")
    logger.info(f"  Output      : {OUTPUT_PATH}")
    logger.info(f"{'='*60}\n")

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        logger.info(f"Eski veritabanı silindi.")

    conn = sqlite3.connect(OUTPUT_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")   # 64MB cache
    conn.executescript(SCHEMA)
    logger.info("Schema oluşturuldu.")

    # Accounts
    t0 = time.time()
    logger.info(f"\n[1/3] Accounts üretiliyor ({N_ACCOUNTS:,})...")
    accounts = gen_accounts(N_ACCOUNTS)
    conn.executemany(
        "INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        accounts
    )
    conn.commit()
    logger.info(f"  ✓ {N_ACCOUNTS:,} hesap eklendi ({time.time()-t0:.1f}s)")

    # Merchants
    t0 = time.time()
    logger.info(f"\n[2/3] Merchants üretiliyor ({N_MERCHANTS:,})...")
    merchants = gen_merchants(N_MERCHANTS)
    conn.executemany(
        "INSERT INTO merchants VALUES (?,?,?,?,?,?,?)",
        merchants
    )
    conn.commit()
    logger.info(f"  ✓ {N_MERCHANTS:,} satıcı eklendi ({time.time()-t0:.1f}s)")

    # Transactions (batched)
    t0 = time.time()
    logger.info(f"\n[3/3] Transactions üretiliyor ({N_TRANSACTIONS:,})...")
    total_inserted = 0

    for batch_start in range(0, N_TRANSACTIONS, BATCH_SIZE):
        batch_n = min(BATCH_SIZE, N_TRANSACTIONS - batch_start)
        batch   = gen_transactions_batch(
            batch_start + 1, batch_n,
            N_ACCOUNTS, N_MERCHANTS, FRAUD_RATE
        )
        conn.executemany(
            "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            batch
        )
        conn.commit()
        total_inserted += batch_n
        elapsed = time.time() - t0
        rate    = total_inserted / elapsed
        eta     = (N_TRANSACTIONS - total_inserted) / rate if rate > 0 else 0
        logger.info(f"  {total_inserted:>10,}/{N_TRANSACTIONS:,} "
              f"({total_inserted/N_TRANSACTIONS*100:.1f}%) "
              f"| {rate:,.0f} row/s | ETA: {eta:.0f}s")

    logger.info(f"\n  ✓ {total_inserted:,} işlem eklendi ({time.time()-t0:.1f}s)")

    # Stats
    logger.info(f"\n{'='*60}")
    logger.info(f"  Final istatistikler:")
    for table in ["accounts","merchants","transactions"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  {table:15s}: {count:>12,} satır")

    fraud_count = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE is_fraud=1"
    ).fetchone()[0]
    logger.info(f"  fraud işlem  : {fraud_count:>12,} (%{fraud_count/N_TRANSACTIONS*100:.2f})")

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    logger.info(f"\n  Dosya boyutu : {size_mb:.1f} MB")
    logger.info(f"  Konum        : {OUTPUT_PATH}")
    logger.info(f"{'='*60}")

    conn.close()
    logger.info("\nTamamlandı!")


if __name__ == "__main__":
    from src.core.logging_config import setup_logging
    setup_logging(process="script")
    main()