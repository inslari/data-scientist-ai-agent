"""
Row Codec testleri (standalone — repo konvansiyonu).

Çalıştır:  uv run python tests/test_row_codec.py

Kapsam:
  1. Round-trip: timestamp+tz / virgül-içeren-string / iç-tırnak / newline / unicode
     / None / NaN / NaT / numpy skalerleri encode→decode'dan KAYIPSIZ, KAYMASIZ geçiyor
     (alan sayısı sabit).
  2. Senaryo B (fraud kurtarma): gerçek fraud CSV → CSV-path serialization (row_codec)
     → _parse_mcp_to_df: quarantine 0/3000, is_fraud {0,1}, fraud ~%2,
     data_quality_status = ok.
  3. Regresyon: kredi riski + bank churn CSV'leri temiz parse, quarantine 0.
  4. Tüketici uyumu: coerced çıktı, dokunulmayan tüketicilerin ast.literal_eval
     idiomuyla parse ediliyor (backward-compat).
  5. DQ-gate hâlâ çalışıyor: kasıtlı gerçekten-bozuk (fazla alanlı) satır → quarantine.

Uzak app DB'ye ASLA dokunmaz (CLAUDE.md backstop).
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

os.environ["API_DATABASE_URL"] = f"sqlite:///{os.path.join(tempfile.gettempdir(), 'rc_test_appdb.sqlite')}"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from decimal import Decimal  # noqa: E402

from src.core.row_codec import encode_result, decode_result, encode_row  # noqa: E402
from src.services.data_executor import _parse_mcp_to_df  # noqa: E402
from src.services.data_quality import run_data_quality  # noqa: E402

_FAILS: list[str] = []
_CSV_DIR = Path(__file__).resolve().parent.parent / "data" / "csvs"
_FRAUD_CSV = _CSV_DIR / "2eb2d968-2193-4bd4-969c-782072f8198c_transactions.csv"


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        _FAILS.append(f"{name} :: {detail}")


def _mcp_from_csv(path, limit=3000, extra_sql=""):
    """CSV-path serialization'ı birebir taklit: read_csv_auto → pandas → encode_result."""
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_csv_auto('{path}') {extra_sql} LIMIT {limit}").df()
    con.close()
    return encode_result(df.columns.tolist(), df.itertuples(index=False, name=None)), df.columns.tolist()


def test_round_trip():
    print("\n[1] Round-trip — zor değerler kayıpsız/kaymasız")
    cols = ["id", "ts", "note", "amt", "flag", "uni", "na"]
    rows = [
        (1, pd.Timestamp("2024-01-06 07:09:39", tz="Europe/Istanbul"), "O'Brien, Jr.",
         Decimal("10.5"), True, "café 日本 — تجربة", None),
        (2, pd.Timestamp("2024-02-01"), "line1\nline2, still", np.float64(3.14), False, 'x"y', float("nan")),
        (np.int64(3), None, "düz, virgüllü", 7, None, "z", pd.NaT),
    ]
    text = encode_result(cols, rows)
    c, r, s = decode_result(text)
    check("kolonlar korunuyor", c == cols, str(c))
    check("3 satır, 0 karantina", len(r) == 3 and s["quarantined_rows"] == 0, str(s))
    check("her satır tam alan sayısı (kayma yok)", all(len(x) == len(cols) for x in r),
          str([len(x) for x in r]))
    check("iç virgül+tırnak string bütün kaldı", r[0][2] == "O'Brien, Jr.", r[0][2])
    check("newline+virgül string bütün kaldı", r[1][2] == "line1\nline2, still", repr(r[1][2]))
    check("unicode korunuyor", r[0][5] == "café 日本 — تجربة", r[0][5])
    check("timestamp+tz virgülsüz ISO string", "," not in r[0][1] and "2024-01-06" in r[0][1], r[0][1])
    check("NaN → None", r[1][6] is None, r[1][6])
    check("NaT → None", r[2][6] is None, r[2][6])
    check("numpy int64 → int", r[2][0] == 3 and isinstance(r[2][0], int), repr(r[2][0]))
    check("Decimal → güvenli", r[0][3] == "10.5", r[0][3])


def test_consumer_backward_compat():
    print("\n[2] Tüketici uyumu — dokunulmayan tüketicilerin ast idiomu çalışıyor")
    # sql_node/mcp_client/ttd/dataviz/mcp.py hepsi bu idiomu kullanıyor:
    text = encode_result(["a", "b", "c"],
                         [(1, pd.Timestamp("2024-01-06 07:09:39", tz="Europe/Istanbul"), "x,y")])
    parsed = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            parsed.append(ast.literal_eval(line))   # tüm 5 tüketicideki birebir çağrı
    check("ast.literal_eval coerced satırı parse etti", len(parsed) == 1, str(parsed))
    check("3 alan (kayma yok)", len(parsed[0]) == 3, str(parsed[0]))
    check("iç-virgüllü string bütün", parsed[0][2] == "x,y", parsed[0][2])


def _profile_from_df(df, target):
    n = len(df)
    cols = {}
    for c in df.columns:
        nun = int(df[c].nunique(dropna=True))
        nnull = int(df[c].isna().sum())
        cols[c] = {"dtype": str(df[c].dtype), "unique_count": nun,
                   "unique_rate": round(nun / max(n, 1), 4), "null_count": nnull}
    return {"shape": {"rows": n, "columns": len(df.columns)}, "columns": cols}


def _scenario(name, csv_path, target, intent="classification"):
    if not csv_path.exists():
        check(f"{name}: CSV mevcut", False, f"bulunamadı: {csv_path.name} (atlandı)")
        return
    text, _ = _mcp_from_csv(csv_path, limit=3000)
    df = _parse_mcp_to_df(text, "t")
    s = df.attrs.get("_dq_structural", {})
    check(f"{name}: quarantine 0 (kayma yok)", s.get("quarantined_rows") == 0, str(s))
    check(f"{name}: temiz satır > 0", len(df) > 0, f"len={len(df)}")
    st = {"intent": intent, "target_col": target, "data_path": "",
          "profiling_report": _profile_from_df(df, target),
          "data_quality_report": {"structural": {"checked": True, "total_rows": s.get("total_rows", 0),
                                  "quarantined_rows": 0, "mismatch_rate": 0.0,
                                  "tables": [{"table": "t", "expected_cols": s.get("expected_cols")}]}}}
    r = run_data_quality(st)
    check(f"{name}: data_quality_status = ok", r["data_quality_status"] == "ok",
          f"{r['data_quality_status']} / {r['data_quality_report']['decision']}")
    return df


def test_scenario_b_fraud():
    print("\n[3] Senaryo B — fraud verisi artık uçtan uca kurtuluyor")
    df = _scenario("fraud", _FRAUD_CSV, "is_fraud")
    if df is not None and "is_fraud" in df.columns:
        distinct = sorted(pd.Series(df["is_fraud"]).dropna().unique().tolist())
        check("is_fraud distinct = {0,1}", distinct == [0, 1], str(distinct))
        frac = float((df["is_fraud"] == 1).mean())
        check("fraud oranı makul (0<oran<0.2)", 0 < frac < 0.2, f"{frac:.4f}")
        print(f"       → fraud oranı (3000 örnek): %{frac*100:.2f}")


def test_regression_clean_datasets():
    print("\n[4] Regresyon — kredi riski + bank churn temiz akıyor")
    credit = _CSV_DIR / "1b097ea4-bcc9-432f-ad92-d1ce02d62a52_credit_risk_dataset.csv"
    churn = _CSV_DIR / "7723fd96-87eb-4548-89e7-87899f05badf_Bank Customer Churn Prediction.csv"
    # target adları veri setine göre; sadece quarantine=0 + parse temizliği kritik.
    for name, path in [("credit_risk", credit), ("bank_churn", churn)]:
        if not path.exists():
            check(f"{name}: CSV mevcut", False, f"{path.name} yok (atlandı)"); continue
        text, cols = _mcp_from_csv(path, limit=3000)
        df = _parse_mcp_to_df(text, "t")
        s = df.attrs.get("_dq_structural", {})
        check(f"{name}: quarantine 0 (serialization bozmadı)", s.get("quarantined_rows") == 0, str(s))
        check(f"{name}: tüm kolonlar korunuyor", list(df.columns) == cols,
              f"{len(df.columns)} vs {len(cols)}")


def test_dq_gate_still_fires():
    print("\n[5] DQ-gate hâlâ çalışıyor — gerçekten bozuk satır yakalanıyor")
    # Kanonik metne kasıtlı FAZLA-alanlı satır enjekte et (üretici değil, bozuk kaynak taklidi).
    good = encode_result(["a", "b", "c"], [(1, 2, 3), (4, 5, 6)])
    corrupted = good + "\n(7, 8, 9, 10)"   # 4 alan / 3 kolon
    c, r, s = decode_result(corrupted)
    check("fazla-alanlı satır karantinaya alındı", s["quarantined_rows"] == 1, str(s))
    check("temiz 2 satır kaldı", len(r) == 2, str(r))


def main():
    print("=" * 70)
    print("ROW CODEC — TEST")
    print("=" * 70)
    test_round_trip()
    test_consumer_backward_compat()
    test_scenario_b_fraud()
    test_regression_clean_datasets()
    test_dq_gate_still_fires()

    print("\n" + "=" * 70)
    if _FAILS:
        print(f"SONUÇ: {len(_FAILS)} BAŞARISIZ")
        for f in _FAILS:
            print("  ✗ " + f)
        sys.exit(1)
    print("SONUÇ: TÜM TESTLER GEÇTİ ✓")


if __name__ == "__main__":
    main()
