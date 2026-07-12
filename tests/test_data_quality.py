"""
Veri Kalitesi Katmanı testleri (standalone script — repo konvansiyonu).

Çalıştır:  uv run python tests/test_data_quality.py

Kapsam:
  1. Dinamiklik: uydurma kolon adlarıyla (is_fraud/flag_suspicious/is_churned/…)
     rol-içerik çelişkisi doğru tetikleniyor mu — hiçbir ad hard-code değil.
  2. Regresyon: temiz binary target (is_churned/churn/loan_status) TETİKLENMEZ;
     kredi riski + bank churn 'ok' + eskisiyle AYNI node yoluna gider (false-pozitif
     DURDURMA yok).
  3. Yapısal kayma: alan-sayısı invaryantı satırları karantinaya alıyor; target
     varken failed, yokken degraded.
  4. TR/EN mesaj tutarlılığı.
  5. Sampling fix: dengesiz veride USING SAMPLE 0 satır döndürmüyor.
  6. Ucuzluk: doğrulama yolu tüm veriyi pandas'a yüklemiyor (DuckDB-only).

Uzak app DB'ye ASLA dokunmaz (CLAUDE.md): en başta API_DATABASE_URL yerel sqlite'a
sabitlenir — zaten hiçbir test app DB'ye bağlanmaz, bu bir backstop.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# --- Hard backstop: testler remote app DB'ye bağlanamasın ---
_SANDBOX = os.path.join(tempfile.gettempdir(), "dq_test_appdb.sqlite")
os.environ["API_DATABASE_URL"] = f"sqlite:///{_SANDBOX}"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from src.services.data_quality import run_data_quality, _name_implies_binary  # noqa: E402
from src.services.data_executor import _parse_mcp_to_df, DuckDBEngine  # noqa: E402
from src.graph import route_after_data_quality, route_after_execution  # noqa: E402
from src.core.report_i18n import R  # noqa: E402
from src.core.config import settings  # noqa: E402

_FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        _FAILS.append(f"{name} :: {detail}")


def col(dtype, nunique, unique_rate, null_count=0):
    return {"dtype": dtype, "unique_count": nunique, "unique_rate": unique_rate,
            "null_count": null_count}


def make_state(rows, columns, target=None, intent="classification", data_path=""):
    st = {
        "intent": intent,
        "profiling_report": {"shape": {"rows": rows, "columns": len(columns)}, "columns": columns},
        "data_path": data_path,
    }
    if target:
        st["target_col"] = target
    return st


# ─────────────────────────────────────────────────────────────────────────────
def test_dynamism_target_conflict():
    print("\n[1] Dinamiklik — rol-içerik çelişkisi (uydurma adlar, hard-code YOK)")

    # Fraud vakası: is_fraud DOUBLE, 73k distinct → sürekli, ad ikili ima ediyor.
    st = make_state(299695,
                    {"is_fraud": col("DOUBLE", 73447, 0.245),
                     "amount": col("DOUBLE", 50000, 0.17)},
                    target="is_fraud", intent="regression")
    r = run_data_quality(st)
    check("is_fraud → failed", r["data_quality_status"] == "failed", r["data_quality_status"])
    check("reason=target_role_content_conflict",
          r["data_quality_report"]["decision"]["reason"] == "target_role_content_conflict",
          r["data_quality_report"]["decision"]["reason"])

    # Farklı uydurma ad: flag_suspicious, aynı bozulma → yine tetiklenmeli (dinamik).
    st2 = make_state(120000, {"flag_suspicious": col("DOUBLE", 9000, 0.42)},
                     target="flag_suspicious", intent="regression")
    r2 = run_data_quality(st2)
    check("flag_suspicious → failed (dinamik)", r2["data_quality_status"] == "failed",
          r2["data_quality_status"])


def test_clean_binary_not_triggered():
    print("\n[2a] Regresyon — temiz binary target TETİKLENMEZ")

    # is_churned: 2 distinct → nunique==2 KESİN binary, asla continuous.
    st = make_state(10000, {"is_churned": col("BIGINT", 2, 0.0002)}, target="is_churned")
    r = run_data_quality(st)
    check("is_churned → ok", r["data_quality_status"] == "ok", r["data_quality_status"])
    check("is_churned verdict=binary_ok",
          r["data_quality_report"]["target_health"]["verdict"] == "binary_ok")

    # Küçük-n: 8 satır, is_x 8 distinct sürekli görünse de non_null<30 → korumalı.
    st_small = make_state(8, {"is_x": col("DOUBLE", 8, 1.0)}, target="is_x")
    rs = run_data_quality(st_small)
    check("küçük-n (8 satır) → tetiklenmez", rs["data_quality_status"] != "failed",
          rs["data_quality_status"])
    check("küçük-n verdict=insufficient_rows",
          rs["data_quality_report"]["target_health"]["verdict"] == "insufficient_rows")

    # 8 satır temiz binary → binary_ok (distinct_ratio 0.25 > 0.20 OLSA DA nunique==2 kazanır).
    st_sb = make_state(8, {"churn": col("BIGINT", 2, 0.25)}, target="churn")
    rsb = run_data_quality(st_sb)
    check("8 satır 2-distinct binary → ok (nunique birincil)",
          rsb["data_quality_status"] == "ok", rsb["data_quality_status"])


def test_legit_continuous_feature():
    print("\n[2b] Regresyon — meşru sürekli FEATURE (target değil) DURDURMAZ")
    # target temiz binary; flag_score binary-adlı ama meşru sürekli bir FEATURE.
    st = make_state(10000,
                    {"churn": col("BIGINT", 2, 0.0002),
                     "flag_score": col("DOUBLE", 5000, 0.5),
                     "is_amount_ratio": col("DOUBLE", 8000, 0.8)},
                    target="churn")
    r = run_data_quality(st)
    check("meşru sürekli feature → status ok (durdurma YOK)",
          r["data_quality_status"] == "ok", r["data_quality_status"])
    notes = r["data_quality_report"]["semantic"]["notes"]
    noted_cols = {n["column"] for n in notes}
    check("flag_score sadece NOT olarak işaretlendi", "flag_score" in noted_cols, str(noted_cols))
    check("is_amount_ratio sadece NOT olarak işaretlendi", "is_amount_ratio" in noted_cols,
          str(noted_cols))


def test_credit_churn_same_path():
    print("\n[2c] Regresyon — kredi riski + bank churn: ok VE aynı node yolu")
    scenarios = {
        "credit_risk (loan_status, classification)":
            make_state(45000, {"loan_status": col("BIGINT", 2, 0.00004),
                               "income": col("DOUBLE", 30000, 0.66)},
                       target="loan_status", intent="classification"),
        "bank_churn (churn, classification)":
            make_state(10000, {"churn": col("BIGINT", 2, 0.0002),
                               "balance": col("DOUBLE", 6000, 0.6)},
                       target="churn", intent="classification"),
    }
    for name, st in scenarios.items():
        r = run_data_quality(st)
        st_after = {**st, **r}
        expected = route_after_execution(st_after)     # eski/mevcut fork (değişmedi)
        actual = route_after_data_quality(st_after)    # yeni router
        check(f"{name} → ok", r["data_quality_status"] == "ok", r["data_quality_status"])
        check(f"{name} → aynı yol ({expected})", actual == expected,
              f"actual={actual} expected={expected}")
        check(f"{name} → entity_resolution_node", actual == "entity_resolution_node", actual)


def test_routing_preserved():
    print("\n[2d] Regresyon — route_after_execution fork'u birebir korunuyor")
    cases = {
        "classification": "entity_resolution_node",
        "regression": "entity_resolution_node",
        "clustering": "entity_resolution_node",
        "deep_analysis": "analysis_node",
        "anomaly_detection": "automl_node",
        "eda": "report_node",
    }
    for intent, expected in cases.items():
        st = {"intent": intent, "data_quality_status": "ok"}
        check(f"intent={intent} → {expected}", route_after_data_quality(st) == expected,
              route_after_data_quality(st))
    # failed → her intent'te report_node kısa-devre
    st_failed = {"intent": "classification", "data_quality_status": "failed"}
    check("DQ failed → report_node", route_after_data_quality(st_failed) == "report_node",
          route_after_data_quality(st_failed))


def test_structural_quarantine_parse():
    print("\n[3] Yapısal — alan-sayısı invaryantı (kolon kayması karantinası)")
    # 5 kolonlu header; iki satır gömülü-virgüllü bare Timestamp içerir → 6 alan → karantina.
    raw = (
        "Columns: transaction_id, amount, transaction_time, shipping_km, is_fraud\n"
        "(1, 84.75, '2024-01-06 07:09:39', 370.95, 0)\n"
        "(2, 107.90, Timestamp('2024-01-09 23:13:47+0300', tz='Europe/Istanbul'), 149.62, 1)\n"
        "(3, 92.36, '2024-01-12 09:20:11', 164.08, 0)\n"
        "(4, 112.47, Timestamp('2024-01-15 20:00:04+0300', tz='Europe/Istanbul'), 397.40, 1)\n"
    )
    df = _parse_mcp_to_df(raw, "transactions")
    s = df.attrs.get("_dq_structural", {})
    check("2 satır karantinaya alındı", s.get("quarantined_rows") == 2, str(s))
    check("2 temiz satır kaldı (df)", len(df) == 2, f"len={len(df)}")
    check("kayan satır df'e SIZMADI (sessiz trim yok)",
          df["is_fraud"].tolist() == [0, 0], df["is_fraud"].tolist())

    # Karar: target varken q>0 → failed (etiket hizası kanıtlanamaz).
    st_seed = {"data_quality_report": {"structural": {
        "checked": True, "total_rows": 100, "quarantined_rows": 1, "mismatch_rate": 0.01,
        "tables": [{"table": "t", "expected_cols": 5}]}},
        "profiling_report": {"shape": {"rows": 99, "columns": 5},
                             "columns": {"is_fraud": col("BIGINT", 2, 0.02)}},
        "target_col": "is_fraud"}
    rt = run_data_quality(st_seed)
    check("target var + karantina → failed", rt["data_quality_status"] == "failed",
          rt["data_quality_status"])
    check("reason=structural_mismatch_target",
          rt["data_quality_report"]["decision"]["reason"] == "structural_mismatch_target")

    # Target YOK (eda) → degraded, temiz satırlarla devam.
    st_noT = {"data_quality_report": st_seed["data_quality_report"],
              "profiling_report": st_seed["profiling_report"], "intent": "eda"}
    rn = run_data_quality(st_noT)
    check("target yok + karantina → degraded", rn["data_quality_status"] == "degraded",
          rn["data_quality_status"])

    # Yüksek oran → her akışta failed.
    st_hi = {"data_quality_report": {"structural": {
        "checked": True, "total_rows": 100, "quarantined_rows": 40, "mismatch_rate": 0.40,
        "tables": [{"table": "t", "expected_cols": 5}]}},
        "profiling_report": {"shape": {"rows": 60, "columns": 5}, "columns": {}}}
    rh = run_data_quality(st_hi)
    check("yüksek karantina oranı → failed", rh["data_quality_status"] == "failed",
          rh["data_quality_status"])
    check("reason=structural_mismatch_high",
          rh["data_quality_report"]["decision"]["reason"] == "structural_mismatch_high")


def test_tr_en_messages():
    print("\n[4] TR/EN mesaj tutarlılığı")
    keys = ["dq_failed_title", "dq_failed_intro", "dq_target_conflict", "dq_structural",
            "dq_structural_high", "dq_action_hint", "dq_degraded_note"]
    for k in keys:
        tr, en = R(k, "tr"), R(k, "en")
        check(f"{k}: tr!=key & en!=key", tr != k and en != k, f"{tr!r}/{en!r}")
        check(f"{k}: tr!=en (gerçekten çevrildi)", tr != en)
    # Parametreli format çalışıyor mu
    msg = R("dq_target_conflict", "en", col="is_fraud", nunique=73447)
    check("param format (en) çalışıyor", "is_fraud" in msg and "73447" in msg, msg[:80])
    msg_tr = R("dq_structural", "tr", n=500, total=10000, cols=17)
    check("param format (tr) çalışıyor", "500" in msg_tr and "17" in msg_tr, msg_tr[:80])


def test_sampling_fix():
    print("\n[5] Sampling fix — dengesiz veride USING SAMPLE 0 döndürmüyor")
    # Dengesiz tablo: 5 minority (sınıf=1) + 95 majority (sınıf=0).
    df = pd.DataFrame({
        "x": list(range(100)),
        "y": [1] * 5 + [0] * 95,
    })

    # 5a) data_executor.DuckDBEngine.stratified_sample (fixli kod yolu)
    eng = DuckDBEngine()
    eng.load("t", df)
    sampled = eng.stratified_sample("t", target_col="y", n=50)
    eng.close()
    minority_n = int((sampled["y"] == 1).sum())
    check("stratified_sample: nadir sınıf (1) 0 DEĞİL", minority_n > 0,
          f"minority={minority_n}, total={len(sampled)}")
    check("stratified_sample: örnek boş değil", len(sampled) > 0, f"total={len(sampled)}")

    # 5b) feature_builder._load_data (stratified cap dalı) — MAX_TRAINING_ROWS'u düşür.
    import src.services.feature_builder as fb
    from src.core.config import settings as _settings
    orig = _settings.MAX_TRAINING_ROWS
    _settings.MAX_TRAINING_ROWS = 30   # 100 > 30 → stratified cap örnekleme tetiklenir
    try:
        with tempfile.TemporaryDirectory() as d:
            pq = os.path.join(d, "imb.parquet")
            df.to_parquet(pq)
            state = {"data_path": pq,
                     "target_definition": {"column_name": "y", "confirmed": True}}
            out = fb._load_data(state)
        check("feature_builder._load_data: boş DEĞİL", len(out) > 0, f"len={len(out)}")
        check("feature_builder._load_data: nadir sınıf korundu",
              int((out["y"] == 1).sum()) > 0, f"minority={(out['y']==1).sum()}")
    finally:
        _settings.MAX_TRAINING_ROWS = orig


def test_numeric_string_note_in_engine():
    print("\n[6] Ucuzluk/in-engine — sayı-ama-string DuckDB TRY_CAST ile yakalanıyor")
    with tempfile.TemporaryDirectory() as d:
        pq = os.path.join(d, "ns.parquet")
        # amount_str: sayısal ama VARCHAR yüklenmiş
        pd.DataFrame({
            "amount_str": [str(round(i * 1.5, 2)) for i in range(200)],
            "label": [0, 1] * 100,
        }).to_parquet(pq)
        st = {
            "intent": "classification", "target_col": "label", "data_path": pq,
            "profiling_report": {"shape": {"rows": 200, "columns": 2}, "columns": {
                "amount_str": col("VARCHAR", 200, 1.0),
                "label": col("BIGINT", 2, 0.01),
            }},
        }
        r = run_data_quality(st)
    notes = {n["column"]: n for n in r["data_quality_report"]["semantic"]["notes"]}
    check("amount_str numeric_loaded_as_string notu var", "amount_str" in notes, str(list(notes)))
    check("in-engine parse_rate ~1.0",
          notes.get("amount_str", {}).get("numeric_parse_rate", 0) >= settings.DQ_NUMERIC_STRING_RATE,
          str(notes.get("amount_str")))
    check("sayı-ama-string status'u DÜŞÜRMEDİ (yalnız not)", r["data_quality_status"] == "ok",
          r["data_quality_status"])


def main():
    print("=" * 70)
    print("VERİ KALİTESİ KATMANI — TEST")
    print("=" * 70)
    test_dynamism_target_conflict()
    test_clean_binary_not_triggered()
    test_legit_continuous_feature()
    test_credit_churn_same_path()
    test_routing_preserved()
    test_structural_quarantine_parse()
    test_tr_en_messages()
    test_sampling_fix()
    test_numeric_string_note_in_engine()

    print("\n" + "=" * 70)
    if _FAILS:
        print(f"SONUÇ: {len(_FAILS)} BAŞARISIZ")
        for f in _FAILS:
            print("  ✗ " + f)
        sys.exit(1)
    print("SONUÇ: TÜM TESTLER GEÇTİ ✓")


if __name__ == "__main__":
    main()
