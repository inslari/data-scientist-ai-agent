"""FAZ A — time-series lag katmani icin gecici standalone test (pytest degil).
Calistir: uv run python test_ts_lag_faza.py
Bu dosya commit edilmeyecek; dogrulama bitince silinebilir.

Uc ayak:
  1) Time-series CSV (gunluk) → lag uretiliyor, frekans=daily, lead satir dustu,
     lag kolonu final feature'larda.
  2) Tabular (modality bos) → lag YOK, aggregation/eski akis aynen.
  3) Frekans kenar durumu (aylik kucuk set) → makul bucket veya WARN, cokme yok.
"""
import os
import tempfile

# App DB'ye ASLA dokunma (CLAUDE.md): testten once sandbox sqlite'a sabitle.
os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_test_ts_faza.sqlite")

import duckdb
import numpy as np
import pandas as pd

from src.services.feature_builder import build_features


def _write_parquet(df: pd.DataFrame) -> str:
    path = os.path.join(tempfile.gettempdir(), f"ts_faza_{np.random.randint(1e9)}.parquet")
    con = duckdb.connect(":memory:")
    con.register("t", df)
    con.execute(f"COPY t TO '{path}' (FORMAT PARQUET)")
    con.close()
    return path


def _meta(columns: list[tuple[str, str]]) -> dict:
    """columns: [(name, semantic_type), ...] → db_intelligence dict."""
    return {
        "columns": [
            {"column_name": n, "semantic_type": s, "importance_score": 0.5}
            for n, s in columns
        ],
        "tables": [],
    }


def _base_state(path, db_intel, feature_plan, modality="", time_col="", intent="regression"):
    return {
        "data_path": path,
        "sql_results": [],
        "profiling_report": {},
        "db_intelligence": db_intel,
        "feature_plan": feature_plan,
        "target_definition": {"column_name": feature_plan["target"]["column_name"], "confirmed": True},
        "modality": modality,
        "time_col": time_col,
        "intent": intent,
    }


def test_1_timeseries_daily():
    print("\n========== TEST 1: time-series gunluk ==========")
    rng = np.random.default_rng(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    # rastgele yuruyus hedef + birkac sayisal feature
    sales = np.cumsum(rng.normal(0, 1, n)) + 100
    df = pd.DataFrame({
        "date": dates,
        "sales": sales,                      # numerik hedef (gecmisi laglenir)
        "temp": rng.normal(20, 5, n),        # sayisal feature
        "promo": rng.integers(0, 2, n),      # sayisal feature
        "store_id": 1,                       # identifier (laglenmez + multi-entity WARN testi)
    })
    path = _write_parquet(df)
    db_intel = _meta([
        ("date", "timestamp"), ("sales", "metric"),
        ("temp", "metric"), ("promo", "metric"), ("store_id", "identifier"),
    ])
    plan = {"target": {"column_name": "sales"}, "analysis_summary": "gunluk satis"}
    state = _base_state(path, db_intel, plan, modality="timeseries", time_col="date")

    out = build_features(state)
    assert out.get("error") is None, f"hata: {out.get('error')}"
    rep = out["feature_report"]
    ts = rep.get("timeseries_lag")
    assert ts is not None, "timeseries_lag raporu yok"
    assert ts["frequency"] == "daily", f"frekans daily bekleniyordu: {ts['frequency']}"
    assert ts["lag_columns"], "lag kolonu uretilmedi"
    assert ts["dropped_lead_rows"] > 0, "lead satir dusurulmedi"
    lag_target = [c for c in ts["lag_columns"] if c.startswith("lag_") and c.endswith("_sales")]
    assert lag_target, "hedefin gecmis lag'i (lag_k_sales) uretilmedi"
    final_lags = [c for c in rep["final_features"] if c.startswith("lag_")]
    assert final_lags, f"hicbir lag kolonu final feature'da yok: {rep['final_features']}"
    # KRITIK: hedefin gecmis lag'i artik kardinaliteden SILINMIYOR → final feature'da
    final_target_lags = [c for c in rep["final_features"] if c.startswith("lag_") and c.endswith("_sales")]
    assert final_target_lags, (
        f"hedefin lag'i (lag_*_sales) final feature'da yok — kardinalite muafiyeti calismadi: "
        f"{rep['final_features']}"
    )
    print(f"  >> hedef lag'i hayatta (final): {final_target_lags}")
    print(f"  frekans={ts['frequency']} pencereler={ts['windows']}")
    print(f"  uretilen lag={ts['lag_columns']}")
    print(f"  dusen lead satir={ts['dropped_lead_rows']}  n_train={rep['n_train']} n_test={rep['n_test']}")
    print(f"  final lag feature'lar={final_lags}")
    print("  TEST 1 GECTI ✔")


def test_2_tabular_unchanged():
    print("\n========== TEST 2: tabular (modality bos) ==========")
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(5, 2, n),
        "f3": rng.integers(0, 100, n),
        "y": rng.integers(0, 2, n),
    })
    path = _write_parquet(df)
    db_intel = _meta([("f1", "metric"), ("f2", "metric"), ("f3", "metric"), ("y", "target")])
    plan = {"target": {"column_name": "y"}, "analysis_summary": "tabular"}
    state = _base_state(path, db_intel, plan, modality="", time_col="", intent="classification")

    out = build_features(state)
    assert out.get("error") is None, f"hata: {out.get('error')}"
    rep = out["feature_report"]
    assert rep.get("timeseries_lag") is None, "tabular yolda timeseries_lag dolu olmamali"
    assert not [c for c in rep["final_features"] if c.startswith("lag_")], "tabular yolda lag kolonu olmamali"
    print(f"  final features={rep['final_features']}  n_train={rep['n_train']} n_test={rep['n_test']}")
    print("  timeseries_lag=None, lag kolonu yok → tabular akis korunmus")
    print("  TEST 2 GECTI ✔")


def test_3_monthly_edge():
    print("\n========== TEST 3: frekans kenar durumu (aylik kucuk set) ==========")
    n = 40
    dates = pd.date_range("2020-01-31", periods=n, freq="ME")  # ay sonu = ~30 gun
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "month": dates,
        "revenue": np.cumsum(rng.normal(0, 1, n)) + 50,
        "cost": rng.normal(10, 3, n),
    })
    path = _write_parquet(df)
    db_intel = _meta([("month", "timestamp"), ("revenue", "metric"), ("cost", "metric")])
    plan = {"target": {"column_name": "revenue"}, "analysis_summary": "aylik gelir"}
    state = _base_state(path, db_intel, plan, modality="timeseries", time_col="month")

    out = build_features(state)
    assert out.get("error") is None, f"hata (cokme): {out.get('error')}"
    rep = out["feature_report"]
    ts = rep.get("timeseries_lag")
    assert ts is not None, "timeseries_lag raporu yok"
    assert ts["frequency"] in ("monthly", "unknown"), f"beklenmeyen frekans: {ts['frequency']}"
    print(f"  frekans={ts['frequency']} pencereler={ts['windows']} dusen lead={ts['dropped_lead_rows']}")
    print(f"  n_train={rep['n_train']} n_test={rep['n_test']}  cokme yok")
    print("  TEST 3 GECTI ✔")


if __name__ == "__main__":
    test_1_timeseries_daily()
    test_2_tabular_unchanged()
    test_3_monthly_edge()
    print("\n=== UC AYAK DA GECTI ===")
