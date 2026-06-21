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

from src.services.feature_builder import (
    build_features,
    _build_timeseries_features,
    _entity_aware_split_positions,
    _chronological_order,
    ColumnMetadata,
)


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
    assert ts.get("entity_col") is None, "entity_col verilmedi → global mod olmali (entity_col=None)"
    assert ts.get("dropped_entities", 0) == 0, "global modda dropped_entities 0 olmali"
    assert rep.get("split_strategy") == "global_chronological", (
        f"entity_col yok → FAZ 1 global kronolojik split bekleniyordu: {rep.get('split_strategy')}")
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

    # ── FAZ B1: rolling ──
    assert ts["rolling_columns"], "rolling kolonu uretilmedi"
    assert any(c.startswith("roll7_mean_") for c in ts["rolling_columns"]), "roll7_mean yok"
    assert any(c.startswith("roll7_std_") for c in ts["rolling_columns"]), "roll7_std yok"
    # 1-birim rolling URETILMEMELI (anlamsiz)
    assert not any(c.startswith("roll1_") for c in ts["rolling_columns"]), "roll1_* uretilmemeli"
    assert ts["roll_windows"] == [7, 30], f"roll pencereleri [7,30] bekleniyordu: {ts['roll_windows']}"
    # rolling kolonlari kardinaliteden SILINMEMELI (muafiyet); leakage_dropped'ta olmamali
    card_dropped = {d["column"] for d in rep["leakage_dropped"] if "kardinalite" in d.get("reason", "")}
    roll_card_dropped = [c for c in ts["rolling_columns"] if c in card_dropped]
    assert not roll_card_dropped, f"rolling kolonu kardinaliteden silindi (muafiyet calismadi): {roll_card_dropped}"
    final_rolls = [c for c in rep["final_features"] if c.startswith("roll")]
    assert final_rolls, f"hicbir rolling kolonu final feature'da yok: {rep['final_features']}"
    print(f"  >> rolling hayatta (final): {final_rolls}")
    print(f"  frekans={ts['frequency']} lag-pencere={ts['windows']} roll-pencere={ts['roll_windows']}")
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
    assert not [c for c in rep["final_features"] if c.startswith("roll")], "tabular yolda rolling kolonu olmamali"
    assert rep.get("split_strategy") == "random", f"tabular → random split bekleniyordu: {rep.get('split_strategy')}"
    print(f"  final features={rep['final_features']}  n_train={rep['n_train']} n_test={rep['n_test']}")
    print("  timeseries_lag=None, lag kolonu yok → tabular akis korunmus")
    print("  TEST 2 GECTI ✔")


def test_3_no_current_leak():
    """KRITIK: rolling penceresi simdiki satiri ICERMEMELI (shift(1)).
    Monoton seri sales=0,1,2,... ile:
      roll7_mean_sales[t] = mean(sales[t-7..t-1]) = sales[t] - 4   (t HARIC)
    Icerseydi mean(sales[t-6..t]) = sales[t] - 3 olurdu. -4 vs -3 → shift(1) kaniti.
    """
    print("\n========== TEST 3: shift(1) kaniti (simdiki satir disli) ==========")
    n = 20
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "sales": [float(i) for i in range(n)],   # 0,1,2,...,19
    })
    meta = ColumnMetadata(_meta([("date", "timestamp"), ("sales", "metric")]), {})
    plan = {"target": {"column_name": "sales"}}

    out_df, info = _build_timeseries_features(df.copy(), plan, "date", meta)
    # daily → lag [1,7,30]; n=20 → w<20 → [1,7]; roll_windows=[7]
    assert info["roll_windows"] == [7], f"roll pencereleri [7] bekleniyordu: {info['roll_windows']}"
    assert "roll7_mean_sales" in out_df.columns, "roll7_mean_sales yok"
    assert not any(c.startswith("roll1_") for c in out_df.columns), "roll1_* uretilmemeli"

    # KANIT: her satirda roll7_mean_sales == sales - 4 (t-7..t-1 ortalamasi, t HARIC)
    diff = (out_df["roll7_mean_sales"].astype(float) - out_df["sales"].astype(float))
    assert (diff.round(6) == -4.0).all(), (
        f"shift(1) KANITI BASARISIZ: roll7_mean - sales == -4 bekleniyordu, "
        f"bulunan essiz degerler={sorted(diff.round(6).unique())} "
        f"(-3 cikarsa simdiki satir pencereye dahil = SIZINTI)"
    )
    print(f"  roll7_mean_sales - sales == -4 (tum satirlar) → simdiki satir HARIC, sizinti YOK ✔")
    print(f"  ornek: sales={out_df['sales'].tolist()[:5]} roll7_mean={out_df['roll7_mean_sales'].round(2).tolist()[:5]}")
    print("  TEST 3 GECTI ✔")


def test_4_monthly_edge():
    print("\n========== TEST 4: frekans kenar durumu (aylik kucuk set) ==========")
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
    print(f"  frekans={ts['frequency']} lag-pencere={ts['windows']} roll-pencere={ts.get('roll_windows')} "
          f"dusen lead={ts['dropped_lead_rows']}")
    print(f"  n_train={rep['n_train']} n_test={rep['n_test']}  cokme yok")
    print("  TEST 4 GECTI ✔")


def test_5_multi_entity_integration():
    """Leg 1 (integration): build_features uctan uca, state'ten entity_col okunuyor,
    grup-ici mod aktif, pipeline tamamlaniyor."""
    print("\n========== TEST 5: multi-entity entegrasyon (state→entity_col) ==========")
    rng = np.random.default_rng(11)
    frames = []
    for sid, base in [("A", 100.0), ("B", 500.0)]:
        n = 120
        frames.append(pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=n, freq="D"),
            "sales": np.cumsum(rng.normal(0, 1, n)) + base,
            "temp": rng.normal(20, 5, n),
            "store_id": sid,
        }))
    df = pd.concat(frames, ignore_index=True)
    path = _write_parquet(df)
    db_intel = _meta([("date", "timestamp"), ("sales", "metric"),
                      ("temp", "metric"), ("store_id", "identifier")])
    plan = {"target": {"column_name": "sales"}, "analysis_summary": "panel satis"}
    state = _base_state(path, db_intel, plan, modality="timeseries", time_col="date")
    state["entity_col"] = "store_id"   # B2-a: state'ten okunmali

    out = build_features(state)
    assert out.get("error") is None, f"hata: {out.get('error')}"
    ts = out["feature_report"]["timeseries_lag"]
    assert ts["entity_col"] == "store_id", f"grup-ici mod bekleniyordu: entity_col={ts.get('entity_col')}"
    assert ts["lag_columns"] and ts["rolling_columns"], "lag/rolling uretilmedi"
    assert out["feature_report"]["split_strategy"] == "entity_aware", (
        f"entity_col dolu → entity-aware split bekleniyordu: {out['feature_report'].get('split_strategy')}")
    print(f"  entity_col={ts['entity_col']} frekans={ts['frequency']} "
          f"lag={len(ts['lag_columns'])} rolling={len(ts['rolling_columns'])} "
          f"dusen-entity={ts['dropped_entities']}")
    print("  TEST 5 GECTI ✔")


def test_6_group_boundary_proof():
    """Leg 1+2+3 (matematiksel): grup-ici transform entity sinirini KORUR; global gecirir.
    A: sales 0..15, B: sales 100..115 (ayri tarih araliklari).
      roll3_mean == sales-2 her iki entity'de (grup ICINDE) → transform sinir korur.
      B'nin lag'i hep >=100 (B'den), A'nin <100 (A'dan) → sinir ihlali yok.
    GLOBAL modda (entity_col bos) ESKI davranis: sinir GECILIR (B'nin lag'i A'dan gelir)."""
    print("\n========== TEST 6: grup-sinir kaniti (transform vs global) ==========")
    a = pd.DataFrame({
        "month": pd.date_range("2020-01-31", periods=16, freq="ME"),
        "sales": [float(i) for i in range(16)],          # 0..15
        "store": "A",
    })
    b = pd.DataFrame({
        "month": pd.date_range("2022-01-31", periods=16, freq="ME"),   # ayri aralik
        "sales": [float(100 + i) for i in range(16)],    # 100..115
        "store": "B",
    })
    df = pd.concat([a, b], ignore_index=True)
    meta = ColumnMetadata(_meta([("month", "timestamp"), ("sales", "metric"),
                                 ("store", "identifier")]), {})
    plan = {"target": {"column_name": "sales"}}

    # ── GRUP-ICI ──
    out_g, info_g = _build_timeseries_features(df.copy(), plan, "month", meta, entity_col="store")
    assert info_g["entity_col"] == "store", "grup-ici mod entity_col yazmali"
    assert not any(c.startswith("roll1_") for c in out_g.columns), "roll1_* olmamali"
    # KANIT 1: roll3_mean == sales - 2 (her iki entity, grup ICINDE — sinir gecilmedi)
    dg = (out_g["roll3_mean_sales"].astype(float) - out_g["sales"].astype(float))
    assert (dg.round(6) == -2.0).all(), (
        f"grup-ici roll3_mean - sales == -2 bekleniyordu: essiz={sorted(dg.round(6).unique())}")
    # KANIT 2: entity sinirinda lag karismiyor
    b_rows = out_g[out_g["store"] == "B"]
    a_rows = out_g[out_g["store"] == "A"]
    assert b_rows["lag_1_sales"].min() >= 100, "B'nin lag'i A'dan geldi (sinir ihlali!)"
    assert a_rows["lag_1_sales"].max() < 100, "A'nin lag'i B'den geldi (sinir ihlali!)"
    print(f"  GRUP-ICI: roll3_mean-sales==-2 (tum satirlar), B.lag>=100 & A.lag<100 → sinir KORUNDU ✔")

    # ── GLOBAL (entity_col bos) → ESKI davranis: sinir GECILIR ──
    out_x, info_x = _build_timeseries_features(df.copy(), plan, "month", meta, entity_col="")
    assert info_x["entity_col"] is None and info_x["dropped_entities"] == 0, "global mod info yanlis"
    crossed = (out_x[out_x["store"] == "B"]["lag_1_sales"].astype(float) < 100).any()
    formula_holds_globally = ((out_x["roll3_mean_sales"].astype(float)
                               - out_x["sales"].astype(float)).round(6) == -2.0).all()
    assert crossed, "global modda B'nin lag'i A'dan gelmeliydi (eski global davranis)"
    assert not formula_holds_globally, "global modda formul TUTMAMALI (sinir gecilir)"
    print(f"  GLOBAL (entity_col bos): B.lag A'dan geliyor, formul tutmuyor → ESKI davranis BIREBIR ✔")
    print("  TEST 6 GECTI ✔")


def test_7_small_entity_report():
    """Leg 5: penceresinden kisa entity KOMPLE duser AMA sessiz degil — log + info dict."""
    print("\n========== TEST 7: kucuk-entity raporu (sessiz degil) ==========")
    rng = np.random.default_rng(5)
    big = pd.DataFrame({
        "month": pd.date_range("2020-01-31", periods=24, freq="ME"),
        "sales": np.cumsum(rng.normal(0, 1, 24)) + 50,
        "store": "BIG",
    })
    small = pd.DataFrame({   # 3 satir < en buyuk pencere (12) → komple duser
        "month": pd.date_range("2021-01-31", periods=3, freq="ME"),
        "sales": [10.0, 11.0, 12.0],
        "store": "SMALL",
    })
    df = pd.concat([big, small], ignore_index=True)
    meta = ColumnMetadata(_meta([("month", "timestamp"), ("sales", "metric"),
                                 ("store", "identifier")]), {})
    plan = {"target": {"column_name": "sales"}}

    out, info = _build_timeseries_features(df.copy(), plan, "month", meta, entity_col="store")
    assert info["dropped_entities"] >= 1, f"kucuk entity dusmedi: {info['dropped_entities']}"
    assert info["dropped_rows_from_small_entities"] >= 3, (
        f"dusen kucuk-entity satiri raporlanmadi: {info['dropped_rows_from_small_entities']}")
    assert "SMALL" not in set(out["store"].unique()), "kucuk entity hala veride"
    assert "BIG" in set(out["store"].unique()), "buyuk entity de dustu (olmamali)"
    print(f"  dropped_entities={info['dropped_entities']} "
          f"dropped_rows_from_small_entities={info['dropped_rows_from_small_entities']} "
          f"(SMALL dustu, BIG kaldi, RAPORLANDI) ✔")
    print("  TEST 7 GECTI ✔")


def test_8_entity_aware_split_aligned():
    """Leg 1: hizali zaman, multi-entity → her entity'nin son %20'si test; her entity
    HEM train HEM test'te temsil ediliyor (hicbiri tamamen bir tarafta degil)."""
    print("\n========== TEST 8: entity-aware split (hizali zaman) ==========")
    entity_key = pd.Series(["A"] * 10 + ["B"] * 10)
    a = pd.date_range("2020-01-01", periods=10, freq="D")
    b = pd.date_range("2020-01-01", periods=10, freq="D")   # hizali (ayni aralik)
    time_key = pd.Series(list(a) + list(b))
    idx = entity_key.index
    ent = entity_key.to_numpy()

    train_pos, test_pos, imb = _entity_aware_split_positions(time_key, entity_key, idx, 0.2)
    assert imb == 0, f"dengeli veride imbalanced 0 olmali: {imb}"
    assert len(test_pos) == 4 and len(train_pos) == 16, f"10+10, %20 → 4 test/16 train: {len(test_pos)}/{len(train_pos)}"
    assert set(ent[test_pos]) == {"A", "B"}, "her entity test'te olmali"
    assert set(ent[train_pos]) == {"A", "B"}, "her entity train'de olmali"
    # her entity'nin test'i KENDI son 2'si (en yeni); train/test kesismiyor
    assert set(test_pos).isdisjoint(set(train_pos)), "train/test ortusuyor"
    print(f"  test entity'leri={sorted(set(ent[test_pos]))} train entity'leri={sorted(set(ent[train_pos]))} "
          f"imbalanced={imb} → her entity iki tarafta da ✔")
    print("  TEST 8 GECTI ✔")


def test_9_staggered_global_vs_entity():
    """Leg 2 (KRITIK): kaydirmali zaman — patolojinin cozuldugu kanit.
    A erken donem, B gec donem. GLOBAL son-%20 → test TAMAMEN B (A'nin gelecegi hic test
    edilmez, model B'yi hic gormeden B'de degerlendirilir). ENTITY-AWARE → test HER iki
    entity'den (her biri kendi gecmisiyle train, gelecegiyle test)."""
    print("\n========== TEST 9: kaydirmali zaman — GLOBAL vs ENTITY-AWARE ==========")
    entity_key = pd.Series(["A"] * 10 + ["B"] * 10)
    a = pd.date_range("2020-01-01", periods=10, freq="D")   # ERKEN
    b = pd.date_range("2021-01-01", periods=10, freq="D")   # GEC (ayri aralik)
    time_key = pd.Series(list(a) + list(b))
    idx = entity_key.index
    ent = entity_key.to_numpy()

    # GLOBAL (mevcut FAZ 1 davranisi): son %20 = en yeni 4 satir → hepsi B
    order = _chronological_order(time_key, idx)
    n_test = max(1, int(round(len(idx) * 0.2)))
    global_test = order[-n_test:]
    assert set(ent[global_test]) == {"B"}, f"GLOBAL test tek entity (B) olmaliydi: {set(ent[global_test])}"

    # ENTITY-AWARE: test her iki entity'den
    train_pos, test_pos, imb = _entity_aware_split_positions(time_key, entity_key, idx, 0.2)
    assert set(ent[test_pos]) == {"A", "B"}, f"entity-aware test cok-entity olmali: {set(ent[test_pos])}"
    assert set(ent[train_pos]) == {"A", "B"}, "entity-aware train cok-entity olmali"
    print(f"  GLOBAL test entity'leri={sorted(set(ent[global_test]))}  ← PATOLOJI (tek entity)")
    print(f"  ENTITY-AWARE test entity'leri={sorted(set(ent[test_pos]))}  ← COZUM (her iki entity)")
    print("  TEST 9 GECTI ✔")


def test_10_small_entity_imbalanced():
    """Leg 5: 1-satirlik entity → cokme yok, dengesiz split raporlaniyor (imbalanced>=1)."""
    print("\n========== TEST 10: kucuk entity dengesiz split (sessiz degil) ==========")
    entity_key = pd.Series(["BIG"] * 10 + ["SMALL"] * 1)
    big = pd.date_range("2020-01-01", periods=10, freq="D")
    time_key = pd.Series(list(big) + [pd.Timestamp("2020-06-01")])
    idx = entity_key.index
    ent = entity_key.to_numpy()

    train_pos, test_pos, imb = _entity_aware_split_positions(time_key, entity_key, idx, 0.2)
    assert imb >= 1, f"1-satirlik entity dengesiz olmali (train bos): imbalanced={imb}"
    small_pos = [i for i in range(len(idx)) if ent[i] == "SMALL"][0]
    assert small_pos in set(test_pos.tolist()), "SMALL'in tek satiri test'te olmali"
    assert small_pos not in set(train_pos.tolist()), "SMALL train'de olmamali (tek satir)"
    # BIG normal bolunuyor, cokme yok
    assert set(ent[test_pos]) == {"BIG", "SMALL"} or "BIG" in set(ent[train_pos])
    print(f"  imbalanced={imb} (SMALL train'i bos, RAPORLANDI), BIG normal bolundu, cokme yok ✔")
    print("  TEST 10 GECTI ✔")


if __name__ == "__main__":
    test_1_timeseries_daily()
    test_2_tabular_unchanged()
    test_3_no_current_leak()
    test_4_monthly_edge()
    test_5_multi_entity_integration()
    test_6_group_boundary_proof()
    test_7_small_entity_report()
    test_8_entity_aware_split_aligned()
    test_9_staggered_global_vs_entity()
    test_10_small_entity_imbalanced()
    print("\n=== ON AYAK DA GECTI ===")
