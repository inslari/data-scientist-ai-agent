"""Katman 1 — ImputeStep parity + #4 delta testi.

  * İncelik 1 (test_oracle_is_impute_only): ``_clean_data``'nın impute-DIŞI hiçbir
    hücreyi oynatmadığını ÖNCE kanıtla — yoksa parity yanlış sebeple kırılır ve
    ImputeStep suçlanır. Oracle'ın temiz olduğunu umma, assert et.
  * Mod A (test_mod_a_equivalence_full_df): ImputeStep'e TÜM-df verildiğinde (#4 farkı
    nötr) çıktı ``_clean_data`` ile BİREBİR → reimplementasyon sadakati.
  * Mod B (test_mod_b_delta_raw_vs_rounded): train-fit vs full-df median deltası; İncelik
    2 gereği HAM (dağılım kayması) ve YUVARLANMIŞ-int (modelin gördüğü) ayrı ölçülür.

Bu dosya sonraki her adımda (ratio, count-encode, scaler ...) büyüyecek — parity
baştan kurulur, 12 adım biriktikten sonra iğne aranmaz.
"""

import glob
import inspect
import os

import numpy as np
import pandas as pd
import pytest
import sklearn

from src.services.feature_builder import (
    _add_missing_indicators, _apply_domain_leakage_drops_tt, _build_ratio_features,
    assemble_pipeline_ctx,
    _build_zero_missing_flags, _clean_data, _count_encode_tt, _detect_leakage_tt,
    _encode_categorical_tt, _finalize_ratio_features, _groupby_deviation_tt,
    _scale_features_tt, _select_features_tt, _select_univariate_tt,
)
from src.services.inference.orchestrator import (
    PIPELINE, StepSpec, _SCOPE_RULES, FrameShapeDivergence, MissingFeatureProduction,
    StructuralPreconditionViolation, UnsupportedPipelineScope, check_pipeline_scope,
    check_structural_preconditions,
    fit_pipeline, transform_pipeline,
)
from src.services.inference.shadow import run_shadow_comparison
from src.services.inference.pickle_io import (
    PickleRef, SklearnVersionMismatch, check_sklearn_compat, enforce_sklearn_compat,
    load_pickle_ref, make_pickle_ref,
)
from src.services.inference.protocol import DefaultFitTransform
from src.services.inference.steps import (
    CountEncodeStep, GroupbyDeviationStep, ImputeStep, InfImputeStep, NormalizeStep,
    DropColumnsStep, MissingIndicatorStep, ZeroFlagStep, ZScoreStep,
    OneHotStep, RatioConstructStep, RatioFinalizeStep, RobustScaleStep, SelectDropStep,
    StandardScaleStep, TargetEncodeStep, ZScoreStep, decide_encode_route, decide_scaler_route,
)


def _find_csv(pattern: str) -> "str | None":
    """Gerçek-kolon parity için gitignored data/csvs altında CSV bul (yoksa None → skip)."""
    m = glob.glob(os.path.join("data", "csvs", pattern))
    return m[0] if m else None


def _fixture() -> pd.DataFrame:
    """Duplicate'siz, currency-string'siz, StringDtype'sız → ``_clean_data``'nın TEK
    değer-değişimi impute olur (temiz oracle). Null'lar: Float64 + Int64 + object; ayrıca
    null'suz bir kolon (dokunulmamalı) ve round+int yolunu tetikleyen Int64."""
    return pd.DataFrame({
        "num_float": pd.array([1.0, 2.0, None, 4.0, 100.0], dtype="Float64"),
        "num_int":   pd.array([1, 2, 3, None, 10], dtype="Int64"),   # median 2.5 → round → 2
        "cat":       pd.Series(["a", "a", None, "b", "a"], dtype="object"),  # mode "a"
        "full":      [1.0, 2.0, 3.0, 4.0, 5.0],   # null-suz → hiç dokunulmamalı
    })


def _as_str(s: pd.Series) -> list:
    return [str(v) for v in s.tolist()]


def test_oracle_is_impute_only():
    """İncelik 1: _clean_data impute-dışı hücre oynatmıyor mu?"""
    fx = _fixture()
    out = _clean_data(fx.copy())
    assert len(out) == len(fx), "drop_duplicates satır düşürdü — fixture dup içeriyor olmalı"
    for col in fx.columns:
        notnull_mask = fx[col].notna().to_numpy()
        left = _as_str(fx[col][notnull_mask])
        right = _as_str(out[col][notnull_mask])
        assert left == right, f"oracle impute-DIŞI '{col}' hücresini oynattı: {left} → {right}"


def test_mod_a_equivalence_full_df():
    """Mod A: ImputeStep(full-df) impute çıktısı _clean_data ile birebir (#4 nötr)."""
    fx = _fixture()
    oracle = _clean_data(fx.copy())
    step = ImputeStep()
    params = step.fit(fx.copy())              # TÜM-df → #4 farkı yok, salt reimplementasyon
    out = step.transform(fx.copy(), params)
    for col in ["num_float", "num_int", "cat", "full"]:
        assert _as_str(oracle[col]) == _as_str(out[col]), \
            f"'{col}' parity kırıldı: oracle={_as_str(oracle[col])} step={_as_str(out[col])}"


def test_mod_b_delta_raw_vs_rounded():
    """Mod B: train-fit vs full-df median deltası — ham ve yuvarlanmış AYRI (İncelik 2)."""
    fx = _fixture()
    # split her iki null'u da train'e almalı (num_float idx2, num_int idx3) — aksi halde
    # o kolonda train-null yok ve fit dolgu öğrenmez (doğru davranış, ama delta ölçülemez).
    train = fx.iloc[:4]                       # ilk 4 satır
    step = ImputeStep()
    p_train = step.fit(train.copy())
    p_full = step.fit(fx.copy())

    # num_int: train [1,2,3,None]→median 2.0 (round→2) ; full [1,2,3,None,10]→median 2.5 (round→2)
    #   → HAM delta 0.5 (dağılım kayması VAR) ama YUVARLANMIŞ delta 0 (model aynı görür)
    assert "num_int" in p_train["numeric_median_raw"] and "num_int" in p_full["numeric_median_raw"]
    raw_delta = p_full["numeric_median_raw"]["num_int"] - p_train["numeric_median_raw"]["num_int"]
    applied_delta = p_full["numeric_median"]["num_int"] - p_train["numeric_median"]["num_int"]
    assert raw_delta != 0.0, "ham median kayması bekleniyordu (dağılım farkı)"
    assert applied_delta == 0, "num_int yuvarlanmış delta 0 olmalı — yuvarlama ham farkı yuttu"

    # İncelik 2 makinesi çalışıyor: ham ve yuvarlanmış AYRI erişilebilir, karışmıyor
    assert "num_int" in p_full["integer_cols"], "int kolon round+int olarak işaretlenmeli"
    assert isinstance(p_full["numeric_median"]["num_int"], int)
    assert isinstance(p_full["numeric_median_raw"]["num_int"], float)


# ── ratio_finalize adımı ─────────────────────────────────────
# Bu adımda A/B YOK: winsor zaten train-fit'ti (davranış değişmiyor, yalnız imza) →
# ölçülecek delta yok. Yalnız parity. İlke: yalnız SIZINTI-DÜZELTEN adımlar (impute #4,
# zscore #5) A/B taşır; sadık-yeniden-yazım adımları yalnız parity.


def _ratio_fixture() -> pd.DataFrame:
    """Oran kolonları: sonlu + inf (payda-sıfır) + NaN (kaynak-missing) + uç değerler
    (winsor sınavı). 'other' ratio değil → dokunulmamalı."""
    return pd.DataFrame({
        "ratio_a_b": [0.5, 1.0, 2.0, np.inf, np.nan, 100.0, -50.0, 1.5, 0.8, 3.0],
        "ratio_c_d": [1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85, 1.0],
        "other":     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


def test_ratio_mod_a_equivalence_full_df():
    """Mod A: RatioFinalizeStep(full-df) == eski _finalize_ratio_features birebir.

    fit→params→transform ayrışması, eski tek-atış _apply_ratio_stats(_fit_ratio_stats)
    ile aynı sonucu vermeli (mantık değişmiyor, yalnız imza). Eşleşmezse replika sadık değil."""
    fx = _ratio_fixture()
    cols = ["ratio_a_b", "ratio_c_d"]
    step = RatioFinalizeStep({"columns": cols})
    params = step.fit(fx.copy())
    out = step.transform(fx.copy(), params)
    oracle = _finalize_ratio_features(fx.copy(), cols)
    for col in fx.columns:
        assert _as_str(out[col]) == _as_str(oracle[col]), \
            f"'{col}' ratio parity kırıldı: step={_as_str(out[col])} oracle={_as_str(oracle[col])}"


def test_ratio_none_drop():
    """None-drop: train'de finite değeri OLMAYAN oran kolonu → fit None → transform düşürür
    (eski _apply_ratio_stats ile aynı). İnference'ta bu kolon zaten final_features'ta yok +
    reindex de eler → çakışma yok; None-drop donmuş params'tan, tek karar."""
    fx = pd.DataFrame({
        "ratio_all_nan": [np.nan, np.inf, -np.inf, np.nan, np.inf],   # finite YOK
        "ratio_ok":      [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    cols = ["ratio_all_nan", "ratio_ok"]
    step = RatioFinalizeStep({"columns": cols})
    params = step.fit(fx.copy())
    assert params["ratio_all_nan"] is None, "finite-yok kolon None (drop sinyali) olmalı"
    out = step.transform(fx.copy(), params)
    assert "ratio_all_nan" not in out.columns, "None-stat kolon düşmeli"
    assert "ratio_ok" in out.columns
    oracle = _finalize_ratio_features(fx.copy(), cols)
    assert "ratio_all_nan" not in oracle.columns, "eski yol da düşürmeli (parity)"
    assert _as_str(out["ratio_ok"]) == _as_str(oracle["ratio_ok"])


# ── count_encode adımı ───────────────────────────────────────
# A/B YOK (sadık-yeniden-yazım: zaten train-fit→test-map). Parity artık STANDART olarak
# fixture (mantık doğru) + gerçek-kolon (ölçek kenar-durumu yok) İKİSİ birden.


def _cat_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "country":   ["FR", "FR", "DE", "ES", "DE", "FR", "ES", "DE"],   # card 3 → uygun
        "other_num": [1, 2, 3, 4, 5, 6, 7, 8],                           # numeric → atlanır
    })


def test_count_mod_a_equivalence():
    """Fixture Mod A: CountEncodeStep(full) == eski _count_encode_tt train çıktısı birebir."""
    fx = _cat_fixture()
    step = CountEncodeStep()
    params = step.fit(fx.copy())
    out = step.transform(fx.copy(), params)
    old_train, _, added = _count_encode_tt(fx.copy(), fx.copy())
    assert set(added) == set(params.keys()), "aynı count_ kolonları üretilmeli"
    for name in added:
        assert _as_str(out[name]) == _as_str(old_train[name]), f"'{name}' count parity kırıldı"


def test_count_unseen_maps_to_zero():
    """Görülmemiş-kategori → 0, params'taki KANONİK unseen kaydından sürülür (koda gömülü değil)."""
    train = pd.DataFrame({"country": ["FR", "FR", "DE", "ES", "DE"]})
    step = CountEncodeStep()
    params = step.fit(train.copy())
    spec = params["count_country"]
    assert set(spec["unseen"]["known"]) == {"FR", "DE", "ES"}, "bilinen set params'ta açık olmalı"
    assert spec["unseen"]["strategy"] == "map_to_zero"
    assert "known_limitation" in spec["unseen"], "Faz 0 sentinel'i işaretlenmeli"
    assert spec["unseen"]["monitor"] == "count_rows_with_unseen"
    prod = pd.DataFrame({"country": ["FR", "US", "DE"]})   # US görülmemiş
    out = step.transform(prod.copy(), params)
    assert out["count_country"].tolist() == [2, 0, 2], "FR→2, US(unseen)→0, DE→2"
    _, old_test, _ = _count_encode_tt(train.copy(), prod.copy())
    assert out["count_country"].tolist() == old_test["count_country"].tolist(), "eski yol da unseen→0"


def test_count_real_column_parity_credit_loan_grade():
    """GERÇEK-KOLON parity: credit loan_grade (nadir G/F kategorileri fixture'da YOK).
    Ölçek kenar-durumu + gerçek frekans dağılımı kanıtı — fixture yalnız 'mantık doğru' der."""
    path = _find_csv("*credit_risk_dataset.csv")
    if not path:
        pytest.skip("credit_risk_dataset.csv yok (gitignored data) — gerçek-kolon parity atlandı")
    df = pd.read_csv(path)[["loan_grade"]].copy()
    step = CountEncodeStep()
    params = step.fit(df.copy())
    out = step.transform(df.copy(), params)
    old_train, _, added = _count_encode_tt(df.copy(), df.copy())
    assert "count_loan_grade" in params, "loan_grade count-encode edilmeli"
    for name in added:
        assert _as_str(out[name]) == _as_str(old_train[name]), f"gerçek-kolon '{name}' parity kırıldı"
    known = params["count_loan_grade"]["unseen"]["known"]
    assert any(g in known for g in ("F", "G")), "nadir grade'ler (F/G) bilinen sette olmalı"


# NOT: elle-zincir testleri (impute→ratio→count[→onehot], scaler-chain) SİLİNDİ — orkestratör
# çok-adım parity'si (test_orchestrator_multistep_parity) onları devraldı (tek çok-adım yolu).
# Geçici `_build_ratio_stand_in` de emekliye ayrıldı: RatioConstructStep'e formelleşti
# (test_ratio_construct_inf_nan_parity eski _build_ratio_features'la parity'yi tutar).


# ── encode_route + onehot + target adımları ──────────────────
# Bu blok üç concern'i kapsar: (A) kardinalite kararı bir kez fit'te + donmuş invariant'ı,
# (D) onehot unseen≡referans (count'un OOD-sentinel'inden ayrı), (E) fit_transform≠transform
# cross-fitting + serving imza-invariant'ı, (C) sklearn-pickle sürüm kapısı fail-closed.


def _deterministic_y(n: int) -> pd.Series:
    """Seedli, sinyal taşıyan y (cross-fitting'i düz transform'dan ayırt edecek kadar düzensiz)."""
    return pd.Series(np.random.RandomState(0).randint(0, 2, size=n))


# ── A: kardinalite-dallanması bir kez, dondur ────────────────


def test_route_partitions_by_train_cardinality():
    """decide_encode_route: ≤10 → onehot, >10 → target; nunique yalnız fit-anı."""
    train = pd.DataFrame({
        "geo": ["FR", "DE", "ES"] * 5,                          # card 3 → onehot
        "hi":  [f"c{i % 15}" for i in range(15)],               # card 15 → target
        "num": range(15),
    })
    route = decide_encode_route(train)
    assert route["onehot"] == ["geo"] and route["target"] == ["hi"], route
    assert route["threshold"] == 10
    assert route["cardinalities"] == {"geo": 3, "hi": 15}       # audit izi


def test_route_invariant_inflated_cardinality_stays_onehot():
    """INVARIANT A: inference'ta kardinalite şişse bile karar DEĞİŞMEZ (donmuş atama uygulanır,
    kardinalite yeniden ÖLÇÜLMEZ). Şişmiş kolon hâlâ one-hot'a gider, unseen → all_zero."""
    train = pd.DataFrame({"geo": ["FR", "DE", "ES"] * 4})       # card 3 → onehot
    route = decide_encode_route(train)
    assert "geo" in route["onehot"]
    oh = OneHotStep({"columns": route["onehot"]})
    params = oh.fit(train.copy())

    prod = pd.DataFrame({"geo": [f"C{i}" for i in range(12)]})  # kardinalite 3 → 12'ye şişti
    out = oh.transform(prod.copy(), params)
    # hâlâ one-hot: train dummy vocab kolonları mevcut, hepsi 0 (unseen → all_zero)
    for dcol in params["dummy_vocab"]:
        assert dcol in out.columns
        assert int(out[dcol].sum()) == 0, "şişmiş-kardinalite unseen → all_zero (karar değişmedi)"
    # kanıt: karar DONDURULMASAYDI prod kardinalitesi target eşiğini aşardı — ama kullanılmıyor
    reroute = decide_encode_route(prod)
    assert "geo" in reroute["target"], "prod card>10 target'a giderdi; donmuş karar bunu yok sayar"


# ── D: onehot fixture + gerçek-kolon + unseen≡referans ────────


def _onehot_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "geo": ["FR", "FR", "DE", "ES", "DE", "FR"],   # card 3 → onehot
        "num": [1, 2, 3, 4, 5, 6],                     # numeric → dokunulmaz
    })


def test_onehot_mod_a_equivalence():
    """Fixture Mod A: OneHotStep == eski _encode_categorical_tt düşük-dal (drop_first) birebir."""
    fx = _onehot_fixture()
    route = decide_encode_route(fx)
    assert route["onehot"] == ["geo"] and route["target"] == []
    oh = OneHotStep({"columns": route["onehot"]})
    params = oh.fit(fx.copy())
    out = oh.transform(fx.copy(), params)
    old_train, _ = _encode_categorical_tt(fx.copy(), fx.copy(), _deterministic_y(len(fx)))
    assert set(params["dummy_vocab"]).issubset(set(old_train.columns)), "aynı dummy kolonları"
    for d in params["dummy_vocab"]:
        assert _as_str(out[d]) == _as_str(old_train[d]), f"onehot '{d}' parity kırıldı"
    assert _as_str(out["num"]) == _as_str(old_train["num"]), "numeric kolon dokunulmamalı"


def test_onehot_unseen_all_zero_and_limitation():
    """Görülmemiş kategori → tüm-dummy-0; eski yolla parity. D: known_limitation onehot'un
    all_zero'sunu count'un OOD-sentinel'inden AYIRIR (unseen ≡ referans kategori, iyi huylu)."""
    train = pd.DataFrame({"geo": ["FR", "FR", "DE", "ES", "DE"]})
    y = _deterministic_y(len(train))
    oh = OneHotStep({"columns": ["geo"]})
    params = oh.fit(train.copy())
    prod = pd.DataFrame({"geo": ["FR", "US", "DE"]})            # US görülmemiş
    out = oh.transform(prod.copy(), params)
    _, old_test = _encode_categorical_tt(train.copy(), prod.copy(), y)
    for d in params["dummy_vocab"]:
        assert _as_str(out[d]) == _as_str(old_test[d]), f"unseen parity '{d}' kırıldı"
    us_row = out.iloc[1]
    assert all(int(us_row[d]) == 0 for d in params["dummy_vocab"]), "US (unseen) → all_zero"
    # D — kanonik unseen kaydı + count'tan ayrı known_limitation
    assert params["unseen"]["strategy"] == "all_zero"
    assert params["unseen"]["monitor"] == "count_rows_with_unseen"
    lim = params["unseen"]["known_limitation"].lower()
    assert "referans" in lim and "ood" in lim, "onehot limitation count'tan AYRI (referans, OOD-değil)"


def test_onehot_real_column_parity_credit():
    """GERÇEK-KOLON parity: credit loan_grade/loan_intent/home_ownership (hepsi ≤10 → onehot)."""
    path = _find_csv("*credit_risk_dataset.csv")
    if not path:
        pytest.skip("credit_risk_dataset.csv yok (gitignored) — gerçek-kolon onehot atlandı")
    df = pd.read_csv(path)[["loan_grade", "loan_intent", "person_home_ownership"]].copy()
    route = decide_encode_route(df)
    assert set(route["onehot"]) == {"loan_grade", "loan_intent", "person_home_ownership"}
    assert route["target"] == [], "credit'te yüksek-kardinalite kolon yok (target inert)"
    oh = OneHotStep({"columns": route["onehot"]})
    params = oh.fit(df.copy())
    out = oh.transform(df.copy(), params)
    y = pd.Series((df.index % 2).astype(int))
    old_train, _ = _encode_categorical_tt(df.copy(), df.copy(), y)
    for d in params["dummy_vocab"]:
        assert _as_str(out[d]) == _as_str(old_train[d]), f"gerçek-kolon onehot '{d}' parity kırıldı"


# ── E: fit_transform ≠ transform + serving imza-invariant'ı ───


def test_e_serving_transform_has_no_target_param():
    """İMZA-DÜZEYİ invariant: hiçbir adımın transform'unda y YOK → serving hedef geçemez →
    cross-fitting (y gerektirir) transform'dan ULAŞILAMAZ."""
    for cls in (ImputeStep, RatioFinalizeStep, CountEncodeStep, OneHotStep, TargetEncodeStep):
        pnames = list(inspect.signature(cls.transform).parameters)
        assert "y" not in pnames and "y_train" not in pnames, f"{cls.__name__}.transform y içermemeli"


def test_e_mixin_split_targetencode_overrides():
    """Cross-fitting yapan tek adım (TargetEncode) mixin'i KULLANMAZ; diğerleri kullanır →
    kasıtlı-fark tekil ve görünür."""
    assert not issubclass(TargetEncodeStep, DefaultFitTransform), "cross-fitting override, mixin değil"
    for cls in (ImputeStep, RatioFinalizeStep, CountEncodeStep, OneHotStep):
        assert issubclass(cls, DefaultFitTransform), f"{cls.__name__} cross-fitting yapmaz → mixin"


def test_e_fit_transform_requires_target_fail_closed():
    """TargetEncode.fit_transform y=None → RuntimeError (fail-closed). Serving (hedefsiz)
    bu train-only yolu ÇAĞIRAMAZ → cross-fitting prodüksiyona sızamaz."""
    X = pd.DataFrame({"hi": [f"c{i % 15}" for i in range(30)]})
    step = TargetEncodeStep({"columns": ["hi"]})
    with pytest.raises(RuntimeError):
        step.fit_transform(X.copy(), y_train=None)


def test_e_cross_fit_differs_from_plain_transform():
    """fit_transform (cross-fitted) ≠ transform (düz) — fark GERÇEK, dolayısıyla yanlış yolu
    serving'de çağırmak çıktıyı değiştirir (guard yük taşıyor). Serving düz olanı kullanır."""
    X = pd.DataFrame({"hi": [f"c{i % 15}" for i in range(60)]})
    y = _deterministic_y(len(X))
    step = TargetEncodeStep({"columns": ["hi"]})
    Xtr, params = step.fit_transform(X.copy(), y)         # cross-fitted train
    Xplain = step.transform(X.copy(), params)             # düz serving
    assert not np.allclose(Xtr["hi"].to_numpy(), Xplain["hi"].to_numpy()), \
        "cross-fitting düz transform'dan farklı olmalı (E — sklearn dokümanı da böyle diyor)"


# ── target parity: serving + train, fixture + gerçek-kolon ────


def _highcard_fixture(n: int = 60, k: int = 15) -> "tuple[pd.DataFrame, pd.Series]":
    return pd.DataFrame({"hi": [f"c{i % k}" for i in range(n)], "num": list(range(n))}), _deterministic_y(n)


def test_target_fixture_serving_and_train_parity():
    """İKİ AYRI iddia: serving-parity (new.transform == eski düz .transform) VE train-parity
    (new.fit_transform cross-fitted == eski fit_transform). Sentetik yüksek-kardinalite fixture."""
    X, y = _highcard_fixture()
    route = decide_encode_route(X)
    assert route["target"] == ["hi"], route
    step = TargetEncodeStep({"columns": route["target"]})
    Xtr_new, params = step.fit_transform(X.copy(), y)
    old_train, old_test = _encode_categorical_tt(X.copy(), X.copy(), y)
    # TRAIN-parity (cross-fitted)
    assert np.allclose(Xtr_new["hi"].to_numpy(), old_train["hi"].to_numpy()), "train-parity (cross-fitted)"
    # SERVING-parity (düz transform)
    out = step.transform(X.copy(), params)
    assert np.allclose(out["hi"].to_numpy(), old_test["hi"].to_numpy()), "serving-parity (düz transform)"


def test_target_unseen_maps_to_prior():
    """Görülmemiş → prior (hedef ortalaması); eski yolla parity. D: strategy='prior', OOD değil."""
    X, y = _highcard_fixture()
    step = TargetEncodeStep({"columns": ["hi"]})
    _, params = step.fit_transform(X.copy(), y)
    assert params["unseen"]["strategy"] == "prior"
    prod = pd.DataFrame({"hi": ["c0", "ZZZ_unseen", "c1"]})
    out = step.transform(prod.copy(), params)
    prior = float(y.mean())
    assert np.isclose(out["hi"].iloc[1], prior), f"unseen → prior ({prior})"
    _, old_test = _encode_categorical_tt(X.copy(), prod.copy(), y)
    assert np.allclose(out["hi"].to_numpy(), old_test["hi"].to_numpy()), "unseen→prior eski yolla parity"


def test_target_real_column_parity_superstore():
    """GERÇEK yüksek-kardinalite: superstore 'Sub-Category' (~17). Referans set DEĞİL → y SENTETİK.
    YARI-GERÇEK: bu test 'iki implementasyon aynı' kanıtlar, 'target-encoding gerçek target'ta
    doğru' DEĞİL (y uydurma). Parity eski==yeni olduğundan geçerli (ikisi de aynı sahte y'yi görür)."""
    path = _find_csv("*superstore*.csv") or _find_csv("*Superstore*.csv")
    if not path:
        pytest.skip("superstore yok (gitignored) — gerçek yüksek-kardinalite target parity atlandı")
    df = pd.read_csv(path)
    col = "Sub-Category"
    if col not in df.columns or df[col].nunique() <= 10:
        pytest.skip(f"{col} yüksek-kardinalite değil — gerçek-kolon target atlandı")
    X = df[[col]].copy()
    y = _deterministic_y(len(df))
    route = decide_encode_route(X)
    assert col in route["target"], f"{col} (card {df[col].nunique()}) target'a gitmeli"
    step = TargetEncodeStep({"columns": route["target"]})
    Xtr_new, params = step.fit_transform(X.copy(), y)
    old_train, old_test = _encode_categorical_tt(X.copy(), X.copy(), y)
    out = step.transform(X.copy(), params)
    assert np.allclose(Xtr_new[col].to_numpy(), old_train[col].to_numpy()), "gerçek train-parity"
    assert np.allclose(out[col].to_numpy(), old_test[col].to_numpy()), "gerçek serving-parity"


# ── C: sklearn-pickle sürüm kapısı (fail-closed) ─────────────


def test_check_sklearn_compat_verdicts():
    """patch farkı → warn; minor|major → abort; eşit → ok."""
    assert check_sklearn_compat("1.8.0", "1.8.0") == "ok"
    assert check_sklearn_compat("1.8.0", "1.8.3") == "warn"    # yalnız patch
    assert check_sklearn_compat("1.8.0", "1.9.0") == "abort"   # minor (pickle-binary kırılabilir)
    assert check_sklearn_compat("1.8.0", "2.0.0") == "abort"   # major
    assert check_sklearn_compat("1.7.5", "1.8.0") == "abort"   # minor (geriye de)


def test_enforce_sklearn_compat_fail_closed():
    """abort → SklearnVersionMismatch FIRLAT (estimator dönmeden); warn/ok → devam."""
    enforce_sklearn_compat("1.8.0", "1.8.0")                   # ok → sessiz
    enforce_sklearn_compat("1.8.0", "1.8.9")                   # warn → raise YOK
    with pytest.raises(SklearnVersionMismatch):
        enforce_sklearn_compat("1.8.0", "1.9.0")              # abort → raise


def test_target_transform_aborts_on_version_mismatch():
    """SERVING fail-closed: sürüm uyumsuzsa transform HİÇ skorlamaz (yanlış skorlamaktansa dur).
    inline-JSON adımlarında OLMAYAN başarısızlık modu — ilk kez bu pickle adımında."""
    X, y = _highcard_fixture()
    step = TargetEncodeStep({"columns": ["hi"]})
    _, params = step.fit_transform(X.copy(), y)
    bad_ref = PickleRef(params["ref"].estimator, "99.0.0")     # sahte gelecek sürüm (major atla)
    params_bad = {**params, "ref": bad_ref}
    with pytest.raises(SklearnVersionMismatch):
        step.transform(X.copy(), params_bad)                  # abort → hiç skorlamaz
    out = step.transform(X.copy(), params)                    # aynı sürüm → sorunsuz skorlar
    assert "hi" in out.columns


# ══════════════════════ SCALER AİLESİ (inf-impute + route + Robust + Standard) ═════════════════
# pickle_io soyutlaması (3 tüketici tek kapı) + TargetEncode refactor davranış-parity'si +
# inf-peek deterministik kanıt + inference-inf-sonucu + route invariant + mixin doğrulaması.


# ── pickle_io + TargetEncode refactor (davranış değişmedi) ───


def test_pickle_ref_passthrough_and_gate():
    """PickleRef: sürüm uyumlu → estimator KİMLİĞİ korunur (soyutlama davranış eklemez, yalnız kapı);
    uyumsuz → estimator DÖNMEZ (fail-closed)."""
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    ref = make_pickle_ref(sc)
    assert ref.sklearn_version == sklearn.__version__
    assert load_pickle_ref(ref) is sc, "uyumlu sürüm → aynı estimator (kimlik) döner"
    with pytest.raises(SklearnVersionMismatch):
        load_pickle_ref(PickleRef(sc, "0.0.0"))    # uyumsuz → skorlamadan önce abort


def test_target_refactor_behavior_unchanged():
    """PickleRef refactor DAVRANIŞ değiştirmemeli: çağrı yolu ``params['ref']`` + gate oldu, ama
    çıktı = gate'siz doğrudan ``estimator.transform`` ile BİREBİR. 'Soyutlarken davranış kaydı' riski yok."""
    X, y = _highcard_fixture()
    step = TargetEncodeStep({"columns": ["hi"]})
    _, params = step.fit_transform(X.copy(), y)
    out = step.transform(X.copy(), params)
    direct = params["ref"].estimator.transform(X[["hi"]])    # gate'siz, doğrudan estimator
    assert np.allclose(out["hi"].to_numpy(dtype=float), np.asarray(direct).ravel()), \
        "refactor çıktıyı değiştirdi — soyutlama davranış eklememeliydi"


# ── inf-impute: #4-değil (train-fit) + peek kapatma deterministik + inference sonlu ──


def test_inf_peek_closed_deterministic():
    """inf yalnız TEST/prod'da (train'de yok): yeni fit ``inf_cols``'a ALMAZ (peek kapalı, train-only);
    eski SEÇİM mantığı (train OR test) ALIRDI. DETERMİNİSTİK küme farkı — ROC A/B DEĞİL (selection-sınıfı)."""
    train = pd.DataFrame({"c": [1.0, 2.0, 3.0, 4.0]})          # train'de inf YOK
    prod  = pd.DataFrame({"c": [1.0, np.inf, 3.0, 4.0]})       # inf yalnız prod'da
    params = InfImputeStep().fit(train.copy())
    assert "c" not in params["inf_cols"], "train-only → 'c' öğrenilmez (peek kapalı)"

    def _old_inf_select(tr, te):   # eski yolun train-OR-test seçimi (peek)
        num = tr.select_dtypes(include="number").columns
        return [col for col in num if InfImputeStep._has_inf(tr[col]) or InfImputeStep._has_inf(te[col])]
    assert "c" in _old_inf_select(train, prod), "eski peek 'c'yi seçerdi (test'e bakar) — kontrast"


def test_inf_inference_stays_finite():
    """Peek kapatmanın postcondition'ı: öğrenilmemiş kolonda prod-inf gelse bile çıktı SONLU (0.0
    fallback) → ham inf downstream scaler'a geçip modeli patlatmaz. nan'a DOKUNULMAZ (ImputeStep'in işi)."""
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})   # inf yok
    params = InfImputeStep().fit(train.copy())
    assert params["inf_cols"] == [], "train'de inf yok → hiçbir kolon öğrenilmez"
    prod = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0], "b": [1.0, np.nan, 3.0, 4.0]})
    out = InfImputeStep().transform(prod.copy(), params)
    assert np.isfinite(out["a"].to_numpy()).all(), "öğrenilmemiş prod-inf → sonlu"
    assert out["a"].iloc[1] == 0.0 and out["a"].iloc[2] == 0.0, "±inf → 0.0 (tanımlı fallback)"
    assert np.isnan(out["b"].iloc[1]), "nan'a dokunulmaz (scaler-inf değil, ImputeStep'in işi)"


# ── scaler real-column parity (family-internal; sentetik-chain SİLİNDİ → orkestratör devraldı) ──


def test_scaler_real_column_parity_credit():
    """GERÇEK-KOLON scaler parity: credit numeric kolonlar (aynı outlier_cols ikisine)."""
    path = _find_csv("*credit_risk_dataset.csv")
    if not path:
        pytest.skip("credit_risk_dataset.csv yok (gitignored) — gerçek-kolon scaler atlandı")
    df = pd.read_csv(path).select_dtypes(include="number").fillna(0.0)
    cols = list(df.columns)[:5]
    df = df[cols].copy()
    outlier_cols = cols[:2]
    xn = df.copy()
    p_inf = InfImputeStep().fit(xn); xn = InfImputeStep().transform(xn, p_inf)
    route = decide_scaler_route(xn, outlier_cols)
    p_rob = RobustScaleStep({"columns": route["robust"]}).fit(xn)
    xn = RobustScaleStep({"columns": route["robust"]}).transform(xn, p_rob)
    p_std = StandardScaleStep({"columns": route["standard"]}).fit(xn)
    xn = StandardScaleStep({"columns": route["standard"]}).transform(xn, p_std)
    xo, _ = _scale_features_tt(df.copy(), df.copy(), outlier_cols)
    for col in cols:
        assert np.allclose(xn[col].to_numpy(dtype=float), xo[col].to_numpy(dtype=float)), \
            f"gerçek-kolon scaler '{col}' parity kırıldı"


# ── route invariant (şişmiş outlier → hâlâ standard) + shared shape ──


def test_scaler_route_invariant_inflated_outlier():
    """INVARIANT: inference'ta 'norm' outlier-ağır olsa bile scaler seçimi DEĞİŞMEZ (donmuş atama).
    StandardScaleStep donmuş TRAIN scaler'ını uygular (yeniden ölçmez); RobustScaleStep norm'a dokunmaz."""
    train = pd.DataFrame({"out": [1.0, 2.0, 3.0, 4.0, 100.0], "norm": [1.0, 2.0, 3.0, 4.0, 5.0]})
    route = decide_scaler_route(train, outlier_cols=["out"])
    assert route["robust"] == ["out"] and route["standard"] == ["norm"]
    std_step = StandardScaleStep({"columns": route["standard"]})
    p_std = std_step.fit(train.copy())
    prod = pd.DataFrame({"out": [2.0, 3.0], "norm": [5.0, 9999.0]})   # norm outlier-ağır (profil değişti)
    out = std_step.transform(prod.copy(), p_std)
    from sklearn.preprocessing import StandardScaler
    expected = StandardScaler().fit(train[["norm"]]).transform(prod[["norm"]])
    assert np.allclose(out[["norm"]].to_numpy(), expected), "donmuş TRAIN scaler uygulandı (yeniden ölçmedi)"
    # norm robust route'ta değil → RobustScaleStep ona dokunmaz (atama donmuş, outlier profilinden değil)
    rob_step = RobustScaleStep({"columns": route["robust"]})
    p_rob = rob_step.fit(train.copy())
    out_rob = rob_step.transform(prod.copy(), p_rob)
    assert np.allclose(out_rob["norm"].to_numpy(), prod["norm"].to_numpy()), "robust 'norm'a dokunmaz"


def test_routes_share_frozen_choice_shape():
    """İki decide_* AYNI frozen_choice deseni: executor-anahtarlı DISJOINT kolon listeleri
    (union = uygun girdi) + audit alanı. Orkestratör iki route'u tek desenle ele alsın diye."""
    enc_df = pd.DataFrame({"geo": ["a", "b", "c"] * 4, "hi": [f"c{i % 15}" for i in range(12)]})
    er = decide_encode_route(enc_df)
    sc_df = pd.DataFrame({"out": [1.0, 2.0, 3.0, 100.0], "norm": [1.0, 2.0, 3.0, 4.0]})
    sr = decide_scaler_route(sc_df, ["out"])
    for route, execs, elig, audit in [
        (er, ("onehot", "target"), {"geo", "hi"}, "cardinalities"),
        (sr, ("robust", "standard"), {"out", "norm"}, "outlier_cols"),
    ]:
        a, b = set(route[execs[0]]), set(route[execs[1]])
        assert a & b == set(), "executor listeleri disjoint olmalı"
        assert a | b == elig, "union = uygun girdi kolonları olmalı"
        assert audit in route, f"audit alanı '{audit}' mevcut olmalı"


def test_scaler_mixin_no_crossfit():
    """Mixin doğrulaması: inf/robust/standard üçü de DefaultFitTransform + fit_transform==fit+transform
    (cross-fit YOK). Kontrast: yalnız TargetEncode override eder (bkz. test_e_cross_fit_differs...) →
    bir override + gerisi mixin = doğru granülarite kanıtı."""
    for cls in (InfImputeStep, RobustScaleStep, StandardScaleStep):
        assert issubclass(cls, DefaultFitTransform), f"{cls.__name__} mixin olmalı"
    X = pd.DataFrame({"out": [1.0, 2.0, 3.0, 4.0, 100.0, 2.0, 3.0, 1.0],
                      "norm": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    route = decide_scaler_route(X, ["out"])
    for cls, key in [(RobustScaleStep, "robust"), (StandardScaleStep, "standard")]:
        step = cls({"columns": route[key]})
        Xft, _ = step.fit_transform(X.copy(), None)
        Xt = step.transform(X.copy(), step.fit(X.copy()))
        col = route[key][0]
        assert np.allclose(Xft[col].to_numpy(), Xt[col].to_numpy()), \
            f"{cls.__name__}: fit_transform == fit+transform (cross-fit yok)"
    Xi = pd.DataFrame({"c": [1.0, 2.0, np.inf, 4.0]})
    inf = InfImputeStep()
    Xft, _ = inf.fit_transform(Xi.copy(), None)
    Xt = inf.transform(Xi.copy(), inf.fit(Xi.copy()))
    assert np.allclose(Xft["c"].to_numpy(), Xt["c"].to_numpy()), "InfImpute: fit_transform == fit+transform"


def test_scaler_pickle_gate_shared():
    """Robust + Standard AYNI pickle_io kapısından: sahte sürüm → abort (fail-closed), 3 pickle tek kapı."""
    train = pd.DataFrame({"out": [1.0, 2.0, 3.0, 100.0], "norm": [1.0, 2.0, 3.0, 4.0]})
    route = decide_scaler_route(train, ["out"])
    for cls, key in [(RobustScaleStep, "robust"), (StandardScaleStep, "standard")]:
        step = cls({"columns": route[key]})
        p = step.fit(train.copy())
        p_bad = {**p, "ref": PickleRef(p["ref"].estimator, "99.0.0")}
        with pytest.raises(SklearnVersionMismatch):
            step.transform(train.copy(), p_bad)               # abort → hiç skorlamaz
        step.transform(train.copy(), p)                       # aynı sürüm → sorunsuz


# ══════════════════════ ZSCORE / NORMALIZE (#5 fix — plan-güdümlü, train-fit) ═══════════════════
# Eski _FEATURE_OPS full-df μ/σ (min/max) → split-öncesi → #5 sızıntı. Yeni adım train-fit. Bu blok:
# FORMÜL-parity (transform == elle train-only) + SIZINTI-delta (full-df ≠ train-only, deterministik) +
# gerçek-kolon DİNAMİK (glob+programatik seçim, hardcode YOK) + orkestratör (specs→kolon, serving==training).


def test_zscore_normalize_formula_parity():
    """FORMÜL-sadakati: transform çıktısı, μ/σ (min/max) elle fit-çerçevesinden hesaplanınca birebir
    (_FEATURE_OPS:1577/1581 formülleri). fit=full → train-fit istatistiği full ile aynı → saf formül testi."""
    fx = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 8.0, 9.0, 10.0]})
    s = pd.to_numeric(fx["x"], errors="coerce")

    zp = ZScoreStep({"specs": [("z_x", "x")]}).fit(fx.copy())
    z_out = ZScoreStep({"specs": [("z_x", "x")]}).transform(fx.copy(), zp)
    z_exp = ((s - s.mean()) / (s.std(ddof=0) + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
    assert np.allclose(z_out["z_x"].to_numpy(), z_exp.to_numpy()), "zscore formül parity kırıldı"

    npp = NormalizeStep({"specs": [("n_x", "x")]}).fit(fx.copy())
    n_out = NormalizeStep({"specs": [("n_x", "x")]}).transform(fx.copy(), npp)
    n_exp = ((s - s.min()) / ((s.max() - s.min()) + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
    assert np.allclose(n_out["n_x"].to_numpy(), n_exp.to_numpy()), "normalize formül parity kırıldı"


def test_zscore_normalize_leakage_delta():
    """SIZINTI-delta (#5 deterministik kanıt): train-dist ≠ full-dist → train-fit μ/σ (min/max) full-df'ten
    FARKLI → test satırlarında çıktı farklı. Eski op full-df'ten hesaplardı = test dağılımı eğitime sızardı."""
    full = pd.DataFrame({"x": [1.0] * 10 + [100.0] * 10})   # ilk 10 = train (düşük), son 10 = test (yüksek)
    train = full.iloc[:10]
    s_all = pd.to_numeric(full["x"], errors="coerce")

    # ── zscore ──
    zp = ZScoreStep({"specs": [("z_x", "x")]}).fit(train.copy())     # μ/σ YALNIZ train
    z_new = ZScoreStep({"specs": [("z_x", "x")]}).transform(full.copy(), zp)
    z_old = ((s_all - s_all.mean()) / (s_all.std(ddof=0) + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
    assert not np.isclose(zp["z_x"]["mean"], float(s_all.mean())), "train μ ≠ full μ (dist farkı — sızıntı kaynağı)"
    assert not np.allclose(z_new["z_x"].to_numpy()[10:], z_old.to_numpy()[10:]), \
        "train-fit vs full-df zscore test satırlarında FARKLI (sızıntı kaldırıldı — deterministik)"

    # ── normalize ──
    npp = NormalizeStep({"specs": [("n_x", "x")]}).fit(train.copy())  # min/max YALNIZ train (hepsi 1.0 → 0)
    n_new = NormalizeStep({"specs": [("n_x", "x")]}).transform(full.copy(), npp)
    n_old = ((s_all - s_all.min()) / ((s_all.max() - s_all.min()) + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
    assert not np.allclose(n_new["n_x"].to_numpy()[10:], n_old.to_numpy()[10:]), \
        "train-fit vs full-df normalize test satırlarında FARKLI (sızıntı kaldırıldı)"


def test_zscore_normalize_real_column_dynamic():
    """GERÇEK-KOLON DİNAMİK (kolon adı GÖMÜLÜ DEĞİL): glob CSV → numeric kolonu programatik seç.
    (a) formül-parity train-fit; (b) sızıntı-delta full-df ≠ train-only (train μ ≠ full μ olduğunda)."""
    path = _find_csv("*credit_risk_dataset.csv")
    if not path:
        pytest.skip("credit_risk_dataset.csv yok (gitignored) — gerçek-kolon zscore atlandı")
    df = pd.read_csv(path)
    num = df.select_dtypes(include="number").columns.tolist()
    if not num:
        pytest.skip("numeric kolon yok")
    col = num[0]                                             # programatik seçim — hiçbir kolon adı yazılmadı
    cut = len(df) // 2
    train = df.iloc[:cut]
    s_tr, s_all = pd.to_numeric(train[col], errors="coerce"), pd.to_numeric(df[col], errors="coerce")

    zp = ZScoreStep({"specs": [("z_col", col)]}).fit(train.copy())
    out = ZScoreStep({"specs": [("z_col", col)]}).transform(df.copy(), zp)
    expected = ((s_all - s_tr.mean()) / (s_tr.std(ddof=0) + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
    assert np.allclose(out["z_col"].to_numpy(), expected.to_numpy()), "gerçek-kolon zscore formül parity kırıldı"

    if not np.isclose(s_tr.mean(), s_all.mean()):           # gerçek veride dist farkı varsa sızıntı-delta göster
        old = ((s_all - s_all.mean()) / (s_all.std(ddof=0) + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
        assert not np.allclose(out["z_col"].to_numpy(), old.to_numpy()), \
            "gerçek-kolon: train-fit ≠ full-df zscore (sızıntı deterministik)"


def test_zscore_in_orchestrator():
    """Orkestratör: ctx zscore_specs → adım Impute↔RatioConstruct arasında çalışır, μ/σ (min/max) TRAIN'den
    param_store'a DONAR, serving==training. NOT: z_amt afin(amt) → select_drop high-corr onu (ham ile gereksiz)
    düşürür — bu DOĞRU pipeline davranışı; üretim kanıtı param_store'dan (final_features üyeliği değil)."""
    n = 30
    fx = pd.DataFrame({
        "amt":   pd.Series([float(i * 2 + 1) for i in range(n)]),
        "other": pd.Series([float((i * 5) % 13) for i in range(n)]),
    })
    y = _deterministic_y(n)
    ctx = {"ratio_pairs": [], "encode_exclude": [], "outlier_cols": [], "intent": "",
           "zscore_specs": [("z_amt", "amt")], "normalize_specs": [("n_other", "other")]}
    X_fit, artifact = fit_pipeline(fx.copy(), y=y, ctx=ctx)
    assert any(e["id"] == "zscore_feature" for e in artifact.manifest), "zscore adımı manifest'te olmalı"
    assert any(e["id"] == "normalize_feature" for e in artifact.manifest), "normalize adımı manifest'te olmalı"
    # üretim + train-fit kanıtı: param_store'da donmuş μ/σ (min/max) train'den
    z_params = artifact.param_store.get("zscore_feature", {})
    n_params = artifact.param_store.get("normalize_feature", {})
    assert "z_amt" in z_params and "n_other" in n_params, "zscore/normalize kolonları fit'te üretilmeli"
    assert np.isclose(z_params["z_amt"]["mean"], float(fx["amt"].mean())), "train-fit μ (tek arm → full=train)"
    # serving == training (donmuş adımlar → hayatta kalan kolonlarda birebir)
    X_serve = transform_pipeline(fx.copy(), artifact)
    assert list(X_serve.columns) == artifact.final_features
    for c in artifact.final_features:
        assert np.allclose(X_serve[c].to_numpy(dtype=float), X_fit[c].to_numpy(dtype=float)), \
            f"orkestratör zscore/normalize serving '{c}' != training"


# ══════════════════════ GROUPBY-DEVIATION (dev / zdev / grpstd — fitted_stat) ═══════════════════
# İki-boyutlu donma: seçim (cat/num/çift + cap + güvenilir-grup) records'a gömülü + istatistik
# (grp_mean/std + global) train'den. Unseen/unguvenilir grup → GLOBAL (count'un map_to_zero'sundan
# AYRI). _GROUPBY_MIN_GROUP_SIZE=30 → fixture ≥2 güvenilir grup (30'ar satır) içermeli.


def _groupby_fixture() -> "tuple[pd.DataFrame, pd.DataFrame]":
    """train: 'A'×30 + 'B'×30 (güvenilir) + 'C'×5 (unguvenilir → global). test: bilinen A/B,
    unguvenilir-bilinen C, unseen Z. amount grup-ayrık (A~10s, B~100s) → net grup ortalaması."""
    grp = ["A"] * 30 + ["B"] * 30 + ["C"] * 5
    amount = [10.0 + i for i in range(30)] + [100.0 + i for i in range(30)] + [1.0, 2.0, 3.0, 4.0, 5.0]
    train = pd.DataFrame({"grp": pd.Series(grp, dtype="object"), "amount": amount})
    test = pd.DataFrame({
        "grp":    pd.Series(["A", "B", "A", "C", "Z"], dtype="object"),   # Z = unseen, C = unguvenilir-bilinen
        "amount": [15.0, 105.0, 25.0, 3.0, 50.0],
    })
    return train, test


def test_groupby_deviation_fixture_parity():
    """dev/zdev/grpstd fixture parity: yeni GroupbyDeviationStep == eski _groupby_deviation_tt
    (train + test tarafı AYRI). Unseen (Z) + unguvenilir grup (C) → global fallback birebir."""
    train, test = _groupby_fixture()
    xo_tr, xo_te, added = _groupby_deviation_tt(train.copy(), test.copy(), meta=None, exclude=set())
    assert added, "fixture en az bir groupby feature üretmeli (2 güvenilir grup)"
    assert (any(a.startswith("dev_") for a in added) and any(a.startswith("zdev_") for a in added)
            and any(a.startswith("grpstd_") for a in added)), "üç aile de (dev/zdev/grpstd) üretilmeli"

    step = GroupbyDeviationStep({"exclude": []})
    params = step.fit(train.copy())
    emitted = [name for rec in params for _, name in rec["emits"]]
    assert set(emitted) == set(added), f"emit seti eşleşmedi (seçim drift): {set(emitted) ^ set(added)}"

    xn_tr = step.transform(train.copy(), params)   # == fit_transform train tarafı (DefaultFitTransform)
    xn_te = step.transform(test.copy(), params)
    for name in added:
        assert np.allclose(xn_tr[name].to_numpy(dtype=float), xo_tr[name].to_numpy(dtype=float),
                           equal_nan=True), f"train '{name}' parity kırıldı"
        assert np.allclose(xn_te[name].to_numpy(dtype=float), xo_te[name].to_numpy(dtype=float),
                           equal_nan=True), f"test '{name}' parity kırıldı"

    # unseen (Z, son satır) → global: dev = amount − global_mean (0 sentinel DEĞİL, global istatistik)
    global_mean = float(pd.to_numeric(train["amount"], errors="coerce").mean())
    assert np.isclose(xn_te["dev_amount_by_grp"].iloc[-1], 50.0 - global_mean), \
        "unseen kategori → global mean (dev = x − global_mean), map_to_zero DEĞİL"


def test_groupby_deviation_real_column_parity_credit():
    """GERÇEK-KOLON groupby parity: credit cat+num (train=test=df → reimplementasyon sadakati)."""
    path = _find_csv("*credit_risk_dataset.csv")
    if not path:
        pytest.skip("credit_risk_dataset.csv yok (gitignored) — gerçek-kolon groupby atlandı")
    df = pd.read_csv(path)
    xo_tr, _xo_te, added = _groupby_deviation_tt(df.copy(), df.copy(), meta=None, exclude=set())
    if not added:
        pytest.skip("credit'te güvenilir grup üretilmedi — veri parity için uygun değil")
    step = GroupbyDeviationStep({"exclude": []})
    params = step.fit(df.copy())
    emitted = [name for rec in params for _, name in rec["emits"]]
    assert set(emitted) == set(added), f"gerçek-kolon emit seti eşleşmedi: {set(emitted) ^ set(added)}"
    xn = step.transform(df.copy(), params)
    for name in added:
        assert np.allclose(xn[name].to_numpy(dtype=float), xo_tr[name].to_numpy(dtype=float),
                           equal_nan=True), f"gerçek-kolon groupby '{name}' parity kırıldı"


def test_groupby_deviation_in_orchestrator():
    """Orkestratör entegrasyonu: reliable-grup fixture'ı fit_pipeline'dan geçince groupby manifest'e
    girer, dev/zdev/grpstd kolonları üretilir ve transform_pipeline (serving) == fit (target yok)."""
    train, _ = _groupby_fixture()
    y = _deterministic_y(len(train))   # encode RouteSpec TargetEncode'u fit_transform(X, y) ile çağırır (fail-closed)
    ctx = {"ratio_pairs": [], "encode_exclude": [], "outlier_cols": []}
    X_fit, artifact = fit_pipeline(train.copy(), y=y, ctx=ctx)
    assert any(e["id"] == "groupby_deviation" for e in artifact.manifest), "groupby manifest'te olmalı"
    gcols = [c for c in artifact.final_features if c.startswith(("dev_", "zdev_", "grpstd_"))]
    assert gcols, "orkestratörde en az bir groupby feature üretilmeli (reliable grup var)"
    X_serve = transform_pipeline(train.copy(), artifact)
    for c in gcols:
        assert np.allclose(X_serve[c].to_numpy(dtype=float), X_fit[c].to_numpy(dtype=float),
                           equal_nan=True), f"orkestratör groupby '{c}' serving != training"


# ══════════════════════ SELECT-DROP (terminus #3 — frozen_choice, delege fit) ═══════════════════
# fit DELEGE eder (feature_builder selection dizisi, eğitim-anı) → drop-liste DONAR; transform pür-drop +
# terminal cleanup (self-contained). Bu blok: delege-wiring parity + re_derive:false (donmuş) + terminus
# invariant (PIPELINE en son) + orkestratör (final_features'tan düşer, serving==training).


def _select_fixture() -> pd.DataFrame:
    """Yüksek-varyanslı 'keep' + sıfır-varyans 'const' (low-var drop) + tam-korele 'corr_a/corr_b'
    (high-corr → biri düşer) + bağımsız 'noise'. 40 satır (leakage/istatistik anlamlı)."""
    n = 40
    a = pd.Series([float(i) for i in range(n)])
    s = pd.Series([float((i * 3) % 11) for i in range(n)])      # a'dan bağımsız
    return pd.DataFrame({
        "keep":   a,
        "const":  pd.Series([1.0] * n),                         # var=0 → low-var
        "corr_a": s,
        "corr_b": s * 2.0 + 1.0,                                # corr_a ile |corr|=1 → high-corr
        "noise":  pd.Series([float((i * 7) % 5) for i in range(n)]),
    })


def test_select_drop_frozen_drop_parity():
    """SelectDropStep (delege) çıktısı == build_features selection dizisi (AYNI fonksiyonlar/sıra/fillna).
    Wiring doğrulaması: low-var (const) + high-corr (corr_a|corr_b) düşer, değerler birebir."""
    fx = _select_fixture()
    y = _deterministic_y(len(fx))
    # oracle: build_features tail (intent yok → univariate atla; meta/domain yok)
    xo, _ = _select_features_tt(fx.copy(), fx.copy())
    xo = xo.apply(pd.to_numeric, errors="coerce").fillna(0)
    xo, _, _ = _detect_leakage_tt(xo, xo.copy(), y, None)
    xo, _, _ = _apply_domain_leakage_drops_tt(xo, xo.copy(), {"dropped": [], "suspected": []})
    xo = xo.apply(pd.to_numeric, errors="coerce").fillna(0)

    step = SelectDropStep({"intent": "", "meta": None, "domain_leakage_report": None, "protected_high_card": None})
    params = step.fit(fx.copy(), y)          # leakage tespiti y gerektirir
    xn = step.transform(fx.copy(), params)

    assert "const" in params["drop_cols"], "sıfır-varyans 'const' düşmeli (low-var)"
    assert ("corr_a" in params["drop_cols"]) or ("corr_b" in params["drop_cols"]), \
        "tam-korele çiftten biri düşmeli (high-corr)"
    assert set(xn.columns) == set(xo.columns), f"selection kolon seti farklı: {set(xn.columns) ^ set(xo.columns)}"
    for col in xn.columns:
        assert np.allclose(xn[col].to_numpy(dtype=float), xo[col].to_numpy(dtype=float)), \
            f"selection '{col}' değer parity kırıldı"


def test_select_drop_transform_is_frozen():
    """re_derive:false — seçim serving'de YENİDEN hesaplanmaz. Train'de düşen 'const' serving'de
    yüksek-varyanslı olsa BİLE düşer; train'de kalan 'keep' serving'de sabit olsa BİLE kalır."""
    fx = _select_fixture()
    step = SelectDropStep({})
    params = step.fit(fx.copy(), _deterministic_y(len(fx)))   # leakage tespiti y gerektirir
    assert "const" in params["drop_cols"], "kurulum: 'const' train'de düşmeli"
    assert "keep" not in params["drop_cols"], "kurulum: 'keep' train'de kalmalı"

    n = len(fx)
    serve = fx.copy()
    serve["const"] = pd.Series([float(i * 100) for i in range(n)])   # profil değişti: artık yüksek varyans
    serve["keep"]  = pd.Series([5.0] * n)                            # artık sabit (low-var OLURDU)
    out = step.transform(serve, params)
    assert "const" not in out.columns, "donmuş: train'de düşen 'const' serving profili değişse de DÜŞER"
    assert "keep" in out.columns, "donmuş: train'de kalan 'keep' serving'de sabit olsa da KALIR (yeniden ölçmez)"


def test_select_drop_is_terminus():
    """TERMINUS #3: SelectDropStep PIPELINE'ın SON adımı — sonrasında hiçbir StepSpec/RouteSpec yok
    (hiçbir fitted_stat seçimden sonra gelemez, aksi halde state seçilmemiş kolona bağlanırdı)."""
    assert getattr(PIPELINE[-1], "step_cls", None) is SelectDropStep, \
        "select_drop PIPELINE'ın EN SON adımı olmalı (terminus)"


def test_select_drop_in_orchestrator():
    """Orkestratör: select_drop manifest'te EN YÜKSEK order (terminus); low-var kolon final_features'ta
    YOK; serving==training. Düşüşü garanti eden low-var+high-corr fixture."""
    n = 40
    a = pd.Series([float(i) for i in range(n)])
    fx = pd.DataFrame({
        "x1":   a,
        "x2":   a * 2.0 + 1.0,                          # scale sonrası x1 ile özdeş → high-corr (biri düşer)
        "flat": pd.Series([3.0] * n),                   # scale sonrası sabit-0 → low-var (düşer)
        "cat":  pd.Series(["p", "q"] * (n // 2), dtype="object"),
    })
    y = _deterministic_y(n)
    ctx = {"ratio_pairs": [], "encode_exclude": [], "outlier_cols": [], "intent": ""}
    X_fit, artifact = fit_pipeline(fx.copy(), y=y, ctx=ctx)

    sd = [e for e in artifact.manifest if e["id"] == "select_drop"]
    assert len(sd) == 1, "select_drop manifest'te tam bir kez olmalı"
    assert sd[0]["order"] == max(e["order"] for e in artifact.manifest), "select_drop EN YÜKSEK order (terminus)"
    assert "flat" not in artifact.final_features, "low-var 'flat' final_features'tan düşmeli"

    X_serve = transform_pipeline(fx.copy(), artifact)
    assert list(X_serve.columns) == artifact.final_features
    for c in artifact.final_features:
        assert np.allclose(X_serve[c].to_numpy(dtype=float), X_fit[c].to_numpy(dtype=float)), \
            f"orkestratör select_drop serving '{c}' != training"


# ══════════════════════════ ORKESTRATÖR İSKELESİ ═══════════════════════════════
# PIPELINE gerçek build_features sırasını replay eder. Bu blok: ratio_construct inf→NaN
# parity + manifest-order=enumerate invariant + legacy_ref kaynak-drift + orkestratör
# çok-adım parity (4 elle-zincirle YAN YANA yeşil, silmeden ÖNCE) + E pipeline-invariant.


# ── ratio_construct portu: inf→NaN devri eski _build_ratio_features ile birebir ──


def test_ratio_construct_inf_nan_parity():
    """RatioConstructStep row-wise çıktısı == eski _build_ratio_features row-wise kısmı (inf→NaN dahil).
    Porlama inf-handling'i kazara değiştirmedi — çift-seçim eski motordan, bölme yeni adımdan."""
    n = 20
    a = pd.Series([float(i + 1) * 3.0 for i in range(n)])          # varied, pozitif
    b = pd.Series([float((i % 7) + 1) for i in range(n)]); b.iloc[3] = 0.0   # 1 sıfır → %5 (kapıyı geçer)
    fx = pd.DataFrame({"aa": a, "bb": b})
    df_old, added, _, ratio_meta = _build_ratio_features(
        fx.copy(), original_cols={"aa", "bb"}, meta=None,
        intent="classification", aggregated=False, leaky_cols_high=set())
    assert added, "fixture en az bir oran üretmeli (kapıları geçen çift)"
    assert "ratio_aa_bb" in added and df_old["ratio_aa_bb"].isna().iloc[3], "payda-sıfır satır NaN (inf→NaN)"

    pairs = [(name, num, den) for name, (num, den) in ratio_meta.items()]
    out = RatioConstructStep({"pairs": pairs}).transform(fx.copy(), None)
    for name in added:
        left = out[name].to_numpy(dtype="float64")
        right = df_old[name].to_numpy(dtype="float64")
        # NaN konumları + sonlu değerler birebir (equal_nan)
        assert np.array_equal(left, right, equal_nan=True), f"ratio_construct '{name}' inf→NaN parity kırıldı"


# ── orkestratör fixture (target YOK → serving==training temiz) ──


def _orchestrator_fixture() -> "tuple[pd.DataFrame, pd.Series, dict]":
    """impute-null (ratio kaynağı a) + ratio çifti + düşük-kard cat (onehot) + outlier (robust)
    + inf kolon (inf_impute). Yüksek-kardinalite YOK → target route boş → serving==training tam."""
    fx = pd.DataFrame({
        "a":       pd.array([10.0, 20.0, None, 40.0, 50.0, 60.0, 70.0, 80.0], dtype="Float64"),
        "b":       [2.0, 5.0, 4.0, 8.0, 10.0, 3.0, 6.0, 2.0],       # payda, sıfırsız
        "cat":     ["x", "x", "y", "x", "y", "y", "x", "y"],        # card 2 → onehot
        "big":     [1.0, 2.0, 3.0, 4.0, 1000.0, 2.0, 3.0, 1.0],     # outlier → robust
        "inf_src": [1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0],     # inf → inf_impute
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    ctx = {"ratio_pairs": [("ratio_a_b", "a", "b")], "encode_exclude": [], "outlier_cols": ["big"]}
    return fx, y, ctx


def _old_oracle_sequence(fx: pd.DataFrame, y: pd.Series, ctx: dict) -> pd.DataFrame:
    """Eski feature_builder fonksiyon dizisi = PIPELINE'ın TAM karşılığı (split-first sırada, X_train'de).
    Config-tam: her adım ctx'ten (zscore/normalize, ratio, exclude, outlier, intent/univariate, meta/domain).
    fit_pipeline bununla birebir olmalı (terminus'un uçtan-uca sadakat referansı). 8-satır fixture'da
    ekstra config yok → mevcut davranışla aynı (zscore/univariate no-op)."""
    exclude = set(ctx.get("encode_exclude", []))
    xo = _clean_data(fx.copy())                                                # ImputeStep
    for name, src in ctx.get("zscore_specs", []):                             # ZScoreStep (train-fit: frame=X_train)
        if src in xo.columns and name not in xo.columns:
            s = pd.to_numeric(xo[src], errors="coerce")
            xo[name] = ((s - s.mean()) / (s.std(ddof=0) + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
    for name, src in ctx.get("normalize_specs", []):                          # NormalizeStep
        if src in xo.columns and name not in xo.columns:
            s = pd.to_numeric(xo[src], errors="coerce")
            xo[name] = ((s - s.min()) / ((s.max() - s.min()) + 1e-9)).replace([np.inf, -np.inf], 0).fillna(0)
    for name, num, den in ctx.get("ratio_pairs", []):                         # RatioConstructStep (rowwise)
        if num in xo.columns and den in xo.columns and name not in xo.columns:
            n = pd.to_numeric(xo[num], errors="coerce"); d = pd.to_numeric(xo[den], errors="coerce")
            xo[name] = (n / d).replace([np.inf, -np.inf], np.nan)
    xo = _finalize_ratio_features(xo, [n for n, _, _ in ctx.get("ratio_pairs", [])])   # RatioFinalizeStep (train=frame)
    xo, _, _ = _count_encode_tt(xo, xo.copy(), exclude=exclude)               # CountEncodeStep
    xo, _, _ = _groupby_deviation_tt(xo, xo.copy(), meta=None, exclude=exclude)  # GroupbyDeviationStep
    xo, _ = _encode_categorical_tt(xo, xo.copy(), y)                          # encode route (OneHot+Target)
    xo, _ = _scale_features_tt(xo, xo.copy(), ctx.get("outlier_cols", []))    # InfImpute + scaler route
    # TERMINUS selection (SelectDropStep = fit dizisi + terminal cleanup)
    xo, _ = _select_features_tt(xo, xo.copy())
    xo = xo.apply(pd.to_numeric, errors="coerce").fillna(0)
    intent = (ctx.get("intent") or "").strip().lower()
    if intent in ("classification", "regression"):
        xo, _, _ = _select_univariate_tt(xo, xo.copy(), y, intent)
    xo, _, _ = _detect_leakage_tt(xo, xo.copy(), y, ctx.get("meta"),
                                  protected_high_card=ctx.get("protected_high_card"))
    xo, _, _ = _apply_domain_leakage_drops_tt(
        xo, xo.copy(), ctx.get("domain_leakage_report") or {"dropped": [], "suspected": []})
    return xo.apply(pd.to_numeric, errors="coerce").fillna(0)   # terminal cleanup (step transform)


def test_orchestrator_multistep_parity():
    """Orkestratör çok-adım parity: fit_pipeline == eski-oracle-dizisi; transform_pipeline ==
    fit çıktısı (serving==training, target YOK). Bu, 4 elle-zinciri DEĞİŞTİRECEK test (silme ÖNCESİ yeşil)."""
    fx, y, ctx = _orchestrator_fixture()
    X_fit, artifact = fit_pipeline(fx, y, ctx)
    xo = _old_oracle_sequence(fx, y, ctx)

    # (1) fit_pipeline == eski build_features ara çıktısı (o 8 adıma karşılık)
    common = [c for c in xo.columns if c in X_fit.columns]
    assert set(X_fit.columns) == set(xo.columns), f"kolon seti farklı: {set(X_fit.columns) ^ set(xo.columns)}"
    for col in common:
        assert np.allclose(X_fit[col].to_numpy(dtype=float), xo[col].to_numpy(dtype=float)), \
            f"orkestratör fit '{col}' eski-oracle'dan farklı"

    # (2) transform_pipeline (serving) == fit çıktısı (target yok → cross-fit farkı yok)
    X_serve = transform_pipeline(fx, artifact)
    assert list(X_serve.columns) == artifact.final_features
    for col in artifact.final_features:
        assert np.allclose(X_serve[col].to_numpy(dtype=float), X_fit[col].to_numpy(dtype=float)), \
            f"serving '{col}' != training (serving==training kırıldı)"


def test_orchestrator_manifest_order_is_enumerate():
    """Manifest-sadakat invariant: order ELLE atanmaz, fiili fit-döngüsü indeksinden türer →
    manifest[i].order == i (kesintisiz, sıralı)."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    orders = [e["order"] for e in artifact.manifest]
    assert orders == list(range(len(artifact.manifest))), f"order enumerate değil: {orders}"
    # route kalemleri de sayaçta yer alır (encode + scaler = 2 route), executor'lar ayrı kalem
    types = [e["type"] for e in artifact.manifest]
    assert types.count("route") == 2, "encode + scaler = 2 route karar kalemi"
    assert "rowwise" in types, "ratio_construct rowwise kalemi"


def test_legacy_ref_source_drift():
    """legacy_ref KAYNAK-DRİFT: PIPELINE'daki adımların legacy_ref'leri build_features KAYNAĞINDA
    aynı GÖRELİ sırada görünmeli. KOŞULLU-ATLAMA: yalnız PIPELINE'da TEMSİL edilenlerin göreli sırası
    ('temsil ettiklerimiz doğru sırada'); ts-lag/aggregate/groupby/select gibi temsil edilmeyenler
    atlanır → henüz-yazılmamış koşullu adımlar testi kırmaz."""
    import inspect as _inspect
    from src.services import feature_builder
    src = _inspect.getsource(feature_builder.build_features)

    refs: list = []
    for spec in PIPELINE:
        r = spec.legacy_ref
        if not refs or refs[-1] != r:           # ardışık tekrar dedup (scaler ailesi tek ref)
            refs.append(r)
    positions = {}
    for r in set(refs):
        idx = src.find(r + "(")                 # ÇAĞRI olarak (yorum-mention değil)
        assert idx != -1, f"legacy_ref '{r}' build_features'ta çağrı olarak yok (drift/yanlış ad)"
        positions[r] = idx
    ordered = sorted(refs, key=lambda r: positions[r])
    assert refs == ordered, \
        f"PIPELINE legacy_ref sırası build_features kaynağıyla uyuşmuyor:\n  PIPELINE={refs}\n  kaynak ={ordered}"


def test_orchestrator_transform_pipeline_never_fit_transforms(monkeypatch):
    """E pipeline-düzeyi invariant: transform_pipeline hiçbir step.fit_transform ÇAĞIRMAZ (yalnız
    transform) → cross-fitting serving'e sızamaz. Target'lı artifact'te bile fit_transform patlatılırsa
    transform_pipeline etkilenmemeli."""
    # target'lı fixture (yüksek-kardinalite kolon) — target executor artifact'e girer
    fx = pd.DataFrame({
        "hi":  [f"c{i % 15}" for i in range(30)],                   # card 15 → target
        "num": [float(i) for i in range(30)],
    })
    y = _deterministic_y(len(fx))
    ctx = {"ratio_pairs": [], "encode_exclude": [], "outlier_cols": []}
    _, artifact = fit_pipeline(fx, y, ctx)
    assert any(e["id"] == "target_encode" for e in artifact.manifest), "target executor manifest'te olmalı"

    for cls in (ImputeStep, RatioConstructStep, RatioFinalizeStep, CountEncodeStep, OneHotStep,
                TargetEncodeStep, InfImputeStep, RobustScaleStep, StandardScaleStep):
        monkeypatch.setattr(cls, "fit_transform",
                            lambda *a, **k: pytest.fail("transform_pipeline fit_transform çağırdı (E ihlali)"))
    out = transform_pipeline(fx, artifact)                          # yalnız transform → fail YOK
    assert list(out.columns) == artifact.final_features


# ══════════════════ UÇTAN-UCA GERÇEK-VERİ PARITY HARNESS (terminus ön-koşulu) ═══════════════════
# İDDİANIN TAM SINIRI (abartma yok — okuyan yanlış güven duymasın):
#   KANITLANAN     : fit_pipeline == `_old_oracle_sequence` (aşağıdaki ELLE rekonstrüksiyon),
#                    gerçek datasette, gerçek kolon çeşitliliğinde, birebir.
#   KANITLANMAYAN  : fit_pipeline == `build_features`'ın KENDİSİ. build_features state/LLM/meta
#                    bağımlı olduğu için doğrudan çağrılamıyor; oracle onun post-split zincirinin
#                    ELLE yazılmış karşılığıdır (aynı feature_builder FONKSİYONLARINI çağırır, ama
#                    sırayı/config'i biz kuruyoruz). Bu, silinen 4 elle-zincirin yerine geçen TEK
#                    merkezî elle-zincirdir — daha iyi, ama hâlâ bir elle-zincir borcu.
#   BORCU KAPATAN  : oracle'ın sadakati bugün YALNIZ `test_legacy_ref_source_drift`'in GÖRELİ SIRA
#                    kontrolüyle korunuyor (değer/config sadakati DEĞİL). Terminus'ta build_features
#                    fiilen orkestratörü çağırınca bu borç KENDİLİĞİNDEN düşer (oracle gereksizleşir).
#   KAPSAM         : `scripts/layer1_coverage_inventory.py` gösterdi ki harness pre-split katmanını
#                    ölçmüyor (fit ve transform AYNI girdi sözleşmesini görüyor → eksik port fark
#                    doğurmuyor). Kapsam borcu oradan takip edilir, buradan DEĞİL.
# ÖN-TEMİZLİK (_clean_data: dedup+currency+impute — fit_pipeline'ın HENÜZ port etmediği kısım) upfront
# bir kez → ImputeStep/_clean_data ikisi de no-op → #4 nötr. ctx DİNAMİK (numeric kolonlar programatik,
# hardcode YOK). y sentetik-deterministik (parity y-realizmi gerektirmez).


def _real_dataset_parity_harness(csv_glob: str, intent: str) -> None:
    path = _find_csv(csv_glob)
    if not path:
        pytest.skip(f"{csv_glob} yok (gitignored) — gerçek-veri harness atlandı")
    df = pd.read_csv(path)
    df = _clean_data(df).reset_index(drop=True)      # dedup+currency+impute BİR KEZ (port-dışı ön-temizlik)
    if len(df) < 50:
        pytest.skip("çok az satır")
    num = df.select_dtypes(include="number").columns.tolist()
    if len(num) < 2:
        pytest.skip("harness için >=2 numeric kolon gerekir")

    y = _deterministic_y(len(df))
    cut = int(len(df) * 0.7)
    X_train, X_test = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    y_train = y.iloc[:cut].reset_index(drop=True)

    # ctx DİNAMİK — hiçbir kolon adı gömülü değil, hepsi numeric kolonlardan programatik türetilir
    ctx = {
        "encode_exclude": [], "intent": intent, "outlier_cols": num[:1],
        "ratio_pairs":     [(f"ratio_{num[0]}_{num[1]}", num[0], num[1])],
        "zscore_specs":    [(f"z_{num[0]}", num[0])],
        "normalize_specs": [(f"n_{num[1]}", num[1])],
    }

    X_fit, artifact = fit_pipeline(X_train.copy(), y_train, ctx)
    xo = _old_oracle_sequence(X_train.copy(), y_train, ctx)

    # (1) UÇTAN-UCA parity: fit_pipeline == oracle (aynı feature_builder fonksiyonları) — gerçek veride birebir
    assert set(X_fit.columns) == set(xo.columns), \
        f"[{csv_glob}] kolon seti farklı: {(set(X_fit.columns) ^ set(xo.columns))}"
    for c in X_fit.columns:
        assert np.allclose(X_fit[c].to_numpy(dtype=float), xo[c].to_numpy(dtype=float), equal_nan=True), \
            f"[{csv_glob}] uçtan-uca '{c}' fit_pipeline ≠ oracle (gerçek veri)"

    # (2) SERVING yolu gerçek veride: transform_pipeline(X_test) donmuş params'la çalışır, final_features üretir
    X_serve = transform_pipeline(X_test.copy(), artifact)
    assert list(X_serve.columns) == artifact.final_features, f"[{csv_glob}] serving final_features uyuşmadı"
    assert len(X_serve) == len(X_test), f"[{csv_glob}] serving satır sayısı korunmalı"


def test_orchestrator_real_dataset_parity_classification():
    """UÇTAN-UCA (classification): fit_pipeline == build_features zinciri, credit gerçek verisinde."""
    _real_dataset_parity_harness("*credit_risk_dataset.csv", "classification")


def test_orchestrator_real_dataset_parity_regression():
    """UÇTAN-UCA (regression): fit_pipeline == build_features zinciri, housing gerçek verisinde."""
    _real_dataset_parity_harness("housing.csv", "regression")


# ══════════════ ÜRETİM-KANITI GUARD'I (sessiz 0-doldurma fail-closed) ══════════════
# Terminal reindex(fill_value=0) sessiz bir tehlikeydi: portlanmamış bir adımın kolonu
# hatasız 0 dolup model yanlış girdiyle skorluyordu. Guard bunu iki farklı BAŞARISIZLIK
# MODUna ayırarak fail-closed'a çevirir (sebepleri farklı, karıştırılmamalı).


def test_guard_never_produced_raises_fail_closed():
    """final_features'ta HİÇ üretilmemiş kolon → MissingFeatureProduction, skorlama YOK.
    (Port borcu modu: hiçbir adım o kolonu hiçbir aşamada oluşturmadı.)"""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    artifact.final_features = list(artifact.final_features) + ["hic_uretilmeyen_kolon"]

    with pytest.raises(MissingFeatureProduction) as ei:
        transform_pipeline(fx, artifact)
    assert "hic_uretilmeyen_kolon" in ei.value.never_produced
    assert ei.value.produced_then_dropped == [], "bu kolon hiç üretilmedi → dropped kovasında OLMAMALI"


def test_guard_separates_produced_then_dropped_from_never_produced():
    """Üretilip sonra düşürülen kolon AYRI kovada raporlanır — port etmek onu ÇÖZMEZ
    (artifact tutarsızlığı). İki modun karışması envanteri şişirirdi."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    # 'cat' ham girdide VAR ama one-hot onu tüketip düşürüyor → üretildi-sonra-düşürüldü
    artifact.final_features = list(artifact.final_features) + ["cat", "hic_yok"]

    with pytest.raises(MissingFeatureProduction) as ei:
        transform_pipeline(fx, artifact)
    assert ei.value.never_produced == ["hic_yok"]
    assert ei.value.produced_then_dropped == ["cat"]


def test_guard_silent_zero_fill_is_closed():
    """REGRESYON KİLİDİ: eksik kolon SESSİZCE 0 dolmamalı. Guard'dan önceki davranış
    (hatasız 0 kolon) geri gelirse bu test kırılır."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    artifact.final_features = list(artifact.final_features) + ["sessizce_sifirlanacakti"]
    try:
        out = transform_pipeline(fx, artifact)
    except MissingFeatureProduction:
        return                                        # doğru davranış
    pytest.fail(f"sessiz 0-doldurma geri geldi: kolon={out['sessizce_sifirlanacakti'].tolist()[:3]}")


def test_guard_passes_on_healthy_artifact():
    """Sağlıklı artifact (kendi final_features'ı) → guard patlamaz, çıktı değişmez.
    Guard'ın yanlış-pozitif üretmediğinin kanıtı."""
    fx, y, ctx = _orchestrator_fixture()
    X_fit, artifact = fit_pipeline(fx, y, ctx)
    out = transform_pipeline(fx, artifact)
    assert list(out.columns) == artifact.final_features
    for col in artifact.final_features:
        assert np.allclose(out[col].to_numpy(dtype=float), X_fit[col].to_numpy(dtype=float))


def test_onehot_zero_fill_policy_is_declared():
    """Politikası TANIMLI eksik (one-hot dummy vocab) izinli — guard'ın izin verdiği tek
    sıfır sınıfı beyan edilmiş olandır (§2.8 D: all_zero ≡ referans kategori)."""
    fx = pd.DataFrame({"cat": ["x", "y", "x", "y", "x", "y"]})
    step = OneHotStep(config={"columns": ["cat"]})
    params = step.fit(fx)
    assert set(step.zero_fill_policy(params)) == set(params["dummy_vocab"])
    assert params["dummy_vocab"], "fixture dummy üretmeli (aksi halde test boş geçer)"


# ══════════════ frozen_choice ŞEKİL TUTARLILIĞI (route ≡ adım-içi seçim) ══════════════


def test_frozen_choice_single_shape_across_route_and_step():
    """Katman 2 'seçim nerede' sorusunu adım tipine göre DEĞİŞTİRMEDEN cevaplayabilmeli:
    route kararı da groupby çift-seçimi de manifest'te AYNI {kind, selection} alanında."""
    train, _ = _groupby_fixture()
    ctx = {"ratio_pairs": [], "encode_exclude": [], "outlier_cols": []}
    _, artifact = fit_pipeline(train, _deterministic_y(len(train)), ctx)

    choices = [e for e in artifact.manifest if e.get("frozen_choice")]
    kinds = {e["frozen_choice"]["kind"] for e in choices}
    assert "route" in kinds, "route kararları frozen_choice taşımalı"
    assert "pairs" in kinds, "groupby çift-seçimi frozen_choice taşımalı (artık params'a gömülü değil)"
    for e in choices:                                  # TEK şekil: her biri kind + selection
        assert set(("kind", "selection")) <= set(e["frozen_choice"]), \
            f"frozen_choice şekli tutarsız: {e['id']} → {e['frozen_choice'].keys()}"


def test_groupby_frozen_choice_matches_param_store():
    """frozen_choice bir GÖRÜNÜRLÜK katmanı — kaynak hâlâ param_store; ikisi tutarlı olmalı
    (kopya değil türev; sapmaları bu test yakalar)."""
    train, _ = _groupby_fixture()
    _, artifact = fit_pipeline(train, _deterministic_y(len(train)),
                               {"ratio_pairs": [], "encode_exclude": [], "outlier_cols": []})
    entry = next(e for e in artifact.manifest if e["id"] == "groupby_deviation")
    records = artifact.param_store["groupby_deviation"]
    sel = entry["frozen_choice"]["selection"]
    assert [(s["group"], s["value"]) for s in sel] == [(r["cat"], r["num"]) for r in records]


# ══════════ KAPSAM KAPISI (desteklenmeyen yol → artifact KURULMAZ, fail-closed) ══════════
# Guard "gördüğünü" kanıtlar; tetiklenmemiş yollar hakkında hiçbir şey söylemez. Doğru cevap
# o yolları tahminen portlamak değil, DESTEKLEMEDİĞİMİZİ söylemek — ve bunu serving'de değil
# artifact-build anında yapmak (eksik artifact hiç doğmasın).


def test_scope_gate_rejects_unsupported_modality():
    """Desteklenmeyen koşul beyan edildiyse artifact KURULMAZ (istisna), sessiz eksik artifact YOK."""
    fx, y, ctx = _orchestrator_fixture()
    with pytest.raises(UnsupportedPipelineScope) as ei:
        fit_pipeline(fx, y, {**ctx, "modality": "timeseries"})
    assert any(v["fact"] == "modality" for v in ei.value.violations)


def test_scope_gate_rejects_every_declared_unsupported_path():
    """Kapsam kurallarının HEPSİ fiilen reddediyor — kural tablosuna eklenip unutulmuş
    (hiç kontrol edilmeyen) bir olgu kalmasın."""
    fx, y, ctx = _orchestrator_fixture()
    tetikleyici = {
        "modality": "timeseries", "aggregated": True, "timeseries_lag": {"lag_columns": ["x"]},
        "domain_features": ["dom_x"], "date_features": ["d_year"],
        "column_renames": [{"a": "b"}], "unported_ops": ["log"],
        "deduplicated": False,        # YAPISAL ön-koşul: dedup'sız çerçeve fitted_stat'ları kaydırır
    }
    for fact, deger in tetikleyici.items():
        with pytest.raises(UnsupportedPipelineScope) as ei:
            fit_pipeline(fx, y, {**ctx, fact: deger})
        assert [v["fact"] for v in ei.value.violations] == [fact], \
            f"'{fact}' tek başına reddetmeli (kural tablosu ile kontrol uyuşmuyor)"


def test_scope_gate_undeclared_builds_but_not_deployable():
    """Beyan EDİLMEMİŞ olgu ihlal DEĞİLDİR (test/deney artifact'i kurulabilsin) ama
    sessizce "destekleniyor" da sayılmaz → deployable=False. Fail-open yok."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)                    # hiçbir kapsam olgusu beyan edilmemiş
    scope = artifact.supported_scope
    assert scope["violations"] == []
    assert scope["deployable"] is False, "beyansız artifact deploy edilebilir sayılmamalı"
    assert set(scope["undeclared"]) == {f for f, _, _ in _SCOPE_RULES}


def test_scope_gate_fully_declared_clean_is_deployable():
    """Tüm olgular beyan edilmiş ve hiçbiri ihlal değilse → deployable=True."""
    fx, y, ctx = _orchestrator_fixture()
    temiz = {"modality": "tabular", "aggregated": False, "timeseries_lag": None,
             "domain_features": [], "date_features": [], "column_renames": [], "unported_ops": [],
             "deduplicated": True}
    _, artifact = fit_pipeline(fx, y, {**ctx, **temiz})
    assert artifact.supported_scope["deployable"] is True
    assert artifact.supported_scope["undeclared"] == []


def test_scope_gate_runs_before_any_step():
    """Kapsam ihlali adımlar KOŞMADAN yakalanmalı (yarı-kurulmuş state bırakma)."""
    fx, y, ctx = _orchestrator_fixture()
    import src.services.inference.orchestrator as orch
    cagrildi = []
    orijinal = ImputeStep.fit_transform
    try:
        ImputeStep.fit_transform = lambda *a, **k: cagrildi.append(1) or orijinal(*a, **k)
        with pytest.raises(UnsupportedPipelineScope):
            orch.fit_pipeline(fx, y, {**ctx, "aggregated": True})
    finally:
        ImputeStep.fit_transform = orijinal
    assert cagrildi == [], "kapsam kapısı ilk adımdan ÖNCE çalışmalı"


def test_ratio_pairs_recorded_as_frozen_choice():
    """Ratio çift seçimi manifest'te kanonik {kind, selection} olarak DONAR (8.1 kararı:
    'pre-split sabit, artifact'e frozen yaz'). Kaynak alanı da kayıtlı olmalı."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    entry = next(e for e in artifact.manifest if e["id"] == "ratio_construct")
    fc = entry["frozen_choice"]
    assert fc["kind"] == "pairs" and fc["source"] == "feature_report.ratio_feature_meta"
    assert [(s["output"], s["numerator"], s["denominator"]) for s in fc["selection"]] \
        == list(ctx["ratio_pairs"])


def test_ratio_meta_is_unfiltered_superset_of_survivors():
    """8.1 kısayolunun DAYANAĞI: ratio_feature_meta X_train'e göre filtrelenmez (selection'da
    düşenler de içinde) → frozen kaynağı olarak yeterli; ratio_features (hayatta-kalanlar)
    kullanılsaydı select_drop'un düşürdükleri kaybolurdu. Kaynakta bu ayrım korunmalı."""
    import inspect as _inspect
    from src.services import feature_builder
    src = _inspect.getsource(feature_builder.build_features)
    meta_satiri = next(l for l in src.splitlines() if '"ratio_feature_meta"' in l)
    kalan_satiri = next(l for l in src.splitlines() if '"ratio_features"' in l)
    assert "X_train.columns" not in meta_satiri, \
        "ratio_feature_meta filtrelenmiş → frozen kaynağı artık güvenli DEĞİL (8.1 dayanağı düştü)"
    assert "X_train.columns" in kalan_satiri, "ratio_features hayatta-kalan listesi olmalı (kontrast)"


# ══════════ MissingIndicator / ZeroFlag / DropColumns (kalan kök boşluklar) ══════════


def _flag_fixture() -> pd.DataFrame:
    """null taşıyan kolon (_isna) + sıfır taşıyan 3+ değerli kolon (is_zero) + binary
    (is_zero ÜRETİLMEMELİ, #10) + kategorik."""
    return pd.DataFrame({
        "amount":  [10.0, 0.0, 30.0, 0.0, 50.0, 60.0, 0.0, 80.0],
        "emp_len": [1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0],     # null → _isna
        "flag":    [0, 1, 0, 1, 0, 1, 0, 1],                        # binary → is_zero YOK (#10)
        "cat":     ["x", "y", "x", "y", "x", "y", "x", "y"],
    })


def test_missing_indicator_parity():
    """`<col>_isna` değer + DTYPE parity: yeni MissingIndicatorStep == _add_missing_indicators."""
    fx = _flag_fixture()
    eski = fx.copy()
    added = _add_missing_indicators(eski, exclude=set())

    step = MissingIndicatorStep(config={"columns": [n[:-len("_isna")] for n in added]})
    yeni = step.transform(fx.copy(), None)

    assert added, "fixture en az bir _isna üretmeli (aksi halde test boş geçer)"
    for name in added:
        assert name in yeni.columns, f"{name} üretilmedi"
        assert yeni[name].tolist() == eski[name].tolist(), f"{name} değerleri eski yoldan farklı"
        assert yeni[name].dtype == eski[name].dtype, f"{name} dtype farkı (int8 sözleşmesi)"


def test_missing_indicator_runs_before_impute():
    """SIRA KİLİDİ: eksiklik bayrağı fillna'dan ÖNCE. Sonra çalışsaydı null kalmaz, bayrak
    hep 0 olurdu → sinyal SESSİZCE silinirdi. Hem PIPELINE sırası hem DAVRANIŞ kanıtlanır."""
    ids = [s.step_cls.id for s in PIPELINE if isinstance(s, StepSpec)]
    assert ids.index("missing_indicator") < ids.index("impute_missing"), \
        "missing_indicator PIPELINE'da impute'tan ÖNCE olmalı"

    fx = _flag_fixture()
    step = MissingIndicatorStep(config={"columns": ["emp_len"]})
    once = step.transform(fx.copy(), None)                       # impute ÖNCESİ (doğru sıra)
    sonra = step.transform(ImputeStep().fit_transform(fx.copy())[0], None)   # impute SONRASI (yanlış)
    assert once["emp_len_isna"].sum() == 2, "impute öncesi null sinyali görülmeli"
    assert sonra["emp_len_isna"].sum() == 0, \
        "impute sonrası bayrak hep 0 olmalı — sıra kilidinin NEDEN var olduğunun kanıtı"


def test_zero_flag_parity_and_binary_guard():
    """is_zero değer parity + #10 guard: binary kolonda is_zero ÜRETİLMEZ (corr −1.0 kopya)."""
    fx = _flag_fixture()
    eski = fx.copy()
    _, added = _build_zero_missing_flags(eski, original_cols=set(fx.columns),
                                         intent="classification", aggregated=False,
                                         force_zero_cols=set())
    step = ZeroFlagStep(config={"columns": [n[len("is_zero_"):] for n in added]})
    yeni = step.transform(fx.copy(), None)

    assert "is_zero_amount" in added, "3+ değerli sıfırlı kolon is_zero üretmeli"
    assert "is_zero_flag" not in added, "#10: binary kolonda is_zero üretilmemeli"
    for name in added:
        assert yeni[name].tolist() == eski[name].tolist(), f"{name} değerleri eski yoldan farklı"


def test_drop_columns_removes_frozen_list():
    """Donmuş liste düşer, gerisi dokunulmaz (re_derive:false — serving'de yeniden karar YOK)."""
    fx = _flag_fixture()
    out = DropColumnsStep(config={"columns": ["flag", "yok_olan_kolon"]}).transform(fx.copy(), None)
    assert "flag" not in out.columns
    assert set(out.columns) == set(fx.columns) - {"flag"}, "yalnız listedekiler düşmeli"


def test_target_in_frame_distorts_groupby_selection():
    """ÖLÇÜLMÜŞ KÖK NEDEN — REGRESYON KİLİDİ (2026-07-23).

    Guard'ın KÖR NOKTASI: eksik bir KALDIRMA "eksik üretim" göstermez; downstream SEÇİM
    SAPMASI üretir. Hedef kolon çerçevede kalırsa GroupbyDeviation'ın `-norm_var` sıralamasına
    girip gerçek bir kolonu top-N'den atar (binary hedefte normalize varyans yüksektir).
    Bu, 'kaskad' sanılan sapmanın gerçek kökü çıktı. DropColumnsStep bunu kapatır."""
    n = 120
    rng = np.random.default_rng(0)
    fx = pd.DataFrame({
        "grp":    ["A"] * 60 + ["B"] * 60,
        "hedef":  ([0] * 14 + [1] * 46) * 2,                 # binary, dengeli-ish → yüksek norm_var
        "olcum":  rng.normal(100, 1, n),                     # dar dağılım → düşük norm_var
        "ikinci": rng.normal(50, 0.5, n),
    })
    sec = GroupbyDeviationStep(config={"exclude": []})
    _, num_hedefli = sec._select(fx)
    _, num_temiz = sec._select(DropColumnsStep(config={"columns": ["hedef"]}).transform(fx, None))

    assert "hedef" in num_hedefli, "hedef aday havuzuna giriyor (sorunun kendisi)"
    assert "hedef" not in num_temiz, "DropColumnsStep sonrası hedef aday olmamalı"
    assert num_hedefli != num_temiz, "hedefin varlığı seçimi DEĞİŞTİRİYOR (ölçülmüş kök neden)"


def test_target_in_frame_creates_target_derived_feature():
    """SIZINTI YÜZEYİ: hedef çerçevede kalırsa groupby HEDEFTEN feature türetir
    (dev_<hedef>_by_grp = hedefin grup ortalamasından sapması) → doğrudan target leakage.
    build_features'ta imkânsız (hedef split öncesi düşer); orkestratörde engelleyen YOKTU."""
    n = 120
    fx = pd.DataFrame({
        "grp":   ["A"] * 60 + ["B"] * 60,
        "hedef": ([0] * 14 + [1] * 46) * 2,
        "olcum": [float(i % 17) for i in range(n)],
    })
    step = GroupbyDeviationStep(config={"exclude": []})
    uretilen = set(step.fit_transform(fx.copy())[0].columns) - set(fx.columns)
    assert any("hedef" in c for c in uretilen), "hedefli çerçevede hedef-türevi üretiliyor (yüzey gerçek)"

    temiz = DropColumnsStep(config={"columns": ["hedef"]}).transform(fx.copy(), None)
    uretilen_temiz = set(step.fit_transform(temiz)[0].columns) - set(temiz.columns)
    assert not any("hedef" in c for c in uretilen_temiz), "drop sonrası hedef-türevi ÜRETİLMEMELİ"


# ══════════ SİMETRİK GUARD (eksik üretim VE fazla mevcudiyet) ══════════
# Asimetri 4. kökü SAKLAMIŞTI: guard "üretilmedi mi" soruyordu, "fazladan var mı" sormuyordu.


def test_symmetric_guard_catches_extra_column():
    """FAZLA mevcudiyet artık hata — eskiden sessizdi. Serving'de eğitimde olmayan bir kolon
    ara çerçevede duruyorsa sapma vardır; beyansız geçemez."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    kirli = fx.copy()
    kirli["beklenmeyen_kolon"] = 1.0

    with pytest.raises(FrameShapeDivergence) as ei:
        transform_pipeline(kirli, artifact)
    assert "beklenmeyen_kolon" in ei.value.extra
    assert ei.value.missing == [], "bu senaryoda eksik yok, yalnız fazla"


def test_symmetric_guard_extra_can_be_declared_tolerated():
    """Tolerans MÜMKÜN ama BEYAN edilmeli (§2.4 uzlaşması) — beyan edilince geçer."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    artifact.tolerated_extra_columns = ["beklenmeyen_kolon"]
    kirli = fx.copy()
    kirli["beklenmeyen_kolon"] = 1.0
    out = transform_pipeline(kirli, artifact)               # beyan var → hata YOK
    assert list(out.columns) == artifact.final_features


def test_symmetric_guard_localizes_divergence_to_step():
    """Sapma ADIM düzeyinde raporlanır (sonda değil) → hangi adımdan sonra ayrıldığı belli."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    kirli = fx.copy()
    kirli["beklenmeyen_kolon"] = 1.0
    with pytest.raises(FrameShapeDivergence) as ei:
        transform_pipeline(kirli, artifact)
    ilk_adim = min(artifact.manifest, key=lambda e: e["order"])
    assert ei.value.step_id == ilk_adim["id"], "ilk sapan adımda patlamalı"


def test_manifest_records_training_columns_per_step():
    """Simetrik karşılaştırmanın referansı: her adım kaleminde eğitim kolon-seti kayıtlı."""
    fx, y, ctx = _orchestrator_fixture()
    _, artifact = fit_pipeline(fx, y, ctx)
    for e in artifact.manifest:
        assert e.get("columns_out") is not None, f"{e['id']} columns_out taşımıyor"
        assert e["columns_out"] == sorted(e["columns_out"]), "deterministik sıra"


def test_symmetric_guard_does_not_false_positive():
    """Sağlıklı yolda guard sessiz — yanlış-pozitif üretmez."""
    fx, y, ctx = _orchestrator_fixture()
    X_fit, artifact = fit_pipeline(fx, y, ctx)
    out = transform_pipeline(fx, artifact)
    assert list(out.columns) == artifact.final_features
    for c in artifact.final_features:
        assert np.allclose(out[c].to_numpy(dtype=float), X_fit[c].to_numpy(dtype=float))


# ══════════ YAPISAL ÖN-KOŞULLAR (eski yolun ADIM DEĞİL, YAPI kaynaklı garantileri) ══════════
# Ders: build_features'ın bazı garantileri sıradan/yapıdan geliyordu. Adımlara çevirirken
# kaybolduğu için eski yolda İMKÂNSIZ olan sızıntı yüzeyleri açıldı. Artık açık ön-koşul.


def test_structural_precondition_target_must_leave_frame():
    """Hedef, drop adımından sonra çerçevede kalamaz → artifact KURULMAZ."""
    fx, y, ctx = _orchestrator_fixture()
    fx = fx.assign(hedef_kolon=y.to_numpy())
    with pytest.raises(StructuralPreconditionViolation) as ei:
        fit_pipeline(fx, y, {**ctx, "target_col": "hedef_kolon", "drop_columns": []})
    assert "hedef_kolon" in str(ei.value)


def test_structural_precondition_drop_list_must_be_effective():
    """'Drop adımı var ama kablolanmamış' sessiz hatası: liste beyan edilmiş ama kolon hâlâ
    çerçevede → ihlal. Invariant doğrudan denetlenir (adımın işini yaptığının kanıtı)."""
    fx = pd.DataFrame({"cust_id": ["A", "B"], "x": [1.0, 2.0]})
    assert check_structural_preconditions(fx, {"drop_columns": ["cust_id"]}), \
        "beyan edilmiş drop kolonu çerçevede kalmışsa ihlal raporlanmalı"
    assert not check_structural_preconditions(fx.drop(columns=["cust_id"]),
                                              {"drop_columns": ["cust_id"]}), \
        "drop uygulandıysa ihlal OLMAMALI (yanlış-pozitif yok)"


def test_structural_precondition_passes_when_wired():
    """Doğru kablolandığında (ctx drop listesi → DropColumnsStep) ön-koşul sağlanır."""
    fx, y, ctx = _orchestrator_fixture()
    fx = fx.assign(cust_id=[f"C{i}" for i in range(len(fx))])
    _, artifact = fit_pipeline(fx, y, {**ctx, "drop_columns": ["cust_id"]})
    kalem = next(e for e in artifact.manifest if e["id"] == "drop_columns")
    assert "cust_id" not in kalem["columns_out"], "drop adımından sonra kolon çerçevede olmamalı"


def test_unique_id_target_encoded_reproduces_y_exactly():
    """ÖLÇÜLMÜŞ SIZINTI YÜZEYİ — REGRESYON KİLİDİ. Benzersiz ID çerçevede kalırsa
    encode_route eşiğini aşıp TargetEncoder'a gider ve serving çıktısı y'nin BİREBİR
    kendisi olur (kolon = target'ın kopyası). Eski yolda İMKÂNSIZ (ID split öncesi düşer)."""
    n = 60
    fx = pd.DataFrame({"cust_id": [f"C{i}" for i in range(n)]})
    y = _deterministic_y(n)
    te = TargetEncodeStep(config={"columns": ["cust_id"]})
    _, params = te.fit_transform(fx.copy(), y)
    serve = te.transform(fx.copy(), params)
    assert np.allclose(serve["cust_id"].to_numpy(dtype=float), y.to_numpy(dtype=float)), \
        "yüzeyin gerçekliği: benzersiz ID target-encode → çıktı == y"

    temiz = DropColumnsStep(config={"columns": ["cust_id"]}).transform(fx.copy(), None)
    assert "cust_id" not in temiz.columns, "DropColumnsStep yüzeyi kapatır"


def test_dedup_changes_fitted_stats_so_it_is_a_precondition():
    """ÖLÇÜLMÜŞ — 'drop_duplicates skip'tir, masumdur' YANLIŞ: satır üyeliğini değiştirdiği
    için ondan SONRAKİ her fitted_stat kayar. Bu yüzden kapsam olgusu (`deduplicated`)."""
    base = pd.DataFrame({"x": [1.0, 2.0, 3.0, None]})
    dup = pd.concat([base] + [pd.DataFrame({"x": [3.0]})] * 8, ignore_index=True)
    med_dupsuz = ImputeStep().fit(dup)["numeric_median"]["x"]
    med_duplu = ImputeStep().fit(dup.drop_duplicates())["numeric_median"]["x"]
    assert med_dupsuz != med_duplu, "dedup fitted_stat'ı değiştirmiyorsa ön-koşul gereksizdi"

    fx, y, ctx = _orchestrator_fixture()
    with pytest.raises(UnsupportedPipelineScope) as ei:
        fit_pipeline(fx, y, {**ctx, "deduplicated": False})
    assert any(v["fact"] == "deduplicated" for v in ei.value.violations)


# ══════════ T0 — BEYAN YÜZEYİ (assemble_pipeline_ctx) + filtrelenmemiş meta çitleri ══════════


def test_pipeline_ctx_declares_every_scope_fact():
    """BEYAN MEKANİZMASININ KİLİDİ: `assemble_pipeline_ctx` her kapsam olgusunu beyan etmeli.
    Biri `_SCOPE_RULES`'a kural ekleyip ctx'e eklemeyi unutursa artifact SESSİZCE
    'destekleniyor' olmaz (deployable=False olur) — ama bu test unutmayı ANINDA yakalar."""
    ctx = assemble_pipeline_ctx(
        intent="classification", modality="", aggregated=False, target_name="y",
        drop_columns=[], ratio_feature_meta={}, missing_indicator_meta={},
        derived_flag_meta={}, feature_plan={}, encode_exclude=set(), outlier_cols=[],
        timeseries_lag=None, domain_feature_cols=[], date_feature_cols=[], column_renames=[])
    eksik = [f for f, _, _ in _SCOPE_RULES if f not in ctx]
    assert not eksik, f"beyan yüzeyi bu kapsam olgularını atlıyor: {eksik}"


def test_pipeline_ctx_is_deployable_and_normalizes_modality():
    """Boş modality build_features'ta 'tabular' demek. Ham bırakılsaydı kapsam kuralı
    `bool(v)` üzerinden onu SESSİZCE geçerdi (beyan var, değer yok = fail-open)."""
    ctx = assemble_pipeline_ctx(
        intent="classification", modality="", aggregated=False, target_name="y",
        drop_columns=[], ratio_feature_meta={}, missing_indicator_meta={},
        derived_flag_meta={}, feature_plan={}, encode_exclude=set(), outlier_cols=[],
        timeseries_lag=None, domain_feature_cols=[], date_feature_cols=[], column_renames=[])
    assert ctx["modality"] == "tabular", "boş modality açıkça tabular'a normalize edilmeli"
    assert check_pipeline_scope(ctx)["deployable"] is True, "tam beyan → deploy edilebilir"


def test_pipeline_ctx_flags_unported_ops():
    """Plan portlanmamış bir op seçerse kapsam kapısı reddetmeli (zscore/normalize portlu)."""
    plan = {"derived_features": [
        {"op": "zscore", "name": "z_a", "inputs": ["a"]},
        {"op": "log",    "name": "l_b", "inputs": ["b"]},      # PORTLANMAMIŞ
    ]}
    ctx = assemble_pipeline_ctx(
        intent="classification", modality="tabular", aggregated=False, target_name="y",
        drop_columns=[], ratio_feature_meta={}, missing_indicator_meta={},
        derived_flag_meta={}, feature_plan=plan, encode_exclude=set(), outlier_cols=[],
        timeseries_lag=None, domain_feature_cols=[], date_feature_cols=[], column_renames=[])
    assert ctx["unported_ops"] == ["log"], "portlanmamış op beyan edilmeli"
    assert ctx["zscore_specs"] == [["z_a", "a"]], "portlu op spec'e çevrilmeli"
    assert check_pipeline_scope(ctx)["violations"], "portlanmamış op → kapsam ihlali"


def test_flag_metas_are_unfiltered_supersets():
    """T0 BORCU — RATIO DERSİNİN TEKRARI (kaynak-düzeyi çit). `missing_indicator_meta` ve
    `derived_flag_meta` X_train'e göre FİLTRELENMEZ (frozen seçim kaynağı); `auto_features`
    ve `derived_flags` ise hayatta-kalan listeleridir. Biri meta'ya filtre koyarsa bu test
    kırılır → kısayolun dayanağı sessizce çürümez."""
    import inspect as _inspect
    from src.services import feature_builder
    src = _inspect.getsource(feature_builder.build_features)

    for anahtar in ('"missing_indicator_meta"', '"derived_flag_meta"'):
        satir = next(l for l in src.splitlines() if anahtar in l and ":" in l)
        assert "X_train.columns" not in satir, \
            f"{anahtar} filtrelenmiş → frozen kaynağı artık güvenli DEĞİL"
    for anahtar in ('"auto_features"', '"derived_flags"'):
        satir = next(l for l in src.splitlines() if anahtar in l and ":" in l)
        assert "X_train.columns" in satir, f"{anahtar} hayatta-kalan listesi olmalı (kontrast)"


def test_flag_meta_superset_on_real_data():
    """DAVRANIŞSAL kanıt (kaynak-okuma değil): gerçek veride meta ⊋ hayatta-kalanlar.
    credit'te iki `_isna` kurulur, selection birini düşürür → filtreli liste kullanılsaydı
    o kolon frozen seçimden DÜŞERDİ ve ara çerçeve eski yoldan saparadı."""
    path = _find_csv("*credit_risk_dataset.csv")
    if not path:
        pytest.skip("credit CSV yok (gitignored)")
    from scripts.offline_report import build_state
    from src.services.feature_builder import build_features
    fr = build_features(build_state(path, "loan_status", "classification", False, None))["feature_report"]

    meta_kaynaklar = set(fr["missing_indicator_meta"])
    hayatta = {c for c in fr["auto_features"] if c.endswith("_isna")}
    assert meta_kaynaklar >= hayatta, "meta hayatta-kalanları KAPSAMALI"
    assert meta_kaynaklar - hayatta, \
        "bu datasette selection en az bir _isna düşürmeli (aksi halde test boş geçer)"


# ══════════ T1 — GÖLGE KOŞU (shadow): üretim çıktısına DOKUNAMAZ ══════════


def _shadow_ctx() -> dict:
    """Tam beyan edilmiş, temiz kapsam (deployable=True)."""
    return {"ratio_pairs": [], "encode_exclude": [], "outlier_cols": [], "intent": "classification",
            "drop_columns": [], "missing_indicator_cols": [], "zero_flag_cols": [],
            "modality": "tabular", "aggregated": False, "timeseries_lag": None,
            "domain_features": [], "date_features": [], "column_renames": [],
            "unported_ops": [], "deduplicated": True, "target_col": None}


def test_shadow_reports_identical_when_paths_agree():
    fx, y, _ = _orchestrator_fixture()
    ctx = _shadow_ctx()
    beklenen, _ = fit_pipeline(fx.copy(), y, ctx)
    rapor = run_shadow_comparison(fx.copy(), y, ctx, beklenen)
    assert rapor["verdict"] == "identical", rapor
    assert rapor["deployable"] is True


def test_shadow_never_raises_when_fit_explodes(monkeypatch):
    """MUTLAK KURAL: gölge koşu üretimi BOZAMAZ. fit_pipeline patlasa bile fırlatmaz."""
    import src.services.inference.shadow as sh
    fx, y, _ = _orchestrator_fixture()

    def _patla(*a, **k):
        raise ValueError("kasitli patlama")
    monkeypatch.setattr("src.services.inference.orchestrator.fit_pipeline", _patla)

    rapor = sh.run_shadow_comparison(fx.copy(), y, _shadow_ctx(), fx.copy())
    assert rapor["verdict"] == "shadow_failed"
    assert "kasitli patlama" in rapor["detail"]


def test_shadow_reports_scope_rejection_as_finding_not_crash():
    """Kapsam reddi gölgede HATA değil BULGUDUR — gölgenin asıl amacı bunu görmek."""
    fx, y, _ = _orchestrator_fixture()
    rapor = run_shadow_comparison(fx.copy(), y, {**_shadow_ctx(), "modality": "timeseries"},
                                  fx.copy())
    assert rapor["verdict"] == "scope_rejected"
    assert rapor["scope"]["deployable"] is False
    assert any(v["fact"] == "modality" for v in rapor["scope"]["violations"])


def test_shadow_detects_column_and_value_divergence():
    """Sapma sessiz kalmamalı — iki tür ayrı verdict'le raporlanır."""
    fx, y, _ = _orchestrator_fixture()
    ctx = _shadow_ctx()
    beklenen, _ = fit_pipeline(fx.copy(), y, ctx)

    kolon_sapmasi = run_shadow_comparison(fx.copy(), y, ctx,
                                          beklenen.assign(fazladan_kolon=0.0))
    assert kolon_sapmasi["verdict"] == "column_divergence"
    assert "fazladan_kolon" in kolon_sapmasi["missing"]

    bozuk = beklenen.copy()
    bozuk[beklenen.columns[0]] = bozuk[beklenen.columns[0]] + 1.0
    deger_sapmasi = run_shadow_comparison(fx.copy(), y, ctx, bozuk)
    assert deger_sapmasi["verdict"] == "value_divergence"
    assert deger_sapmasi["n_diverging"] >= 1


def test_shadow_is_off_by_default():
    """Varsayılan `legacy` — gölge maliyeti SIFIR, feature_report'ta boş kalır."""
    from src.core.config import settings
    assert settings.FEATURE_PIPELINE_MODE == "legacy", \
        "varsayılan legacy OLMALI (üretim davranışı flag'siz değişmez)"


def test_zscore_step_is_inert_in_shadow():
    """DÜRÜSTLÜK KİLİDİ: gölgede zscore'un 'identical' çıkması #5'i DOĞRULAMAZ.
    Gölge çerçevesinde z kolonu eski yol tarafından (full-df, leaky) ZATEN üretilmiştir;
    `ZScoreStep.fit` `name in X_train.columns` görüp ATLAR → adım no-op → eşitlik bedava.
    #5'in gerçek kanıtı deterministik sızıntı-delta testidir, gölge DEĞİL."""
    fx = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "z_a": [9.0, 9.0, 9.0, 9.0]})
    params = ZScoreStep(config={"specs": [["z_a", "a"]]}).fit(fx)
    assert params == {}, "çıktı kolonu zaten varsa adım ATLAMALI (gölgede inert)"
    params2 = ZScoreStep(config={"specs": [["z_a", "a"]]}).fit(fx.drop(columns=["z_a"]))
    assert "z_a" in params2, "kolon yoksa adım fiilen çalışmalı (T4'teki hâli)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
