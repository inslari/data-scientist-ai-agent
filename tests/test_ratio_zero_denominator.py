"""Regresyon testi: oran (ratio) motorunda sifir-payda sentinel'inin GERI DONMESINI yakala.

Kok bug (FAZ 0): `r.clip(lo,hi).fillna(0.0)` — payda-sifir (inf→NaN) ve kaynak-missing
satirlarini, gercek dagilimin ALTINDA (0.0 < q01) benzersiz bir sentinel'e cokurtuyordu.
Bu, mevcut is_zero_<payda> flag'ini gizlice yeniden kodluyor (tenure=0 proxy'si).

Duzeltme (Secenek B + 3 ekleme):
  #1 winsor sinirlari + impute median SPLIT SONRASI train'e fit → test'e transform (sizinti yok)
  #2 sifir-payda kolonlarinin is_zero flag'i prevalence+cap'ten muaf (invariant)
  #3 payda-sifir ∪ kaynak-missing ikisi de dagilim-ici median ile dolar (0.0 sentinel yok)

TAMAMEN DINAMIK: uydurma cok-kelimeli kolon adlariyla calisir; churn'e ozel yama YOK.
Standalone script (repo konvansiyonu): `uv run python tests/test_ratio_zero_denominator.py`
"""
import sys
import numpy as np
import pandas as pd

from src.services.feature_builder import (
    _build_ratio_features,
    _build_zero_missing_flags,
    _finalize_ratio_features_tt,
    _RATIO_WINSOR_Q,
    _FLAG_MIN_PREVALENCE,
)

RNG = np.random.default_rng(42)
_fail = []


def check(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        _fail.append(msg)


def _make_df(n=3000):
    """Uydurma, cok-kelimeli paydali kolonlarla sentetik veri (dinamiklik kaniti)."""
    df = pd.DataFrame({
        # numerator (ad-sirali 'a' → pay): 'active_score' < 'monthly_active_users'
        "active_score":         RNG.uniform(1.0, 100.0, n),
        # DENOMINATOR #1 — cok kelimeli, ~%4 sifir (prevalence gate ICINDE → normal flag)
        "monthly_active_users": RNG.integers(1, 50, n).astype(float),
        # DENOMINATOR #2 — ~%0.5 sifir (prevalence gate ALTINDA → invariant force gerekli)
        "rare_zero_divisor":    RNG.uniform(5.0, 20.0, n),
    })
    # ~%4 sifir enjekte (monthly_active_users) — zero_frac ratio guard'i (0.05) icinde
    zi1 = RNG.choice(n, size=int(0.04 * n), replace=False)
    df.loc[zi1, "monthly_active_users"] = 0.0
    # ~%0.5 sifir enjekte (rare_zero_divisor) — _FLAG_MIN_PREVALENCE (0.01) ALTINDA
    zi2 = RNG.choice(n, size=max(1, int(0.005 * n)), replace=False)
    df.loc[zi2, "rare_zero_divisor"] = 0.0
    return df


def test_sentinel_gone_and_in_distribution():
    print("\n[1] Sentinel yok + impute dagilim-ici (payda-sifir)")
    df = _make_df()
    orig = set(df.columns)
    df, ratio_cols, denom_zero, _ = _build_ratio_features(
        df, orig, None, "classification", False, set())
    target = "ratio_active_score_monthly_active_users"
    check(target in ratio_cols, f"cok-kelimeli paydali oran uretildi: {target}")
    check("monthly_active_users" in denom_zero, "payda sifir-payda setinde bildirildi")

    # Split → train'e fit, test'e transform (Addition #1: sizinti yok)
    n = len(df)
    idx = RNG.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    Xtr, Xte = df.iloc[tr].copy(), df.iloc[te].copy()
    Xtr, Xte = _finalize_ratio_features_tt(Xtr, Xte, ratio_cols)

    finite_tr = pd.to_numeric(df.iloc[tr]["active_score"]) / pd.to_numeric(df.iloc[tr]["monthly_active_users"])
    finite_tr = finite_tr[np.isfinite(finite_tr)]
    lo = float(finite_tr.quantile(_RATIO_WINSOR_Q))
    hi = float(finite_tr.quantile(1 - _RATIO_WINSOR_Q))
    med = float(finite_tr.clip(lo, hi).median())

    col_tr = Xtr[target]
    zero_denom_tr = (pd.to_numeric(df.iloc[tr]["monthly_active_users"]) == 0).to_numpy()
    fill_val = col_tr.to_numpy()[zero_denom_tr]

    # Kabul: min dagilim ICINDE (eski bug'da 0.0 < lo idi)
    check(col_tr.min() >= lo - 1e-9,
          f"final min ({col_tr.min():.4f}) >= train q01 ({lo:.4f}) — out-of-range sentinel YOK")
    check(np.allclose(fill_val, med, atol=1e-6),
          f"payda-sifir satirlari train-median'a ({med:.4f}) impute edildi")
    # AYIRT EDICI degil: median interior → tek esikle sifir-payda izole edilemez
    fin_mask = ~zero_denom_tr
    fin_vals = col_tr.to_numpy()[fin_mask]
    interior = (fin_vals < med - 1e-9).any() and (fin_vals > med + 1e-9).any()
    check(interior, "impute degeri sonlu-payda dagiliminin ICINDE (hem alt hem ust komsu var) → ayirt edici sentinel yok")


def test_source_missing_covered():
    print("\n[3] Kaynak-missing de median'a doluyor (0.0 sentinel degil)")
    df = _make_df()
    # numerator'e NaN enjekte (payda sifir DEGIL ama kaynak missing) — churn'de %0 idi, gizli bug
    miss = RNG.choice(len(df), size=int(0.03 * len(df)), replace=False)
    df.loc[miss, "active_score"] = np.nan
    orig = set(df.columns)
    df, ratio_cols, _, _ = _build_ratio_features(df, orig, None, "classification", False, set())
    target = "ratio_active_score_monthly_active_users"
    Xtr, Xte = df.copy(), df.copy()
    Xtr, Xte = _finalize_ratio_features_tt(Xtr, Xte, ratio_cols)
    finite = (pd.to_numeric(df["active_score"]) / pd.to_numeric(df["monthly_active_users"]))
    finite = finite[np.isfinite(finite)]
    lo = float(finite.quantile(_RATIO_WINSOR_Q))
    src_missing = pd.to_numeric(df["active_score"]).isna().to_numpy()
    vals = Xtr[target].to_numpy()[src_missing]
    check(len(vals) > 0 and (vals >= lo - 1e-9).all(),
          f"kaynak-missing satirlari dagilim-ici deger aldi (min {vals.min():.4f} >= q01 {lo:.4f}), 0.0 sentinel yok")


def test_is_zero_invariant():
    print("\n[2] is_zero invariant: prevalence-alti payda flag'i force uretiliyor")
    df = _make_df()
    orig = set(df.columns)
    df, ratio_cols, denom_zero, _ = _build_ratio_features(
        df, orig, None, "classification", False, set())
    # rare_zero_divisor: %0.5 sifir → normal prevalence gate ALTINDA
    p = float((pd.to_numeric(df["rare_zero_divisor"]) == 0).mean())
    check(p < _FLAG_MIN_PREVALENCE, f"rare_zero_divisor prevalence ({p:.4f}) < gate ({_FLAG_MIN_PREVALENCE}) — normal olsa flag DUSER")
    check("rare_zero_divisor" in denom_zero, "rare_zero_divisor sifir-payda seti icinde")

    # force OLMADAN: flag uretilmemeli (bug'in eski davranisi — bilgi silinir)
    df_a = df.copy()
    df_a, added_no_force = _build_zero_missing_flags(df_a, orig, "classification", False)
    check("is_zero_rare_zero_divisor" not in added_no_force,
          "force olmadan prevalence-alti flag URETILMEZ (bilgi silinme riski dogrulandi)")

    # force ILE: invariant → flag garanti
    df_b = df.copy()
    df_b, added_force = _build_zero_missing_flags(
        df_b, orig, "classification", False, force_zero_cols=denom_zero)
    check("is_zero_rare_zero_divisor" in added_force,
          "force ILE prevalence-alti payda flag'i GARANTI uretildi (invariant)")
    check("is_zero_monthly_active_users" in added_force,
          "prevalence-ici payda flag'i de mevcut (invariant kapsar)")


def test_train_fit_no_leakage():
    print("\n[1b] Winsor/impute train'e fit → test outlier'i train sinirina klipslenir (sizinti yok)")
    df = _make_df(2000)
    orig = set(df.columns)
    df, ratio_cols, _, _ = _build_ratio_features(df, orig, None, "classification", False, set())
    target = "ratio_active_score_monthly_active_users"
    n = len(df)
    tr = np.arange(0, int(0.8 * n))
    te = np.arange(int(0.8 * n), n)
    Xtr, Xte = df.iloc[tr].copy(), df.iloc[te].copy()
    # Test'e ASIRI outlier enjekte (train'de yok) — sizinti olsaydi test kendi hi'sine klipslerdi
    Xte.iloc[0, Xte.columns.get_loc(target)] = 1e9
    Xtr, Xte = _finalize_ratio_features_tt(Xtr, Xte, ratio_cols)
    finite_tr = (pd.to_numeric(df.iloc[tr]["active_score"]) / pd.to_numeric(df.iloc[tr]["monthly_active_users"]))
    finite_tr = finite_tr[np.isfinite(finite_tr)]
    hi_tr = float(finite_tr.quantile(1 - _RATIO_WINSOR_Q))
    check(Xte[target].max() <= hi_tr + 1e-6,
          f"test max ({Xte[target].max():.4f}) <= TRAIN q99 ({hi_tr:.4f}) — sinir train'den, test'ten sizmiyor")


if __name__ == "__main__":
    test_sentinel_gone_and_in_distribution()
    test_source_missing_covered()
    test_is_zero_invariant()
    test_train_fit_no_leakage()
    print("\n" + ("=" * 60))
    if _fail:
        print(f"FAIL ({len(_fail)}):")
        for m in _fail:
            print("  - " + m)
        sys.exit(1)
    print("TÜM TESTLER GEÇTİ ✓")
