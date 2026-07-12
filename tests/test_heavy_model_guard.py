"""Büyük-N model guard testleri (standalone — repo konvansiyonu).

Çalıştır:  uv run python tests/test_heavy_model_guard.py

Kapsam:
  1. Eşik-ALTI no-op: credit(26k)/churn(8k)/fraud@50k(40k)/fraud@100k(80k) → roster
     BİREBİR değişmez (aynı anahtarlar, RF/ET n_estimators=200, adaboost duruyor),
     applied=False. Referans ROC'lar (0.9478/0.866/0.971) kayamaz.
  2. Eşik-ÜSTÜ (240k): adaboost DÜŞER, RF/ET n_estimators→100, boosting trio+logreg
     KALIR, roster boş değil, note.applied=True + dropped/reduced doğru.
  3. Güvenlik: roster {adaboost} + büyük-N → boşaltılmaz (no-op).
  4. Optuna cap: _rfet_n_estimators_high büyük-N'de RF/ET→100, boosting→300.
  5. TR/EN not render.
  6. Boosting/logreg n_estimators eşik-üstünde de dokunulmaz.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["API_DATABASE_URL"] = f"sqlite:///{os.path.join(tempfile.gettempdir(), 'hmg_appdb.sqlite')}"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.ml_trainer import _apply_heavy_model_guard, _rfet_n_estimators_high  # noqa: E402
from src.services.ml_trainer import _get_classification_models_balanced  # noqa: E402
from src.core.config import settings  # noqa: E402
from src.core.report_i18n import R  # noqa: E402

_FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        _FAILS.append(f"{name} :: {detail}")


def _roster():
    # Fraud yolundaki dengesiz roster: RF, ET, AdaBoost, LogReg, XGB, LGBM, (CatBoost)
    return _get_classification_models_balanced({0: 1.0, 1: 20.0})


def _n_est(model):
    try:
        return model.get_params().get("n_estimators")
    except Exception:
        return None


def test_below_threshold_noop():
    print("\n[1] Eşik-ALTI no-op — referans veri setleri roster'ı DEĞİŞMEZ")
    print(f"    (default HEAVY_MODEL_MAX_ROWS={settings.HEAVY_MODEL_MAX_ROWS:,})")
    for label, n_train in [("churn~8k", 8_000), ("credit~26k", 26_000),
                           ("fraud@50k~40k", 40_000), ("fraud@100k~80k", 80_000)]:
        roster = _roster()
        keys_before = list(roster.keys())
        rf_before = _n_est(roster["random_forest"])
        out, note = _apply_heavy_model_guard(roster, n_train)
        check(f"{label}: applied=False", note["applied"] is False, str(note))
        check(f"{label}: anahtarlar birebir", list(out.keys()) == keys_before,
              f"{list(out.keys())} vs {keys_before}")
        check(f"{label}: adaboost duruyor", "adaboost" in out, str(list(out.keys())))
        check(f"{label}: RF n_estimators korundu ({rf_before})",
              _n_est(out["random_forest"]) == rf_before, str(_n_est(out["random_forest"])))


def test_above_threshold_fires():
    print("\n[2] Eşik-ÜSTÜ (240k) — ağır modeller kısılır")
    roster = _roster()
    out, note = _apply_heavy_model_guard(roster, 240_000)
    check("applied=True", note["applied"] is True, str(note))
    check("adaboost DÜŞTÜ", "adaboost" not in out, str(list(out.keys())))
    check("dropped=['adaboost']", note["dropped"] == ["adaboost"], str(note["dropped"]))
    check("RF n_estimators=100", _n_est(out.get("random_forest")) == settings.HEAVY_MODEL_TREES,
          str(_n_est(out.get("random_forest"))))
    check("ET n_estimators=100", _n_est(out.get("extra_trees")) == settings.HEAVY_MODEL_TREES,
          str(_n_est(out.get("extra_trees"))))
    check("RF/ET reduced kaydı var", set(note["reduced"]) >= {"random_forest", "extra_trees"},
          str(note["reduced"]))
    # Boosting trio + logreg KALIR ve dokunulmaz
    for m in ("xgboost", "lightgbm", "logistic_regression"):
        check(f"{m} roster'da kaldı", m in out, str(list(out.keys())))
    check("xgboost n_estimators dokunulmadı (200)", _n_est(out.get("xgboost")) == 200,
          str(_n_est(out.get("xgboost"))))
    check("roster boş değil", len(out) > 0, str(len(out)))


def test_never_empties():
    print("\n[3] Güvenlik — roster asla boşaltılmaz")
    roster = _roster()
    only_ada = {"adaboost": roster["adaboost"]}
    out, note = _apply_heavy_model_guard(only_ada, 240_000)
    check("sadece-adaboost + büyük-N → no-op (boşaltmaz)",
          "adaboost" in out and note["applied"] is False, str(note))


def test_optuna_cap():
    print("\n[4] Optuna n_estimators cap")
    thr = settings.HEAVY_MODEL_MAX_ROWS
    check("RF büyük-N → HEAVY_MODEL_TREES",
          _rfet_n_estimators_high("random_forest", thr + 1) == settings.HEAVY_MODEL_TREES)
    check("ET büyük-N → HEAVY_MODEL_TREES",
          _rfet_n_estimators_high("extra_trees", thr + 1) == settings.HEAVY_MODEL_TREES)
    check("RF eşik-altı → 300", _rfet_n_estimators_high("random_forest", thr - 1) == 300)
    check("xgboost → hiç kısılmaz (300)", _rfet_n_estimators_high("xgboost", thr + 1) == 300)


def test_tr_en_note():
    print("\n[5] TR/EN not render")
    for k in ("heavy_model_guard_note", "heavy_guard_dropped", "heavy_guard_reduced"):
        tr, en = R(k, "tr"), R(k, "en")
        check(f"{k}: tr!=key & en!=key", tr != k and en != k, f"{tr!r}/{en!r}")
        check(f"{k}: tr!=en", tr != en)
    msg = R("heavy_model_guard_note", "en", n=240000, thr=100000, detail="adaboost skipped")
    check("param format çalışıyor", "240,000" in msg and "100,000" in msg, msg)


def main():
    print("=" * 70)
    print("BÜYÜK-N MODEL GUARD — TEST")
    print("=" * 70)
    test_below_threshold_noop()
    test_above_threshold_fires()
    test_never_empties()
    test_optuna_cap()
    test_tr_en_note()

    print("\n" + "=" * 70)
    if _FAILS:
        print(f"SONUÇ: {len(_FAILS)} BAŞARISIZ")
        for f in _FAILS:
            print("  ✗ " + f)
        sys.exit(1)
    print("SONUÇ: TÜM TESTLER GEÇTİ ✓")


if __name__ == "__main__":
    main()
