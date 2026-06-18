"""
FAZ 4 doğrulama — regresyon target-lock'unun evaluation + quality_gate'e bağlanması.

Çalıştır:  uv run python test_phase4.py

Kontroller:
  1. Regresyon (temiz veri)   → target_lock_leakage boş/low, quality_gate çökmüyor,
                                 "high" target-lock WARN'ı YOK.
  2. Regresyon (sentetik sızıntı) → _evaluate_regression target_lock_leakage'i dolduruyor,
                                     quality_gate "high" target-lock WARN üretiyor ve mesaj
                                     regresyona özgü ("varyansı çöküyor" / "dar bir bant").
  3. Classification → quality_gate target-lock mesajları FAZ 1 öncesiyle BİREBİR aynı.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.services.evaluator import _evaluate_regression
from src.services.quality_gate import evaluate_quality_gate

warnings.filterwarnings("ignore")


def _split(X: pd.DataFrame, y: np.ndarray, frac: float = 0.7):
    n = len(y)
    k = int(n * frac)
    return X.iloc[:k], X.iloc[k:], y[:k], y[k:]


def test_regression_clean():
    print("=== 1) Regresyon (temiz veri) ===")
    rng = np.random.default_rng(42)
    n = 2000
    X = pd.DataFrame({
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
    })
    # Hedef feature'larla zayıf/orta ilişkili, hiçbir dilimde çökmüyor
    y = (3.0 * X["f1"].to_numpy() + 1.5 * X["f2"].to_numpy()
         + rng.standard_normal(n) * 2.0) * 10000

    X_tr, X_te, y_tr, y_te = _split(X, y)
    model = RandomForestRegressor(n_estimators=60, random_state=42).fit(X_tr, y_tr)

    evaluation = _evaluate_regression(model, X_tr, y_tr, X_te, y_te, list(X.columns))
    locks = evaluation.get("target_lock_leakage", [])
    print(f"  target_lock_leakage: {len(locks)} kayıt (anahtar mevcut: "
          f"{'target_lock_leakage' in evaluation})")
    high = [c for c in locks if c.get("confidence") == "high"]

    gate = evaluate_quality_gate({"evaluation_report": evaluation})
    codes = [w["code"] for w in gate["warnings"]]
    print(f"  quality_gate status={gate['status']} codes={codes}")

    assert "target_lock_leakage_high" not in codes, "Temiz veride high target-lock WARN olmamalı!"
    print("  ✓ Temiz veride high target-lock WARN yok, quality_gate çökmedi.\n")


def test_regression_leakage():
    print("=== 2) Regresyon (sentetik sızıntı — tek dilimde varyans çöküşü) ===")
    rng = np.random.default_rng(42)
    n = 2000
    f = rng.random(n)                      # uniform → qcut deciles temiz
    y = rng.standard_normal(n) * 100000    # f'ten bağımsız
    # Tek decile'da hedefi neredeyse sabitle → komşulardan ANİ kopuş → high
    mask = (f >= 0.4) & (f < 0.5)
    y[mask] = 50000 + rng.standard_normal(mask.sum()) * 5.0

    X = pd.DataFrame({"f_leak": f})
    X_tr, X_te, y_tr, y_te = _split(X, y)
    model = RandomForestRegressor(n_estimators=60, random_state=42).fit(X_tr, y_tr)

    evaluation = _evaluate_regression(model, X_tr, y_tr, X_te, y_te, list(X.columns))
    locks = evaluation.get("target_lock_leakage", [])
    print(f"  target_lock_leakage: {len(locks)} kayıt")
    for c in locks:
        print(f"    [{c['confidence'].upper()}] {c['feature']} purity={c['purity']}")
    assert locks, "Sentetik sızıntıda target_lock_leakage boş olmamalı!"

    gate = evaluate_quality_gate({"evaluation_report": evaluation})
    codes = [w["code"] for w in gate["warnings"]]
    print(f"  quality_gate status={gate['status']} codes={codes}")
    high_warns = [w for w in gate["warnings"] if w["code"] == "target_lock_leakage_high"]
    assert high_warns, "Sentetik sızıntıda 'target_lock_leakage_high' WARN bekleniyordu!"

    msg = high_warns[0]["message"]
    print(f"  WARN mesajı: {msg}")
    assert ("varyansı çöküyor" in msg) or ("dar bir bant" in msg), \
        "Regresyon high WARN mesajı regresyona özgü dilde olmalı!"
    print("  ✓ Sentetik sızıntı regresyon raporunda high WARN olarak göründü.\n")


# FAZ 1 öncesi classification mesajları (birebir korunmalı)
_EXPECTED_CLS_HIGH = (
    "Olası veri sızıntısı: feat_x. "
    "Bu özellikler belirli aralıklarda sonucu neredeyse kesin belirliyor — "
    "yani model 'tahmin' etmek yerine cevabı doğrudan görüyor olabilir. "
    "Gerçek kullanımda bu bilgi elde olmayabilir; bu özelliklerin çıkarılması önerilir."
)
_EXPECTED_CLS_LOW = (
    "Güçlü belirleyici özellik: feat_y. "
    "Sonuçla güçlü ama tutarlı bir ilişkisi var — bu genelde normaldir, "
    "sızıntı değil gerçek bir sinyaldir. Yine de gerçek kullanımda bu bilginin "
    "elde olacağından emin olun."
)


def test_classification_unchanged():
    print("=== 3) Classification mesajları değişmedi mi? ===")
    report = {
        "task": "binary_classification",
        "test_metrics": {"roc_auc": 0.9, "f1_weighted": 0.85},
        "target_lock_leakage": [
            {"feature": "feat_x", "confidence": "high", "importance": 0.6,
             "purity": 0.99, "reason": "..."},
            {"feature": "feat_y", "confidence": "low", "importance": 0.3,
             "purity": 0.98, "reason": "..."},
        ],
    }
    gate = evaluate_quality_gate({"evaluation_report": report})
    by_code = {w["code"]: w["message"] for w in gate["warnings"]}
    assert by_code.get("target_lock_leakage_high") == _EXPECTED_CLS_HIGH, \
        "Classification HIGH mesajı değişmiş!"
    assert by_code.get("target_lock_signal_low") == _EXPECTED_CLS_LOW, \
        "Classification LOW mesajı değişmiş!"
    print("  ✓ Classification HIGH/LOW mesajları birebir korundu.\n")


if __name__ == "__main__":
    test_regression_clean()
    test_regression_leakage()
    test_classification_unchanged()
    print("TÜM FAZ 4 KONTROLLERİ GEÇTİ ✓")
