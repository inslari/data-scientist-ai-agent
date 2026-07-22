"""Job 3 — payda guvenligi: sifiri GECEN payda ile oran turetilmez (offline, LLM/DB yok).

Neden: a/b'nin bozuklugu "payda kucuk" degil, paydanin veri araliginda SIFIRI GECMESI. Isaret
degistiginde b=+e ve b=-e olan iki neredeyse-ozdes satir zit sonsuzluklara firlar; winsorize
kuyrugu keser ama SIRALAMAYI onaramaz. Tek-tarafli kucuk payda ise agir kuyruklu ama MONOTON —
winsorize onu gercekten onarir.

Kural SADECE bu matematiksel bozuklugu hedefler; "bu feature ise yaramaz" yargisi ICERMEZ.
Bu dosyanin asil isi motorun URETIM GUCUNUN korundugunu kanitlamak: kredi ve e-ticaret fraud'un
mevcut ratio_* feature'larinin HEPSI aynen uretilmeye devam etmeli ve importance siralarini
korumali. Olculdu (4 veri seti, 48 aday kolon): kredi/churn/e-ticaret paydalari 0.000 kutup
kutlesi; ULB'nin V1-V28'i 0.380-0.498 — 0.38 genisliginde bosluk, sifir yanlis siniflama.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.services import feature_builder as fb
from src.services.feature_builder import _build_ratio_features

CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "csvs")
CREDIT = os.path.join(CSV_DIR, "1b097ea4-bcc9-432f-ad92-d1ce02d62a52_credit_risk_dataset.csv")
ECOMM = os.path.join(CSV_DIR, "2eb2d968-2193-4bd4-969c-782072f8198c_transactions.csv")
ULB = os.path.join(CSV_DIR, "a93ec4b3-4f11-4ff2-b012-d55405f46297_creditcard.csv")


def _ratios(df: pd.DataFrame, drop: list[str], gate: bool = True) -> list[str]:
    """Motorun bu df icin urettigi ratio_* kolonlari. gate=False → kural KAPALI (eski davranis)."""
    df = df.drop(columns=drop, errors="ignore").copy()
    original = set(df.columns)
    limit = fb._RATIO_DENOM_MAX_SIGN_MASS if gate else 1.0
    with patch.object(fb, "_RATIO_DENOM_MAX_SIGN_MASS", limit):
        _, added, _, _ = _build_ratio_features(df, original, None, "classification",
                                            aggregated=False, leaky_cols_high=set())
    return added


class SignCrossingDenominatorTest(unittest.TestCase):
    """Sentetik: kuralin tam olarak neyi eledigi / elemedigi."""

    def setUp(self):
        rng = np.random.default_rng(42)
        n = 5000
        self.df = pd.DataFrame({
            "a_signal": rng.normal(100, 20, n),          # pay
            "b_centered": rng.normal(0, 1, n),           # SIFIRI GECIYOR → payda olamaz
            "c_positive": rng.gamma(2.0, 50.0, n) + 1,   # tek-tarafli, sifira yaklasiyor → payda OLABILIR
        })

    def test_zero_crossing_denominator_is_skipped(self):
        added = _ratios(self.df, [])
        self.assertNotIn("ratio_a_signal_b_centered", added)

    def test_positive_denominator_with_small_values_survives(self):
        """Tek-tarafli agir kuyruk bozukluk DEGIL — winsorize onariyor. Kural buna dokunmamali."""
        added = _ratios(self.df, [])
        self.assertIn("ratio_a_signal_c_positive", added)
        self.assertIn("ratio_b_centered_c_positive", added)

    def test_zero_crossing_column_is_still_free_as_a_NUMERATOR(self):
        """Yargi yok: sifiri gecen kolon yasakli degil, yalniz PAYDA olamaz."""
        added = _ratios(self.df, [])
        self.assertTrue(any(c.startswith("ratio_b_centered_") for c in added),
                        f"b_centered pay olarak da elenmis: {added}")

    def test_single_stray_negative_does_not_kill_the_feature(self):
        """Kutle esigin ALTINDA (%0.02) → winsorizer zaten yutuyor, feature yasamali."""
        df = self.df.copy()
        df.loc[0, "c_positive"] = -3.0                    # 1/5000 = %0.02 < %1
        added = _ratios(df, [])
        self.assertIn("ratio_a_signal_c_positive", added)

    def test_threshold_is_derived_from_the_winsor_quantile_not_a_magic_number(self):
        self.assertEqual(fb._RATIO_DENOM_MAX_SIGN_MASS, fb._RATIO_WINSOR_Q)

    def test_sign_mass_just_above_threshold_is_skipped(self):
        """Esik civari: %2 negatif kutle → winsorizer soguramaz → elenmeli."""
        df = self.df.copy()
        df.loc[: int(0.02 * len(df)), "c_positive"] = -3.0
        self.assertNotIn("ratio_a_signal_c_positive", _ratios(df, []))


@unittest.skipUnless(os.path.exists(CREDIT) and os.path.exists(ECOMM),
                     "referans veri setleri yok")
class ProductionPowerPreservedTest(unittest.TestCase):
    """ASIL SART: motorun uretim gucu korunuyor — mevcut ratio_*'lar birebir ayni."""

    def test_credit_ratio_list_is_bit_for_bit_identical(self):
        df = pd.read_csv(CREDIT)
        before = _ratios(df, ["loan_status"], gate=False)
        after = _ratios(df, ["loan_status"], gate=True)
        self.assertEqual(before, after, "kredi oranlari degisti — uretim gucu kaybi")
        self.assertEqual(len(after), 16, f"kredi 16 oran uretmeliydi, {len(after)} uretti")

    def test_ecommerce_fraud_ratio_list_is_bit_for_bit_identical(self):
        df = pd.read_csv(ECOMM)
        drop = ["is_fraud", "transaction_id", "user_id"]
        before = _ratios(df, drop, gate=False)
        after = _ratios(df, drop, gate=True)
        self.assertEqual(before, after, "e-ticaret oranlari degisti — uretim gucu kaybi")
        self.assertEqual(len(after), 10, f"e-ticaret 10 oran uretmeliydi, {len(after)} uretti")

    def test_credit_ratio_importance_ranking_is_unchanged(self):
        """Feature'lar ayni degil sadece — modeldeki AGIRLIKLARI da ayni yerde duruyor."""
        from sklearn.ensemble import RandomForestClassifier

        raw = pd.read_csv(CREDIT)
        y = raw["loan_status"]

        def _rank(gate: bool) -> list[str]:
            df = raw.drop(columns=["loan_status"]).copy()
            original = set(df.columns)
            limit = fb._RATIO_DENOM_MAX_SIGN_MASS if gate else 1.0
            with patch.object(fb, "_RATIO_DENOM_MAX_SIGN_MASS", limit):
                out, added, _, _ = _build_ratio_features(df, original, None, "classification",
                                                      aggregated=False, leaky_cols_high=set())
            X = out.select_dtypes(include="number").fillna(0.0)
            m = RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=-1).fit(X, y)
            imp = pd.Series(m.feature_importances_, index=X.columns)
            return [c for c in imp.sort_values(ascending=False).index if c in added]

        self.assertEqual(_rank(gate=False), _rank(gate=True),
                         "kredi ratio importance sirasi kaydi")


@unittest.skipUnless(os.path.exists(ULB), "ULB veri seti yok")
class PolePoisonedRatiosAreRemovedTest(unittest.TestCase):
    """ULB (PCA V1-V28): paydasi sifiri gecen oranlar gitmeli."""

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(ULB, nrows=60000)

    def test_pca_component_denominators_are_all_eliminated(self):
        after = _ratios(self.df, ["Class"])
        poles = [c for c in after if c.rsplit("_", 1)[-1].startswith("V")]
        self.assertEqual(poles, [], f"kutup paydali oranlar hala uretiliyor: {poles}")

    def test_the_one_mathematically_sound_ratio_survives(self):
        """Time/Amount sifiri gecmiyor → ratio_Amount_Time mesru, kalmali."""
        self.assertIn("ratio_Amount_Time", _ratios(self.df, ["Class"]))

    def test_gate_off_reproduces_the_old_broken_behaviour(self):
        """Kural olmadan motor gercekten kutup uretiyordu — regresyon degil, duzeltme."""
        before = _ratios(self.df, ["Class"], gate=False)
        self.assertTrue(any(c.rsplit("_", 1)[-1].startswith("V") for c in before),
                        "eski davranis kutup uretmiyorsa bu testin dayanagi yok")
        self.assertGreater(len(before), len(_ratios(self.df, ["Class"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
