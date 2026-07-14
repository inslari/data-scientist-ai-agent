"""Job 1 — ince test setinde overfit verdict'i susturulur (offline, LLM/DB yok).

Wilson disiplininin aynisi: ORNEKLEMIN DESTEKLEMEDIGI IDDIADA BULUNMA. ULB'de test setinde
yalniz 34 pozitif var; PR-AUC'un bootstrap sd'si ~0.07. Gozlenen gap (0.0899) bu gurultunun
ICINDE kaliyor — "overfitting_detected" deyip regularization onermek, olcemedigimiz bir seyi
iddia etmektir.

Sinama: CV skoru, test skorunun bootstrap %95 GA'sinin ICINDE mi? Icindeyse fark sifirdan
ayirt edilemiyor → status="inconclusive", regularization YOK, critic retry YOK, quality_gate
sessiz. Sabit "min pozitif" esigi yok: ince test seti zaten GA'yi genisletip kurali tetikler.

Iki yonlu: GERCEK overfitting (dar GA, CV disarda) hala yakalanmali — kural bir susturma
mekanizmasi degil, bir ANLAMLILIK sinamasi.
"""

import unittest

from src.services.critic_advisor import _diagnose_overfitting
from src.services.evaluator import _analyze_overfitting
from src.services.quality_gate import evaluate_quality_gate

# ULB gercek rakamlari (Job 2 kosusu): CV-PR 0.8584, test-PR 0.7685, 34 pozitif.
ULB_TRAIN = {"roc_auc": 0.9999, "pr_auc": 0.9994, "accuracy": 0.9999}
ULB_TEST = {"roc_auc": 0.9247, "pr_auc": 0.7685, "accuracy": 0.9995}
ULB_CI = {"roc_auc": {"lower": 0.8449, "upper": 0.9712},
          "pr_auc": {"lower": 0.6103, "upper": 0.8977}}   # genis: 34 pozitif


class ThinTestSetIsInconclusiveTest(unittest.TestCase):

    def _ulb(self) -> dict:
        return _analyze_overfitting(ULB_TRAIN, ULB_TEST, cv_score=0.8584, cv_metric="pr_auc",
                                    test_ci=ULB_CI, n_test_positives=34)

    def test_ulb_gap_is_declared_inconclusive_not_overfitting(self):
        out = self._ulb()
        self.assertAlmostEqual(out["gap"], 0.0899, places=3)   # gap hala RAPORLANIYOR
        self.assertEqual(out["status"], "inconclusive",
                         "34 pozitifle olculen gap overfitting IDDIASI'na donusmemeli")

    def test_regularization_is_not_recommended(self):
        desc = self._ulb()["description"]
        self.assertIn("önerilmez", desc)
        self.assertNotIn("Regularization artırılabilir", desc)
        self.assertIn("34", desc, "gerekce (pozitif sayisi) kullaniciya gosterilmeli")

    def test_the_uncertainty_evidence_is_exposed_not_just_the_verdict(self):
        out = self._ulb()
        self.assertEqual(out["ci_lower"], 0.6103)
        self.assertEqual(out["ci_upper"], 0.8977)
        self.assertEqual(out["n_test_positives"], 34)
        self.assertEqual(out["metric"], "pr_auc")   # gap HANGI metrikten — karismasin

    def test_critic_does_not_trigger_retry_or_regularization(self):
        self.assertIsNone(_diagnose_overfitting(self._ulb()))

    def test_quality_gate_stays_silent(self):
        state = {"evaluation_report": {"task": "classification", "test_metrics": ULB_TEST,
                                       "overfitting": self._ulb()}}
        gate = evaluate_quality_gate(state)
        codes = [c["code"] for c in gate.get("failures", []) + gate.get("warnings", [])]
        self.assertNotIn("severe_overfitting", codes)
        self.assertNotIn("mild_overfitting", codes)

    def test_ci_of_the_MATCHING_metric_is_used(self):
        """CV PR-AUC ile kostuysa PR-AUC'un GA'si okunmali. ROC'un dar GA'sina bakmak,
        Job 2'de kapattigimiz metrik-karistirma hatasinin tekrari olurdu."""
        narrow_pr = {"roc_auc": {"lower": 0.84, "upper": 0.97},
                     "pr_auc": {"lower": 0.74, "upper": 0.79}}   # CV 0.8584 DISARDA
        out = _analyze_overfitting(ULB_TRAIN, ULB_TEST, cv_score=0.8584, cv_metric="pr_auc",
                                   test_ci=narrow_pr, n_test_positives=34)
        self.assertEqual(out["status"], "overfitting_detected")


class RealOverfittingSurvivesTest(unittest.TestCase):
    """Kural bir SUSTURMA mekanizmasi degil — kalin test setinde gercek overfit hala yakalanir."""

    def test_wide_gap_with_narrow_ci_is_still_overfitting(self):
        out = _analyze_overfitting(
            {"roc_auc": 0.99}, {"roc_auc": 0.72}, cv_score=0.88, cv_metric="roc_auc",
            test_ci={"roc_auc": {"lower": 0.70, "upper": 0.74}},   # kalin test seti → dar GA
            n_test_positives=1500,
        )
        self.assertEqual(out["status"], "overfitting_detected")
        self.assertIsNotNone(_diagnose_overfitting(out))

    def test_credit_like_normal_run_is_unchanged(self):
        """Kredi referansi: gap negatif/kucuk → zaten `normal`, kural buna hic dokunmaz."""
        out = _analyze_overfitting(
            {"roc_auc": 0.9985}, {"roc_auc": 0.9468}, cv_score=0.9413, cv_metric="roc_auc",
            test_ci={"roc_auc": {"lower": 0.9350, "upper": 0.9580}}, n_test_positives=1421,
        )
        self.assertEqual(out["status"], "normal")
        self.assertLess(out["gap"], 0.0)

    def test_no_cv_means_no_suppression(self):
        """CV yoksa gap resubstitution farkidir; onun bootstrap GA'si YOK → susturma calismaz."""
        out = _analyze_overfitting({"roc_auc": 0.99}, {"roc_auc": 0.80}, cv_score=None,
                                   test_ci={"roc_auc": {"lower": 0.70, "upper": 0.99}})
        self.assertEqual(out["basis"], "train_vs_test")
        self.assertEqual(out["status"], "overfitting_detected")

    def test_missing_ci_falls_back_to_old_behaviour(self):
        out = _analyze_overfitting(ULB_TRAIN, ULB_TEST, cv_score=0.8584, cv_metric="pr_auc",
                                   test_ci=None, n_test_positives=34)
        self.assertEqual(out["status"], "overfitting_detected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
