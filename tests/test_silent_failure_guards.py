"""Sessiz basarisizlik yasagi — eksik metrik SESSIZCE 0 sayilmaz, kullanici BOS KART gormez.

Kaynak: Job 2'de bir `except Exception: pass`, fonksiyon-ici import'un yol actigi
UnboundLocalError'i yutuyordu; PR-AUC kodu DURUYORDU ama HIC CALISMIYORDU. Envanterde
34 ayni desen bulundu; kritik 8'i logger.warning'e cevrildi. Bu dosya, sadece log yeterli
olmayan IKI tanesinin DAVRANISINI kilitler:

  1) predict_proba dustu → ROC/PR-AUC/kalibrasyon/threshold hepsi yok. Rapor bunu ACIKCA
     soylemeli (`metrics_unavailable`), bos kart gostermemeli.
  2) test ROC yok → quality_gate eskiden `0` sayip haksiz "dusuk performans" FAIL'i veriyordu.
     Olcemedigimiz sey hakkinda hukum vermeyiz: kontrol atlanir, WARN ile GORUNUR kalir.
"""

import unittest

from src.core.report_i18n import R
from src.services.quality_gate import evaluate_quality_gate


def _gate(test_metrics: dict, lang: str = "tr") -> dict:
    return evaluate_quality_gate({
        "report_language": lang,
        "evaluation_report": {
            "task": "classification",
            "test_metrics": test_metrics,
            "overfitting": {"status": "normal", "gap": 0.01},
        },
    })


class MissingAucIsNotZeroTest(unittest.TestCase):

    def test_missing_roc_auc_does_not_produce_a_low_auc_failure(self):
        """ASIL BUG: metrik yoklugu `0` sanilip mukemmel model bile FAIL ediyordu."""
        gate = _gate({"accuracy": 0.97, "f1_weighted": 0.96})     # ROC YOK
        codes = [f["code"] for f in gate["failures"]]
        self.assertNotIn("low_auc", codes)

    def test_missing_roc_auc_is_surfaced_as_a_warning_not_swallowed(self):
        """Sessizce gecmek de yanlis — olculemedigi GORUNUR olmali."""
        gate = _gate({"accuracy": 0.97, "f1_weighted": 0.96})
        codes = [w["code"] for w in gate["warnings"]]
        self.assertIn("auc_unavailable", codes)

    def test_genuinely_low_auc_still_fails(self):
        """Kural bir susturma degil: ROC GERCEKTEN dusukse FAIL korunur."""
        codes = [f["code"] for f in _gate({"roc_auc": 0.52})["failures"]]
        self.assertIn("low_auc", codes)

    def test_good_auc_passes(self):
        gate = _gate({"roc_auc": 0.9478})     # kredi referansi
        self.assertNotIn("low_auc", [f["code"] for f in gate["failures"]])
        self.assertNotIn("auc_unavailable", [w["code"] for w in gate["warnings"]])

    def test_multiclass_ovr_key_is_honoured(self):
        gate = _gate({"roc_auc_ovr": 0.88})
        self.assertNotIn("auc_unavailable", [w["code"] for w in gate["warnings"]])


class MetricsUnavailableIsExplainedTest(unittest.TestCase):
    """Bos kart degil, acik neden — TR ve EN."""

    def test_message_exists_in_both_languages(self):
        for lang in ("tr", "en"):
            msg = R("eval_no_probabilities", lang)
            self.assertTrue(msg and len(msg) > 40, f"{lang} mesaji bos")
        self.assertIn("predict_proba", R("eval_no_probabilities", "tr"))
        self.assertIn("ROC-AUC", R("eval_no_probabilities", "en"))

    def test_report_prints_the_reason(self):
        """Eksik metrik BOŞ KART değil, açık neden.

        Bu güvence FALLBACK GÖVDESİNDEN katmanlı Model Performansı bölümüne taşındı
        (metrikler blob'dan çıkarken). Yer değişti, ilke değişmedi — üstelik güçlendi:
        bölüm LLM'den bağımsız üretildiği için hem normal hem fallback yolunda basılır,
        artık her etkilenen metriği tek tek sebebiyle sayar (eskiden tek satırdı).
        """
        from src.core.report_schema import render_section
        from src.services.report_sections import build_performance_section
        state = {
            "intent": "classification",
            "evaluation_report": {
                "test_metrics": {"accuracy": 0.97},
                "metrics_unavailable": {"reason": "predict_proba_failed",
                                        "detail": "AttributeError: no predict_proba",
                                        "affected": ["roc_auc", "pr_auc"]},
                "reliability_score": {"grade": "B", "score": 70},
            },
            "feature_report": {"n_test": 1000},
        }
        md = render_section(build_performance_section(state, "tr"), "tr")
        self.assertIn("predict_proba", md)
        self.assertIn("roc_auc", md)
        # Sessiz sıfır yasağı: yok olan metrik 0 diye de, iyi haber diye de sunulamaz
        self.assertNotIn("0.000", md)
        self.assertIn("iyi olduğu anlamına", md)

    def test_auc_unavailable_warning_is_localised(self):
        tr = [w["message"] for w in _gate({"accuracy": 0.9}, "tr")["warnings"]
              if w["code"] == "auc_unavailable"][0]
        en = [w["message"] for w in _gate({"accuracy": 0.9}, "en")["warnings"]
              if w["code"] == "auc_unavailable"][0]
        self.assertNotEqual(tr, en)
        self.assertIn("ROC-AUC", en)


if __name__ == "__main__":
    unittest.main(verbosity=2)
