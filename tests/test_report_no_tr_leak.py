"""Faz 1 #1 KALICI GUARD: EN modunda üretilen raporda TÜRKÇE SIZINTI olmasın.

Türkçe sızıntı iki kez kaçtı (oof_blend ← seçilen; başka yerler). Tek seferlik script
yetmez — suite'in parçası, HER koşuda churn + credit ikisinde de taranır. Üçüncü kez
kaçmasın. Gerçek servisler (build_features→train→evaluate→render), LLM prose HARİÇ.

Tespit: (a) Türkçe'ye ÖZGÜ harfler (ğ ş ı İ ç — İngilizce'de asla yok) + (b) yalnız
ASCII harf kullanan yaygın TR rapor kelimeleri (whole-word). CSV yoksa skip (CI güvenli).
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sandbox_tr_leak.sqlite")

from scripts.offline_report import build_state, run_pipeline, render

_CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "csvs")
_DATASETS = [
    ("7723fd96-87eb-4548-89e7-87899f05badf_Bank Customer Churn Prediction.csv", "churn"),
    ("1b097ea4-bcc9-432f-ad92-d1ce02d62a52_credit_risk_dataset.csv", "loan_status"),
]

# Türkçe'ye özgü harfler — İngilizce metinde ASLA bulunmaz (feature adları da İngilizce).
_TR_CHARS = set("ğĞşŞıİçÇ")   # ö/ü/â hariç: bazı özel ad/loanword'lerde çıkabilir, gürültü
# Yalnız ASCII harfli, İNGİLİZCE'DE OLMAYAN yaygın TR rapor kelimeleri (whole-word).
# NOT: EN/TR ortak kelimeler ("risk", "model", "data") HARİÇ — aksi halde İngilizce
# "risk"/"model" false-positive verir (nitekim verdi). Türkçe'ye özgü olanları tut.
_TR_WORDS = [
    "secilen", "oran", "deger", "veri", "hedef", "dogru", "yanlis",
    "gecerli", "orani", "puan", "adet", "kontrol", "guven",
]
_TR_WORD_RE = re.compile(r"(?<![a-zçğıöşü])(" + "|".join(_TR_WORDS) + r")(?![a-zçğıöşü])", re.IGNORECASE)


def _scan_turkish(text: str) -> list[str]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        bad_chars = sorted(set(ch for ch in line if ch in _TR_CHARS))
        if bad_chars:
            hits.append(f"L{i} TR-harf {bad_chars}: {line.strip()[:120]}")
        for m in _TR_WORD_RE.finditer(line):
            hits.append(f"L{i} TR-kelime '{m.group(1)}': {line.strip()[:120]}")
    return hits


class EnglishReportHasNoTurkishTest(unittest.TestCase):
    def test_en_report_clean_on_real_datasets(self):
        ran = 0
        for fname, target in _DATASETS:
            csv = os.path.join(_CSV_DIR, fname)
            if not os.path.exists(csv):
                continue
            ran += 1
            state = build_state(csv, target, "classification", full=False, models=None, nrows=2000)
            state = run_pipeline(state)
            text_en = render(state, "en")
            hits = _scan_turkish(text_en)
            self.assertEqual(hits, [], f"\n[{fname}] EN raporunda Türkçe sızıntı:\n" + "\n".join(hits))
        if ran == 0:
            self.skipTest("referans CSV'ler yok (data/csvs) — CI'da atlanır")


if __name__ == "__main__":
    unittest.main(verbosity=2)
