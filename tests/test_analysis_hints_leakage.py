"""Offline invariant tests for the analyst → feature_plan hint channel (no LLM/DB).

Bu kanal, analistin bulgularının DS ajanına aktığı TEK yoldur; leakage sınırı burada
çizilir. Testler kanalın HEDEF-KÖR kalmasını sabitler — ileride biri yanlışlıkla
hedefe dokunursa burada patlar:

A) İMZA — extract_plan_hints hiçbir target parametresi kabul etmez. Sızıntı disipline
   değil, tipe bağlı olarak imkânsız olmalı.
B) SIFIR SAYI — payload yalnızca kolon adı taşır. Tek bir float/int bile sızarsa
   ("şu kolonun target korelasyonu 0.83") karar düzeyine indirgeme argümanı çöker.
C) KARIŞTIRILMIŞ HEDEF — hedef sütunun değerleri shuffle edilince ipuçları BİT BİT
   AYNI kalmalı. Kanal hedefe dokunuyorsa çıktı değişir. En ucuz kalıcı koruma budur.
D) DIŞLAMA — exclude_columns'daki kolonlar hiçbir listede ve hiçbir redundant çiftte
   görünmez (bir (feature, target) çifti bile hedefli bilgi taşır).
E) İÇERİK — hedef-agnostik yapısal bulgular gerçekten yakalanıyor (kanal boş değil).
G) PAYLOAD ÖZDEŞLİĞİ — ipuçları TÜM veriden ve YALNIZCA TRAIN satırlarından
   hesaplandığında BİT BİT AYNI çıkmalı. Sızıntı sorusunun asıl bekçisi budur:
   payload iki dünyada özdeşse kanal test satırlarından bilgi taşımıyordur, dolayısıyla
   LLM ne yaparsa yapsın ROC'u şişiremez. Bunu ROC A/B'siyle ölçmek MÜMKÜN DEĞİL —
   LLM planı aynı girdiyle bile ±0.004 ROC salınımı üretiyor (ölçüldü), aradığımız
   etki o gürültünün altında kalıyor. Bu test ise deterministik ve LLM'siz.

   NOT: alttaki örneklem (PLAN_HINT_MAX_ROWS) yüzünden tek koşu yanıltabilir —
   tüm-veri ve train-only farklı satırlara denk gelip eşiğe yakın bir kolonu bir
   yerde içeri bir yerde dışarı atabilir. O yüzden birden çok seed ile tekrarlanır.
"""

import inspect
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import settings
from src.services.analysis_engine import extract_plan_hints, format_plan_hints
from src.services.feature_builder import TEST_SIZE


TARGET = "churn"

# Hedefli bir ipucu üretilseydi burada görünürdü: leaky_feature hedefle neredeyse
# birebir aynı. Hedef-kör kanal bunu GÖRMEMELİ (hedef dışlanıyor), yapısal bulgular
# (sabit/çarpık/high-card/redundant) ise yakalanmalı.
def _frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    tenure = rng.gamma(2.0, 8.0, size=n)                    # çarpık
    df = pd.DataFrame({
        TARGET:           y,
        "leaky_feature":  y * 1.0 + rng.normal(0, 0.01, n),  # hedefle ~1.0 korelasyon
        "tenure":         tenure,
        "tenure_months":  tenure * 12.0,                     # tenure ile redundant (r≈1)
        "monthly_charge": rng.normal(70, 15, size=n),
        "constant_col":   np.ones(n),                        # sabit
        "customer_id":    [f"C{i:06d}" for i in range(n)],   # high-cardinality
        "region":         rng.choice(["tr", "de", "fr"], size=n),
        # Metin tutulan timestamp: hem tarih hem (naif bakışla) yüksek kardinaliteli.
        "signup_at":      pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
    })
    return df


def _profile(df: pd.DataFrame) -> dict:
    """DuckDBEngine.profile_table() çıktısının test eşdeğeri (aynı anahtarlar)."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]

    cols = {}
    for c in df.columns:
        s = df[c]
        cp = {
            "dtype":        "DOUBLE" if c in numeric else "VARCHAR",
            "null_count":   int(s.isna().sum()),
            "null_rate":    round(float(s.isna().mean()), 4),
            "unique_count": int(s.nunique()),
            "unique_rate":  round(float(s.nunique() / max(len(s), 1)), 4),
        }
        if c in numeric:
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            outlier_rate = (
                float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()) if iqr > 0 else 0.0
            )
            cp.update({
                "mean": float(s.mean()), "median": float(s.median()), "std": float(s.std()),
                "q25": q1, "q75": q3, "min": float(s.min()), "max": float(s.max()),
                "has_outliers": round(outlier_rate, 4),
            })
        else:
            cp["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        cols[c] = cp

    return {
        "shape":   {"rows": len(df), "columns": len(df.columns)},
        "summary": {"numeric_columns": numeric, "categorical_columns": categorical,
                    "datetime_columns": [], "missing_rate": 0.0},
        "columns": cols,
    }


def _hints(df: pd.DataFrame, exclude=frozenset({TARGET})) -> dict:
    """Node'un yaptığını birebir taklit et: hedefi df'ten DÜŞÜR + exclude'a koy."""
    profile = _profile(df)
    view = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    return extract_plan_hints(view, profile, exclude_columns=set(exclude))


def _leaf_values(obj):
    """Payload'daki tüm yaprak değerleri düz listeye açar."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaf_values(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _leaf_values(v)
    else:
        yield obj


class TestHintChannelIsTargetBlind(unittest.TestCase):

    # A) Sızıntı tasarım gereği imkânsız: fonksiyon hedefi PARAMETRE OLARAK alamaz.
    def test_signature_has_no_target_parameter(self):
        params = set(inspect.signature(extract_plan_hints).parameters)
        self.assertEqual(params, {"df", "profile", "exclude_columns"})
        for p in params:
            self.assertNotIn("target", p.lower(), f"'{p}' hedefe dair bir parametre olamaz")

    # B) Payload sıfır sayı taşır — yalnızca kolon adı + iki sabit meta alan.
    def test_payload_carries_no_statistics(self):
        hints = _hints(_frame())
        meta = {hints.pop("generated_from"), hints.pop("target_aware")}
        self.assertEqual(meta, {"full_data", False})

        for v in _leaf_values(hints):
            self.assertIsInstance(v, str, f"Payload'a sayı sızdı: {v!r}")
            self.assertNotIsInstance(v, bool)

        text = format_plan_hints(_hints(_frame()))
        digits = [ch for ch in text if ch.isdigit()]
        # Kolon adlarından gelen rakamlar dışında (yok) hiçbir istatistik metne girmez.
        self.assertEqual(digits, [], f"Prompt metnine sayı sızdı: {text}")

    # C) KARIŞTIRILMIŞ HEDEF: hedef değerleri shuffle edilince çıktı DEĞİŞMEMELİ.
    #    Değişiyorsa kanal hedefe dokunuyor demektir.
    def test_shuffled_target_produces_identical_hints(self):
        df = _frame()
        baseline = _hints(df)

        shuffled = df.copy()
        rng = np.random.default_rng(99)
        shuffled[TARGET] = rng.permutation(shuffled[TARGET].to_numpy())

        self.assertEqual(baseline, _hints(shuffled))

        # Hedefi tamamen ters çevirmek de bir şey değiştirmemeli.
        inverted = df.copy()
        inverted[TARGET] = 1 - inverted[TARGET]
        self.assertEqual(baseline, _hints(inverted))

    # D) Dışlanan kolonlar hiçbir listede / hiçbir çiftte görünmez.
    def test_excluded_columns_never_appear(self):
        # leaky_feature'ı da hedef adayı gibi dışla → hiçbir yerde geçmemeli.
        hints = _hints(_frame(), exclude=frozenset({TARGET, "leaky_feature"}))
        mentioned = set(_leaf_values(hints))
        self.assertNotIn(TARGET, mentioned)
        self.assertNotIn("leaky_feature", mentioned)

    # E) Kanal boş değil: hedef-agnostik yapısal bulgular gerçekten yakalanıyor.
    def test_structural_findings_are_detected(self):
        hints = _hints(_frame())
        self.assertIn("constant_col", hints["constant_columns"])
        self.assertIn("customer_id", hints["high_cardinality"])
        self.assertIn("tenure", hints["skewed"])
        self.assertIn(["tenure", "tenure_months"], hints["redundant_pairs"])

    # F) Tarih kolonu yakalanır ve "yüksek kardinalite" listesine DÜŞMEZ — yoksa LLM'e
    #    aynı kolon için çelişkili iki emir giderdi ("at" vs "parçala").
    def test_date_column_is_flagged_and_not_high_cardinality(self):
        hints = _hints(_frame())
        self.assertIn("signup_at", hints["date_like"])
        self.assertNotIn("signup_at", hints["high_cardinality"])
        self.assertIn("date/time", format_plan_hints(hints).lower())

    # Veri okunamazsa (df=None) kanal profiling'ten üretmeye devam eder, çökmez.
    def test_degrades_gracefully_without_dataframe(self):
        df = _frame()
        hints = extract_plan_hints(None, _profile(df), exclude_columns={TARGET})
        self.assertIn("constant_col", hints["constant_columns"])
        self.assertEqual(hints["redundant_pairs"], [])


# ═══════════════════════════════════════════════════════════
#  G) PAYLOAD ÖZDEŞLİĞİ — tüm veri vs yalnızca train satırları
# ═══════════════════════════════════════════════════════════

def _train_rows(df: pd.DataFrame, target: str, seed: int) -> pd.DataFrame:
    """feature_plan_node._train_partition'ın aynısı: aynı TEST_SIZE, stratify, seed."""
    df = df[df[target].notna()]
    try:
        train, _ = train_test_split(df, test_size=TEST_SIZE, random_state=seed,
                                    stratify=df[target])
    except ValueError:
        train, _ = train_test_split(df, test_size=TEST_SIZE, random_state=seed)
    return train


def _identity_gap(df: pd.DataFrame, target: str, seed: int) -> tuple[dict, dict]:
    """(tüm veriden hints, train-only hints) — özdeş olmaları beklenir."""
    profile = _profile(df)
    excl = {target}

    full  = _hints(df, exclude=frozenset(excl))
    train = _hints(_train_rows(df, target, seed), exclude=frozenset(excl))
    # generated_from ölçüm modunda node tarafından yazılır; payload kıyasına girmez.
    for h in (full, train):
        h.pop("generated_from", None)
    return full, train


def _wide_frame(n: int, seed: int) -> pd.DataFrame:
    """PLAN_HINT_MAX_ROWS üstünde → alttaki örneklem gerçekten devreye girer."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    amount = rng.lognormal(3.0, 1.2, size=n)           # çarpık + outlier'lı
    return pd.DataFrame({
        TARGET:         y,
        "amount":       amount,
        "amount_cents": amount * 100.0,                 # redundant çift
        "score":        rng.normal(0, 1, size=n),
        "flag":         rng.integers(0, 2, size=n),
        "segment":      rng.choice(list("abcde"), size=n),
        "created_at":   pd.date_range("2024-01-01", periods=n, freq="min").astype(str),
    })


class TestPayloadIdentityFullVsTrainOnly(unittest.TestCase):
    """Kanal test satırlarından bilgi taşımıyor: payload iki dünyada özdeş."""

    def test_identical_across_seeds_small_frame(self):
        df = _frame(n=1000)
        for seed in (0, 1, 42, 7, 2024):
            full, train = _identity_gap(df, TARGET, seed)
            self.assertEqual(full, train, f"seed={seed}: tüm-veri ve train-only payload farklı")

    def test_identical_when_sampling_kicks_in(self):
        # settings.RANDOM_SEED üretimde split seed'i; örneklem tavanının ÜSTÜNDE
        # veriyle koş → _hint_sample iki kolda FARKLI satır kümeleri çeker.
        df = _wide_frame(n=60_000, seed=3)
        for seed in (settings.RANDOM_SEED, 1, 99):
            full, train = _identity_gap(df, TARGET, seed)
            self.assertEqual(full, train, f"seed={seed}: örneklem altında payload kaydı")

    def test_target_never_leaks_into_payload(self):
        for frame in (_frame(n=1000), _wide_frame(n=60_000, seed=5)):
            full, _ = _identity_gap(frame, TARGET, settings.RANDOM_SEED)
            self.assertNotIn(TARGET, set(_leaf_values(full)))


# Gerçek referans veri setleri repo'da yok (data/ gitignore'lu). Varsa aynı özdeşlik
# kontrolü onlarda da koşar; yoksa test atlanır — sentetik kontroller her yerde koşar.
_REAL = {
    "kredi":    ("data/csvs/1b097ea4-bcc9-432f-ad92-d1ce02d62a52_credit_risk_dataset.csv", "loan_status"),
    "churn":    ("data/csvs/7723fd96-87eb-4548-89e7-87899f05badf_Bank Customer Churn Prediction.csv", "churn"),
    "eticaret": ("data/csvs/2eb2d968-2193-4bd4-969c-782072f8198c_transactions.csv", "is_fraud"),
}


class TestPayloadIdentityOnReferenceDatasets(unittest.TestCase):

    def test_reference_datasets_identical(self):
        ran = 0
        for name, (path, target) in _REAL.items():
            p = Path(path)
            if not p.exists():
                continue
            df = pd.read_csv(p)
            for seed in (settings.RANDOM_SEED, 7):
                full, train = _identity_gap(df, target, seed)
                self.assertEqual(full, train, f"{name} (seed={seed}): payload farklı → sızıntı sinyali")
                self.assertNotIn(target, set(_leaf_values(full)))
            ran += 1
        if ran == 0:
            self.skipTest("Referans veri setleri yok (data/ gitignore'lu) — sentetik testler koştu.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
