"""Katman 1 KAPSAM ENVANTERİ — orkestratörün ham girdiden üretemediği feature'ları KOD söyler.

Neden bu script var
───────────────────
Handoff'un "tüm adım-portları BİTTİ" ifadesi, orkestratörün ``build_features``'ın POST-SPLIT
zincirini kapsadığı için doğruydu; ama PRE-SPLIT katmanının (normalize_column_names,
missing-indicator, date türevleri, is_zero flag'leri, domain recipe'leri, _FEATURE_OPS'un
kalan op'ları, aggregate, ts-lag ...) portlanıp portlanmadığını HİÇBİR test söylemiyordu:
mevcut parity harness'ında ``fit_pipeline`` ile ``transform_pipeline`` AYNI girdi
sözleşmesini görüyor, dolayısıyla eksik port ikisinde birden eksik → fark doğmuyor.

Bu script farkı doğurur: **fit tarafı gerçek ``build_features``** (üretimin fiilen ürettiği
feature seti), **transform tarafı orkestratör**. Aradaki fark = portlanmamış kalemler.
Elle çıkarılmış bir listeye güvenmek yerine ``MissingFeatureProduction`` guard'ının
listesini kullanır — envanteri kod üretir.

Sınıflandırma da bize ait DEĞİL: her eksik kolon, ``feature_report``'un kendi attribution
alanlarıyla (``derived_features`` / ``domain_features`` / ``derived_flags`` /
``ratio_features`` / ``auto_features``) etiketlenir. Hiçbir alana girmeyenler
"attribution YOK" olarak ayrı raporlanır (bunlar tipik olarak ham kolonlar,
normalize/encode türevleri ve date türevleridir).

Kullanım
────────
    API_DATABASE_URL="sqlite:///<abs>" PYTHONIOENCODING=utf-8 \
        uv run python scripts/layer1_coverage_inventory.py

Commit EDİLMEZ diye bir kısıt yok — kalıcı araç: her port turundan sonra yeniden koşulup
listenin kısaldığı görülür (Katman 1 kapanış ölçütü = liste boş).
"""

from __future__ import annotations

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts.offline_report import _infer_semantic_type, build_state
from src.services.feature_builder import (
    _add_missing_indicators, _build_zero_missing_flags, _clean_data, build_features,
)
from src.services.inference.orchestrator import (
    FrameShapeDivergence, MissingFeatureProduction, StructuralPreconditionViolation,
    UnsupportedPipelineScope, fit_pipeline, transform_pipeline,
)

# Hangi feature_report alanı bir kolonu SAHİPLENİYOR (build_features'ın KENDİ attribution'ı)
_ATTRIBUTION_KEYS = [
    "derived_features",   # _apply_plan_derived_features (_FEATURE_OPS)
    "domain_features",    # _apply_feature_specs / _apply_domain_recipes
    "derived_flags",      # _build_zero_missing_flags (is_zero)
    "ratio_features",     # _build_ratio_features (çift seçimi + bölme)
    "auto_features",      # _add_missing_indicators (_isna) + count/deviation
]


def _find_csv(pattern: str) -> str | None:
    hits = sorted(glob.glob(os.path.join("data", "csvs", pattern)))
    return hits[0] if hits else None


def _deterministic_y(n: int) -> pd.Series:
    return pd.Series([(i * 7 + 3) % 2 for i in range(n)])


def _orchestrator_artifact(df_clean: pd.DataFrame, ctx: dict):
    """Orkestratörü **GERÇEK beyan yüzeyiyle** fit et (T0 sonrası — taklit YOK).

    Önceki sürüm ctx'i burada elle kuruyordu (seçim fonksiyonlarını doğrudan çağırarak).
    T0'dan sonra ctx tek kaynaktan gelir: ``feature_report["pipeline_ctx"]`` —
    ``assemble_pipeline_ctx``'in ürettiği beyan. Böylece envanter, üretimin fiilen
    kullanacağı kablolamayı ÖLÇER, benzerini değil.
    """
    y = _deterministic_y(len(df_clean))
    _, artifact = fit_pipeline(df_clean.copy(), y, ctx)
    return artifact


def inventory(csv_pattern: str, target: str, intent: str) -> None:
    path = _find_csv(csv_pattern)
    if not path:
        print(f"\n### {csv_pattern}: CSV YOK (gitignored) — atlandı")
        return

    label = os.path.basename(path)
    print(f"\n{'=' * 78}\n### {label}  (target={target}, intent={intent})\n{'=' * 78}")

    # (1) GERÇEK üretim yolu: build_features'ın fiilen ürettiği feature seti
    state = build_state(path, target, intent, full=False, models=None)
    out = build_features(state)
    if out.get("error"):
        print(f"  build_features HATA: {out['error']}")
        return
    fr = out["feature_report"]
    real_features = fr["final_features"]
    print(f"  build_features → {len(real_features)} final feature")

    # (2) Orkestratörün ham girdiden üretebildikleri
    raw = pd.read_csv(path)

    # GERÇEK beyan yüzeyi — build_features'ın kendi ürettiği ctx (taklit YOK, T0)
    ctx = fr["pipeline_ctx"]
    drop_columns = ctx.get("drop_columns") or []
    # Hedef ham CSV'de duruyor; build_features onu _resolve_target ile ayırıyor. Envanter
    # ham çerçeveyle çalıştığı için hedefi drop listesine dahil ederiz (ön-koşul gereği).
    if target in raw.columns and target not in drop_columns:
        ctx = {**ctx, "drop_columns": list(drop_columns) + [target]}
    df_clean = _clean_data(raw.copy()).reset_index(drop=True)
    try:
        artifact = _orchestrator_artifact(df_clean, ctx)
    except (UnsupportedPipelineScope, StructuralPreconditionViolation) as exc:
        print(f"  ⛔ KAPSAM DIŞI — artifact kurulmadı (fail-closed, DOĞRU davranış):\n     {exc}")
        return
    scope = artifact.supported_scope
    print(f"  fit_pipeline   → {len(artifact.final_features)} feature, "
          f"{len(artifact.manifest)} manifest kalemi, deployable={scope.get('deployable')}"
          + (f" (beyansız: {scope['undeclared']})" if scope.get("undeclared") else ""))

    # (3) GERÇEK feature setini orkestratörden iste → guard envanteri söyler
    artifact.final_features = list(real_features)
    try:
        transform_pipeline(raw.copy(), artifact)
        print("  ✅ EKSİK YOK — orkestratör gerçek feature setini ham girdiden üretebiliyor")
        return
    except MissingFeatureProduction as e:
        never, dropped = e.never_produced, e.produced_then_dropped

    # (4) Sınıflandırma: build_features'ın KENDİ attribution alanları (bizim tahminimiz değil)
    owner: dict = {}
    for key in _ATTRIBUTION_KEYS:
        for col in (fr.get(key) or []):
            owner.setdefault(col, key)

    def _dump(title: str, cols: list, note: str) -> None:
        if not cols:
            return
        print(f"\n  {title}: {len(cols)}/{len(real_features)} kolon "
              f"(%{100 * len(cols) / max(len(real_features), 1):.0f}) — {note}\n")
        groups: dict = {}
        for col in cols:
            groups.setdefault(owner.get(col, "— attribution YOK —"), []).append(col)
        for key in sorted(groups, key=lambda k: -len(groups[k])):
            g = groups[key]
            print(f"    [{key}] × {len(g)}")
            for c in g[:10]:
                print(f"        · {c}")
            if len(g) > 10:
                print(f"        … +{len(g) - 10} tane daha")
            print()

    # ASIL ENVANTER — port borcu
    _dump("❌ HİÇ ÜRETİLMEDİ", never, "adım portlanmamış → 8.1 envanterinin GİRDİSİ")
    _dump("⚠️  ÜRETİLDİ SONRA DÜŞÜRÜLDÜ", dropped,
          "aşağıda tek tek AÇIKLANIYOR (açıklanamayan kalırsa GERÇEK sorun)")
    _explain_dropped(dropped, artifact)


def _explain_dropped(dropped: list, artifact) -> None:
    """``produced_then_dropped`` sepetini AÇIKLA — "ölçüm artefaktı" iddiası kanıtlansın.

    O sepet dolu kaldığı sürece "guard temiz" DENEMEZ: içinde gerçek bir sorun saklanabilir.
    Her kolon için tek tek sorulur: bu kolonu KİM düşürdü ve bu meşru mu?

      * ``select_drop (frozen)``  — artifact'in KENDİ donmuş drop listesinde. Bizim fit'imiz
        bu kolonu eğitimde de düşürdü; yabancı ``final_features``'ta yer alması ölçüm
        kurulumunun sonucu (build_features'ın select'i başka bir çerçevede karar verdi). MEŞRU.
      * ``encoder tüketti``       — kategorik kaynak kolon one-hot/target tarafından tüketildi
        (``get_dummies`` kaynağı düşürür). Eğitimde de böyle olur. MEŞRU.
      * ``AÇIKLANAMADI``          — hiçbiri değil → **gerçek sorun**, incelenmeli.
    """
    if not dropped:
        print("  ✅ produced_then_dropped SEPETİ BOŞ")
        return
    drop_cols = set((artifact.param_store.get("select_drop") or {}).get("drop_cols") or [])
    encoder_sources: set = set()
    for sid in ("onehot_encode", "target_encode"):
        p = artifact.param_store.get(sid) or {}
        encoder_sources |= set(p.get("columns") or [])

    buckets: dict = {}
    for col in dropped:
        if col in drop_cols:
            buckets.setdefault("select_drop (frozen) — MEŞRU", []).append(col)
        elif col in encoder_sources:
            buckets.setdefault("encoder tüketti — MEŞRU", []).append(col)
        else:
            buckets.setdefault("❗ AÇIKLANAMADI — GERÇEK SORUN", []).append(col)

    print("  ── produced_then_dropped açıklaması ──")
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):
        print(f"    {k}: {len(buckets[k])}")
        if k.startswith("❗"):
            for c in buckets[k]:
                print(f"        · {c}")
    unexplained = len(buckets.get("❗ AÇIKLANAMADI — GERÇEK SORUN", []))
    print(f"    → {'✅ hepsi açıklandı (ölçüm artefaktı)' if not unexplained else f'❌ {unexplained} AÇIKLANAMAYAN'}\n")


# Envanterin KAPSAMI dataset-koşulludur: bir adım ancak onu tetikleyen veride görünür
# (date türevleri timestamp kolonu ister, target-encode yüksek kardinalite ister, domain
# recipe'leri pack ister). Bu yüzden liste TEK dataset'ten değil, mevcut veri
# çeşitliliğinin tamamından toplanır — yine de "gördüklerimiz kesin, görmediklerimiz
# bilinmiyor" sınırı geçerlidir (yeni modalite → yeniden koş).
_DATASETS = [
    ("*credit_risk_dataset.csv",  "loan_status",        "classification"),  # düz tabular
    ("housing.csv",               "median_house_value", "regression"),      # düz tabular
    ("*Bank Customer Churn*.csv", "churn",              "classification"),  # identifier kolonu
    ("*superstore*.csv",          "Profit",             "regression"),      # timestamp + yüksek kard.
    ("*transactions*.csv",        "is_fraud",           "classification"),  # timestamp + yüksek kard.
    ("*student_dataset*.csv",     "exam_score",         "regression"),      # saf sayısal
]


if __name__ == "__main__":
    print("KATMAN 1 KAPSAM ENVANTERİ — 'kod söylesin' (guard tabanlı)")
    for pattern, target, intent in _DATASETS:
        try:
            inventory(pattern, target, intent)
        except Exception as exc:                      # bir dataset patlarsa kalanı yine koşsun
            print(f"\n### {pattern}: KOŞU HATASI ({type(exc).__name__}: {exc})")
    print("\nBitti. Liste boşalınca Katman 1 kapsam borcu kapanmıştır.")
