from __future__ import annotations

"""Artifact içerik-üreticileri — ``model_ref`` + ``training_reference`` (task #5).

``serialize.py`` yaz/oku; bu modül İÇERİK üretimi (aynı ayrım: ``pickle_io`` estimator-kapı,
``serialize`` artifact-io, ``references`` alan-içeriği). İkisi de SAF fonksiyon: girdi alır,
dict döndürür, hiçbir şey diske yazmaz → tek başına test edilebilir.

**BU TUR YALNIZ MEKANİZMA** (§2.20 kilidi): alanları ÜRETEN fonksiyonlar + Artifact alanları +
round-trip. Wiring (``fit_pipeline``'ın bunları çağırması, ``evaluation``'ın ``baseline_metrics``
beslemesi, worker'ın ``model_path`` vermesi) AYRI iş — artifact hâlâ diske otomatik yazılmıyor.
"""

import os
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# raw_input_signature ile AYNI cap (tutarlılık): yüksek-kardinalite kategorikte vocab saklanmaz,
# PSI için özet + kuyruk kütlesi tutulur. (orchestrator._SIG_MAX_VOCAB ile hizalı — kopya, import
# döngüsü olmasın diye; ikisi değişirse test yakalar → test_reference_cap_matches_signature.)
_REF_MAX_VOCAB = 50

_REF_QUANTILES = [0.01, 0.25, 0.50, 0.75, 0.99]


class ModelFileMissing(RuntimeError):
    """``model_ref`` bir dosyaya REFERANS veriyor (kopya değil — model 147MB, kopyalamak her
    artifact'i 400× şişirir). Dosya kaybolursa artifact skorlayamaz → FAIL-CLOSED, kullanım
    anında (yükleme değil). Referansın bilinen kırılganlığı; bedeli bilinçli."""


def build_model_ref(model_path: "str | None", model_name: str, optimal_threshold: "float | None",
                    threshold_provenance: "dict | None", sklearn_version: str) -> dict:
    """``model_ref`` alanı — model METADATA'sı (dosyanın KENDİSİ değil, YOL).

    KARAR: REFERANS, kopya DEĞİL (ölçüldü: model 147MB, blend ``fitted_`` list'inde; kopyalamak
    serialize'ı 25-393KB'tan 147MB'a çıkarırdı). Yol taşınır; ``model_present`` kullanım anında
    fail-closed doğrulanır (``resolve_model_path``).

    ``sklearn_version``: model pickle'ının sürüm kapısı için (dördüncü müşteri — Target/Robust/
    Standard'dan sonra). Kapı METADATA düzeyinde: 147MB estimator OKUNMADAN sürüm karşılaştırılır."""
    return {
        "model_path": model_path,
        "model_name": model_name,
        "optimal_threshold": (None if optimal_threshold is None else float(optimal_threshold)),
        # {value, source: oof_cv_f1max|default_0.5, n} — ml_trainer._select_oof_threshold'dan
        "threshold_provenance": threshold_provenance,
        "sklearn_version": sklearn_version,
        "byte_size": (os.path.getsize(model_path)
                      if model_path and os.path.exists(model_path) else None),
    }


def resolve_model_path(model_ref: dict) -> str:
    """Kullanım-anı fail-closed: model dosyası VAR MI? (referansın kırılganlık kontrolü)."""
    p = (model_ref or {}).get("model_path")
    if not p or not os.path.exists(p):
        raise ModelFileMissing(
            f"model_ref dosyaya referans veriyor ama dosya yok: {p!r}. Artifact skorlayamaz "
            f"(REFERANS kopya değil — model 147MB). Çözüm: modeli yeniden eğit ya da yolu düzelt."
        )
    return p


def _continuous_stats(s: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {"kind": "continuous", "mean": None, "std": None, "quantiles": None, "n": 0}
    q = v.quantile(_REF_QUANTILES)
    return {
        "kind": "continuous",
        "mean": float(v.mean()), "std": float(v.std(ddof=0)),
        "quantiles": {str(k): float(q.loc[k]) for k in _REF_QUANTILES},
        "n": int(v.notna().sum()),
    }


def _categorical_stats(s: pd.Series) -> dict:
    vc = s.astype("object").dropna().value_counts(normalize=True)
    n_uniq = int(vc.shape[0])
    if n_uniq <= _REF_MAX_VOCAB:
        return {"kind": "categorical", "value_counts_normalized":
                {str(k): float(v) for k, v in vc.items()}, "n_unique": n_uniq}
    # CAP: yalnız en sık _REF_MAX_VOCAB tutulur. Kesilen KÜTLE kaydedilir (kullanıcı şartı 3) →
    # Katman 2 PSI'ın "other" kovasını bundan kurar; yoksa kalan-% bilinmez, oranlar yanlış normalize olur.
    tut = vc.iloc[:_REF_MAX_VOCAB]
    return {
        "kind": "categorical",
        "value_counts_normalized": {str(k): float(v) for k, v in tut.items()},
        "n_unique": n_uniq,
        "vocab_truncated": [n_uniq, _REF_MAX_VOCAB],   # 542 → 50
        "tail_mass": float(1.0 - tut.sum()),           # kesilen normalize kütle (PSI "other")
    }


def build_training_reference(raw_train: pd.DataFrame, columns: "list | None" = None,
                             baseline_metrics: "dict | None" = None,
                             signature: "dict | None" = None) -> dict:
    """``training_reference`` — Katman 2'nin temeli. İKİ ORTOGONAL YARI:

      * ``per_feature`` = **PSI REFERANSI** (raw-space, INPUT drift). Ham train partisyonundan;
        sürekli → mean/std/quantiles, kategorik → normalize value_counts (yüksek-kard → cap + tail_mass).
      * ``baseline_metrics`` = **CBPE ÇAPASI** (prediction-space, PERFORMANCE drift). evaluation'dan
        kopya; ``per_feature`` ile ORTOGONAL — Katman 2 FARKLI monitörlerde kullanır, karışırsa
        yanlış alarm. **Fit anında YOK** (evaluation sonra) → ``None`` kabul edilir, wiring'e kalır.

    ``columns``: hangi kolonlar izlenecek. **``raw_input_signature.required`` ile AYNI küme
    verilmeli** (id/timestamp drop'ları GİRMEZ — monitoring onları izlemez, tutarlılık için imza
    ile hizalı). ``None`` → ``raw_train``'in tüm kolonları (fixture/test kolaylığı).

    ``signature`` = **TEK TİP-KARARININ KAYNAĞI** (``raw_input_signature``). Kolonun continuous mu
    categorical mı olduğunu buradan (``columns[col].kind``) okur — dtype'tan DEĞİL. Gerekçe: imza
    domain+kardinalite farkında (``float64`` saklanan binary, düşük-kard int → categorical); salt-dtype
    ayrık int'i (products_number/tenure) continuous kurup DEJENERE quantile üretiyordu → sahte PSI.
    İmza serving'in doğruladığı sözleşme olduğundan referansı ona bağlamak ikisini inşaen tutarlı
    kılar (imza ``known_values`` de ``str(v)`` ile kuruluyor → PSI eşleşmesiyle aynı normalizasyon).
    Eşleme: ``continuous`` → sürekli istatistik; ``binary/categorical/datetime/other`` → value_counts.
    ``None`` → dtype heuristiği (yalnız imzasız fixture/test yolu; üretim wiring'i imzayı HER ZAMAN verir)."""
    cols = list(columns) if columns is not None else list(raw_train.columns)
    sig_cols = (signature or {}).get("columns") or {}
    per_feature: dict = {}
    for c in cols:
        if c not in raw_train.columns:
            continue
        s = raw_train[c]
        kind = (sig_cols.get(str(c)) or {}).get("kind")
        if kind is not None:
            # TEK KAYNAK: imzanın domain kararı. continuous dışı hepsi value_counts (binary de
            # dahil — 0/1 quantile'ı dejenere olurdu, PSI'da 2 kategori doğru kova).
            use_continuous = (kind == "continuous")
        else:
            # imzasız fallback (fixture/test): eski dtype heuristiği.
            use_continuous = (pd.api.types.is_numeric_dtype(s)
                              and not pd.api.types.is_bool_dtype(s))
        per_feature[str(c)] = _continuous_stats(s) if use_continuous else _categorical_stats(s)
    return {
        "space": "raw",
        "per_feature": per_feature,        # PSI referansı (raw-space, input drift)
        "baseline_metrics": baseline_metrics,   # CBPE çapası (prediction-space); ORTOGONAL, wiring'e kadar None
        "n_train": int(len(raw_train)),
    }
