from __future__ import annotations

"""
Fairness Analizi
----------------
Grup başına performans ve seçim oranı ölçümü — HAM segment değerleri üzerinden.

BU MODÜL evaluator._analyze_bias'ın YERİNİ ALIR. Eskisi dört ayrı sessiz hata
taşıyordu; hepsi "eksik veri 0/None'a düşüyor, düşüş hiçbir yere yazılmıyor"
desenindeydi:

  1. Rapor ``.get('fpr', 0)`` okuyordu, üretici FPR/FNR'yi HİÇ hesaplamıyordu →
     her raporda "FPR=0.000, FNR=0.000". Artık gerçekten hesaplanıyor (ve zaten
     doğru metrik odur: accuracy_diff dengesiz sınıfta yanıltıcıdır — %99 negatif
     bir veri setinde model herkese "hayır" derse iki grupta da accuracy ~%99
     çıkar ve fark ~0 görünür; oysa FNR iki grupta da %100'dür).
  2. Grup maskesi ölçeklenmiş one-hot'ta boşalıyordu → kalabalık gruplar sessizce
     atlanıyordu. Artık ham değere bakılıyor (segment_carrier).
  3. ``drop_first`` kurbanı kategori hiç görünmüyordu. Ham değerde hepsi var.
  4. DI=0.0 (tam ayrımcılık) ``if disparate_impact else None`` falsy kontrolüyle
     "hesaplanmadı"ya düşüyordu. Artık ``is not None``.

HÜKÜM VERMİYORUZ, KANIT SUNUYORUZ
=================================
Otomatik "High Bias" damgası YOK. Sebep: DI ile performans farkı AYNI ŞEYİ
ÖLÇMEZ ve ikisinin ayrışması bir çelişki değildir.

  - FPR/FNR farkı → PERFORMANS paritesi: model gruplarda aynı doğrulukta mı?
  - DI            → SEÇİM ORANI paritesi: model gruba ne sıklıkla pozitif diyor?

Bir model iki grupta da kusursuz olup DI 0.656 verebilir — eğer grubun GERÇEK
pozitif taban oranı zaten düşükse. O hâlde model ayrımcılık yapmıyor, veriyi
yansıtıyor. Bu yüzden DI'yi tek başına değil, ``y_pred`` taban oranı ile ``y_true``
taban oranının KARŞILAŞTIRMASIYLA sunuyoruz: fark model kaynaklı mı, veri
kaynaklı mı sorusunu ancak bu ikili cevaplar. 4/5 bandı (0.8–1.25) yalnızca
REFERANS EŞİK olarak belirtilir — hukuki bir eşiktir, istatistiksel değil;
etiketlemeyi kullanıcı yapar.

İNCE ÖRNEKLEM
=============
Her oran Wilson skor aralığıyla birlikte döner. 6 kişilik bir grupta DI 0.656
pekâlâ saf gürültü olabilir; aralık bunu görünür kılar. Eşiğin altındaki gruplar
rapordan SİLİNMEZ — görünür sebeple listelenir (``not_computed_reason``).
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Wilson skor aralığı ──────────────────────────────────────────────────────

def wilson_interval(successes: int, n: int, z: float = 1.96) -> Optional[dict]:
    """Bir oranın %95 Wilson skor aralığı.

    Neden Wilson: normal (Wald) yaklaşımı küçük n'de ve uç oranlarda çöker —
    p=0 veya p=1'de sıfır genişlikte aralık üretir ("kesin biliyoruz" der, oysa
    hiçbir şey bilmiyoruzdur). Fairness'ta gruplar tam da küçük ve oranlar tam da
    uç olduğu için bu fark kritiktir.
    """
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return {
        "lower": round(float(max(0.0, centre - margin)), 4),
        "upper": round(float(min(1.0, centre + margin)), 4),
        "method": "wilson",
    }


def _rate(mask: np.ndarray, condition: np.ndarray) -> tuple[Optional[float], int, int]:
    """(oran, pay, payda) — payda 0 ise oran None (0 DEĞİL; tanımsız ≠ sıfır)."""
    denom = int(mask.sum())
    if denom == 0:
        return None, 0, 0
    num = int((condition & mask).sum())
    return num / denom, num, denom


# ── Grup başına metrikler ────────────────────────────────────────────────────

def _group_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, positive_label: Any
) -> dict:
    """Bir grup için performans + seçim oranı metrikleri.

    FPR = gerçekte negatif olanlardan yanlışlıkla pozitif denenlerin oranı
    FNR = gerçekte pozitif olanlardan kaçırılanların oranı
    Her ikisi de TANIMSIZ olabilir (grupta hiç negatif/pozitif yoksa) → None,
    ve nedeni ``*_undefined_reason`` ile taşınır. Sıfıra DÜŞMEZ.
    """
    n = int(mask.sum())
    is_pos_true = (y_true == positive_label)
    is_pos_pred = (y_pred == positive_label)

    # FPR: payda = gruptaki gerçek negatifler
    neg_mask = mask & ~is_pos_true
    fpr, fp, n_neg = _rate(neg_mask, is_pos_pred)

    # FNR: payda = gruptaki gerçek pozitifler
    pos_mask = mask & is_pos_true
    fnr, fn, n_pos = _rate(pos_mask, ~is_pos_pred)

    # Seçim oranı (DI'nin payı): modelin bu gruba pozitif deme sıklığı
    sel_rate, n_sel, _ = _rate(mask, is_pos_pred)
    # Gerçek taban oranı: verinin kendisinde bu grubun pozitif olma sıklığı.
    # DI'yi yorumlamanın TEK dürüst yolu bu ikisini yan yana koymaktır.
    base_rate, n_base, _ = _rate(mask, is_pos_true)

    acc = float((y_true[mask] == y_pred[mask]).mean()) if n else None

    out: dict = {
        "n": n,
        "accuracy": round(acc, 4) if acc is not None else None,
        "fpr": round(fpr, 4) if fpr is not None else None,
        "fpr_ci": wilson_interval(fp, n_neg) if n_neg else None,
        "fpr_undefined_reason": "" if fpr is not None else {"code": "fr_fpr_undefined"},
        "fnr": round(fnr, 4) if fnr is not None else None,
        "fnr_ci": wilson_interval(fn, n_pos) if n_pos else None,
        "fnr_undefined_reason": "" if fnr is not None else {"code": "fr_fnr_undefined"},
        "selection_rate": round(sel_rate, 4) if sel_rate is not None else None,
        "selection_rate_ci": wilson_interval(n_sel, n) if n else None,
        "base_rate": round(base_rate, 4) if base_rate is not None else None,
        "base_rate_ci": wilson_interval(n_base, n) if n else None,
        "n_true_positives": int(n_pos),
        "n_true_negatives": int(n_neg),
    }
    return out


def _group_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "rmse": None, "mae": None, "mean_prediction": None}
    err = y_true[mask] - y_pred[mask]
    return {
        "n": n,
        "rmse": round(float(np.sqrt(np.mean(err**2))), 4),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "mean_prediction": round(float(np.mean(y_pred[mask])), 4),
    }


# ── Ana analiz ───────────────────────────────────────────────────────────────

# 4/5 kuralı — REFERANS, hüküm değil. Otomatik etiketleme için KULLANILMAZ;
# yalnızca kullanıcıya "bu bandın dışı geleneksel olarak incelenmeye değer
# görülür" diye bağlam sunmak için taşınır.
DI_REFERENCE_BAND = (0.8, 1.25)


def analyze_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: Optional[pd.DataFrame],
    segment_meta: dict,
    intent: str,
) -> dict:
    """Segment taşıyıcısı üzerinden grup başına fairness analizi.

    Args:
        y_true, y_pred: test seti gerçek/tahmin (pozisyonel olarak segments ile hizalı)
        segments: HAM grup değerleri (test satırlarına hizalı). None → analiz yok,
            SEBEP segment_meta["reason"]'dan gelir.
        segment_meta: {"columns", "mode", "skipped", "reason"}
        intent: "classification" | "regression"

    Returns:
        {
          "status": "ok" | "not_computed",
          "reason": {"code","params"},      # status != ok ise DOLU (sessiz sıfır yasağı);
                                            # dil-nötr KOD, rapor R_REASON ile çevirir
          "mode": "narrow" | "wide",
          "task": "classification" | "regression",
          "positive_label": Any,
          "skipped_columns": {col: reason}, # taşınmayanlar — görünür
          "segments": {col: {"groups": {...}, "comparisons": {...}, "skipped_groups": {...}}}
        }
    """
    task = "regression" if "regression" in (intent or "") else "classification"
    min_n = settings.FAIRNESS_MIN_GROUP_SIZE

    # Taşıyıcı TÜM ham kolonları getirir (hata analizi de okur). Fairness'ın analiz
    # kapsamı meta["segment_columns"]'dır — taşınan her kolon segment DEĞİLDİR.
    seg_cols = [
        c for c in ((segment_meta or {}).get("segment_columns") or [])
        if segments is not None and c in segments.columns
    ]

    if segments is None or segments.empty or not seg_cols:
        reason = (segment_meta or {}).get("reason") or {"code": "fr_no_carrier"}
        logger.info(f"[fairness] Analiz yapılmadı: {reason}")
        return {
            "status": "not_computed",
            "reason": reason,
            "mode": (segment_meta or {}).get("mode", ""),
            "task": task,
            "positive_label": None,
            "skipped_columns": (segment_meta or {}).get("skipped", {}),
            "segments": {},
        }

    n_rows = min(len(segments), len(y_true))
    if n_rows == 0:
        return {
            "status": "not_computed",
            "reason": {"code": "fr_no_rows"},
            "mode": segment_meta.get("mode", ""),
            "task": task,
            "positive_label": None,
            "skipped_columns": segment_meta.get("skipped", {}),
            "segments": {},
        }

    # Hizalama savunması: taşıyıcı X_test ile pozisyonel hizalı olmalı. Uzunluk
    # tutmuyorsa kırpmak SESSİZ bir yanlış hizalama üretirdi → analizi reddet.
    if len(segments) != len(y_true):
        reason = {"code": "fr_misaligned",
                  "params": {"s": len(segments), "t": len(y_true)}}
        logger.error(f"[fairness] {reason}")
        return {
            "status": "not_computed",
            "reason": reason,
            "mode": segment_meta.get("mode", ""),
            "task": task,
            "positive_label": None,
            "skipped_columns": segment_meta.get("skipped", {}),
            "segments": {},
        }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    positive_label = None
    if task == "classification":
        labels = np.unique(y_true)
        if len(labels) != 2:
            return {
                "status": "not_computed",
                "reason": {"code": "fr_multiclass", "params": {"n": len(labels)}},
                "mode": segment_meta.get("mode", ""),
                "task": task,
                "positive_label": None,
                "skipped_columns": segment_meta.get("skipped", {}),
                "segments": {},
            }
        # Pozitif sınıf: ikiliden büyük olan (1/True/"yes" → 1 konvansiyonu).
        positive_label = labels[-1]

    result_segments: dict = {}

    for col in seg_cols:
        # Ham değer sayısal da olabilir (taşıyıcı tipleri korur) — grup anahtarı
        # olarak kullanmak için burada string'e çevrilir.
        raw = segments[col].astype(str)
        groups: dict = {}
        skipped_groups: dict = {}

        for value in sorted(raw.dropna().unique()):
            mask = (raw == value).to_numpy()
            n = int(mask.sum())
            if n < min_n:
                # SESSİZ ATLAMA YASAK — sebep taşınır, grup raporda görünür.
                skipped_groups[str(value)] = {
                    "code": "fr_thin_group", "params": {"n": n, "min": min_n}}
                continue
            groups[str(value)] = (
                _group_classification_metrics(y_true, y_pred, mask, positive_label)
                if task == "classification"
                else _group_regression_metrics(y_true, y_pred, mask)
            )

        if not groups:
            result_segments[col] = {
                "groups": {},
                "comparisons": {},
                "skipped_groups": skipped_groups,
                "reason": {"code": "fr_no_groups"},
            }
            continue

        result_segments[col] = {
            "groups": groups,
            "comparisons": _build_comparisons(groups, task),
            "skipped_groups": skipped_groups,
            "reason": "",
        }

    return {
        "status": "ok",
        "reason": "",
        "mode": segment_meta.get("mode", ""),
        "task": task,
        "positive_label": (positive_label.item()
                           if hasattr(positive_label, "item") else positive_label),
        "di_reference_band": list(DI_REFERENCE_BAND),
        "skipped_columns": segment_meta.get("skipped", {}),
        "segments": result_segments,
    }


def _build_comparisons(groups: dict, task: str) -> dict:
    """Gruplar arası farkları REFERANS GRUBA göre çıkarır.

    Referans grup = en kalabalık grup. Neden: en dar güven aralığı onda; farkı
    en güvenilir zemine göre ölçmek gerekir. (Eski kod one-hot'un "0" tarafını
    referans alıyordu — o "grup değil, geri kalan herkes"ti ve yorumlanamazdı.)

    DI yalnızca HAM oran olarak döner. Etiket/damga YOK — ``base_rate_ratio`` ile
    birlikte sunulur ki okuyucu "seçim oranı farkı model kaynaklı mı veri kaynaklı
    mı" sorusunu kendisi cevaplayabilsin.
    """
    if not groups:
        return {}

    ref = max(groups.items(), key=lambda kv: kv[1].get("n", 0))
    ref_name, ref_m = ref
    out: dict = {"reference_group": ref_name, "reference_n": ref_m.get("n"), "groups": {}}

    for name, m in groups.items():
        if name == ref_name:
            continue
        cmp: dict = {}
        if task == "classification":
            # Performans paritesi
            for metric in ("fpr", "fnr", "accuracy"):
                a, b = m.get(metric), ref_m.get(metric)
                if a is not None and b is not None:
                    cmp[f"{metric}_diff"] = round(a - b, 4)
                else:
                    cmp[f"{metric}_diff"] = None
                    cmp[f"{metric}_diff_undefined_reason"] = (
                        m.get(f"{metric}_undefined_reason")
                        or ref_m.get(f"{metric}_undefined_reason")
                        or {"code": "fr_diff_undefined"}
                    )

            # Seçim oranı paritesi (DI) — HAM, eşiksiz.
            sel, ref_sel = m.get("selection_rate"), ref_m.get("selection_rate")
            if sel is not None and ref_sel is not None and ref_sel > 0:
                cmp["disparate_impact"] = round(sel / ref_sel, 4)   # DI=0.0 GEÇERLİ değerdir
            else:
                cmp["disparate_impact"] = None
                cmp["disparate_impact_undefined_reason"] = (
                    {"code": "fr_di_undef_zero"}
                    if ref_sel == 0 else {"code": "fr_di_undef_other"}
                )

            # Verinin kendisindeki taban oranı farkı — DI'yi yorumlamanın anahtarı.
            base, ref_base = m.get("base_rate"), ref_m.get("base_rate")
            if base is not None and ref_base is not None and ref_base > 0:
                cmp["base_rate_ratio"] = round(base / ref_base, 4)
            else:
                cmp["base_rate_ratio"] = None
                cmp["base_rate_ratio_undefined_reason"] = (
                    {"code": "fr_brr_undef_zero"}
                    if ref_base == 0 else {"code": "fr_brr_undef_other"}
                )

            # Model kaynaklı mı, veri kaynaklı mı? DI ile taban oranı oranı birbirine
            # yakınsa seçim farkı VERİDEN geliyordur; ayrışıyorsa MODELDEN.
            di, brr = cmp.get("disparate_impact"), cmp.get("base_rate_ratio")
            if di is not None and brr is not None:
                cmp["selection_gap_vs_base"] = round(di - brr, 4)
        else:
            for metric in ("rmse", "mae"):
                a, b = m.get(metric), ref_m.get(metric)
                cmp[f"{metric}_diff"] = round(a - b, 4) if (a is not None and b is not None) else None

        cmp["n"] = m.get("n")
        out["groups"][name] = cmp

    return out
