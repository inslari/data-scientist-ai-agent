from __future__ import annotations

"""
Aksiyon Yoğunlaşması (Action Concentration) — D3
================================================
Soru: modelin AKSİYON aldığı en riskli %10 (top-decile) tek bir gruba mı yığılmış?

Bu ADALET DEĞİL, DAĞILIM gözlemidir. Fairness demografik gruplar arası performans
paritesini sorar; burada soru "modelin işleme aldığı kişiler kim" — grup demografik
olmak zorunda değil (loan_grade tamamen geçerli). Bu yüzden kolon seçimi fairness'ın
demografik carrier'ından AYRI (segment_carrier.select_concentration_columns).

METRİK (grup→decile, tercih edilen tanım)
-----------------------------------------
    rep_ratio = grubun top-decile payı / grubun popülasyon payı
Baseline 1.0× (orantılı). >1 over-temsil, <1 under-temsil. (Alternatif "decile ne
kadar homojen" Herfindahl tek sayı verir ama aksiyona bağlanmaz — reddedildi.)

YÖN NETLİĞİ — yüksek yığılma her zaman kötü DEĞİL
-------------------------------------------------
Model gerçekten riskli bir segmenti buluyor olabilir. Ayrımı ``base_rate_ratio``
(grubun gerçek pozitif oranı / popülasyon prevalansı) ile YAN YANA koyarak yaparız —
NEDENSEL bağ kurmadan (base_rate yığılmanın "sebebi" değil; iki gözlem hizalı ya da
değil). İşaret-uyuşmazlığı flag kuralı (büyüklük değil işaret; decile ekstrem kuyruk
olduğu için riskli grup zaten süper-lineer over-temsil edilir → salt oran yanıltıcı):

  beklenen konsantrasyon : rep yüksek + base yüksek → iki gözlem hizalı        → FLAG YOK
  beklenmedik over-konsant: rep yüksek + base ≤ 1   → risk yokken yığılma       → info
  kapsama boşluğu         : rep düşük  + base yüksek → riskli grubu kaçırıyor    → warn adayı
  orantılı / diğer        :                                                     → FLAG YOK

İKİ AYRI KAPI
-------------
  1. ÖLÇÜLÜR MÜ (n-guard): grup pop_count ≥ min_n; decile n_top ≥ min_n. Aksi →
     görünür sebeple atlanır (sessiz eleme yok).
  2. FLAG'LENİR Mİ (CI kapısı): rep gerçek olmalı — decile_share Wilson aralığı
     pop_share'ı DIŞLAMALI (nokta tahmin gürültüsü band dışına savurmasın, D1 deseni).
     Kapsama boşluğu warn'a ancak base_rate de anlamlıysa (base CI prevalansı dışlar)
     çıkar; aksi halde info kalır — model o grubu decile DIŞINDA yakalıyor olabilir.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.logging_config import get_logger
from src.services.fairness import wilson_interval

logger = get_logger(__name__)

# Orantılılık bandı — dörtte-beş ile AYNI değer (tutarlılık); referans, hüküm değil.
PROPORTIONALITY_BAND = (0.8, 1.25)


def _ci_excludes(ci: Optional[dict], point: float) -> Optional[int]:
    """CI ``point``'i dışlıyor mu? +1 (tamamı üstünde), -1 (tamamı altında), 0 (içeriyor),
    None (CI yok). Nokta tahmininin gürültü mü gerçek mi olduğunu söyler."""
    if not ci or ci.get("lower") is None or ci.get("upper") is None:
        return None
    if ci["lower"] > point:
        return 1
    if ci["upper"] < point:
        return -1
    return 0


def compute_action_concentration(
    prob_pos: np.ndarray,
    raw_test: Optional[pd.DataFrame],
    conc_cols: list[str],
    y_true: np.ndarray,
    *,
    decile: float = 0.10,
    skipped_columns: Optional[dict] = None,
) -> dict:
    """Top-decile grup dağılımı — HAM segment değerleri üzerinden (raw_test carrier).

    prob_pos/y_true test setine, raw_test pozisyonel olarak X_test'e hizalı (carrier
    invariant'ı). Sıralama ``_top_decile_lift`` ile AYNI (argsort(-prob)). Yeni model/
    feature YOK — mevcut tahmin + mevcut ham kolonlar.
    """
    lo, hi = PROPORTIONALITY_BAND
    min_n = settings.FAIRNESS_MIN_GROUP_SIZE
    out: dict = {
        "status": "not_computed", "reason": "", "decile": decile,
        "band": [lo, hi], "columns": {},
        "skipped_columns": dict(skipped_columns or {}),
    }

    if raw_test is None or not len(raw_test) or not conc_cols:
        out["reason"] = {"code": "ac_no_columns"}
        return out
    if prob_pos is None or len(prob_pos) == 0:
        out["reason"] = {"code": "ac_no_probs"}
        return out
    if len(raw_test) != len(prob_pos) or len(y_true) != len(prob_pos):
        # Hizalama savunması — sessiz yanlış-atıf yerine reddet (fairness deseni).
        out["reason"] = {"code": "ac_misaligned",
                         "params": {"r": len(raw_test), "p": len(prob_pos)}}
        return out

    prob_pos = np.asarray(prob_pos, dtype=float)
    y_true = np.asarray(y_true)
    n_total = len(prob_pos)
    n_top = max(1, int(round(decile * n_total)))
    if n_top < min_n:
        out["reason"] = {"code": "ac_thin_decile", "params": {"n": n_top, "min": min_n}}
        return out

    prevalence = float((y_true == 1).mean()) if n_total else 0.0
    prev_pos = int((y_true == 1).sum())
    top_mask = np.zeros(n_total, dtype=bool)
    top_mask[np.argsort(-prob_pos)[:n_top]] = True

    columns: dict = {}
    for col in conc_cols:
        if col not in raw_test.columns:
            continue
        series = raw_test[col]
        raw = series.astype(str).to_numpy()
        values = sorted(series.dropna().astype(str).unique())
        groups: dict = {}
        skipped_groups: dict = {}

        for value in values:
            grp = (raw == value)
            pop_count = int(grp.sum())
            if pop_count < min_n:
                skipped_groups[value] = {"code": "ac_thin_group",
                                         "params": {"n": pop_count, "min": min_n}}
                continue
            decile_count = int((grp & top_mask).sum())
            pop_share = pop_count / n_total
            decile_share = decile_count / n_top
            rep_ratio = (decile_share / pop_share) if pop_share else None

            g_pos = int((y_true[grp] == 1).sum())
            base_rate = g_pos / pop_count
            base_ratio = (base_rate / prevalence) if prevalence else None

            dshare_ci = wilson_interval(decile_count, n_top)
            base_ci = wilson_interval(g_pos, pop_count)
            # rep_ratio güven aralığı = decile_share CI / pop_share (pop_share sabit).
            rep_ci = ({"lower": round(dshare_ci["lower"] / pop_share, 4),
                       "upper": round(dshare_ci["upper"] / pop_share, 4),
                       "method": "wilson"} if (dshare_ci and pop_share) else None)

            rep_side = _ci_excludes(dshare_ci, pop_share)          # decile_share vs pop_share
            base_side = _ci_excludes(base_ci, prevalence)          # base_rate vs prevalence

            kind, severity = _classify(rep_ratio, base_ratio, rep_side, base_side, lo, hi)
            groups[value] = {
                "n_pop": pop_count, "n_decile": decile_count,
                "pop_share": round(pop_share, 4), "decile_share": round(decile_share, 4),
                "rep_ratio": round(rep_ratio, 4) if rep_ratio is not None else None,
                "rep_ci": rep_ci,
                "base_rate": round(base_rate, 4),
                "base_rate_ratio": round(base_ratio, 4) if base_ratio is not None else None,
                "base_rate_ci": base_ci,
                "kind": kind, "flag_severity": severity,
            }

        if groups:
            columns[col] = {"groups": groups, "skipped_groups": skipped_groups}
        else:
            columns[col] = {"groups": {}, "skipped_groups": skipped_groups,
                            "reason": {"code": "ac_no_groups"}}

    if not any((c.get("groups") for c in columns.values())):
        out["reason"] = {"code": "ac_no_groups"}
        out["columns"] = columns
        return out

    out.update({
        "status": "ok", "reason": "", "n_top": n_top, "n_total": n_total,
        "prevalence": round(prevalence, 4), "total_positives": prev_pos,
        "columns": columns,
    })
    return out


def _classify(rep_ratio, base_ratio, rep_side, base_side, lo, hi):
    """(kind, flag_severity) — işaret-uyuşmazlığı + CI kapısı. flag_severity None=flag yok."""
    if rep_ratio is None or base_ratio is None:
        return "undefined", None
    over = rep_ratio >= hi
    under = rep_ratio <= lo
    base_high = base_ratio >= hi
    base_low = base_ratio <= 1.0

    # Beklenen konsantrasyon: iki gözlem hizalı (over + yüksek taban riski) → FLAG YOK.
    if over and base_high:
        return "expected", None
    # Beklenmedik over-konsantrasyon: yığılma var ama grup riskli değil. CI gerçek olmalı.
    if over and base_low and rep_side == 1:
        return "over_unexpected", "info"
    # Kapsama boşluğu: riskli grup decile'da az. rep gerçekten düşük olmalı (CI kapısı).
    if under and base_high and rep_side == -1:
        # warn'a ancak base_rate de anlamlıysa (ikisi de net) çıkar; aksi info.
        return ("coverage_gap", "warn") if base_side == 1 else ("coverage_gap", "info")
    return "proportional" if (not over and not under) else "aligned_or_weak", None
