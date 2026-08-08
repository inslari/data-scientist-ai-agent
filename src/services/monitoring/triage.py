"""Katman 2 Faz 0 — TRIAGE roll-up (`overall_status`). SAF (DB yok, deterministik, tek başına testli).

Dört monitörü — drift(PSI) · tahmini(CBPE) · gerçekleşen(realized) · veri kalitesi — TEK rozete indirger:
``healthy | warning | critical | unknown``. Aggregator pencere hesabının SONUNDA çağırır; endpoint HESAPLAMAZ.

═══════════════════════════════════════════════════════════════════════════════════════════════════════
TRIAGE ↔ "ÇIPLAK SAYI YOK" DİSİPLİNİ — UZLAŞMA (UI turuna referans)
═══════════════════════════════════════════════════════════════════════════════════════════════════════
H2O'nun "5 saniyede sağlık" çipi ile bizim "çıplak sayı yok" disiplinimiz ÇELİŞMEZ:
  * Çip bir SAYI DEĞİL, DETERMİNİSTİK DURUM — burada hesaplanır, endpoint aynen taşır, dashboard yeniden
    yorumlamaz.
  * ``unknown`` DEĞERİ sayesinde çip ölçülemeyeni yeşil GÖSTERMEZ → "temiz ≠ ölçülmedi" rozete taşınır.
    Çekirdek sinyaller (realized/cbpe/drift) hiçbiri ölçülemezse rozet DÜRÜSTÇE unknown (yeşil asla).
  * Çipin yanındaki her sayı provenance taşır (n + gate + assumes). Çip triage; sayılar kanıt. Ayrı katman.

ROLL-UP — ÜÇ ADIM (naif max DEĞİL, sinyal-farkında):
  1. Eksen-başına şiddet: her eksen → healthy/warning/critical/**unmeasured**. Tavanlar:
       realized → critical üretebilen TEK eksen (ölçülmüş gerçek, en güçlü). CBPE → warning TAVANI (belirsiz
       tahmin + concept-drift'te iyimser → asla kırmızı). quality → warning tavanı. drift → critical mümkün.
  2. ÇAPRAZ UZLAŞMA — TEK YÖN (yalnız AŞAĞI): realized GÜVENİLİR (gate=ok ∧ coverage≥trust) ∧ healthy ise
       öncü göstergeleri bastırır (drift critical→warning, cbpe warning→healthy). Gerçek-doğru öncüyü ezer →
       "realized yeşil + PSI kırmızı" çelişik rozeti önlenir. EZİLEN ham şiddet ``suppressed_axes``'ta KORUNUR
       (dashboard "kritik ama ölçülen etki yok" diyebilsin — bilgi gizlenmez, yalnız rozete yansımaz).
       ⚠️ INVARIANT: bastırma realized'ın AYNI PENCEREDE ölçülmüş olmasını gerektirir. Şemada actual_label o
       pencerenin tahminlerine yazılır → yapısal olarak aynı-pencere. İleride "trailing realized" (başka
       pencerenin etiketleri) eklenirse bu ezme SESSİZCE yanlış olur → bastırma yalnız ``r_trusted``
       (aynı-pencere ok+coverage) ile tetiklenir; test bunu çitler.
  3. Birleştir + UNKNOWN TABANI: en az bir çekirdek eksen ölçüldüyse → ölçülenlerin MAX şiddeti. Üç çekirdek
       eksen de unmeasured → ``unknown`` (model sağlığı cevaplanamaz; quality tek başına yeşil YAPMAZ). Quality
       alarmı o durumda ``code=all_core_unmeasured_with_quality_alarm`` ile görünür (UI unknown çipin YANINA
       ayrı kalite uyarısı basar — göz ardı edilmesin).

⚠️ EŞİKLER (config `settings`, tek blok): PSI 0.10/0.25 LİTERATÜR; realized/cbpe/coverage/quality/rmse İCAT
   → shadow koşusu verisiyle KALİBRE edilecek (config.py yorumları damgalı).
"""
from __future__ import annotations

_ORDER = ("healthy", "warning", "critical")
_RANK = {s: i for i, s in enumerate(_ORDER)}


def _realized_axis(gate, metrics, coverage, s):
    """(severity|'unmeasured', trusted, detail). GÜVENİLİR = gate=ok ∧ coverage≥trust — yanlı sayı (düşük
    kapsam, seçilim yanlılığı) badge'i sürüklemesin. Sınıflandırma accuracy/roc; regresyon göreli rmse artışı."""
    if gate != "ok" or not metrics:
        return "unmeasured", False, {"gate": gate}
    if coverage is None or coverage < s.MONITORING_LABEL_COVERAGE_TRUST:
        return "unmeasured", False, {"reason": "coverage_below_trust", "label_coverage": coverage}
    acc, bacc = metrics.get("accuracy"), metrics.get("baseline_accuracy")
    roc, broc = metrics.get("roc_auc"), metrics.get("baseline_roc_auc")
    rmse, brmse = metrics.get("rmse"), metrics.get("baseline_rmse")
    if acc is not None and bacc is not None:
        drop, mu = bacc - acc, "accuracy"
        crit, warn = s.MONITORING_REALIZED_DROP_CRITICAL, s.MONITORING_REALIZED_DROP_WARN
    elif roc is not None and broc is not None:
        drop, mu = broc - roc, "roc_auc"
        crit, warn = s.MONITORING_REALIZED_DROP_CRITICAL, s.MONITORING_REALIZED_DROP_WARN
    elif rmse is not None and brmse is not None and brmse > 0:
        drop, mu = (rmse - brmse) / brmse, "rmse_rel"        # regresyon: göreli artış (0-1 skalası uymaz)
        crit, warn = s.MONITORING_REALIZED_RMSE_INCREASE_CRITICAL, s.MONITORING_REALIZED_RMSE_INCREASE_WARN
    else:
        return "unmeasured", False, {"reason": "no_comparable_metric"}
    sev = "critical" if drop >= crit else "warning" if drop >= warn else "healthy"
    return sev, True, {"metric": mu, "drop": round(drop, 6)}


def _cbpe_axis(gate, est, s):
    """(severity|'unmeasured', detail). TAVAN warning — belirsiz tahmin + concept-drift iyimserliği → asla critical."""
    if gate != "ok" or not est:
        return "unmeasured", {"gate": gate}
    ea, ba = est.get("estimated_accuracy"), est.get("baseline_accuracy")
    er, br = est.get("estimated_roc_auc"), est.get("baseline_roc_auc")
    if ea is not None and ba is not None:
        drop, mu = ba - ea, "accuracy"
    elif er is not None and br is not None:
        drop, mu = br - er, "roc_auc"
    else:
        return "unmeasured", {"reason": "no_comparable_metric"}
    sev = "warning" if drop >= s.MONITORING_CBPE_DROP_WARN else "healthy"    # TAVAN warning
    return sev, {"metric": mu, "drop": round(drop, 6)}


def _drift_axis(psi_status, psi_max, attr_ratio, degen_mass, s):
    """(severity|'unmeasured', coverage_adequate, detail). ÖLÇÜM KAPSAMI eksikse (attr düşük / degen kütle
    yüksek) drift healthy OLAMAZ → warning tabanı (modelin dörtte biri ölçülemezken 'sağlıklı' yanıltıcı)."""
    if psi_status != "ok" or psi_max is None:
        return "unmeasured", None, {"psi_status": psi_status}
    if psi_max >= s.MONITORING_PSI_CRITICAL:
        sev = "critical"
    elif psi_max >= s.MONITORING_PSI_WARN:
        sev = "warning"
    else:
        sev = "healthy"
    adequate = True
    if attr_ratio is not None and attr_ratio < s.MONITORING_DRIFT_ATTR_MIN:
        adequate = False
    if degen_mass is not None and degen_mass > s.MONITORING_DEGEN_MASS_MAX:
        adequate = False
    if not adequate and sev == "healthy":
        sev = "warning"                                       # kısmi ölçümde healthy olamaz
    return sev, adequate, {"psi_max": psi_max, "coverage_adequate": adequate}


def _quality_axis(quality, s):
    """(severity|'unmeasured', detail). TAVAN warning. alarm_eligible değilse (n<min) unmeasured."""
    if not quality or not quality.get("alarm_eligible"):
        return "unmeasured", {"alarm_eligible": bool(quality and quality.get("alarm_eligible"))}
    ur = float((quality.get("unseen") or {}).get("rate") or 0.0)
    oor = float((quality.get("out_of_range") or {}).get("rate") or 0.0)
    if ur >= s.MONITORING_QUALITY_ALARM_UNSEEN_RATE or oor >= s.MONITORING_QUALITY_ALARM_OOR_RATE:
        return "warning", {"unseen_rate": ur, "out_of_range_rate": oor}
    return "healthy", {"unseen_rate": ur, "out_of_range_rate": oor}


def _code_for(driver, status, suppressed):
    """Makine anahtarı (i18n — dashboard TR/EN'e çevirir; prose DEĞİL)."""
    if status == "healthy":
        return "all_measured_healthy"
    if driver == "realized":
        return "realized_drop_critical" if status == "critical" else "realized_drop_warning"
    if driver == "drift":
        if status == "critical":
            return "psi_high_impact_critical"
        return "psi_drift_no_impact_warning" if suppressed.get("drift") else "psi_drift_warning"
    if driver == "cbpe":
        return "cbpe_drop_warning"
    if driver == "quality":
        return "quality_alarm_warning"
    return "unknown"


def compute_overall_status(*, psi_status, psi_max, importance_attributed_ratio, degenerate_importance_mass,
                           cbpe_gate_status, cbpe_estimate, realized_gate_status, realized_metrics,
                           label_coverage, quality_rates, settings) -> "tuple[str, dict]":
    """Dört monitör değerinden (status, reason) üretir. reason = {driver, code, value, core_unmeasured,
    suppressed_axes, measurement_coverage}."""
    s = settings
    r_sev, r_trusted, r_detail = _realized_axis(realized_gate_status, realized_metrics, label_coverage, s)
    c_sev, c_detail = _cbpe_axis(cbpe_gate_status, cbpe_estimate, s)
    d_sev, d_adequate, d_detail = _drift_axis(psi_status, psi_max, importance_attributed_ratio,
                                              degenerate_importance_mass, s)
    q_sev, q_detail = _quality_axis(quality_rates, s)

    # ── Adım 2: çapraz uzlaşma (TEK YÖN, yalnız aşağı; ham şiddet suppressed_axes'ta korunur) ──
    suppressed: dict = {}
    if r_trusted and r_sev == "healthy":
        if d_sev == "critical":
            suppressed["drift"] = "critical"
            d_sev = "warning"                                 # gerçek-doğru öncü göstergeyi ezer
        if c_sev == "warning":
            suppressed["cbpe"] = "warning"
            c_sev = "healthy"

    coverage_block = {"importance_attributed_ratio": importance_attributed_ratio,
                      "degenerate_importance_mass": degenerate_importance_mass, "adequate": d_adequate}
    core = {"realized": r_sev, "cbpe": c_sev, "drift": d_sev}
    all_axes = {**core, "quality": q_sev}
    measured = {a: sev for a, sev in all_axes.items() if sev != "unmeasured"}
    detail_map = {"realized": r_detail, "cbpe": c_detail, "drift": d_detail, "quality": q_detail}

    # ── Adım 3: unknown tabanı — üç çekirdek eksen de unmeasured → model sağlığı cevaplanamaz ──
    if all(core[a] == "unmeasured" for a in core):
        q_alarm = (q_sev == "warning")
        code = "all_core_unmeasured_with_quality_alarm" if q_alarm else "all_core_unmeasured_unknown"
        return "unknown", {"driver": ("quality" if q_alarm else None), "code": code,
                           "value": (q_detail if q_alarm else {}), "core_unmeasured": True,
                           "suppressed_axes": suppressed, "measurement_coverage": coverage_block}

    # ── birleştir: ölçülenlerin MAX şiddeti; driver = tie-break öncelik realized>drift>cbpe>quality ──
    top = max(_RANK[sev] for sev in measured.values())
    status = _ORDER[top]
    driver = next(a for a in ("realized", "drift", "cbpe", "quality")
                  if a in measured and _RANK[measured[a]] == top)
    return status, {"driver": driver, "code": _code_for(driver, status, suppressed),
                    "value": (detail_map.get(driver) or {}), "core_unmeasured": False,
                    "suppressed_axes": suppressed, "measurement_coverage": coverage_block}
