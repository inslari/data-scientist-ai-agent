"""Deterministik risk/WARN listesi (Faz 2 Grup B, #6).

İLKE: risk listesi ÖLÇMEZ, mevcut kanıtları TOPLAR. Her madde raporda zaten hesaplanmış
bir sayıya işaret eder — yeni metrik icat etmez, LLM'e sormaz. LLM yalnız anlatıya çevirir,
madde ekleyemez (statü rozeti dersi).

KONUM-TABANLI WARN≠FAIL GARANTİSİ: bu modül rapor-zamanında (execution'dan SONRA) çalışır,
hiçbir kontrol-akışı yetkisi yoktur → pipeline'ı ASLA durduramaz. Durdurma yalnız model-öncesi
`data_quality_status=failed` (veri bütünlüğü) ile olur. Model-kalitesi riskleri en fazla warn.
KODDA ASSERT: `critical` severity yalnız veri bütünlüğünden (source="data_quality") gelebilir.

"TEMİZ" ≠ "ÖLÇÜLMEDİ": her kontrol ran/not-run izler (rule 3, ran_detectors deseni). Koşmayan
kontrol temiz SAYILMAZ. Fairness #12 gelene kadar not-run ("n/CI verisi yok").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.services.fairness import wilson_interval, intervals_overlap
from src.core.report_i18n import R_REASON
from src.core.report_schema import _fmt_value  # Faz 3 #4: not metnindeki oranlara ondalık disiplini
from src.core.logging_config import get_logger

logger = get_logger(__name__)

# ── Eşikler (türetilmiş / açıkça-operasyonel; sihirli sabit yok) ─────────────
_CONFIDENT_ERROR_FLOOR = 10       # mutlak taban: "5 hatanın 2'sine alarm yok" (sabit taban)
_CONFIDENT_ERROR_RATE = 0.01      # severity SÜRÜCÜSÜ: confident_errors / n_total (toplam tahminin
                                  #   payı) — operasyonel eşik, ayarlanabilir. Hataların payı (_pct)
                                  #   yalnız AÇIKLAYICI, severity'yi ondan türetme.
_RECALL_PARITY_RATIO = 0.8        # yaygın kullanılan 0.8 oranı (REGÜLASYON İDDİASI YOK — sınıflar
                                  #   arası recall için; gerçek gruplar-arası karşılaştırma #12'de).
_BRIER_MARGINAL = 0.9             # brier baseline'ın %90'ının üstündeyse "zar zor geçiyor" (info)

_SEVERITY = {"info": 0, "warn": 1, "critical": 2}


@dataclass
class RiskCheck:
    id: str
    ran: bool
    risk: bool = False
    severity: str = "info"            # info | warn | critical
    source: str = "model"             # "model" | "data_quality" (critical YALNIZ data_quality)
    value: Any = None
    detail: dict = field(default_factory=dict)   # {"tr":..., "en":...}
    not_run_reason: dict = field(default_factory=dict)


def _d(tr: str, en: str) -> dict:
    return {"tr": tr, "en": en}


# ── Tekil kontroller: her biri mevcut bir sayıya bağlanır ────────────────────
def _check_severe_imbalance(ev: dict) -> RiskCheck:
    cb = ev.get("class_balance") or {}
    if not cb.get("distribution"):
        return RiskCheck("severe_imbalance", ran=False,
                         not_run_reason=_d("sınıf dengesi hesaplanmadı", "class balance not computed"))
    lvl = cb.get("level")
    if lvl == "severe":
        mr = cb.get("minority_rate")
        return RiskCheck("severe_imbalance", ran=True, risk=True, severity="warn", value=mr,
                         detail=_d(f"Sınıf dağılımı ciddi dengesiz (azınlık %{round((mr or 0)*100,2):g}); "
                                   f"ROC bu rejimde yanıltıcı, PR-AUC birincil metrik.",
                                   f"Severely imbalanced (minority {round((mr or 0)*100,2):g}%); "
                                   f"ROC is misleading here, PR-AUC is primary."))
    return RiskCheck("severe_imbalance", ran=True, risk=False, value=lvl)


def _class_recall_ci(cm: dict, cls) -> Optional[tuple]:
    """(recall, wilson_ci) — confusion matrix'ten (tp, actual_support). Sayı uydurma yok."""
    matrix = cm.get("matrix") or []
    labels = cm.get("labels") or []
    idx = {str(l): i for i, l in enumerate(labels)}
    i = idx.get(str(cls))
    if i is None or not matrix:
        return None
    tp = matrix[i][i]
    support = sum(matrix[i])
    if not support:
        return None
    return tp / support, wilson_interval(tp, support)


def _check_minority_recall(ev: dict) -> RiskCheck:
    cb = ev.get("class_balance") or {}
    cm = ev.get("confusion_matrix") or {}
    interest = cb.get("interest_classes") or []
    majority = cb.get("majority_class")
    if not interest or majority is None or not cm.get("matrix"):
        return RiskCheck("minority_recall", ran=False,
                         not_run_reason=_d("sınıf-başı recall/CI yok", "per-class recall/CI unavailable"))
    maj = _class_recall_ci(cm, majority)
    if not maj:
        return RiskCheck("minority_recall", ran=False,
                         not_run_reason=_d("çoğunluk recall'ı okunamadı", "majority recall unreadable"))
    maj_recall, maj_ci = maj
    worst = None
    for cls in interest:
        if cls == majority:
            continue
        got = _class_recall_ci(cm, cls)
        if not got:
            continue
        rec, ci = got
        ratio = (rec / maj_recall) if maj_recall else 1.0
        # n/CI KAPISI: yalnız CI'lar ÖRTÜŞMÜYORSA (ilgi sınıfı belirgin düşük) risk.
        ci_ok = bool(ci and maj_ci and ci.get("upper") is not None
                     and maj_ci.get("lower") is not None
                     and ci["upper"] < maj_ci["lower"])
        if ratio < _RECALL_PARITY_RATIO and ci_ok and (worst is None or ratio < worst[1]):
            worst = (cls, ratio, rec, maj_recall)
    if worst is None:
        return RiskCheck("minority_recall", ran=True, risk=False, value=maj_recall)
    cls, ratio, rec, maj_recall = worst
    return RiskCheck("minority_recall", ran=True, risk=True, severity="warn", value=round(ratio, 3),
                     detail=_d(f"İlgi sınıfının yakalama oranı ({round(rec,4)}) çoğunluğunkinden "
                               f"({round(maj_recall,4)}) belirgin düşük (oran {round(ratio,3)} < "
                               f"{_RECALL_PARITY_RATIO}, güven aralıkları örtüşmüyor). Toplam skor bunu gizler.",
                               f"The interest class's recall ({round(rec,4)}) is clearly below the "
                               f"majority's ({round(maj_recall,4)}) (ratio {round(ratio,3)} < "
                               f"{_RECALL_PARITY_RATIO}, non-overlapping intervals). The overall score hides this."))


def _check_confident_errors(ev: dict) -> RiskCheck:
    err = ev.get("error_analysis") or {}
    conf = err.get("confidence_analysis") or {}
    n_total = err.get("n_total")
    hce = conf.get("high_confidence_errors")
    if hce is None or not n_total:
        return RiskCheck("confident_errors", ran=False,
                         not_run_reason=_d("güven analizi yok (olasılık üretilemedi)",
                                           "confidence analysis unavailable (no probabilities)"))
    rate = hce / n_total
    pct_of_errors = conf.get("high_confidence_errors_pct")   # AÇIKLAYICI, severity sürücüsü DEĞİL
    # Severity SÜRÜCÜSÜ: toplam tahminin payı (rate). Mutlak taban + oran birlikte.
    if hce >= _CONFIDENT_ERROR_FLOOR and rate >= _CONFIDENT_ERROR_RATE:
        return RiskCheck("confident_errors", ran=True, risk=True, severity="warn", value=hce,
                         detail=_d(f"{hce} tahmin (toplamın %{round(rate*100,2):g}'i) yüksek güvenle "
                                   f"YANLIŞ — model emin olduğu için kimse ikinci kez bakmıyor "
                                   f"(hataların %{pct_of_errors}'i). Bunlara bakın.",
                                   f"{hce} predictions ({round(rate*100,2):g}% of all) are confidently "
                                   f"WRONG — the model is sure, so nobody double-checks them "
                                   f"({pct_of_errors}% of errors). Look at these."))
    return RiskCheck("confident_errors", ran=True, risk=False, value=hce)


def _check_overfitting(ev: dict) -> RiskCheck:
    ovf = ev.get("overfitting") or {}
    st = ovf.get("status")
    if not st:
        return RiskCheck("overfitting", ran=False,
                         not_run_reason=_d("genelleme analizi yok", "generalization analysis unavailable"))
    if st in ("mild_overfitting", "overfitting_detected"):
        gap = ovf.get("gap")
        return RiskCheck("overfitting", ran=True, risk=True, severity="warn", value=gap,
                         detail=_d(f"CV−Test farkı ({gap}) genelleme zayıflığına işaret ediyor "
                                   f"(durum: {st}). Yeni veride skor düşebilir.",
                                   f"The CV−Test gap ({gap}) points to weak generalization "
                                   f"(status: {st}). The score may drop on new data."))
    # normal / inconclusive (CI içinde, gürültü) / underfit → risk değil
    return RiskCheck("overfitting", ran=True, risk=False, value=st)


def _check_brier(ev: dict) -> RiskCheck:
    cal = ev.get("calibration") or {}
    brier = cal.get("brier_score")
    base = cal.get("brier_baseline")
    if brier is None or base is None:
        return RiskCheck("brier_vs_baseline", ran=False,
                         not_run_reason=_d("Brier/baseline yok", "Brier/baseline unavailable"))
    if brier >= base:
        return RiskCheck("brier_vs_baseline", ran=True, risk=True, severity="warn", value=brier,
                         detail=_d(f"Olasılık kalitesi (Brier {brier}) taban çizgisini ({base}) "
                                   f"geçemiyor — tahminler taban orandan iyi değil.",
                                   f"Probability quality (Brier {brier}) does not beat the baseline "
                                   f"({base}) — estimates are no better than the base rate."))
    if brier >= _BRIER_MARGINAL * base:
        return RiskCheck("brier_vs_baseline", ran=True, risk=True, severity="info", value=brier,
                         detail=_d(f"Olasılık kalitesi (Brier {brier}) tabanı ({base}) zar zor geçiyor.",
                                   f"Probability quality (Brier {brier}) barely beats the baseline ({base})."))
    return RiskCheck("brier_vs_baseline", ran=True, risk=False, value=brier)


def _check_leakage(state: dict, ev: dict) -> RiskCheck:
    domain = state.get("domain_leakage_report") or {}
    fi = ev.get("feature_importance_summary") or {}
    domain_ran = "domain_leakage_report" in state
    importance_ran = bool(fi)
    if not (domain_ran or importance_ran):
        return RiskCheck("leakage", ran=False,
                         not_run_reason=_d("sızıntı kontrolleri koşmadı", "leakage checks did not run"))
    concerns = domain.get("concerns") or []
    tlock = ev.get("target_lock_leakage") or []
    suspects = fi.get("leakage_suspects") or []
    total = len(concerns) + len(tlock) + len(suspects)
    if total:
        return RiskCheck("leakage", ran=True, risk=True, severity="warn", value=total,
                         detail=_d(f"{total} sızıntı işareti bulundu (Sızıntı ve Riskler bölümüne bakın). "
                                   f"Sızıntı test skorunu şişirebilir.",
                                   f"{total} leakage flags were raised (see the Leakage & Risks section). "
                                   f"Leakage can inflate the test score."))
    return RiskCheck("leakage", ran=True, risk=False, value=0)


def _check_near_duplicates(state: dict) -> RiskCheck:
    rel = state.get("relations_report") or {}
    if not rel or "redundant_pairs" not in rel:
        return RiskCheck("near_duplicates", ran=False,
                         not_run_reason=_d("ilişki analizi yok", "relations analysis unavailable"))
    pairs = rel.get("redundant_pairs") or []
    if pairs:
        return RiskCheck("near_duplicates", ran=True, risk=True, severity="info", value=len(pairs),
                         detail=_d(f"{len(pairs)} near-duplicate özellik çifti — bilgi tekrarı "
                                   f"(Veri İlişkileri bölümü).",
                                   f"{len(pairs)} near-duplicate feature pairs — redundant information "
                                   f"(Data Relations section)."))
    return RiskCheck("near_duplicates", ran=True, risk=False, value=0)


def _check_normalization_collisions(state: dict) -> RiskCheck:
    fr = state.get("feature_report") or {}
    if "normalization_collisions" not in fr:
        return RiskCheck("normalization_collisions", ran=False,
                         not_run_reason=_d("normalizasyon bilgisi yok", "normalization info unavailable"))
    cols = fr.get("normalization_collisions") or []
    if cols:
        names = [c.get("normalized_name") for c in cols]
        return RiskCheck("normalization_collisions", ran=True, risk=True, severity="warn", value=len(cols),
                         detail=_d(f"{len(cols)} kolon adı çakışması: farklı-değerli kolonlar aynı normalize "
                                   f"ada düşüyor ({names}) — şema/isim sorunu (veri korundu, silinmedi).",
                                   f"{len(cols)} column-name collisions: distinct-valued columns map to the "
                                   f"same normalized name ({names}) — a schema/naming issue (data preserved)."))
    return RiskCheck("normalization_collisions", ran=True, risk=False, value=0)


def _check_data_quality(state: dict) -> RiskCheck:
    status = state.get("data_quality_status")
    if not status:
        return RiskCheck("data_quality", ran=False, source="data_quality",
                         not_run_reason=_d("veri kalitesi statüsü yok", "data quality status unavailable"))
    if status == "failed":
        return RiskCheck("data_quality", ran=True, risk=True, severity="critical", source="data_quality",
                         value=status, detail=_d("Veri bütünlüğü hatası (data_quality=failed) — kritik.",
                                                 "Data integrity failure (data_quality=failed) — critical."))
    if status == "degraded":
        return RiskCheck("data_quality", ran=True, risk=True, severity="warn", source="data_quality",
                         value=status, detail=_d("Veri kalitesi düşük (degraded) — nedeni Veri Kalitesi bölümünde.",
                                                 "Degraded data quality — see the Data Quality section."))
    return RiskCheck("data_quality", ran=True, risk=False, source="data_quality", value=status)


def _check_fairness(state: dict) -> list[RiskCheck]:
    """Grup parite riski (#12) — fairness.py'nin bastığı sayıları TOPLAR, ölçmez.

    ÜÇ KAPI birlikte (minority_recall deseni):
      1. Dörtte-beş: DI, fairness'ın taşıdığı referans bandı [0.8, 1.25] dışında mı?
         (Bu bölüm GERÇEK gruplar-arası karşılaştırma → 4/5 kuralı adıyla anılabilir;
         Grup B minority-recall'da anılamıyordu, orası sınıflar-arasıydı.)
      2. CI kapısı: grubun ve referansın SEÇİM ORANI aralıkları örtüşüyorsa DI gürültü
         → risk yok (intervals_overlap kesin False değilse geçilir — "aralık yok" da
         geçer, doğrulanamayan zeminde model suçlanmaz).
      3. Taban-oranı atfı: amplification = DI / base_rate_ratio (ölçek-BAĞIMSIZ; gap =
         DI−brr ölçeğe bağımlıydı ve asimetriyi kaçırıyordu). Aynı banda vurulur:
           amp band DIŞI  → model eşitsizliği BÜYÜTÜYOR → warn (model bulgusu)
           amp band İÇİ   → model sadık ama VERİDE eşitsizlik var → info (veri bulgusu,
                            sessiz geçmez — kullanıcının bilmesi gereken bir dağılım gerçeği)

    En fazla bir warn (en uç model) + bir info (en uç veri) döner — bounded, iki satır.
    Rule 3: fairness yoksa/koşamadıysa GERÇEK sebep yazılır (eski "n/CI yok" yalanı KALKTI).
    """
    ev = state.get("evaluation_report") or {}
    fair = ev.get("fairness")
    if not fair:
        return [RiskCheck("fairness", ran=False,
                          not_run_reason=_d("adalet analizi bu çalışmada üretilmedi",
                                            "fairness analysis was not produced in this run"))]
    if fair.get("status") != "ok":
        # GERÇEK sebep — fairness'ın dil-nötr kodunu iki dile çevir (rule 3).
        reason = fair.get("reason")
        return [RiskCheck("fairness", ran=False,
                          not_run_reason=_d(f"adalet analizi koşmadı: {R_REASON(reason, 'tr')}",
                                            f"fairness analysis did not run: {R_REASON(reason, 'en')}"))]
    if fair.get("task") != "classification":
        return [RiskCheck("fairness", ran=False,
                          not_run_reason=_d(
                              "grup parite kapısı (DI / dörtte-beş) ikili sınıflandırma için tanımlı; bu görev regresyon",
                              "the group-parity gate (DI / four-fifths) is defined for binary classification; this task is regression"))]

    lo, hi = fair.get("di_reference_band") or (0.8, 1.25)
    worst_model = None   # (col, g, di, brr, amp, fold)
    worst_data = None
    for col, seg in (fair.get("segments") or {}).items():
        groups = seg.get("groups") or {}
        comp = seg.get("comparisons") or {}
        ref = comp.get("reference_group")
        ref_ci = (groups.get(ref) or {}).get("selection_rate_ci")
        for g, c in (comp.get("groups") or {}).items():
            di, brr = c.get("disparate_impact"), c.get("base_rate_ratio")
            if di is None or brr is None or brr <= 0:
                continue
            if lo <= di <= hi:
                continue   # dörtte-beş içinde → parite tamam
            # CI kapısı: yalnız KESİN örtüşmeyen (ayırt-edilebilir) fark değerlendirilir.
            if intervals_overlap((groups.get(g) or {}).get("selection_rate_ci"), ref_ci) is not False:
                continue
            amp = round(di / brr, 4)
            fold = max(amp, 1.0 / amp) if amp > 0 else float("inf")
            if lo <= amp <= hi:
                if worst_data is None or fold > worst_data[5]:
                    worst_data = (col, g, di, brr, amp, fold)
            else:
                if worst_model is None or fold > worst_model[5]:
                    worst_model = (col, g, di, brr, amp, fold)

    out: list[RiskCheck] = []
    if worst_model is not None:
        col, g, di, brr, amp, _ = worst_model
        di_s, brr_s, amp_s = _fmt_value(di), _fmt_value(brr), _fmt_value(amp)  # Faz 3 #4: 3 hane
        out.append(RiskCheck(
            "fairness", ran=True, risk=True, severity="warn", value=amp,
            detail=_d(f"{col} = {g}: model grup eşitsizliğini BÜYÜTÜYOR. Seçim oranı oranı (DI {di_s}), "
                      f"verideki taban oranı farkını ({brr_s}×) aşıyor → büyütme {amp_s}× (dörtte-beş bandı "
                      f"{lo}–{hi} dışında, seçim oranı güven aralıkları örtüşmüyor). Bu, taban oranıyla "
                      f"AÇIKLANAMAYAN, model kaynaklı bir fark; bakılmaya değer.",
                      f"{col} = {g}: the model AMPLIFIES the group gap. The selection-rate ratio (DI {di_s}) "
                      f"exceeds the data's base-rate gap ({brr_s}×) → amplification {amp_s}× (outside the "
                      f"four-fifths band {lo}–{hi}, selection-rate intervals non-overlapping). This is a "
                      f"model-driven gap NOT explained by the base rate; worth examining.")))
    if worst_data is not None:
        col, g, di, brr, amp, _ = worst_data
        di_s, brr_s, amp_s = _fmt_value(di), _fmt_value(brr), _fmt_value(amp)  # Faz 3 #4: 3 hane
        out.append(RiskCheck(
            "fairness", ran=True, risk=True, severity="info", value=brr,
            detail=_d(f"{col} = {g}: modelin kendisi haksızlık EKLEMİYOR (büyütme {amp_s}×, dörtte-beş bandı "
                      f"{lo}–{hi} içinde) — ama VERİNİN kendisinde belirgin bir grup eşitsizliği var (taban "
                      f"oranı farkı {brr_s}×, DI {di_s}). Bu bir VERİ bulgusu, model bulgusu değil: fark verinin "
                      f"dağılımından geliyor.",
                      f"{col} = {g}: the model itself adds NO unfairness (amplification {amp_s}×, within the "
                      f"four-fifths band {lo}–{hi}) — but the DATA itself carries a marked group inequality "
                      f"(base-rate gap {brr_s}×, DI {di_s}). This is a DATA finding, not a model finding: the gap "
                      f"comes from the data's distribution.")))
    if not out:
        return [RiskCheck("fairness", ran=True, risk=False, value="parity_within_band")]
    return out


def _check_action_concentration(state: dict) -> list[RiskCheck]:
    """Aksiyon yoğunlaşması (D3) — top-decile grup dağılımından FLAG'leri toplar, ölçmez.

    action_concentration.py'nin işaret-uyuşmazlığı + CI-kapısı sonucunu okur:
      - kapsama boşluğu (riskli grup decile'da az, ikisi de CI-anlamlı) → warn (kaçan
        pozitif = doğrudan kayıp; over-konsantrasyondan operasyonel olarak ağır).
      - beklenmedik over-konsantrasyon / CI-belirsiz kapsama boşluğu → info.
      - beklenen konsantrasyon FLAG'LENMEZ (grup gerçekten riskli — model işini yapıyor).
    En fazla bir warn + bir info (bounded). ran=False → GERÇEK sebep (rule 3)."""
    ev = state.get("evaluation_report") or {}
    ac = ev.get("action_concentration")
    if not ac:
        return [RiskCheck("action_concentration", ran=False,
                          not_run_reason=_d("aksiyon yoğunlaşması bu çalışmada üretilmedi",
                                            "action concentration was not produced in this run"))]
    if ac.get("status") != "ok":
        return [RiskCheck("action_concentration", ran=False,
                          not_run_reason=_d(f"aksiyon yoğunlaşması koşmadı: {R_REASON(ac.get('reason'), 'tr')}",
                                            f"action concentration did not run: {R_REASON(ac.get('reason'), 'en')}"))]

    warns: list[tuple] = []
    infos: list[tuple] = []
    for col, cd in (ac.get("columns") or {}).items():
        for value, g in (cd.get("groups") or {}).items():
            sev = g.get("flag_severity")
            if sev == "warn":
                warns.append((col, value, g))
            elif sev == "info":
                infos.append((col, value, g))

    out: list[RiskCheck] = []
    if warns:
        col, value, g = max(warns, key=lambda t: t[2].get("base_rate_ratio") or 0)
        rep, base = g.get("rep_ratio"), g.get("base_rate_ratio")
        out.append(RiskCheck(
            "action_concentration", ran=True, risk=True, severity="warn", value=rep,
            detail=_d(f"{col} = {value}: veri-taban riski yüksek ({base}×) ama modelin en riskli "
                      f"%10'unda az temsil ediliyor ({rep}×, güven aralıkları anlamlı). Model bu riskli "
                      f"grubu aksiyon setinin DIŞINDA bırakıyor olabilir — kaçan pozitif doğrudan kayıptır. "
                      f"İncelenmeye değer. (Dağılım gözlemi; nedensellik iddiası değil.)",
                      f"{col} = {value}: high data base risk ({base}×) yet under-represented in the model's "
                      f"riskiest 10% ({rep}×, intervals significant). The model may be leaving this risky group "
                      f"OUTSIDE the action set — a missed positive is a direct loss. Worth examining. "
                      f"(A distribution observation; no causal claim.)")))
    if infos:
        col, value, g = max(infos, key=lambda t: (lambda r: max(r, 1.0 / r) if r else 1.0)(t[2].get("rep_ratio") or 1.0))
        rep, base, kind = g.get("rep_ratio"), g.get("base_rate_ratio"), g.get("kind")
        if kind == "over_unexpected":
            det = _d(f"{col} = {value}: model en riskli %10'una bu grubu orantısız alıyor ({rep}×) ama grubun "
                     f"veri-taban riski yüksek değil ({base}×) — beklenmedik yoğunlaşma (bir proxy'ye işaret "
                     f"edebilir). Bir dağılım gözlemi, kusur değil; nereye bakılmalı.",
                     f"{col} = {value}: the model over-selects this group into the riskiest 10% ({rep}×) though "
                     f"its data base risk is not high ({base}×) — an unexpected concentration (may point to a "
                     f"proxy). A distribution observation, not a defect; where to look.")
        else:  # coverage_gap, CI-belirsiz
            det = _d(f"{col} = {value}: veri-taban riski yüksek ({base}×) ve en riskli %10'da az temsil ({rep}×) "
                     f"ama fark istatistiksel olarak net değil — model bu grubu decile dışında yakalıyor olabilir. "
                     f"Bakılabilir.",
                     f"{col} = {value}: high data base risk ({base}×) and under-represented in the riskiest 10% "
                     f"({rep}×) but the difference is not statistically clear — the model may catch this group "
                     f"outside the decile. Worth a look.")
        out.append(RiskCheck("action_concentration", ran=True, risk=True, severity="info", value=rep, detail=det))
    if not out:
        return [RiskCheck("action_concentration", ran=True, risk=False, value="no_unexpected_concentration")]
    return out


def build_risk_list(state: dict) -> dict:
    """Deterministik risk listesi — tüm kaynakları toplar, ölçmez. Rapor-zamanı, kontrol yetkisi yok."""
    ev = state.get("evaluation_report") or {}
    is_clf = state.get("intent", "") == "classification"

    checks: list[RiskCheck] = []
    # Sınıf-tabanlı kontroller yalnız classification (regresyon guard)
    if is_clf:
        checks += [_check_severe_imbalance(ev), _check_minority_recall(ev),
                   _check_confident_errors(ev), _check_brier(ev)]
    checks += [_check_overfitting(ev), _check_leakage(state, ev),
               _check_near_duplicates(state), _check_normalization_collisions(state),
               _check_data_quality(state)]
    checks += _check_fairness(state)   # LIST: aynı anda model-warn + veri-info mümkün
    if is_clf:
        checks += _check_action_concentration(state)   # LIST: kapsama-warn + over-info mümkün

    risks = [c for c in checks if c.ran and c.risk]
    clean = [c for c in checks if c.ran and not c.risk]
    not_run = [c for c in checks if not c.ran]

    # YAPISAL ASSERT: critical YALNIZ veri bütünlüğünden (data_quality). Model kalitesi asla critical.
    for r in risks:
        assert not (r.severity == "critical" and r.source != "data_quality"), (
            f"critical severity model kaynağından üretilemez: {r.id} ({r.source})")

    # En yüksek severity (sunum rozeti; YENİ STOP ÜRETMEZ — konum-tabanlı garanti)
    level = max((_SEVERITY[r.severity] for r in risks), default=-1)
    level_name = {2: "critical", 1: "warn", 0: "info", -1: "clean"}[level]
    risks.sort(key=lambda r: -_SEVERITY[r.severity])
    logger.info(f"[risk] {len(risks)} risk (level={level_name}), {len(clean)} temiz, {len(not_run)} koşmadı")
    return {"risks": risks, "clean": clean, "not_run": not_run, "level": level_name}
