from __future__ import annotations

"""
Report Sections
---------------
state → ReportSection (katmanlı rapor bölümleri). Katmanlı yapının PİLOT modülü
fairness'tır; diğer bölümler kademeli olarak buraya taşınacak.

İŞ BÖLÜMÜ (değişmez)
====================
    KOD  → evidence + status + not_computed_reason   (bu dosya)
    LLM  → headline + explanation                    (build_section_prose)

LLM'e verilen şemada (report_schema.LLM_SECTION_SCHEMA) ``evidence`` alanı YOKTUR;
sayı uydurmak yasak değil, İMKÂNSIZDIR. Bu dosyadaki fonksiyonlar kanıtı state'ten
deterministik kurar — hiçbiri LLM çağırmaz.
"""

from typing import Optional

from src.core.llm import get_llm
from src.core.config import settings
from src.core.feature_naming import humanize_feature, is_derived_feature


def _hf(name, lang, state):
    """humanize_feature + ratio_meta: state'teki feature_report.ratio_feature_meta'yı geçirir.
    Böylece ratio etiketleri (Faz 1 #4) her bölümde AYNI ve DOĞRU görünür — belirsiz string-parse
    yerine üretim metadata'sından. Metadata yoksa humanize belirsiz adı ham gösterir."""
    rm = (state.get("feature_report") or {}).get("ratio_feature_meta") or {}
    return humanize_feature(name, lang, rm)
from src.core.logging_config import get_logger
from src.core.report_i18n import R_REASON
from src.core.report_schema import (
    LLM_SECTION_SCHEMA,
    ORDER_DATA_QUALITY,
    ORDER_DATA_RELATIONS,
    ORDER_ERROR_ANALYSIS,
    ORDER_FAIRNESS,
    ORDER_FEATURES,
    ORDER_GENERALIZATION,
    ORDER_LEAKAGE_RISK,
    ORDER_PERFORMANCE,
    ORDER_RECOMMENDATIONS,
    ORDER_SUMMARY,
    Evidence,
    ReportSection,
    _fmt_value,
)
from src.services.fairness import wilson_interval

# Güçlü/Zayıf değerlendirmesi tüm detay bölümlerinden SONRA, önerilerden önce
# gelir — özet en üstte (yönetici), değerlendirme kanıtın ardında, öneriler sonda.
ORDER_ASSESSMENT = 85
from src.services.relations import fisher_z_interval

logger = get_logger(__name__)


# ── Fairness bölümü (pilot) ──────────────────────────────────────────────────

_FAIRNESS_TITLE = {"tr": "Adalet ve Gruplar Arası Farklar", "en": "Fairness & Group Differences"}

_DI_REFERENCE_NOTE = {
    "tr": ("Referans eşik: 0.8–1.25 bandı (4/5 kuralı) geleneksel olarak incelenmeye "
           "değer görülür. Bu hukuki bir başparmak kuralıdır, istatistiksel bir eşik "
           "değildir — otomatik bir yargıya dönüştürmüyoruz."),
    "en": ("Reference band: 0.8–1.25 (the 4/5ths rule) is conventionally treated as "
           "worth examining. It is a legal rule of thumb, not a statistical threshold — "
           "we do not turn it into an automatic verdict."),
}


# Fairness kanıt ETİKETLERİ ve notları — deterministik, rapor diline uyar.
# report_sections'ın kendi UI etiketleri (servis sebepleri değil) → yerel iki dilli
# dict (mevcut _FAIRNESS_TITLE/_DI_REFERENCE_NOTE deseni). {p} = "col = grup(ref)"
# ya da "col: grup / ref" gövdesi; burada yalnız METRİK etiketi çevrilir.
_FAIR_L = {
    "tr": {
        "ref_suffix": " (referans)",
        "fpr": "{p} — yanlış alarm oranı (FPR)",
        "fnr": "{p} — kaçırma oranı (FNR)",
        "sel": "{p} — modelin pozitif deme oranı",
        "base": "{p} — verideki gerçek oran",
        "rmse": "{p} — ortalama hata (RMSE)",
        "di": "{col}: {g} / {ref} — seçim oranı oranı (DI)",
        "brr": "{col}: {g} / {ref} — verideki gerçek oran oranı",
        "gap": "{col}: {g} — seçim farkının veriyle açıklanamayan kısmı",
        "fpr_diff": "{col}: {g} − {ref} — yanlış alarm farkı",
        "fnr_diff": "{col}: {g} − {ref} — kaçırma farkı",
        "rmse_diff": "{col}: {g} − {ref} — ortalama hata farkı",
        "base_note": ("Modelin değil, VERİNİN kendisindeki oran. Modelin seçim "
                      "oranıyla karşılaştırıldığında farkın model kaynaklı mı veri "
                      "kaynaklı mı olduğunu gösterir."),
        "gap_note": ("0'a yakınsa seçim oranı farkı VERİDEN geliyor (model grubun "
                     "gerçek oranını yansıtıyor). Sıfırdan uzaksa fark MODELDEN "
                     "geliyor ve incelenmeye değer."),
        "ci_overlap_note": ("İki grubun güven aralıkları örtüşüyor — bu fark "
                            "istatistiksel olarak ayırt edilemez (gürültüden farksız "
                            "olabilir); bir bulgu olarak okunmamalı."),
    },
    "en": {
        "ref_suffix": " (reference)",
        "fpr": "{p} — false-alarm rate (FPR)",
        "fnr": "{p} — miss rate (FNR)",
        "sel": "{p} — rate at which the model says positive",
        "base": "{p} — actual rate in the data",
        "rmse": "{p} — average error (RMSE)",
        "di": "{col}: {g} / {ref} — selection-rate ratio (DI)",
        "brr": "{col}: {g} / {ref} — actual-rate ratio in the data",
        "gap": "{col}: {g} — part of the selection gap not explained by the data",
        "fpr_diff": "{col}: {g} − {ref} — false-alarm gap",
        "fnr_diff": "{col}: {g} − {ref} — miss-rate gap",
        "rmse_diff": "{col}: {g} − {ref} — average-error gap",
        "base_note": ("This is the rate in the DATA, not the model's. Compared with the "
                      "model's selection rate it shows whether the gap comes from the "
                      "model or from the data itself."),
        "gap_note": ("Near 0, the selection-rate gap comes from the DATA (the model "
                     "reflects the group's actual rate). Far from 0, the gap comes from "
                     "the MODEL and is worth examining."),
        "ci_overlap_note": ("The two groups' confidence intervals overlap — this "
                            "difference is statistically indistinguishable (may be no "
                            "different from noise); it should not be read as a finding."),
    },
}


def _fairness_evidence(fair: dict, lang: str) -> list[Evidence]:
    """Fairness sonucundan kanıt kalemlerini kurar. Tüm sayılar test setinden."""
    ev: list[Evidence] = []
    task = fair.get("task", "classification")
    L = _FAIR_L.get(lang, _FAIR_L["tr"])

    for col, seg in (fair.get("segments") or {}).items():
        groups = seg.get("groups") or {}
        comparisons = seg.get("comparisons") or {}
        ref_name = comparisons.get("reference_group")

        # Grup başına ham metrikler
        for gname, m in groups.items():
            n = m.get("n")
            is_ref = L["ref_suffix"] if gname == ref_name else ""
            p = f"{col} = {gname}{is_ref}"
            if task == "classification":
                ev.append(Evidence(
                    label=L["fpr"].format(p=p),
                    value=m.get("fpr"),
                    provenance="test_set",
                    sample_size=m.get("n_true_negatives"),
                    uncertainty=m.get("fpr_ci"),
                    note=R_REASON(m.get("fpr_undefined_reason"), lang),
                ))
                ev.append(Evidence(
                    label=L["fnr"].format(p=p),
                    value=m.get("fnr"),
                    provenance="test_set",
                    sample_size=m.get("n_true_positives"),
                    uncertainty=m.get("fnr_ci"),
                    note=R_REASON(m.get("fnr_undefined_reason"), lang),
                ))
                ev.append(Evidence(
                    label=L["sel"].format(p=p),
                    value=m.get("selection_rate"),
                    provenance="test_set",
                    sample_size=n,
                    uncertainty=m.get("selection_rate_ci"),
                ))
                ev.append(Evidence(
                    label=L["base"].format(p=p),
                    value=m.get("base_rate"),
                    provenance="test_set",
                    sample_size=n,
                    uncertainty=m.get("base_rate_ci"),
                    note=L["base_note"],
                ))
            else:
                ev.append(Evidence(
                    label=L["rmse"].format(p=p),
                    value=m.get("rmse"), provenance="test_set", sample_size=n,
                ))

        # Referans gruba göre farklar
        for gname, c in (comparisons.get("groups") or {}).items():
            if task == "classification":
                di = c.get("disparate_impact")
                brr = c.get("base_rate_ratio")
                gap = c.get("selection_gap_vs_base")
                # D2: DI'nin dayandığı seçim oranları ayırt edilemezse (CI örtüşür)
                # notu değiştir — sayı görünür kalır ama "bulgu değil" damgası taşır.
                di_note = (L["ci_overlap_note"] if c.get("disparate_impact_indistinguishable")
                           else R_REASON(c.get("disparate_impact_undefined_reason"), lang))
                ev.append(Evidence(
                    label=L["di"].format(col=col, g=gname, ref=ref_name),
                    value=di,
                    provenance="test_set",
                    sample_size=c.get("n"),
                    note=di_note,
                    reference=_DI_REFERENCE_NOTE.get(lang, _DI_REFERENCE_NOTE["tr"]),
                ))
                ev.append(Evidence(
                    label=L["brr"].format(col=col, g=gname, ref=ref_name),
                    value=brr,
                    provenance="test_set",
                    sample_size=c.get("n"),
                    note=R_REASON(c.get("base_rate_ratio_undefined_reason"), lang),
                ))
                if gap is not None:
                    ev.append(Evidence(
                        label=L["gap"].format(col=col, g=gname),
                        value=gap,
                        provenance="test_set",
                        sample_size=c.get("n"),
                        note=L["gap_note"],
                    ))
                for metric in ("fpr_diff", "fnr_diff"):
                    # D2: fark CI-örtüşme ile ayırt edilemezse notu bununla değiştir.
                    diff_note = (L["ci_overlap_note"]
                                 if c.get(f"{metric}_indistinguishable")
                                 else R_REASON(c.get(f"{metric}_undefined_reason"), lang))
                    ev.append(Evidence(
                        label=L[metric].format(col=col, g=gname, ref=ref_name),
                        value=c.get(metric),
                        provenance="test_set",
                        sample_size=c.get("n"),
                        note=diff_note,
                    ))
            else:
                ev.append(Evidence(
                    label=L["rmse_diff"].format(col=col, g=gname, ref=ref_name),
                    value=c.get("rmse_diff"), provenance="test_set", sample_size=c.get("n"),
                ))

    return ev


def _fairness_skipped_section(fair: dict, lang: str) -> Optional[ReportSection]:
    """Ölçülemeyen grup/kolonları GÖRÜNÜR kılan alt bölüm.

    Bu alt bölümün varlık sebebi: eskiden ince gruplar ve drop_first kurbanları
    sessizce kayboluyordu — rapor onların varlığından hiç söz etmiyordu. Artık
    'ölçmedik' de bir bulgudur ve sebebiyle birlikte yazılır.
    """
    lines: list[Evidence] = []

    for col, reason in (fair.get("skipped_columns") or {}).items():
        lines.append(Evidence(
            label=(f"{col} — segment olarak taşınmadı" if lang == "tr"
                   else f"{col} — not carried as a segment"),
            value=None, provenance="test_set", note=R_REASON(reason, lang),
        ))

    for col, seg in (fair.get("segments") or {}).items():
        for gname, reason in (seg.get("skipped_groups") or {}).items():
            lines.append(Evidence(
                label=(f"{col} = {gname} — ölçülmedi" if lang == "tr"
                       else f"{col} = {gname} — not measured"),
                value=None, provenance="test_set", note=R_REASON(reason, lang),
            ))

    if not lines:
        return None

    return ReportSection(
        id="fairness_not_measured",
        title=("Ölçülemeyenler" if lang == "tr" else "Not measured"),
        headline=("Aşağıdaki gruplar için adalet ölçümü yapılmadı — sebepleriyle listelenmiştir."
                  if lang == "tr" else
                  "Fairness was not measured for the groups below — each with its reason."),
        explanation=("Bir grubun burada olması o grupta sorun OLMADIĞI anlamına gelmez; "
                     "yalnızca eldeki veriyle güvenilir bir ölçüm yapılamadığı anlamına gelir."
                     if lang == "tr" else
                     "A group listed here does not mean it is free of problems; it means the "
                     "available data does not support a reliable measurement."),
        evidence=lines,
        status="ok",
    )


def build_fairness_section(state: dict, lang: str = "tr") -> ReportSection:
    """Fairness bölümünü kurar — katmanlı yapının pilotu.

    LLM ÇAĞIRMAZ: yalnızca evidence/status/reason doldurur. headline/explanation
    sonradan build_section_prose() ile yazılır (o da bu kanıtları GİRDİ alır).
    """
    ev_report = state.get("evaluation_report", {}) or {}
    fair = ev_report.get("fairness") or {}
    title = _FAIRNESS_TITLE.get(lang, _FAIRNESS_TITLE["tr"])

    # Fairness hiç koşmadıysa (ör. clustering/anomaly) → görünür sebep.
    if not fair:
        intent = state.get("intent", "")
        return ReportSection(
            id="fairness",
            title=title,
            order=ORDER_FAIRNESS,
            status="not_applicable",
            not_computed_reason=(
                f"Bu analiz tipinde ({intent}) grup başına performans karşılaştırması "
                f"tanımlı değil — ayrı bir test seti ve gerçek etiket yok."
                if lang == "tr" else
                f"Group-wise performance comparison is not defined for this analysis "
                f"type ({intent}) — there is no separate test set with true labels."
            ),
        )

    if fair.get("status") != "ok":
        # SESSİZ SIFIR YASAĞI: analiz yapılamadı → sebep ZORUNLU ve rapora yazılır.
        return ReportSection(
            id="fairness",
            title=title,
            order=ORDER_FAIRNESS,
            status="not_computed",
            not_computed_reason=(
                f"Adalet analizi yapılamadı: {R_REASON(fair.get('reason'), lang)}"
                if lang == "tr" else
                f"Fairness analysis could not be performed: {R_REASON(fair.get('reason'), lang)}"
            ),
        )

    evidence = _fairness_evidence(fair, lang)
    if not evidence:
        return ReportSection(
            id="fairness",
            title=title,
            order=ORDER_FAIRNESS,
            status="not_computed",
            not_computed_reason=(
                "Segment kolonları taşındı ama hiçbir grup ölçüm için yeterli "
                "örnekleme sahip değil."
                if lang == "tr" else
                "Segment columns were carried, but no group has enough samples to measure."
            ),
        )

    section = ReportSection(
        id="fairness",
        title=title,
        order=ORDER_FAIRNESS,
        evidence=evidence,
        status="ok",
    )
    skipped = _fairness_skipped_section(fair, lang)
    if skipped:
        section.subsections.append(skipped)
    return section


# ── LLM prose katmanı ────────────────────────────────────────────────────────

# ── Veri İlişkileri bölümü ───────────────────────────────────────────────────

_RELATIONS_TITLE = {"tr": "Veri İlişkileri", "en": "Data Relationships"}

# NEDENSELLİK KORUMASI — KATMAN 1/3: her ilişki kanıtına KOD tarafından basılan damga.
# LLM'in atlayabileceği bir uyarıyı yapısal hâle getirir: prose nedensellik ima etse
# bile hemen altındaki kanıt satırı bunu reddeder. Status rozeti dersi: talimat
# yetmiyor, yapı gerekiyor.
# KATMAN 2/3: bölüm düzeyinde deterministik uyarı — LLM'in prose'undan ÖNCE,
# kod tarafından basılır. Her raporda vardır, atlanamaz.
_RELATIONS_DISCLAIMER = {
    "tr": ("> ⚠️ **Bu bölüm ilişkiyi tanımlar, sebebini açıklamaz.** Aşağıdaki ölçüler "
           "alanların birlikte nasıl hareket ettiğini gösterir. Hiçbiri bir alanın "
           "diğerine sebep olduğunun kanıtı değildir — arkalarında ölçülmemiş ortak "
           "bir etken de olabilir, ilişkinin yönü ters de olabilir."),
    "en": ("> ⚠️ **This section describes relationships, not their causes.** The measures "
           "below show how fields move together. None of them is evidence that one field "
           "causes another — there may be an unmeasured common driver, or the direction "
           "may be the reverse."),
}

_PROVENANCE_NOTE = {
    "train_only": {
        "tr": ("Yalnızca eğitim verisinden hesaplandı; test satırları bu hesaba GİRMEDİ. "
               "Böylece burada gördüğünüz ilişki, modelin sınandığı veriden bağımsızdır."),
        "en": ("Computed on the training data only; test rows were NOT included, so the "
               "relationship shown here is independent of the data the model was tested on."),
    },
    "full_data_descriptive": {
        "tr": ("TÜM veriden hesaplandı ve yalnızca TANIMLAYICIDIR — veri setinin kendisini "
               "betimler, bir tahmin gücü iddiası taşımaz."),
        "en": ("Computed on the full data and is DESCRIPTIVE only — it characterises the "
               "dataset itself and makes no claim about predictive power."),
    },
}

# Görünür varsayılan: en güçlü N ilişki. Kalanı KIRPILMAZ — alt bölümde katlanır
# ("uzman detayı" hedefi kırpmayı yasaklıyor; okunabilirlik katmanlamayı gerektiriyor).
RELATIONS_VISIBLE_TOP = 10


def _rel_direction(score: float, lang: str) -> str:
    """İlişkinin YÖNÜNÜ nedensellik ima etmeyen dille anlatır."""
    if lang == "en":
        return "move together (same direction)" if score > 0 else "move in opposite directions"
    return "birlikte artıyor" if score > 0 else "ters yönde hareket ediyor"


def build_relations_section(state: dict, lang: str = "tr") -> ReportSection:
    """Veri İlişkileri bölümü — X-y (hedefle) ve X-X (sütunlar arası).

    Kanıt tamamen deterministik; provenance relations_report'tan gelir ve hesabın
    fiilen hangi satırlardan geçtiğini söyler (split anında üretilir, bkz.
    src/services/relations.py).
    """
    rel = state.get("relations_report") or {}
    title = _RELATIONS_TITLE.get(lang, _RELATIONS_TITLE["tr"])

    if not rel:
        return ReportSection(
            id="data_relations", title=title, order=ORDER_DATA_RELATIONS,
            status="not_computed",
            not_computed_reason=(
                "Sütun ilişkileri bu çalışmada hesaplanmadı."
                if lang == "tr" else
                "Column relationships were not computed in this run."
            ),
        )

    prov = rel.get("generated_from") or "full_data_descriptive"
    provenance = "train_only" if prov == "train_only" else "full_data_descriptive"
    n_rows = rel.get("n_rows") or 0
    prov_note = _PROVENANCE_NOTE[provenance].get(lang, _PROVENANCE_NOTE[provenance]["tr"])

    target = rel.get("target") or ""
    rels = rel.get("target_relations") or []
    pairs = rel.get("top_pairs") or []
    redundant = rel.get("redundant_pairs") or []

    if not rels and not pairs:
        return ReportSection(
            id="data_relations", title=title, order=ORDER_DATA_RELATIONS,
            status="not_computed",
            not_computed_reason=(
                R_REASON(rel.get("reason"), lang)
                or ("Ölçülebilir bir sayısal ilişki bulunamadı (yeterli sayısal sütun yok)."
                    if lang == "tr" else
                    "No measurable numeric relationship was found (not enough numeric columns).")
            ),
        )

    def _rel_evidence(r: dict) -> Evidence:
        score = r["score"]
        yon = _rel_direction(score, lang)
        return Evidence(
            label=(f"{_hf(r['feature'], lang, state)} ↔ {target} — {yon}"
                   if target else _hf(r["feature"], lang, state)),
            value=score,
            provenance=provenance,
            sample_size=n_rows,
            uncertainty=r.get("ci"),
            # BOILERPLATE TEKRARI YASAĞI: nedensellik damgası + provenance açıklaması
            # 30 satırda 60 satır aynı metin ediyordu. İkisi de artık bölüm başında
            # BİR KEZ (disclaimer'da) — satırda yalnız değişen bilgi (değer, n, CI).
        )

    evidence: list[Evidence] = [_rel_evidence(r) for r in rels[:RELATIONS_VISIBLE_TOP]]

    # X-X: "neredeyse aynı bilgi" çiftleri — hedefi hiç görmez, yorumu güvenli.
    for p in pairs[:RELATIONS_VISIBLE_TOP]:
        evidence.append(Evidence(
            label=(f"{_hf(p['a'], lang, state)} ↔ {_hf(p['b'], lang, state)} — "
                   f"{_rel_direction(p['corr'], lang)}"),
            value=p["corr"],
            provenance=provenance,
            sample_size=n_rows,
            uncertainty=fisher_z_interval(p["corr"], n_rows),
        ))

    section = ReportSection(
        id="data_relations", title=title, order=ORDER_DATA_RELATIONS,
        evidence=evidence, status="ok",
        # KATMAN 2: bölüm uyarısı — kod basar, LLM dokunamaz, her raporda var.
        # Nedensellik çerçevesi + provenance açıklaması BURADA bir kez toplanır
        # (eskiden her kanıt satırında tekrar ediyordu → 60 satır aynı metin).
        disclaimer=(_RELATIONS_DISCLAIMER.get(lang, _RELATIONS_DISCLAIMER["tr"])
                    + " " + prov_note),
    )

    # Kalan ilişkiler KIRPILMAZ — katlanır (uzman detayı isteğe bağlı derinlikte).
    rest = rels[RELATIONS_VISIBLE_TOP:]
    if rest:
        section.subsections.append(ReportSection(
            id="data_relations_all",
            title=("Tüm ilişkiler" if lang == "tr" else "All relationships"),
            headline=(f"Kalan {len(rest)} ilişki — aynı ölçü, daha zayıf."
                      if lang == "tr" else
                      f"The remaining {len(rest)} relationships — same measure, weaker."),
            evidence=[_rel_evidence(r) for r in rest],
            status="ok",
        ))

    if redundant:
        # #10 — NEAR-DUP TAVSİYESİNİ İKİYE AYIR: ham↔ham çift için "yalnız birini topla"
        # (kullanıcı ham alanı toplamayı bırakabilir); türetilmiş dahil çift için "özellik
        # üretiminde sadeleştir" (kullanıcı MOTORUN ürettiği feature'ı toplamayı BIRAKAMAZ →
        # eski tek-tip "veri toplama fırsatı" tavsiyesi türetilmişte YANLIŞTI). is_derived_feature
        # ile çift-başına sınıflanır; tek liste kalır, her satır kendi doğasını taşır.
        def _redundant_ev(p: dict) -> Evidence:
            derived = is_derived_feature(p["a"]) or is_derived_feature(p["b"])
            if derived:
                note = ("Biri motorun ürettiği bir özellik — bu bir feature-engineering tekrarı; "
                        "veri toplamada DEĞİL, özellik üretiminde sadeleştirilir."
                        if lang == "tr" else
                        "One is an engine-derived feature — a feature-engineering redundancy; "
                        "prune it in feature generation, NOT data collection.")
            else:
                note = ("Her ikisi de ham alan — yalnız birini toplamak yeterli (veri toplama "
                        "sadeleştirmesi)."
                        if lang == "tr" else
                        "Both are raw fields — collecting only one suffices (a data-collection "
                        "simplification).")
            return Evidence(
                label=f"{_hf(p['a'], lang, state)} ↔ {_hf(p['b'], lang, state)}",
                value=p["corr"], provenance=provenance, sample_size=n_rows, note=note,
            )

        section.subsections.append(ReportSection(
            id="data_relations_redundant",
            title=("Neredeyse aynı bilgiyi taşıyan sütunlar"
                   if lang == "tr" else "Columns carrying nearly the same information"),
            headline=(f"{len(redundant)} sütun çifti birbirinin neredeyse kopyası."
                      if lang == "tr" else
                      f"{len(redundant)} column pairs are near-duplicates of each other."),
            explanation=(
                "Bu çiftlerden yalnızca biri modele katkı sağlar; ikisini birden tutmak yeni "
                "bilgi eklemez. Sadeleştirme çiftin doğasına göre YA veri toplamada (iki ham "
                "alan) YA da özellik üretiminde (biri motor türevi) yapılır — her satır hangisi "
                "olduğunu belirtir."
                if lang == "tr" else
                "Only one of each pair adds anything to the model; keeping both adds no new "
                "information. Depending on the pair, the simplification belongs EITHER to data "
                "collection (two raw fields) OR to feature generation (one is engine-derived) — "
                "each row states which."
            ),
            evidence=[_redundant_ev(p) for p in redundant],
            status="ok",
        ))

    return section


# ── Hata Analizi bölümü ──────────────────────────────────────────────────────

# Faz 3 #6: başlık stili tutarlılığı — diğer 10 başlık isim-öbeği (Title Case);
# tek istisna bu cümle-başlıktı. İsim-öbeğine çevrildi.
_ERRORS_TITLE = {"tr": "Hata Analizi", "en": "Error Analysis"}

# NEREDE / NEDEN ayrımı — nedensellik korumasının kardeşi. Çerçeve artık bölüm
# başında bir kez (_ERRORS_DISCLAIMER); satır başına damga KALDIRILDI (boilerplate).
_ERRORS_DISCLAIMER = {
    "tr": ("> ⚠️ **Bu bölüm hatanın nerede yoğunlaştığını gösterir, sebebini açıklamaz.** "
           "Aşağıdaki alanlarda hatalı tahminler doğru tahminlerden ayrışıyor. Bu, o "
           "alanların hataya sebep olduğu anlamına GELMEZ — nerede bakılacağını söyler, "
           "niçin yanıldığını değil."),
    "en": ("> ⚠️ **This section shows where errors concentrate, not why they happen.** "
           "Wrong predictions differ from correct ones on the fields below. That does NOT "
           "mean those fields cause the errors — it tells you where to look, not why."),
}

_COHENS_D_NOTE = {
    "tr": ("Ayrışma büyüklüğü (Cohen's d): 0.2 küçük, 0.5 orta, 0.8 büyük fark. "
           "Alanın ölçü biriminden bağımsızdır, bu yüzden alanlar arasında kıyaslanabilir."),
    "en": ("Separation size (Cohen's d): 0.2 small, 0.5 medium, 0.8 large. It is "
           "independent of the field's unit, so it is comparable across fields."),
}

ERRORS_VISIBLE_TOP = 5


def build_error_analysis_section(state: dict, lang: str = "tr") -> ReportSection:
    """Hata Analizi — hata oranı, yüksek güvenli hatalar, ayrışan alanlar.

    Kanıtın can alıcı kısmı: her alan için HATALI vakalardaki ortalama ile DOĞRU
    vakalardaki ortalama yan yana ("hatalı vakalarda 42 saat, doğrularda 17 saat").
    Ham değer ancak split anındaki taşıyıcı sayesinde mümkün — X_test ölçeklidir.
    """
    ev = (state.get("evaluation_report") or {})
    err = ev.get("error_analysis") or {}
    title = _ERRORS_TITLE.get(lang, _ERRORS_TITLE["tr"])

    if not err:
        return ReportSection(
            id="error_analysis", title=title, order=ORDER_ERROR_ANALYSIS,
            status="not_computed",
            not_computed_reason=(
                "Hata analizi bu çalışmada yapılmadı."
                if lang == "tr" else "Error analysis was not performed in this run."
            ),
        )

    n_err, n_total = err.get("n_errors"), err.get("n_total")
    if n_err == 0:
        return ReportSection(
            id="error_analysis", title=title, order=ORDER_ERROR_ANALYSIS,
            status="not_applicable",
            not_computed_reason=(
                "Test setinde hiç hatalı tahmin yok — incelenecek hata yok."
                if lang == "tr" else
                "There are no wrong predictions on the test set — no errors to examine."
            ),
        )

    evidence: list[Evidence] = [
        Evidence(
            label=("Hatalı tahmin oranı" if lang == "tr" else "Error rate"),
            value=err.get("error_rate"), provenance="test_set", sample_size=n_total,
            note=(f"{n_err:,} kayıt / {n_total:,} kayıt" if lang == "tr"
                  else f"{n_err:,} of {n_total:,} records"),
        ),
    ]

    conf = err.get("confidence_analysis") or {}
    if conf:
        evidence.append(Evidence(
            label=("Emin olduğu hâlde yanıldığı tahminler"
                   if lang == "tr" else "Predictions it was confident about but got wrong"),
            value=conf.get("high_confidence_errors"), provenance="test_set", sample_size=n_err,
            note=(f"Hataların %{conf.get('high_confidence_errors_pct', 0)}'i. 'Emin' = model olasılığı "
                  f">0.7 veya <0.3 (yani |olasılık−0.5|>0.2). Bunlar en riskli hatalardır: model "
                  f"emin olduğu için kimse kontrol etmez."
                  if lang == "tr" else
                  f"{conf.get('high_confidence_errors_pct', 0)}% of errors. 'Confident' = model "
                  f"probability >0.7 or <0.3 (i.e. |probability−0.5|>0.2). These are the riskiest: "
                  f"because the model is confident, nobody double-checks them."),
        ))

    # Ayrışan alanlar — ham değer varsa "42 vs 17", yoksa görünür sebeple ölçekli.
    diffs = err.get("feature_distribution_diff") or {}
    items = list(diffs.items())

    def _feat_evidence(col: str, info: dict) -> Evidence:
        has_raw = info.get("raw_available")
        if has_raw:
            # Faz 3 #4: ham ortalama not metnine gömülü — TEK doğruluk kaynağı
            # _fmt_value ile aynı ondalık disiplininden geçir (91571.9019 → 91,571.90;
            # küçük ortalama (yaş 42.3) → "42.3"). Aksi hâlde ham 4-hane sızar.
            m_err = _fmt_value(info.get("mean_in_errors_raw"))
            m_cor = _fmt_value(info.get("mean_in_correct_raw"))
            detail = (f"Hatalı vakalarda ortalama **{m_err}**, doğru vakalarda **{m_cor}**."
                      if lang == "tr" else
                      f"Average **{m_err}** in wrong cases, **{m_cor}** in correct ones.")
        else:
            _rur = R_REASON(info.get("raw_unavailable_reason"), lang)
            detail = (f"Ham karşılığı yok — {_rur}" if lang == "tr"
                      else f"No raw equivalent — {_rur}")
        return Evidence(
            label=(f"{_hf(col, lang, state)} — ayrışma büyüklüğü"
                   if lang == "tr" else f"{_hf(col, lang, state)} — separation size"),
            value=info.get("cohens_d"),
            provenance="test_set",
            sample_size=n_total,
            # BOILERPLATE TEKRARI YASAĞI: Cohen's d ölçeği + "nerede/neden" damgası
            # her alanda tekrar ediyordu. İkisi de artık bölüm başında BİR KEZ
            # (disclaimer'da) — satırda yalnız değişen bilgi (ortalamalar / ham-yok).
            note=detail,
        )

    evidence += [_feat_evidence(c, i) for c, i in items[:ERRORS_VISIBLE_TOP]]

    section = ReportSection(
        id="error_analysis", title=title, order=ORDER_ERROR_ANALYSIS,
        evidence=evidence, status="ok",
        # "Nerede/neden" çerçevesi + Cohen's d ölçeği BURADA bir kez toplanır
        # (eskiden her alanın kanıt satırında tekrar ediyordu).
        disclaimer=(_ERRORS_DISCLAIMER.get(lang, _ERRORS_DISCLAIMER["tr"])
                    + " " + _COHENS_D_NOTE.get(lang, _COHENS_D_NOTE["tr"])),
    )

    rest = items[ERRORS_VISIBLE_TOP:]
    if rest:
        section.subsections.append(ReportSection(
            id="error_analysis_all",
            title=("Ayrışan diğer alanlar" if lang == "tr" else "Other separating fields"),
            headline=(f"Kalan {len(rest)} alan — aynı ölçü, daha zayıf ayrışma."
                      if lang == "tr" else
                      f"The remaining {len(rest)} fields — same measure, weaker separation."),
            evidence=[_feat_evidence(c, i) for c, i in rest],
            status="ok",
        ))

    # HAM DEĞER MERCEĞİ — sıralamayı DEĞİŞTİRMEZ, yeniden gruplar.
    # Ölçülen desen: en çok ayrışan alanlar türetilmiş olabilir (one-hot, count_*,
    # grpstd_*) ve ham karşılıkları yoktur; "hatalı vakalarda 42, doğrularda 17"
    # cümlesi ancak ham alanlarda kurulur ve bunlar alt sıralara düşebilir.
    # Sıralamayı ham-öncelikli yapmak en güçlü ayrıştırıcıyı gizlerdi (dürüst değil);
    # bunun yerine aynı alanlara ikinci bir mercekten bakılır.
    raw_items = [(c, i) for c, i in items if i.get("raw_available")]
    if raw_items:
        section.subsections.append(ReportSection(
            id="error_analysis_raw",
            title=("Gerçek birimlerle okunabilen alanlar"
                   if lang == "tr" else "Fields readable in real-world units"),
            headline=(f"{len(raw_items)} alanın ham karşılığı var — hatalı ve doğru "
                      f"vakalardaki ortalamalar gerçek birimleriyle okunabiliyor."
                      if lang == "tr" else
                      f"{len(raw_items)} fields have a raw equivalent — the averages in "
                      f"wrong and correct cases can be read in their real-world units."),
            explanation=(
                "Bu liste yukarıdaki alanların bir ALT KÜMESİDİR, yeni bir ölçüm değil: "
                "aynı alanlar, yalnızca gerçek birimlerinde okunabilenler ayrı toplandı. "
                "Türetilmiş alanların (sayım, oran, kategori göstergesi) ham karşılığı "
                "olmadığı için burada yer almazlar — ayrışmaları daha güçlü olsa bile."
                if lang == "tr" else
                "This list is a SUBSET of the fields above, not a new measurement: the same "
                "fields, with only those readable in their real-world units gathered here. "
                "Derived fields (counts, ratios, category indicators) have no raw equivalent "
                "and are absent — even when they separate more strongly."
            ),
            evidence=[_feat_evidence(c, i) for c, i in raw_items],
            status="ok",
        ))

    # Kaç feature TARANDI — "en çok ayrışan" iddiasının kapsamı görünür olmalı.
    n_examined = err.get("n_features_examined")
    if n_examined:
        section.subsections.append(ReportSection(
            id="error_analysis_scope",
            title=("Tarama kapsamı" if lang == "tr" else "Scope of the scan"),
            headline=(f"{n_examined} sayısal alanın tamamı tarandı; yalnızca belirgin "
                      f"ayrışma gösterenler ({len(items)}) listelendi."
                      if lang == "tr" else
                      f"All {n_examined} numeric fields were scanned; only those with a "
                      f"clear separation ({len(items)}) are listed."),
            explanation=(
                "Belirgin ayrışma eşiği Cohen's d ≥ 0.2'dir (küçük etki). Bu eşiğin "
                "altındakiler hatalı ve doğru vakalarda pratikte aynı davranıyor demektir."
                if lang == "tr" else
                "The clear-separation threshold is Cohen's d ≥ 0.2 (small effect). Below it, "
                "a field behaves practically the same in wrong and correct cases."
            ),
            status="ok",
        ))

    return section


# ── Model Performansı bölümü ─────────────────────────────────────────────────

_PERF_TITLE = {"tr": "Model Performansı", "en": "Model Performance"}

# Her metrik DÜZ DİLLE açıklanır — metrik adı tek başına okuyucuya hiçbir şey
# söylemez. Açıklamalar TANIMLAYICIDIR: "0.92 iyidir" demezler, 0.92'nin NE
# ölçtüğünü söylerler. Yargı okuyucunundur (fairness dersi: kanıt sunuyoruz).
_METRIC_META: dict[str, dict[str, tuple[str, str]]] = {
    "roc_auc": {
        "tr": ("Ayırt etme gücü (ROC-AUC)",
               "Rastgele bir 'evet' kaydı ile rastgele bir 'hayır' kaydı verildiğinde, "
               "modelin 'evet' olanı daha yüksek sıralama olasılığı. 0.5 yazı-tura ile "
               "aynı demektir, 1.0 kusursuz ayrım."),
        "en": ("Discriminative power (ROC-AUC)",
               "Given a random 'yes' record and a random 'no' record, the chance the model "
               "ranks the 'yes' one higher. 0.5 is the same as a coin flip, 1.0 is perfect "
               "separation."),
    },
    "roc_auc_ovr": {
        "tr": ("Ayırt etme gücü (ROC-AUC, her sınıf diğerlerine karşı)",
               "Her sınıf tek tek 'diğer hepsine karşı' sınandığında modelin doğru sınıfı "
               "daha yüksek sıralama olasılığı. 0.5 yazı-tura, 1.0 kusursuz ayrım."),
        "en": ("Discriminative power (ROC-AUC, one-vs-rest)",
               "The chance the model ranks the correct class higher when each class is "
               "tested against all the others. 0.5 is a coin flip, 1.0 is perfect."),
    },
    "pr_auc": {
        "tr": ("Az rastlanan durumu yakalama gücü (PR-AUC)",
               "Aranan durum seyrekse ayırt etme gücünden daha bilgilendiricidir: model "
               "'evet' derken ne kadar isabetli, aynı anda ne kadarını yakalıyor."),
        "en": ("Power to catch the rare case (PR-AUC)",
               "More informative than ROC when the case of interest is rare: how accurate "
               "the model is when it says 'yes', and how much of it it catches."),
    },
    "accuracy": {
        "tr": ("Doğruluk",
               "Modelin yaptığı tüm tahminlerin ne kadarı doğru çıktı. Dengesiz veride tek "
               "başına yanıltıcıdır: her kayda 'hayır' diyen bir model de yüksek doğruluk alır."),
        "en": ("Accuracy",
               "What share of all the model's predictions were correct. On imbalanced data it "
               "misleads on its own: a model that says 'no' to everything also scores high."),
    },
    "f1_weighted": {
        "tr": ("Yakalama–isabet dengesi (F1)",
               "'Evet' dediklerinin ne kadarının doğru olduğu ile yakalaması gerekenlerin ne "
               "kadarını yakaladığının dengesi; sınıf büyüklüklerine göre ağırlıklandırılmış."),
        "en": ("Catch–accuracy balance (F1)",
               "The balance between how many of its 'yes' calls are right and how many of the "
               "real cases it catches; weighted by class size."),
    },
    "f1_macro": {
        "tr": ("Sınıf başına eşit ağırlıklı denge (F1-macro)",
               "Aynı denge, ama her sınıf eşit sayılır — küçük sınıftaki başarısızlık burada "
               "büyük sınıfın arkasına gizlenemez."),
        "en": ("Per-class equal-weight balance (F1-macro)",
               "The same balance, but every class counts equally — failure on a small class "
               "cannot hide behind a large one here."),
    },
    "precision_weighted": {
        "tr": ("İsabet oranı",
               "Model 'evet' dediğinde ne kadar sık haklı çıkıyor."),
        "en": ("Precision", "How often the model is right when it says 'yes'."),
    },
    "recall_weighted": {
        "tr": ("Yakalama oranı",
               "Gerçekten 'evet' olan kayıtların ne kadarını model yakalayabiliyor."),
        "en": ("Recall", "What share of the truly 'yes' records the model manages to catch."),
    },
    "mcc": {
        "tr": ("Dengeli başarı ölçüsü (MCC)",
               "-1 ile 1 arasında: 0 rastgele tahminle aynı, 1 kusursuz, -1 tam ters demektir. "
               "Dengesiz veride doğruluktan daha dürüst bir özettir çünkü dört hücrenin hepsini "
               "hesaba katar."),
        "en": ("Balanced quality score (MCC)",
               "Between -1 and 1: 0 means no better than guessing, 1 is perfect, -1 is fully "
               "inverted. On imbalanced data it is a more honest summary than accuracy because "
               "it uses all four cells."),
    },
    "log_loss": {
        "tr": ("Olasılık hatası (log-loss)",
               "Modelin verdiği olasılıkların gerçekten ne kadar uzak düştüğü. Düşük olması "
               "iyidir; emin olup yanılmayı ağır cezalandırır."),
        "en": ("Probability error (log-loss)",
               "How far the model's stated probabilities fall from reality. Lower is better; "
               "it punishes being confident and wrong."),
    },
    "r2": {
        "tr": ("Açıklanan değişkenlik (R²)",
               "Hedefteki oynamanın ne kadarını model açıklayabiliyor. 0, her seferinde "
               "ortalamayı söylemekle aynı demektir."),
        "en": ("Explained variation (R²)",
               "How much of the target's variation the model accounts for. 0 is the same as "
               "always guessing the average."),
    },
    "rmse": {
        "tr": ("Ortalama hata büyüklüğü (RMSE)",
               "Hedefin kendi biriminde ortalama sapma; büyük hataları küçüklerden daha ağır "
               "cezalandırır."),
        "en": ("Typical error size (RMSE)",
               "Average deviation in the target's own unit; it penalises large errors more "
               "heavily than small ones."),
    },
    "mae": {
        "tr": ("Ortalama mutlak hata (MAE)",
               "Hedefin kendi biriminde ortalama sapma; her hata eşit ağırlıkta sayılır."),
        "en": ("Mean absolute error (MAE)",
               "Average deviation in the target's own unit, counting every error equally."),
    },
    "mape": {
        "tr": ("Ortalama yüzde hata (MAPE)",
               "Sapmanın gerçek değere oranı, yüzde olarak."),
        "en": ("Mean percentage error (MAPE)",
               "The deviation as a percentage of the true value."),
    },
}

# Rapordaki gösterim sırası — sözlük sırası değil, OKUYUCU önceliği:
# ayırt etme gücü → yakalama → denge → ham doğruluk.
# NOT: 'recall_weighted' KASITLI YOK — weighted recall MATEMATİKSEL OLARAK accuracy'ye
# eşittir (aynı sayıyı iki ayrı metrik gibi göstermek sıfır bilgi taşır, yanıltır). Azınlık/
# ilgi sınıfının recall+precision'ı ayrıca gösterilir (bkz. build_performance_section, #7).
_METRIC_ORDER = (
    "roc_auc", "roc_auc_ovr", "pr_auc", "f1_weighted", "f1_macro",
    "precision_weighted", "accuracy", "mcc", "log_loss",
    "r2", "rmse", "mae", "mape",
)

# test_metrics anahtarı → confidence_intervals anahtarı.
# Çok sınıflıda test_metrics 'roc_auc_ovr' yazar ama bootstrap aynı değeri
# 'roc_auc' altında tutar; eşleştirilmezse GA sessizce DÜŞER (metrik güvenilir
# görünür, oysa aralığı hiç basılmamıştır).
_CI_KEY_ALIAS = {"roc_auc_ovr": "roc_auc"}

# Bu metrikler ORANDIR (0-1) → Evidence.reliable aralık genişliğine bakabilir.
# RMSE/MAE gibi birimli metriklerde 0.30 eşiği anlamsız olurdu.
_RATE_METRICS = frozenset({
    "roc_auc", "roc_auc_ovr", "pr_auc", "accuracy", "f1_weighted", "f1_macro",
    "precision_weighted", "recall_weighted",
})


def _ci_for(ci: dict, metric_key: str) -> Optional[dict]:
    """Bootstrap güven aralığını Evidence.uncertainty biçimine çevirir."""
    vals = ci.get(_CI_KEY_ALIAS.get(metric_key, metric_key))
    if not isinstance(vals, dict):
        return None
    lo, hi = vals.get("lower"), vals.get("upper")
    if lo is None or hi is None:
        return None
    return {"lower": lo, "upper": hi, "method": "bootstrap"}


def _cm_usable(cm: dict) -> bool:
    """Confusion matrix kxk ve etiketleriyle tutarlı mı?

    TEK doğrulama noktası. Ayrı ayrı kontrol edilirse tüketiciler ayrışır: tablo
    kurucusu "bozuk" deyip çekilirken sınıf-başına kanıt aynı matrise indeksle
    dalıp IndexError atar ve RAPORUN TAMAMI çöker (ölçüldü —
    test_bozuk_matris_coktürmüyor_sessizce_atlaniyor).
    """
    matrix = cm.get("matrix") or []
    labels = cm.get("labels") or []
    return bool(
        matrix and labels
        and len(matrix) == len(labels)
        and all(isinstance(r, (list, tuple)) and len(r) == len(labels) for r in matrix)
    )


def _class_display(labels: list, lang: str) -> dict:
    """Sınıf etiketlerini okunabilir ada çevirir. Çeviremediğini AYNEN bırakır.

    "Gerçekte: 1" okuyucuya hiçbir şey söylemez. YALNIZCA etiket kümesi tam
    {0, 1} olduğunda çeviri yapılır: bu kod tabanında ikili hedefin pozitif
    sınıfı her yerde ``classes_sorted[1]``'dir (bkz. evaluator eşik analizi),
    yani 1 = "evet". Bu bir varsayım değil, mevcut sözleşmenin okunması.

    Etiketler zaten anlamlıysa ("churn"/"stay", "A"/"B"/"C") DOKUNULMAZ — onlara
    isim uydurmak, sayıyı olduğu gibi bırakmaktan daha kötüdür.
    """
    if sorted(str(l) for l in labels) != ["0", "1"]:
        return {}
    return ({"0": "hayır", "1": "evet"} if lang == "tr" else {"0": "no", "1": "yes"})


def _confusion_table(cm: dict, lang: str) -> Optional[dict]:
    """Confusion matrix'i DÜZ DİLLİ bir tabloya çevirir.

    Ham 2x2 liste ("[[4500, 300], [700, 1000]]") okuyucuya hiçbir şey söylemez;
    hangi sayının 'yanlış alarm' hangisinin 'kaçırılan' olduğu tam da iş kararını
    belirleyen ayrımdır. İkili problemde hücreler adlandırılır; çok sınıflıda
    köşegen işaretlenir.
    """
    if not _cm_usable(cm):
        return None
    matrix = cm["matrix"]
    labels = cm["labels"]

    tr = lang == "tr"
    said = "Model: {} dedi" if tr else "Model said: {}"
    actual = "Gerçekte: {}" if tr else "Actually: {}"
    disp = _class_display(labels, lang)
    name = lambda l: disp.get(str(l), str(l))

    if len(labels) == 2:
        neg, pos = name(labels[0]), name(labels[1])
        tn, fp = matrix[0][0], matrix[0][1]
        fn, tp = matrix[1][0], matrix[1][1]
        cell = ("{n:,} — {t}" if tr else "{n:,} — {t}").format
        return {
            "caption": (
                "Satırlar gerçek durumu, sütunlar modelin tahminini gösterir."
                if tr else
                "Rows are the true state, columns are the model's prediction."
            ),
            "headers": ["", said.format(neg), said.format(pos)],
            "rows": [
                [f"**{actual.format(neg)}**",
                 cell(n=tn, t="✅ doğru" if tr else "✅ correct"),
                 cell(n=fp, t="❌ yanlış alarm" if tr else "❌ false alarm")],
                [f"**{actual.format(pos)}**",
                 cell(n=fn, t="❌ kaçırılan" if tr else "❌ missed"),
                 cell(n=tp, t="✅ doğru" if tr else "✅ correct")],
            ],
        }

    return {
        "caption": (
            "Satırlar gerçek sınıfı, sütunlar modelin tahminini gösterir. "
            "Köşegendeki (✅) sayılar doğru tahminlerdir; satırdaki diğer sayılar o "
            "sınıfın hangi sınıfla karıştırıldığını gösterir."
            if lang == "tr" else
            "Rows are the true class, columns are the model's prediction. The diagonal "
            "(✅) holds the correct predictions; the other numbers in a row show which "
            "class it was confused with."
        ),
        "headers": [""] + [said.format(name(c)) for c in labels],
        "rows": [
            [f"**{actual.format(name(labels[i]))}**"] + [
                (f"✅ {matrix[i][j]:,}" if i == j else f"{matrix[i][j]:,}")
                for j in range(len(labels))
            ]
            for i in range(len(labels))
        ],
    }


def _per_class_evidence(per_class: dict, cm: dict, lang: str) -> list[Evidence]:
    """Sınıf başına yakalama/isabet — ORANI DEĞİL, SAYIYI da söyleyerek.

    "class 1 recall 0.52" cümlesi okuyucuya hiçbir şey ifade etmez. Sayılar
    confusion matrix'ten AYNEN alınır (recall x support çarpımıyla yeniden
    üretilmez — yuvarlama gürültüsü kanıt katmanında kabul edilemez).

    Wilson aralığı burada da kritik: küçük bir sınıfın "%52 yakalama"sı aslında
    üç kayıttan biri olabilir; Evidence.reliable bunu kendiliğinden damgalar.
    """
    matrix = cm.get("matrix") or []
    labels = cm.get("labels") or []
    idx = {str(lbl): i for i, lbl in enumerate(labels)}
    # Tabloyla AYNI adlandırma: tablo "evet" derken kanıt "1" derse okuyucu
    # iki ayrı şeyden bahsedildiğini sanır.
    disp = _class_display(labels, lang)
    tr = lang == "tr"
    ev: list[Evidence] = []

    for raw_cls, m in per_class.items():
        i = idx.get(str(raw_cls))
        cls = disp.get(str(raw_cls), str(raw_cls))
        support = m.get("support")
        # Sayıları matristen çek; eşleşme kurulamazsa oranı yine göster ama
        # SAYI UYDURMA — sessizce yaklaşık bir değer basmaktansa notu kısa tut.
        tp = pred_n = None
        if i is not None and matrix:
            tp = matrix[i][i]
            pred_n = sum(matrix[r][i] for r in range(len(matrix)))

        recall_note = (
            (f"Gerçekte '{cls}' olan {support:,} kayıttan {tp:,} tanesini yakaladı."
             if tr else
             f"It caught {tp:,} of the {support:,} records that truly are '{cls}'.")
            if tp is not None and support else ""
        )
        ev.append(Evidence(
            label=(f"'{cls}' sınıfı — yakalama oranı" if tr else f"Class '{cls}' — recall"),
            value=m.get("recall"), provenance="test_set", sample_size=support,
            uncertainty=(wilson_interval(tp, support) if tp is not None and support else None),
            note=recall_note,
        ))

        precision_note = (
            (f"Modelin '{cls}' dediği {pred_n:,} kayıttan {tp:,} tanesi gerçekten '{cls}'ydı."
             if tr else
             f"Of the {pred_n:,} records the model called '{cls}', {tp:,} really were.")
            if tp is not None and pred_n else ""
        )
        ev.append(Evidence(
            label=(f"'{cls}' sınıfı — isabet oranı" if tr else f"Class '{cls}' — precision"),
            value=m.get("precision"), provenance="test_set", sample_size=pred_n,
            uncertainty=(wilson_interval(tp, pred_n) if tp is not None and pred_n else None),
            note=precision_note,
        ))

    return ev


def _metrics_unavailable_subsection(ev_report: dict, lang: str) -> Optional[ReportSection]:
    """Üretilemeyen metrikleri GÖRÜNÜR kılar — eksik metrik 0 DEĞİLDİR.

    Ölçülmüş hata sınıfı: model olasılık üretemediğinde ROC/PR-AUC/kalibrasyon
    hiç var olmaz; bunları sessizce atlamak raporu "bu model ölçülmedi" yerine
    "bu model iyi" okutur.
    """
    unavail = ev_report.get("metrics_unavailable") or {}
    if not unavail:
        return None
    affected = unavail.get("affected") or []
    return ReportSection(
        id="performance_unavailable",
        title=("Üretilemeyen ölçümler" if lang == "tr" else "Measurements not produced"),
        headline=(f"{len(affected)} ölçüm bu çalışmada hiç üretilemedi."
                  if lang == "tr" else
                  f"{len(affected)} measurements could not be produced in this run."),
        explanation=(
            "Seçilen model tahminlerine olasılık ataması yapamadı; olasılığa dayanan "
            "ölçümler bu yüzden YOK. Yok olmaları modelin kötü olduğu anlamına da "
            "gelmez, iyi olduğu anlamına da — bu sorular bu çalışmada yanıtsız kaldı."
            if lang == "tr" else
            "The selected model cannot attach probabilities to its predictions, so every "
            "probability-based measurement is ABSENT. Their absence means neither that the "
            "model is bad nor that it is good — these questions went unanswered in this run."
        ),
        evidence=[
            Evidence(label=name, value=None, provenance="test_set",
                     note=(f"Üretilemedi: {unavail.get('detail', '')}" if lang == "tr"
                           else f"Not produced: {unavail.get('detail', '')}"))
            for name in affected
        ],
        status="ok",
    )


# ── Aksiyon Yoğunlaşması (D3) alt-bölümü ─────────────────────────────────────

def _ac_note(kind: str, rep, base, lang: str) -> str:
    """Grup notu — NEDENSEL bağ kurmadan iki gözlemi yan yana koyar (yön netliği)."""
    r = f"{_fmt_value(rep)}×" if rep is not None else "—"   # Faz 3 #4: oran 3 hane
    b = f"{_fmt_value(base)}×" if base is not None else "—"
    if lang == "tr":
        return {
            "expected": (f"Aksiyon setinde over-temsil ediliyor ({r}) VE veride yüksek taban "
                         f"riskine sahip ({b}) — iki gözlem hizalı, beklenen konsantrasyon (kusur değil)."),
            "over_unexpected": (f"Aksiyon setinde over-temsil ({r}) ama grubun veri-taban riski yüksek "
                                f"değil ({b}) — beklenmedik; nereye bakılmalı."),
            "coverage_gap": (f"Veri-taban riski yüksek ({b}) ama aksiyon setinde az temsil ({r}) — model "
                             f"bu grubu decile DIŞINDA yakalıyor olabilir ya da kaçırıyor; incelenmeye değer."),
        }.get(kind, f"Popülasyon payına göre {r} temsil (taban riski {b}).")
    return {
        "expected": (f"Over-represented in the action set ({r}) AND carries a high base risk in the "
                     f"data ({b}) — the two observations are aligned, an expected concentration (not a defect)."),
        "over_unexpected": (f"Over-represented in the action set ({r}) but the group's data base risk is "
                            f"not high ({b}) — unexpected; worth a look."),
        "coverage_gap": (f"High data base risk ({b}) but under-represented in the action set ({r}) — the "
                         f"model may catch this group OUTSIDE the decile, or may be missing it; worth examining."),
    }.get(kind, f"Represented at {r} of its population share (base risk {b}).")


def _action_concentration_subsection(ev_report: dict, lang: str) -> Optional[ReportSection]:
    """En riskli %10'a KİMİN girdiği — dağılım gözlemi (adalet değil). Nedensellik yasağı:
    'yoğunlaşıyor' der, 'çünkü riskli' demez; base_rate_ratio yalnız hizalı-mı diye yan yana."""
    ac = ev_report.get("action_concentration") or {}
    if not ac:
        return None
    title = ("Aksiyon yoğunlaşması: en riskli %10 kim?" if lang == "tr"
             else "Action concentration: who is the riskiest 10%?")
    if ac.get("status") != "ok":
        return ReportSection(
            id="action_concentration", title=title, status="not_computed",
            not_computed_reason=R_REASON(ac.get("reason"), lang),
        )

    ev: list[Evidence] = []
    for col, cd in (ac.get("columns") or {}).items():
        for value, g in (cd.get("groups") or {}).items():
            if g.get("kind") in ("proportional", "aligned_or_weak", "undefined"):
                # Orantılıya yakın gruplar da GÖRÜNÜR (iki yön de raporlanır) ama nötr not.
                pass
            # Güvenilirlik n_decile'dan (oran ölçeğindeki rep_ci, 0–1 için tasarlı
            # width≤0.30 kuralını yanlış tetikler → Germany n=104'e sahte "ince" damgası).
            # rep_ci yine görünür ama NOT metninde; damga gerçek örneklemden.
            rc = g.get("rep_ci") or {}
            _ci_lbl = "%95 aralık" if lang == "tr" else "95% CI"
            ci_txt = (f" · {_ci_lbl} [{_fmt_value(rc.get('lower'))} — {_fmt_value(rc.get('upper'))}]"
                      if rc.get("lower") is not None else "")
            ev.append(Evidence(
                label=f"{col} = {value}",
                value=g.get("rep_ratio"),
                provenance="test_set",
                sample_size=g.get("n_decile"),
                note=_ac_note(g.get("kind"), g.get("rep_ratio"), g.get("base_rate_ratio"), lang) + ci_txt,
            ))
        for value, reason in (cd.get("skipped_groups") or {}).items():
            ev.append(Evidence(
                label=(f"{col} = {value} — ölçülmedi" if lang == "tr"
                       else f"{col} = {value} — not measured"),
                value=None, provenance="test_set", note=R_REASON(reason, lang),
            ))
    for col, reason in (ac.get("skipped_columns") or {}).items():
        ev.append(Evidence(
            label=(f"{col} — yoğunlaşma adayı değil" if lang == "tr"
                   else f"{col} — not a concentration candidate"),
            value=None, provenance="test_set", note=R_REASON(reason, lang),
        ))
    if not ev:
        return None

    n_top = ac.get("n_top")
    return ReportSection(
        id="action_concentration", title=title,
        headline=(f"Modelin işleme aldığı en riskli %10'un ({n_top} kayıt) grup dağılımı — "
                  f"1.0× orantılı, yüksek=over-temsil, düşük=under-temsil."
                  if lang == "tr" else
                  f"The group split of the riskiest 10% ({n_top} rows) the model would act on — "
                  f"1.0× is proportional, high=over-represented, low=under-represented."),
        explanation=(
            "Bu bölüm bir DAĞILIM gözlemidir, adalet ölçümü DEĞİL — en riskli %10'a kimin "
            "girdiğini TANIMLAR, sebebini açıklamaz. Yüksek yoğunlaşma tek başına kusur değildir: "
            "model gerçekten riskli bir grubu buluyor olabilir. Grubun veri-taban riski "
            "(base_rate) yalnızca 'iki gözlem hizalı mı' diye YAN YANA konur — yoğunlaşmanın "
            "SEBEBİ olarak değil."
            if lang == "tr" else
            "This is a DISTRIBUTION observation, not a fairness measurement — it DESCRIBES who "
            "falls in the riskiest 10%, it does not explain why. High concentration is not a "
            "defect on its own: the model may be finding a genuinely risky group. The group's "
            "data base risk (base_rate) is placed ALONGSIDE only to show whether the two "
            "observations are aligned — not as the CAUSE of the concentration."),
        evidence=ev, status="ok",
    )


def build_performance_section(state: dict, lang: str = "tr") -> ReportSection:
    """Model Performansı — test metrikleri + güven aralıkları + confusion + sınıf başına.

    Blob'dan taşınan ilk büyük bölüm. Buradaki her sayı TEST SETİNDEN gelir:
    model o satırları hiç görmedi, dolayısıyla en güçlü kanıt sınıfıdır.
    """
    ev_report = state.get("evaluation_report") or {}
    title = _PERF_TITLE.get(lang, _PERF_TITLE["tr"])
    test_m = ev_report.get("test_metrics") or {}
    intent = state.get("intent", "")

    # Denetimsiz analizlerde "test metrikleri yok" DEMEK yanlış çerçevedir: eksik
    # bir ölçüm değil, tanımsız bir sorudur. Gerçek etiket olmadan "doğru/yanlış"
    # diye bir karşılaştırma kurulamaz (fairness'ta da aynı ayrım yapılıyor).
    if intent and intent not in ("classification", "regression"):
        return ReportSection(
            id="performance", title=title, order=ORDER_PERFORMANCE,
            status="not_applicable",
            not_computed_reason=(
                f"Bu analiz tipinde ({intent}) 'doğru tahmin' diye bir karşılaştırma "
                f"tanımlı değil — karşılaştırılacak gerçek bir cevap yok. Bu analizin "
                f"kendi kalite ölçüleri raporun ilgili bölümünde veriliyor."
                if lang == "tr" else
                f"There is no notion of a 'correct prediction' for this analysis type "
                f"({intent}) — there is no true answer to compare against. This analysis "
                f"has its own quality measures, reported in their own section."
            ),
        )

    if not test_m:
        return ReportSection(
            id="performance", title=title, order=ORDER_PERFORMANCE,
            status="not_computed",
            not_computed_reason=(
                "Test metrikleri bu çalışmada üretilmedi — modelin başarısı ölçülemedi."
                if lang == "tr" else
                "Test metrics were not produced in this run — model performance is unmeasured."
            ),
        )

    cm = ev_report.get("confusion_matrix") or {}
    cm_ok = _cm_usable(cm)
    ci = ev_report.get("confidence_intervals") or {}
    # Matris kullanılabilirse örneklem ORADAN gelsin: rapordaki tablo ile n aynı
    # kaynaktan okunur, ikisi birbirini tutar.
    n_test = (
        sum(sum(row) for row in cm["matrix"]) if cm_ok
        else (state.get("feature_report") or {}).get("n_test")
    )

    evidence: list[Evidence] = []
    for key in _METRIC_ORDER:
        if key not in test_m:
            continue
        meta = _METRIC_META.get(key, {}).get(lang) or _METRIC_META.get(key, {}).get("tr")
        if not meta:
            continue
        label, note = meta
        uncertainty = _ci_for(ci, key)
        reference = ""
        # PR-AUC tek başına okunamaz: "0.78" iyi mi kötü mü, pozitif sınıfın
        # oranına bağlıdır. Taban oranı REFERANS olarak verilir — hüküm değil.
        if key == "pr_auc" and test_m.get("pr_auc_baseline") is not None:
            base = test_m["pr_auc_baseline"]
            reference = (
                f"Taban çizgisi {base}: hiçbir şey öğrenmeyen, rastgele tahmin eden bir "
                f"model bu değeri alırdı (verideki 'evet' oranı). Karşılaştırma bu sayıya göre yapılır."
                if lang == "tr" else
                f"Baseline {base}: a model that learned nothing and guessed at random would "
                f"score this (the share of 'yes' in the data). Compare against this number."
            )
        # #5 accuracy baseline: hep ÇOĞUNLUK sınıfını diyen modelin doğruluğu. Dengesizde
        # kritik — %79.65 taban → %83.9 model farkını okuyucu ancak böyle görür.
        if key == "accuracy" and test_m.get("accuracy_baseline") is not None:
            base = test_m["accuracy_baseline"]
            reference = (
                f"Taban çizgisi {base}: hep en sık görülen sınıfı diyen (hiçbir şey öğrenmeyen) "
                f"model bu doğruluğu alırdı. Modelin katkısı bu sayının ÜSTÜNE ne eklediğidir."
                if lang == "tr" else
                f"Baseline {base}: always predicting the most common class (learning nothing) "
                f"gives this accuracy. The model's contribution is what it adds ON TOP of this."
            )
        evidence.append(Evidence(
            label=label, value=test_m[key], provenance="test_set",
            sample_size=n_test,
            uncertainty=uncertainty if key in _RATE_METRICS else None,
            note=note, reference=reference,
        ))

    if ci:
        evidence.append(Evidence(
            label=("Skorun oynama payı ölçüldü mü" if lang == "tr"
                   else "Was the score's margin measured"),
            value=True, provenance="test_set", sample_size=n_test,
            note=("Yukarıdaki oransal ölçümlerin yanındaki %95 aralık, aynı model başka bir "
                  "test örneklemiyle sınansaydı skorun hangi bandda kalacağını gösterir "
                  "(tekrarlı yeniden örnekleme ile ölçüldü). Dar band = sonuç kararlı."
                  if lang == "tr" else
                  "The 95% interval next to each rate above shows the band the score would "
                  "fall in if the same model were tested on a different sample (measured by "
                  "repeated resampling). A narrow band means a stable result."),
        ))

    # #7 İLGİ SINIFI (azınlık/pozitif) recall + precision — weighted recall'ın (==accuracy)
    # yerine. Dengesizde toplam skor azınlığı gizler; ilgi sınıfının yakalama/isabeti ayrı
    # gösterilir ve digest'e girer → özet bunu yansıtır. Pozitif≠azınlık ise İKİSİ de.
    cb = ev_report.get("class_balance") or {}
    per_class = ev_report.get("per_class_metrics") or {}
    disp = _class_display(list((cb.get("distribution") or {}).keys()), lang)
    for lab in cb.get("interest_classes", []):
        pc = per_class.get(str(lab))
        if not pc:
            continue
        nm = disp.get(str(lab), str(lab))
        role = ""
        if lab == cb.get("minority_class") and lab == cb.get("positive_class"):
            role = (" (azınlık + pozitif)" if lang == "tr" else " (minority + positive)")
        elif lab == cb.get("positive_class"):
            role = (" (pozitif)" if lang == "tr" else " (positive)")
        elif lab == cb.get("minority_class"):
            role = (" (azınlık)" if lang == "tr" else " (minority)")
        evidence.append(Evidence(
            label=(f"İlgi sınıfı '{nm}'{role} — yakalama (recall)" if lang == "tr"
                   else f"Interest class '{nm}'{role} — catch (recall)"),
            value=pc.get("recall"), provenance="test_set", sample_size=pc.get("support"),
            note=(f"Gerçekte '{nm}' olanların yüzde kaçını model yakaladı. Toplam skor bunu "
                  f"gizler; dengesiz veride ASIL bakılacak sayı budur."
                  if lang == "tr" else
                  f"Of those truly '{nm}', what share the model caught. The overall score hides "
                  f"this; on imbalanced data this is the number that matters."),
        ))
        evidence.append(Evidence(
            label=(f"İlgi sınıfı '{nm}'{role} — isabet (precision)" if lang == "tr"
                   else f"Interest class '{nm}'{role} — accuracy of hits (precision)"),
            value=pc.get("precision"), provenance="test_set", sample_size=pc.get("support"),
            note=(f"Model '{nm}' dediğinde yüzde kaç haklı çıktı."
                  if lang == "tr" else
                  f"When the model said '{nm}', what share it got right."),
        ))

    # ── #9 KULLANILAN EŞİK + provenance + eşik-bağımlı/bağımsız ayrımı ──
    thr_info = ev_report.get("threshold") or {}
    if thr_info.get("value") is not None:
        prov = thr_info.get("provenance") or {}
        src = prov.get("source", "default_0.5")
        src_txt = ("katman-dışı çapraz-doğrulama (OOF, F1-max)" if src == "oof_cv_f1max"
                   else "varsayılan 0.5") if lang == "tr" else \
                  ("out-of-fold cross-validation (OOF, F1-max)" if src == "oof_cv_f1max" else "default 0.5")
        evidence.append(Evidence(
            label=("Kullanılan sınıflandırma eşiği" if lang == "tr" else "Classification threshold used"),
            value=thr_info["value"], provenance="cv", sample_size=prov.get("n"),
            note=(f"Bu olasılık kesme noktasının üstü '{disp.get('1','pozitif')}' sayılır. "
                  f"AYARLANABİLİR — değiştirirseniz aşağıdaki EŞİK-BAĞIMLI metrikler oynar: "
                  f"accuracy, F1, precision, recall, confusion matrix. EŞİK-BAĞIMSIZ (değişmez): "
                  f"ROC-AUC, PR-AUC, top-decile lift."
                  if lang == "tr" else
                  f"Scores above this probability cutoff count as '{disp.get('1','positive')}'. "
                  f"ADJUSTABLE — if you change it, the THRESHOLD-DEPENDENT metrics move: "
                  f"accuracy, F1, precision, recall, confusion matrix. THRESHOLD-INDEPENDENT "
                  f"(unchanged): ROC-AUC, PR-AUC, top-decile lift."),
            reference=(f"Kaynak: {src_txt} — eşik test setinde DEĞİL, sızıntısız seçildi."
                       if lang == "tr" else
                       f"Source: {src_txt} — chosen leakage-free, NOT on the test set."),
        ))

    # ── #9 EŞİK TAKASI: F1 düz-aralığının iki ucu (SEÇİM OOF, ÖLÇÜM test) ──
    tr_off = ev_report.get("threshold_tradeoff") or {}
    if tr_off.get("high_recall") and tr_off.get("high_precision"):
        hr, hp, ch = tr_off["high_recall"], tr_off["high_precision"], tr_off.get("chosen", {})
        evidence.append(Evidence(
            label=("Eşik takası (F1 bu aralıkta düz)" if lang == "tr"
                   else "Threshold trade-off (F1 is flat across this band)"),
            value=None, provenance="test_set",
            note=(f"Eşik bir İŞ KARARI: F1 bu komşulukta neredeyse sabit, ama recall↔precision "
                  f"takası kayar. Yüksek-recall ucu (eşik {hr['threshold']}): recall {hr['recall']}, "
                  f"precision {hr['precision']}. Yüksek-precision ucu (eşik {hp['threshold']}): "
                  f"recall {hp['recall']}, precision {hp['precision']}. Seçili (eşik {ch.get('threshold')}): "
                  f"recall {ch.get('recall')}, precision {ch.get('precision')}. Ürün slider'ı bunu canlı yapar."
                  if lang == "tr" else
                  f"The threshold is a BUSINESS DECISION: F1 is nearly constant across this band, but "
                  f"the recall↔precision trade-off shifts. High-recall end (threshold {hr['threshold']}): "
                  f"recall {hr['recall']}, precision {hr['precision']}. High-precision end "
                  f"(threshold {hp['threshold']}): recall {hp['recall']}, precision {hp['precision']}. "
                  f"Chosen (threshold {ch.get('threshold')}): recall {ch.get('recall')}, "
                  f"precision {ch.get('precision')}. The product slider does this live."),
            reference=("Aralık katman-dışı (OOF) eğriden seçildi · değerler test setinde ölçüldü."
                       if lang == "tr" else
                       "Band selected on the out-of-fold (OOF) curve · values measured on the test set."),
        ))

    # ── #9 TOP-DECILE LIFT: eşikten bağımsız iş metriği (baseline + tavan zorunlu) ──
    tdl = ev_report.get("top_decile_lift") or {}
    if tdl.get("lift") is not None and tdl.get("total_positives"):
        pct = round(tdl["capture"] * 100, 1)
        evidence.append(Evidence(
            label=("En riskli %10'un yakalama gücü (top-decile lift)" if lang == "tr"
                   else "Catching power of the riskiest 10% (top-decile lift)"),
            value=tdl["lift"], provenance="test_set", sample_size=tdl.get("n_top"),
            note=(f"En riskli %10'u alırsanız gerçek pozitiflerin %{pct}'ini yakalarsınız "
                  f"({tdl['positives_in_top']}/{tdl['total_positives']}). EŞİKTEN BAĞIMSIZ — "
                  f"sıralama kalitesi, kesme noktası değişse de sabit (bir avantaj)."
                  if lang == "tr" else
                  f"Taking the riskiest 10% catches {pct}% of the true positives "
                  f"({tdl['positives_in_top']}/{tdl['total_positives']}). THRESHOLD-INDEPENDENT — "
                  f"a ranking quality; unchanged if the cutoff moves (an advantage)."),
            reference=(f"Baseline (rastgele) 1.0× · tavan (mükemmel sıralama) {_fmt_value(tdl.get('ceiling'))}×. "
                       f"Referanssız lift anlamsız."
                       if lang == "tr" else
                       f"Baseline (random) 1.0× · ceiling (perfect ranking) {_fmt_value(tdl.get('ceiling'))}×. "
                       f"Lift is meaningless without these."),
        ))

    section = ReportSection(
        id="performance", title=title, order=ORDER_PERFORMANCE,
        evidence=evidence, status="ok",
        table=_confusion_table(cm, lang) if cm_ok else None,
    )

    # D3: "en riskli %10 kimi yakalıyor" (lift) → "peki KİM bu %10" — doğal akış, lift'in ardına.
    _ac = _action_concentration_subsection(ev_report, lang)
    if _ac:
        section.subsections.append(_ac)

    # #8 SINIF DENGESİZLİĞİNİN ADI KONUR — headline (digest'e girer, tüm metrikleri çerçeveler).
    # Kademe (balanced/imbalanced/severe) üniform-pay türevli eşikten (evaluator). Dağılım HER
    # ZAMAN yazılır; "dengesiz" nitelemesi + uyarı yalnız eşik altında (flag'i sonuçtan ayır).
    if cb.get("distribution"):
        _dist = " / ".join(
            (f"%{round(r*100, 2):g} {disp.get(str(lab), str(lab))}" if lang == "tr"
             else f"{round(r*100, 2):g}% {disp.get(str(lab), str(lab))}")
            for lab, r in sorted(cb["distribution"].items(), key=lambda kv: -kv[1]))
        _lvl = cb.get("level", "balanced")
        if _lvl == "severe":
            section.headline = (
                f"Sınıf dengesi: {_dist} — CİDDİ DENGESİZ. Bu rejimde ROC-AUC yanıltıcıdır; "
                f"PR-AUC birincil metriktir, doğruluk taban çizgisine kıyasla okunmalıdır."
                if lang == "tr" else
                f"Class balance: {_dist} — SEVERELY IMBALANCED. ROC-AUC is misleading in this "
                f"regime; PR-AUC is the primary metric, and accuracy must be read against its baseline.")
        elif _lvl == "imbalanced":
            section.headline = (
                f"Sınıf dengesi: {_dist} — DENGESİZ. Tüm metrikler bu ışıkta okunmalı; toplam "
                f"skorlar azınlık/ilgi sınıfının recall'ını gizler (aşağıda ayrı verildi)."
                if lang == "tr" else
                f"Class balance: {_dist} — IMBALANCED. Read all metrics in this light; overall "
                f"scores hide the minority/interest class's recall (given separately below).")
        else:
            section.headline = (f"Sınıf dengesi: {_dist} — dengeli." if lang == "tr"
                                else f"Class balance: {_dist} — balanced.")
    if per_class and cm_ok:
        section.subsections.append(ReportSection(
            id="performance_per_class",
            title=("Sınıf başına başarı" if lang == "tr" else "Per-class performance"),
            headline=(f"{len(per_class)} sınıfın her biri için yakalama ve isabet ayrı ölçüldü."
                      if lang == "tr" else
                      f"Catch and accuracy were measured separately for each of the "
                      f"{len(per_class)} classes."),
            explanation=(
                "Toplam skor sınıflar arasındaki farkı gizler: model büyük sınıfta çok iyi, "
                "küçük sınıfta zayıf olabilir ve ortalama yine yüksek çıkar. Aşağıdaki "
                "sayılar bu ayrımı açar. Az kayıtlı bir sınıfın oranı 'ölçüm ince' "
                "damgası taşıyorsa, o oran tek başına bir yargıya dayanak olmaz."
                if lang == "tr" else
                "An overall score hides differences between classes: the model can be strong "
                "on a large class and weak on a small one while the average still looks high. "
                "The numbers below open that up. Where a small class carries a 'thin "
                "measurement' stamp, its rate is not a basis for a judgement on its own."
            ),
            evidence=_per_class_evidence(per_class, cm, lang),
            status="ok",
        ))

    unavail = _metrics_unavailable_subsection(ev_report, lang)
    if unavail:
        section.subsections.append(unavail)

    all_scores = (state.get("metrics") or {}).get("all_model_scores") or {}
    if all_scores:
        chosen = state.get("model_name", "")
        section.subsections.append(ReportSection(
            id="performance_model_comparison",
            title=("Denenen diğer modeller" if lang == "tr" else "Other models tried"),
            headline=(f"{len(all_scores)} model denendi; en yüksek skoru alan seçildi."
                      if lang == "tr" else
                      f"{len(all_scores)} models were tried; the highest scoring one was chosen."),
            # Faz 3 #3: her satırda tekrar eden aynı not → BİR KEZ, listenin başında.
            explanation=("Aşağıdaki sayılar çapraz doğrulama skorlarıdır — modeller birbiriyle "
                         "tam bu skora göre karşılaştırıldı ve seçim bu sayıya dayanır."
                         if lang == "tr" else
                         "The numbers below are cross-validation scores — models were compared on "
                         "exactly this, and the choice rests on this number."),
            evidence=[
                # Faz 3 #4 İSTİSNASI: burada 3. ve 4. hane iki modeli AYIRIR (0.8581 vs
                # 0.8569); değer 4-haneli STRING olarak verilir → genel 3-hane kuralına
                # tabi olmaz (_fmt_value str dalını aynen basar). Bölüm-tipine özel, dinamik.
                Evidence(
                    label=(f"{name}{(' ← seçilen' if lang == 'tr' else ' ← selected') if name == chosen else ''}"),
                    value=f"{float(score):.4f}", provenance="cv",
                )
                for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            ],
            status="ok",
        ))

    return section


# ── Nedensellik dedektörü — KATMAN 3/3 ───────────────────────────────────────
# Prompt'a güvenmiyoruz (status rozetinde güvendik, tutmadı). LLM prose'u nedensel
# dil içeriyorsa YAKALANIR: bir kez daha, sertleştirilmiş talimatla denenir; yine
# kaçırırsa açıklama DÜŞÜRÜLÜR (başlık + kanıt kalır). Rapor asla nedensel iddia
# taşımaz — bu bir temenni değil, kod yolu.
#
# Kalıplar YÜKSEK GÜVENLİ olacak şekilde seçildi: "etkili/etki alanı" gibi masum
# kullanımlar yanlış alarm üretmesin diye fiil çekimleri hedeflenir.
_CAUSAL_PATTERNS = [
    r"\bneden\s+ol\w*", r"\bsebep\s+ol\w*", r"\byol\s+aç\w*",
    r"\betkiliyor\b", r"\betkiler\b", r"\betkilemekte\b", r"\betkisiyle\b",
    r"\bsürüklü\w*", r"\bkaynaklan\w*", r"\bbelirliyor\b", r"\bbelirler\b",
    r"\bsayesinde\b", r"\byüzünden\b", r"\bdolayısıyla\s+artı\w*",
    r"\bnedeniyle\b", r"\bsonucunda\s+artı\w*", r"\byol\s+açı\w*",
    r"\bcauses?\b", r"\bcaused\s+by\b", r"\bleads?\s+to\b", r"\bdrives?\b",
    r"\bdriven\s+by\b", r"\binfluences?\b", r"\bresults?\s+in\b",
    r"\bdue\s+to\b", r"\bbecause\s+of\b", r"\bdetermines?\b",
    # "age, gender play a crucial role" — ölçülmemiş bir SEBEP iddiası (özellik
    # önemi katkıyı ölçer, hedefi ürettiğini değil). Sentez raporunda ölçüldü.
    r"\bplays?\s+(?:a|an)\s+[\w\s-]{0,20}?\brole\b",
    # "insights into why customers leave" / "why certain customers may churn" —
    # hedefin NEDENİNİ açıklama iddiası. "why <özne>" kalıbı domain-özneleriyle
    # sınırlı tutulur ("why this matters" gibi meşru kullanımı yakalamasın diye).
    r"\binsights?\s+into\s+why\b",
    r"\bwhy\s+(?:certain\s+|some\s+|many\s+|the\s+)?(?:customers?|clients?|users?|people|they|accounts?|individuals?|churn\w*)\b",
]


def detect_causal_language(text: str) -> list[str]:
    """Metindeki nedensel ifadeleri bulur. Boş liste = temiz.

    Olumsuzlanan kullanımlar TEMİZ sayılır: "birinin diğerine sebep olduğunu
    GÖSTERMEZ" cümlesi istediğimiz çerçevenin ta kendisidir (bkz. _is_negated).
    """
    import re
    hits: list[str] = []
    for pat in _CAUSAL_PATTERNS:
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
            if _is_negated(text, m.start(), m.end()):
                continue
            hits.append(m.group())
    return hits


# Hata analizi için SPEKÜLASYON kalıpları: "model şu YÜZDEN yanılıyor" tarzı,
# ölçülmemiş bir açıklama iddiası. Nedensel kalıpların üstüne eklenir — bu bölümde
# hem "X hataya sebep oluyor" hem "muhtemelen şundandır" yasaktır.
_SPECULATION_PATTERNS = [
    r"\bçünkü\b", r"\bsebebi\b", r"\bnedeni\b", r"\bnedenler\w*\b",
    r"\bmuhtemelen\b", r"\bbüyük\s+ihtimalle\b", r"\bgörünüşe\s+göre\b",
    r"\bolduğu\s+için\b", r"\bbundan\s+dolayı\b", r"\bdemek\s+ki\b",
    r"\bbecause\b", r"\bthe\s+reason\b", r"\blikely\s+due\b", r"\bprobably\b",
    r"\bpresumably\b", r"\bsuggests?\s+that\b", r"\bimplies?\s+that\b",
    r"\bthis\s+is\s+why\b",
]

# Faz 3 #7 — VERİ/MODEL'i niteleyen editoryal fluff. KÖR TEK-KELİME DEĞİL, İSİM
# TAMLAMASI (sıfat + veri/model/analiz kelimesi): "robust standard errors" (meşru
# istatistik terimi) veya "comprehensive scan" (leakage bölümünün 'üç bağımsız
# kontrol' kullanımı) YANLIŞLIKLA yakalanmasın diye. Yalnız boş övgü kalıbı hedeflenir.
_EDITORIAL_FLUFF_PATTERNS = [
    r"\b(?:robust|comprehensive|rich|clean|high[-\s]quality|solid|powerful|excellent)\s+"
    r"(?:data\s?set|data|model|analysis|pipeline|feature\s+set)\b",
    r"\b(?:sağlam|kapsamlı|zengin|temiz|yüksek\s+kaliteli|güçlü|mükemmel)\s+"
    r"(?:veri\s*seti|veri|model|analiz(?:i|imiz)?|boru\s+hattı)\b",
    # modeli niteleyen pazarlama dili — bu raporlarda kanıta dayalı meşru kullanımı yok
    r"\bstate[-\s]of[-\s]the[-\s]art\b", r"\bcutting[-\s]edge\b", r"\bson\s+teknoloji\b",
]


# OLUMSUZLAMA PENCERESİ — yanlış alarm koruması.
# "hataların NEDENLERİNİ AÇIKLAMAMAKTADIR" cümlesi tam da İSTEDİĞİMİZ çerçevedir,
# ama içinde "nedenlerini" geçer. Dedektör bunu spekülasyon sanıp açıklamayı
# düşürdü (ölçüldü: LLM doğru cümleyi yazdı, ceza aldı). Bu yüzden bir eşleşmenin
# yakınında olumsuzlama varsa eşleşme SAYILMAZ.
#
# YÖN ÖNEMLİ — iki dil iki yerde olumsuzlar:
#   Türkçe: fiil EKİ, eşleşmeden SONRA  → "sebep olduğunu göster-MEZ"
#   İngilizce: yardımcı fiil, eşleşmeden ÖNCE → "does NOT show that X causes Y"
# Tek yönlü pencere İngilizceyi kaçırır (ölçüldü).
#
# Pencere dar tutulur: tipik kalıp ("sebep olduğunu göstermez" ~24, "does not show
# that" ~19 karakter) rahat sığar, ama cümlenin uzağındaki alakasız bir "değil"
# gerçek bir nedensel iddiayı aklayamaz.
_NEGATION_WINDOW = 40
_NEG_FORWARD_RE = (
    r"(m[ae]z\b|m[ae]y[ae]n\b|m[ıiuü]yor\b|m[ae]makta|m[ae]mekte|m[ae]dı\b|m[ae]di\b"
    # -mAmIş anlatı-geçmiş olumsuzu: "ölçülMEMİŞtir", "açıklanMAMIŞtır",
    # "belirlenMEMİŞ" — "X'in sebebi ölçülmemiştir" tam da istenen hedge'dir,
    # spekülasyon değil (hata analizindeki 'nedenlerini açıklamamaktadır' yanlış
    # alarmının kardeşi; \b yok çünkü -tir eki gelebilir).
    r"|m[ae]m[ıiuü]ş"
    r"|değil|yok\b)"
)
_NEG_BACKWARD_RE = r"(\bnot\b|n't\b|\bno\b|\bnever\b|\bnothing\b)"

# BU KALIPLAR OLUMSUZLANAMAZ: kendileri birer spekülasyon işaretidir, ardındaki
# fiilin olumsuz olması iddiayı ortadan kaldırmaz. "Görünüşe göre model bu grubu
# tanıMIYOR" hâlâ tahmindir — ölçülmemiş bir açıklamadır (ölçüldü: eski pencere
# bunu yanlışlıkla temiz sayıyordu).
_ALWAYS_SPECULATIVE = (
    r"\bçünkü\b", r"\bmuhtemelen\b", r"\bbüyük\s+ihtimalle\b", r"\bgörünüşe\s+göre\b",
    r"\bdemek\s+ki\b", r"\bprobably\b", r"\bpresumably\b",
)


def _is_negated(text: str, start: int, end: int) -> bool:
    """Eşleşmenin yakınında olumsuzlama var mı? (ileri: TR eki, geri: EN yardımcı fiili)"""
    import re
    after = text[end:end + _NEGATION_WINDOW]
    before = text[max(0, start - _NEGATION_WINDOW):start]
    return bool(re.search(_NEG_FORWARD_RE, after, flags=re.IGNORECASE)
                or re.search(_NEG_BACKWARD_RE, before, flags=re.IGNORECASE))


def detect_speculation(text: str) -> list[str]:
    """Nedensel VE spekülatif ("çünkü", "muhtemelen") ifadeleri bulur.

    Hata analizi bölümü için: bölüm hatanın NEREDE olduğunu gösterir, NEDEN
    olduğunu iddia etmez. "Model yaşlı müşterilerde yanılıyor çünkü ..." cümlesi
    elimizdeki kanıtın taşımadığı bir açıklamadır.

    Olumsuzlanan kullanımlar ("nedenlerini açıklamamaktadır") TEMİZ sayılır —
    aksi hâlde doğru çerçeveyi yazan cümle cezalandırılır.
    """
    import re
    hits = detect_causal_language(text)
    for pat in _SPECULATION_PATTERNS:
        always = pat in _ALWAYS_SPECULATIVE
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
            if not always and _is_negated(text, m.start(), m.end()):
                continue
            hits.append(m.group())
    # Faz 3 #7: veri/model'i niteleyen editoryal fluff. Olumsuzlanmışsa ("sağlam
    # DEĞİL") gerçek bir eleştiridir → temiz sayılır, aynı pencere korumasıyla.
    for pat in _EDITORIAL_FLUFF_PATTERNS:
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
            if _is_negated(text, m.start(), m.end()):
                continue
            hits.append(m.group())
    return hits


# ── Etkili Özellikler bölümü ─────────────────────────────────────────────────

_FEATURES_TITLE = {"tr": "En Etkili Özellikler", "en": "Most Influential Features"}

# NEDENSELLİK DERSİNİN TÜREVİ — önem sıralaması TANIMLAR, sebebini AÇIKLAMAZ.
# "X önemli çünkü hedefi belirliyor" cümlesi elimizdeki ölçünün taşımadığı bir
# iddiadır: importance yalnızca modelin bu alana ne kadar YASLANDIĞINI söyler,
# alanın hedefi ürettiğini değil. İlişkiler bölümündeki damganın kardeşi.
_IMPORTANCE_STAMP = {
    "tr": ("Bu bir KATKI ölçüsüdür: modelin kararını verirken bu alana ne kadar "
           "yaslandığını gösterir. Alanın hedefi BELİRLEDİĞİNİ göstermez — model "
           "başka bir alanla ilişkili olduğu için de bu alana yaslanmış olabilir."),
    "en": ("This is a CONTRIBUTION measure: it shows how much the model leaned on this "
           "field when deciding. It does NOT show the field DETERMINES the target — the "
           "model may lean on it only because it is correlated with another field."),
}

_FEATURES_DISCLAIMER = {
    "tr": ("> ⚠️ **Bu bölüm hangi alanların modelin kararında ağırlık taşıdığını gösterir, "
           "hedefi neyin belirlediğini değil.** Bir alanın üst sırada olması onu 'sebep' "
           "yapmaz; alt sırada olması da 'gereksiz' yapmaz. Sıralama modele özeldir — "
           "başka bir model başka bir sıralama verebilir."),
    "en": ("> ⚠️ **This section shows which fields carry weight in the model's decision, not "
           "what determines the target.** A field ranking high does not make it a 'cause'; "
           "ranking low does not make it 'useless'. The ranking is specific to this model — "
           "another model could rank them differently."),
}

_SHAP_NOTE = {
    "tr": ("SHAP: her alanın tahmine katkısını, TEST verisi üzerinde tek tek ölçen bir "
           "yöntem. Model bu satırları eğitimde görmedi, bu yüzden en güçlü katkı kanıtıdır."),
    "en": ("SHAP: a method that measures each field's contribution to the prediction, "
           "one by one, on the TEST data. The model never saw these rows in training, so "
           "this is the strongest contribution evidence."),
}

_FI_NOTE = {
    "tr": ("Modelin kendi iç ağırlığı: eğitim sırasında bu alana atadığı önem. Test "
           "verisinden değil, modelin iç yapısından okunur — SHAP'tan farklı bir ölçüdür."),
    "en": ("The model's own internal weight: the importance it assigned to this field "
           "during training. Read from the model's internals, not the test data — a "
           "different measure from SHAP."),
}

FEATURES_VISIBLE_TOP = 5


def build_features_section(state: dict, lang: str = "tr") -> ReportSection:
    """En Etkili Özellikler — SHAP (test_set) ve model iç önemi (model_internal).

    İki ölçü KARIŞTIRILMAZ: SHAP test verisinde ölçülen katkıdır, feature_importance
    modelin kendi iç ağırlığıdır. Her biri kendi provenance etiketini taşır; hangi
    sayının nereden geldiği okuyucuya açıktır (kullanıcı kararı).
    """
    ev_report = state.get("evaluation_report") or {}
    title = _FEATURES_TITLE.get(lang, _FEATURES_TITLE["tr"])

    shap = ev_report.get("shap_importance") or {}
    fi_sum = ev_report.get("feature_importance_summary") or {}
    shap_top = shap.get("top5_shap_features") or {}
    fi_top = fi_sum.get("top5_features") or {}

    if not shap_top and not fi_top:
        return ReportSection(
            id="features", title=title, order=ORDER_FEATURES,
            status="not_computed",
            not_computed_reason=(
                "Özellik önem analizi bu çalışmada üretilmedi (model önem skoru vermedi "
                "veya SHAP paketi yok)."
                if lang == "tr" else
                "Feature importance was not produced in this run (the model exposed no "
                "importance scores, or the SHAP package is unavailable)."
            ),
        )

    stamp = _IMPORTANCE_STAMP.get(lang, _IMPORTANCE_STAMP["tr"])
    shap_note = _SHAP_NOTE.get(lang, _SHAP_NOTE["tr"])
    fi_note = _FI_NOTE.get(lang, _FI_NOTE["tr"])

    def _shap_ev() -> list[Evidence]:
        return [
            Evidence(
                label=(f"{_hf(feat, lang, state)} — tahmine katkı payı"
                       if lang == "tr" else f"{_hf(feat, lang, state)} — contribution share"),
                value=pct, provenance="test_set", sample_size=shap.get("n_samples_used"),
                note=(f"Modelin kararına %{pct} katkı. {shap_note} {stamp}"
                      if lang == "tr" else f"{pct}% of the model's decision. {shap_note} {stamp}"),
            )
            for feat, pct in list(shap_top.items())[:FEATURES_VISIBLE_TOP]
        ]

    def _fi_ev() -> list[Evidence]:
        return [
            Evidence(
                label=(f"{_hf(feat, lang, state)} — modelin iç ağırlığı"
                       if lang == "tr" else f"{_hf(feat, lang, state)} — internal weight"),
                value=pct, provenance="model_internal",
                note=f"%{pct}. {fi_note} {stamp}" if lang == "tr" else f"{pct}%. {fi_note} {stamp}",
            )
            for feat, pct in list(fi_top.items())[:FEATURES_VISIBLE_TOP]
        ]

    # BİRİNCİL kanıt hangisi? SHAP varsa o (test setinde ölçülen katkı — en güçlü);
    # yoksa model iç önemi ÜST DÜZEYE çıkar. Aksi hâlde SHAP üretilmeyen modellerde
    # (ensemble'larda feature_importances_ yok → SHAP çöker) bölümün üst düzey kanıtı
    # boşalır, build_section_prose zemin bulamaz ve yorum SESSİZCE düşerdi. Ölçüldü:
    # adult'ta ensemble → SHAP boş → prose boş kalıyordu.
    primary_is_shap = bool(shap_top)
    section = ReportSection(
        id="features", title=title, order=ORDER_FEATURES,
        evidence=_shap_ev() if primary_is_shap else _fi_ev(),
        status="ok",
        disclaimer=_FEATURES_DISCLAIMER.get(lang, _FEATURES_DISCLAIMER["tr"]),
    )

    # Model iç önemi SHAP'la BİRLİKTE varsa AYRI alt bölümde (iki ölçü karışmasın).
    # SHAP yoksa zaten üstte; ikinci kez basılmaz.
    if primary_is_shap and fi_top:
        top5_pct = fi_sum.get("top5_total_contribution_pct")
        insignificant = fi_sum.get("insignificant_features_count")
        head = (f"Modelin kendi iç ağırlığına göre ilk {len(list(fi_top)[:FEATURES_VISIBLE_TOP])} alan."
                if lang == "tr" else
                f"The top {len(list(fi_top)[:FEATURES_VISIBLE_TOP])} fields by the model's own "
                f"internal weight.")
        exp_parts = []
        if top5_pct is not None:
            exp_parts.append(
                f"İlk 5 alan modelin toplam ağırlığının %{top5_pct}'ini taşıyor."
                if lang == "tr" else
                f"The top 5 fields carry {top5_pct}% of the model's total weight.")
        if insignificant:
            exp_parts.append(
                f"{insignificant} alan pratikte hiç ağırlık taşımıyor (%1 altı)."
                if lang == "tr" else
                f"{insignificant} fields carry practically no weight (under 1%).")
        section.subsections.append(ReportSection(
            id="features_model_internal",
            title=("Modelin iç önem sıralaması" if lang == "tr"
                   else "The model's internal importance"),
            headline=head, explanation=" ".join(exp_parts),
            evidence=_fi_ev(), status="ok",
        ))

    # SHAP birincilken model iç önemi yoksa: SHAP'ın test setinde ölçüldüğü not düşülür.
    # (İki ölçü de yoksa zaten yukarıda not_computed dönmüştük.)
    return section


# ── Sızıntı & Riskler bölümü ─────────────────────────────────────────────────

_LEAKAGE_TITLE = {"tr": "Sızıntı ve Riskler", "en": "Leakage & Risks"}

# Üç DEDEKTÖR, üç FARKLI yöntem. Karıştırılırsa okuyucu "aynı şey üç kez mi
# söyleniyor" sanır. Her biri kalem bazında ayrı alt bölüm + kendi yöntem etiketi.
# Not: sabit "üç" cümlesi KALDIRILDI. Koşan dedektör sayısı tek kaynaktan (ran_detectors)
# üretilir → intro sayısı ile gövdedeki alt bölüm sayısı YAPISAL olarak ayrışamaz (bkz.
# build_leakage_section: _checks_phrase + _LEAKAGE_DETECTOR_NAMES).
_LEAKAGE_INTRO = {
    "tr": ("Veri sızıntısı: modelin, gerçekte tahmin anında elde OLMAYACAK bir bilgiye "
           "yaslanması. Sızıntılı bir model testte parlak görünür ama sahada çöker."),
    "en": ("Data leakage: the model leaning on information that would NOT actually be "
           "available at prediction time. A leaky model looks brilliant in testing but "
           "collapses in the field."),
}

# Koşan sızıntı dedektörlerinin GÖRÜNEN adları — TEK KAYNAK. ran_detectors bu anahtarları
# içerir; intro sayısı, "temiz" başlığındaki ad listesi ve gövdedeki alt bölümler hep buradan
# türer. Yeni dedektör → buraya bir ad + ran_detectors'a bir anahtar + gövdeye render bloğu;
# sayı her yerde kendiliğinden güncellenir (status rozeti dersi).
_LEAKAGE_DETECTOR_NAMES = {
    "domain":      {"tr": "anlam bazlı",  "en": "semantic"},
    "target_lock": {"tr": "dilim kilidi", "en": "slice-lock"},
    "importance":  {"tr": "baskınlık",    "en": "dominance"},
}


def _checks_phrase(n: int, lang: str) -> str:
    """Koşan kontrol sayısını dilbilgisiyle üretir (intro sonu). Tek kaynak: n."""
    if lang == "tr":
        return f"{n} bağımsız kontrol yapıldı; her biri farklı bir açıdan bakar."
    unit = "check was" if n == 1 else "checks were"
    return f"{n} independent {unit} run, each from a different angle."

# not: purity_gain'in DOĞRU okunuşu — Wilson düzeltmesi bu dedektörün eski sessiz
# yanlış-alarmını kapatıyor. Ham oran değil, GENEL ORANIN ÜSTÜNE ne kadar EKLEDİĞİ.
_TARGET_LOCK_METHOD = {
    "tr": ("Yöntem: her sayısal alan dilimlere bölünür; bir dilimde sonuç neredeyse "
           "tek bir sınıfa kilitleniyorsa işaretlenir. Ölçü, o dilimdeki oranın veri "
           "genelindeki orana göre ne kadar ARTTIĞIDIR (küçük dilimlerde ham oran "
           "yanıltıcı olduğu için güven alt sınırıyla düzeltilir). Yalnızca işaretlenir, "
           "otomatik atılmaz — karar sizde."),
    "en": ("Method: each numeric field is split into slices; a slice is flagged if the "
           "outcome there locks onto almost a single class. The measure is how much the "
           "rate in that slice RISES above the overall rate (corrected with a confidence "
           "lower bound, because raw rates mislead in small slices). Flagged only, never "
           "auto-dropped — the call is yours."),
}


def _leakage_status_note(confidence: str, lang: str) -> str:
    """Dedektörün kendi 'sızıntı mı gerçek sinyal mi' yargısını taşır."""
    if confidence == "high":
        return ("Ani kopuş var → olası sızıntı, incelenmeli."
                if lang == "tr" else "Sharp break → possible leakage, worth examining.")
    return ("Kademeli ilişki → büyük olasılıkla gerçek sinyal, sızıntı değil."
            if lang == "tr" else "Gradual relationship → likely a real signal, not leakage.")


def build_leakage_section(state: dict, lang: str = "tr") -> ReportSection:
    """Sızıntı ve Riskler — üç bağımsız dedektör, kalem bazında ayrıştırılmış.

    1. domain_leakage_report  → LLM şema/anlam yargısı (schema_semantic)
    2. target_lock_leakage    → dilim saflığı + Wilson (test_set)  [ilk kez görünür]
    3. leakage_suspects       → importance yüzdesi sezgiseli (model_internal)

    KRİTİK: flag varsa NEDEN (hangi alan, kazanç ne) görünür; flag yoksa "temiz"
    diye AÇIKÇA söylenir — sessiz boşluk yok. "Kontrol edildi, bir şey çıkmadı"
    ile "kontrol edilmedi" farklı şeylerdir ve ayrı ayrı yazılır.
    """
    title = _LEAKAGE_TITLE.get(lang, _LEAKAGE_TITLE["tr"])
    ev_report = state.get("evaluation_report") or {}

    domain = state.get("domain_leakage_report") or {}
    domain_concerns = domain.get("concerns") or []
    target_lock = ev_report.get("target_lock_leakage") or []
    fi_sum = ev_report.get("feature_importance_summary") or {}
    suspects = fi_sum.get("leakage_suspects") or []

    # Hangi kontroller FİİLEN koştu? ("koştu ama temiz" ≠ "hiç koşmadı")
    domain_ran = "domain_leakage_report" in state
    importance_ran = bool(fi_sum)              # özet varsa importance+target_lock koştu
    any_ran = domain_ran or importance_ran

    if not any_ran:
        return ReportSection(
            id="leakage", title=title, order=ORDER_LEAKAGE_RISK,
            status="not_computed",
            not_computed_reason=(
                "Sızıntı kontrolleri bu çalışmada koşmadı (özellik önem bilgisi yok)."
                if lang == "tr" else
                "Leakage checks did not run in this analysis (no feature importance available)."
            ),
        )

    total_flags = len(domain_concerns) + len(target_lock) + len(suspects)

    # TEK KAYNAK: hangi dedektörler FİİLEN koştu. intro sayısı, "temiz" başlığındaki ad
    # listesi ve gövdedeki alt bölümler hep bu listeden türer → ayrışamazlar. domain bir
    # dedektör; importance koştuysa target_lock + importance (iki ayrı yöntem) koşar.
    ran_detectors: list[str] = []
    if domain_ran:
        ran_detectors.append("domain")
    if importance_ran:
        ran_detectors += ["target_lock", "importance"]
    n_checks = len(ran_detectors)
    ran_names = [_LEAKAGE_DETECTOR_NAMES[d][lang] for d in ran_detectors]

    section = ReportSection(
        id="leakage", title=title, order=ORDER_LEAKAGE_RISK, status="ok",
        disclaimer=("> " + _LEAKAGE_INTRO.get(lang, _LEAKAGE_INTRO["tr"])
                    + " " + _checks_phrase(n_checks, lang)),
    )

    # ── Alt bölüm 1: Domain (şema/anlam) ─────────────────────────────────────
    if domain_ran:
        if domain_concerns:
            high = [c for c in domain_concerns if c.get("confidence") == "high"]
            conf_label = {"high": "yüksek şüphe", "medium": "orta şüphe", "low": "düşük şüphe"} \
                if lang == "tr" else {"high": "high suspicion", "medium": "medium suspicion",
                                      "low": "low suspicion"}
            ev = [
                Evidence(
                    label=(f"{_hf(c.get('column_name', '?'), lang, state)} "
                           f"({c.get('relationship_type', '')})"),
                    value=conf_label.get(c.get("confidence", "low"), c.get("confidence")),
                    provenance="schema_semantic",
                    note=c.get("reasoning", ""),
                )
                for c in domain_concerns
            ]
            head = (f"{len(domain_concerns)} alan anlam olarak şüpheli bulundu"
                    + (f"; {len(high)} tanesi modelden otomatik çıkarıldı." if high else ".")
                    if lang == "tr" else
                    f"{len(domain_concerns)} fields were flagged as semantically suspicious"
                    + (f"; {len(high)} were auto-removed from the model." if high else "."))
            exp = (("Yüksek şüpheli alanlar otomatik çıkarıldığı için aşağıdaki performans "
                    "onlarsız ölçüldü — sahte yüksek skor değil, gerçek performans."
                    if high else
                    "Bu alanlar modelde bırakıldı ama anlamsal olarak dikkat çekiyor.")
                   if lang == "tr" else
                   ("Because the high-suspicion fields were auto-removed, the performance above "
                    "was measured without them — real performance, not an inflated score."
                    if high else
                    "These fields were kept in the model but stand out semantically."))
            section.subsections.append(ReportSection(
                id="leakage_domain", status="ok",
                title=("Anlam bazlı kontrol" if lang == "tr" else "Semantic check"),
                headline=head, explanation=exp, evidence=ev,
            ))
        else:
            section.subsections.append(ReportSection(
                id="leakage_domain", status="ok",
                title=("Anlam bazlı kontrol" if lang == "tr" else "Semantic check"),
                headline=("Anlam bazlı kontrol yapıldı; şüpheli alan bulunmadı."
                          if lang == "tr" else
                          "The semantic check ran; no suspicious field was found."),
                explanation=(domain.get("overall_assessment") or "").strip(),
            ))

    # ── Alt bölüm 2: Target-lock (dilim saflığı + Wilson) — İLK KEZ GÖRÜNÜR ───
    if importance_ran:
        method = _TARGET_LOCK_METHOD.get(lang, _TARGET_LOCK_METHOD["tr"])
        if target_lock:
            high = [c for c in target_lock if c.get("confidence") == "high"]
            ev = []
            for c in target_lock:
                gain = c.get("purity_gain")
                base = c.get("baseline")
                locked = c.get("locked_class")
                if gain is not None and base is not None:
                    detail = (f"'{c.get('bin')}' aralığında sonuçların çoğu '{locked}' sınıfına "
                              f"kilitleniyor; bu, veri genelindeki %{round(base*100, 1)} oranın "
                              f"belirgin üstünde (kazanç {gain})."
                              if lang == "tr" else
                              f"In the '{c.get('bin')}' range the outcome locks onto class "
                              f"'{locked}', clearly above the overall {round(base*100, 1)}% rate "
                              f"(gain {gain}).")
                else:
                    detail = (f"'{c.get('bin')}' aralığında hedefin oynaması neredeyse "
                              f"sıfıra düşüyor (saflık {c.get('purity')})."
                              if lang == "tr" else
                              f"In the '{c.get('bin')}' range the target's variation nearly "
                              f"collapses (purity {c.get('purity')}).")
                ev.append(Evidence(
                    label=(f"{_hf(c.get('feature', '?'), lang, state)} — dilim kilidi"
                           if lang == "tr" else
                           f"{_hf(c.get('feature', '?'), lang, state)} — slice lock"),
                    value=gain if gain is not None else c.get("purity"),
                    provenance="test_set",
                    sample_size=c.get("bin_size"),
                    note=f"{detail} {_leakage_status_note(c.get('confidence', 'low'), lang)}",
                ))
            head = (f"{len(target_lock)} alanda dilim kilidi işaretlendi"
                    + (f" ({len(high)} tanesi ani kopuş — olası sızıntı)." if high
                       else " (hepsi kademeli — büyük olasılıkla gerçek sinyal).")
                    if lang == "tr" else
                    f"{len(target_lock)} fields were flagged for a slice lock"
                    + (f" ({len(high)} with a sharp break — possible leakage)." if high
                       else " (all gradual — likely real signal)."))
            section.subsections.append(ReportSection(
                id="leakage_target_lock", status="ok",
                title=("Dilim kilidi kontrolü" if lang == "tr" else "Slice-lock check"),
                headline=head, explanation=method, evidence=ev,
            ))
        else:
            section.subsections.append(ReportSection(
                id="leakage_target_lock", status="ok",
                title=("Dilim kilidi kontrolü" if lang == "tr" else "Slice-lock check"),
                headline=("Dilim kilidi kontrolü yapıldı; işaretlenen alan yok."
                          if lang == "tr" else
                          "The slice-lock check ran; no field was flagged."),
                explanation=method,
            ))

        # ── Alt bölüm 3: Importance sezgiseli ────────────────────────────────
        if suspects:
            sev_label = {"high": "yüksek", "medium": "orta"} if lang == "tr" \
                else {"high": "high", "medium": "medium"}
            ev = [
                Evidence(
                    label=(f"{_hf(s.get('column', '?'), lang, state)} — {sev_label.get(s.get('severity'), s.get('severity'))} şüphe"
                           if lang == "tr" else
                           f"{_hf(s.get('column', '?'), lang, state)} — {sev_label.get(s.get('severity'), s.get('severity'))} suspicion"),
                    value=None, provenance="model_internal", note=s.get("reason", ""),
                )
                for s in suspects
            ]
            section.subsections.append(ReportSection(
                id="leakage_importance", status="ok",
                title=("Aşırı baskın alan kontrolü" if lang == "tr"
                       else "Dominant-field check"),
                headline=(f"{len(suspects)} alan tek başına modelin çok büyük kısmını taşıyor."
                          if lang == "tr" else
                          f"{len(suspects)} fields each carry an outsized share of the model."),
                explanation=(
                    "Tek bir alanın modelin neredeyse tamamını taşıması sızıntının klasik "
                    "işaretidir: o alan hedefin gizli bir kopyası olabilir. Kesin değildir, "
                    "ama bakmaya değer."
                    if lang == "tr" else
                    "One field carrying almost the entire model is a classic leakage sign: it "
                    "may be a hidden copy of the target. Not a certainty, but worth a look."),
                evidence=ev,
            ))
        else:
            # "koştu ama temiz" ≠ "hiç koşmadı" (bölüm ilkesi): baskınlık dedektörü de
            # temizken AÇIKÇA görünür. Böylece gövdedeki alt bölüm sayısı = ran_detectors
            # (intro sayısı) — yapısal olarak ayrışamazlar.
            section.subsections.append(ReportSection(
                id="leakage_importance", status="ok",
                title=("Aşırı baskın alan kontrolü" if lang == "tr"
                       else "Dominant-field check"),
                headline=("Aşırı baskın alan kontrolü yapıldı; işaretlenen alan yok."
                          if lang == "tr" else
                          "The dominant-field check ran; no field was flagged."),
                explanation=(
                    "Tek bir alanın modelin neredeyse tamamını taşıması sızıntının klasik "
                    "işaretidir; böyle bir alan bulunmadı."
                    if lang == "tr" else
                    "One field carrying almost the entire model is a classic leakage sign; "
                    "no such field was found."),
            ))

    # Üst başlık DETERMİNİSTİK: bu bölümün üst düzeyinde kendi kanıtı yok (kanıt alt
    # bölümlerde), yani LLM'e verilecek zemin yok. "Temiz" ile "işaret var" ayrımı
    # koddan yazılır — sessiz boşluk da, kanıtsız yorum da olmaz.
    if total_flags == 0:
        # ran_names/n_checks TEK KAYNAKTAN (yukarıdaki ran_detectors) — intro ile ayrışmaz.
        section.headline = (
            f"Sızıntı belirtisi yok: yapılan {n_checks} kontrolün "
            f"({', '.join(ran_names)}) hiçbiri bir işaret bulmadı."
            if lang == "tr" else
            f"No sign of leakage: none of the {n_checks} checks "
            f"({', '.join(ran_names)}) flagged anything.")
        section.explanation = (
            f"Bu, sızıntının kesin olmadığı anlamına gelmez — yalnızca bu {n_checks} yöntemin "
            "yakaladığı türden bir sızıntı görünmüyor. Yine de test skorları gerçekçi "
            "kabul edilebilir."
            if lang == "tr" else
            "This does not prove there is no leakage — only that no leakage of the kind "
            f"these {n_checks} methods catch is visible. The test scores can be taken as realistic.")
    else:
        section.headline = (
            f"{total_flags} işaret bulundu — her biri aşağıda yöntemiyle birlikte."
            if lang == "tr" else
            f"{total_flags} flags were raised — each shown below with its method.")
        section.explanation = (
            "Aşağıdaki her alt bölüm FARKLI bir yöntemin bulgusudur; aynı alan birden "
            "fazla kontrolde çıkabilir. Bir işaret kesin sızıntı demek değildir — "
            "nereye bakılacağını söyler."
            if lang == "tr" else
            "Each subsection below is a finding from a DIFFERENT method; the same field may "
            "appear in more than one. A flag is not proof of leakage — it says where to look.")

    return section


# ── Veri Kalitesi bölümü (kanıt-dayalı, kendi verisi var) ────────────────────

_DQ_TITLE = {"tr": "Veri Kalitesi", "en": "Data Quality"}


def build_data_quality_section(state: dict, lang: str = "tr") -> ReportSection:
    """Veri Kalitesi — boyut, karantina, örnekleme, final feature sayısı.

    SENTEZ DEĞİL: kendi deterministik verisi var (kaç satır karantinaya alındı,
    eksik oran, örneklendi mi). O yüzden diğer kanıt bölümleri gibi kurulur.
    """
    title = _DQ_TITLE.get(lang, _DQ_TITLE["tr"])
    profiling = state.get("profiling_report") or {}
    shape = profiling.get("shape") or {}
    fr = state.get("feature_report") or {}
    dq = state.get("data_quality_report") or {}

    rows = shape.get("rows")
    cols = shape.get("columns")
    if not rows and not fr.get("n_train"):
        return ReportSection(
            id="data_quality", title=title, order=ORDER_DATA_QUALITY,
            status="not_computed",
            not_computed_reason=(
                "Veri profili bu çalışmada üretilmedi."
                if lang == "tr" else "No data profile was produced in this run."),
        )

    evidence: list[Evidence] = []
    if rows is not None:
        evidence.append(Evidence(
            label=("Ham veri boyutu" if lang == "tr" else "Raw data size"),
            value=rows, provenance="full_data_descriptive",
            note=(f"{rows:,} satır × {cols:,} kolon" if lang == "tr" and cols
                  else (f"{rows:,} rows × {cols:,} columns" if cols else "")),
        ))

    missing = (profiling.get("summary") or {}).get("missing_rate")
    if missing is not None:
        evidence.append(Evidence(
            label=("Genel eksik veri oranı" if lang == "tr" else "Overall missing-data rate"),
            value=missing, provenance="full_data_descriptive",
            note=("Tüm hücrelerin ne kadarı boştu (0 = hiç eksik yok)."
                  if lang == "tr" else "Share of all cells that were empty (0 = none missing)."),
        ))

    # Karantina — kaç satır düşürüldü, NEDEN (sessiz düşme yok).
    quar = dq.get("quarantine") or {}
    q_rows = quar.get("rows") or 0
    if quar:
        evidence.append(Evidence(
            label=("Karantinaya alınan satır" if lang == "tr" else "Quarantined rows"),
            value=q_rows, provenance="full_data_descriptive",
            sample_size=quar.get("total"),
            note=(f"Hedefi tutarsız/bozuk olduğu için {q_rows:,} satır modele "
                  f"alınmadı (uyumsuzluk oranı {quar.get('mismatch_rate')})."
                  if lang == "tr" else
                  f"{q_rows:,} rows were excluded for an inconsistent/broken target "
                  f"(mismatch rate {quar.get('mismatch_rate')})."),
        ))

    # Modele fiilen giren
    if fr.get("n_train") is not None:
        evidence.append(Evidence(
            label=("Modele giren veri" if lang == "tr" else "Data used by the model"),
            value=fr.get("n_train"), provenance="train_only",
            note=(f"Eğitim {fr.get('n_train'):,} · test {fr.get('n_test'):,} satır; "
                  f"{fr.get('n_features')} özellik üretildi."
                  if lang == "tr" else
                  f"Train {fr.get('n_train'):,} · test {fr.get('n_test'):,} rows; "
                  f"{fr.get('n_features')} features built."),
        ))

    # Örnekleme (MAX_TRAINING_ROWS aşıldı) — görünür, sessiz değil.
    samp = (fr.get("sampling") or {})
    if samp.get("sampled"):
        evidence.append(Evidence(
            label=("Eğitim için örneklendi" if lang == "tr" else "Sampled for training"),
            value=samp.get("sampled_rows"), provenance="full_data_descriptive",
            sample_size=samp.get("original_rows"),
            note=(f"Veri üst sınırı ({samp.get('cap'):,}) aşıldığı için eğitim "
                  f"{samp.get('original_rows'):,} satırdan örneklendi."
                  if lang == "tr" else
                  f"Training was sampled from {samp.get('original_rows'):,} rows because the "
                  f"cap ({samp.get('cap'):,}) was exceeded."),
        ))

    status = "ok"
    dq_status = state.get("data_quality_status")
    section = ReportSection(
        id="data_quality", title=title, order=ORDER_DATA_QUALITY,
        evidence=evidence, status=status,
    )
    if dq_status == "degraded" and q_rows > 0:
        section.disclaimer = (
            f"> ⚠️ **Veri kalitesi kısmen düşük.** {q_rows:,} satır karantinaya alındı ama "
            f"modele devam edildi; aşağıdaki sonuçlar kalan veriyle üretildi."
            if lang == "tr" else
            f"> ⚠️ **Data quality is partly degraded.** {q_rows:,} rows were quarantined but "
            f"the run continued; the results below were produced on the remaining data.")
    return section


# ── Genelleme bölümü (kanıt-dayalı) — GÜVENİLİR mayınının yapısal panzehiri ──
# "Ölçüm kararlılığı ✓ / model güvenilirliği ✗" ayrımını burada KOD kurar: overfit
# verdict'i (CV−Test farkı) deterministik yazılır, LLM'in hükmüne bırakılmaz.

_GEN_TITLE = {"tr": "Genelleme ve Güvenilirlik", "en": "Generalization & Reliability"}

_OVF_STATUS_LABEL = {
    "tr": {
        "normal": "Model yeni veride de neredeyse test kadar iyi — genelleme sağlam.",
        "mild_overfitting": "Yeni veride hafif bir düşüş var — sınırda, izlenmeli.",
        "overfitting_detected": "Yeni veride belirgin düşüş — model eğitim verisine fazla uymuş olabilir.",
        "underfitting": "Model eğitim verisinde bile zayıf — kapasitesi yetersiz olabilir.",
        "inconclusive": "Test seti bu soruyu yanıtlayacak kadar kalın değil — genelleme kesinleştirilemedi.",
    },
    "en": {
        "normal": "The model does almost as well on new data as in testing — generalization is sound.",
        "mild_overfitting": "A slight drop on new data — borderline, worth monitoring.",
        "overfitting_detected": "A clear drop on new data — the model may have over-fit the training set.",
        "underfitting": "The model is weak even on training data — it may lack capacity.",
        "inconclusive": "The test set is too thin to answer this — generalization could not be settled.",
    },
}


def build_generalization_section(state: dict, lang: str = "tr") -> ReportSection:
    """Genelleme — CV−Test farkı (dürüst) + kalibrasyon. GÜVENİLİR mayınının panzehiri.

    Burada verdict KOD tarafından yazılır: LLM'e 'model güvenilir mi' diye sormayız,
    ölçülen CV−Test farkını deterministik yorumlarız. Böylece 'ölçüm kararlı' ile
    'model güvenilir' karışmaz — biri buradaki kanıt, diğeri hiç iddia edilmez.
    """
    title = _GEN_TITLE.get(lang, _GEN_TITLE["tr"])
    ev_report = state.get("evaluation_report") or {}
    ovf = ev_report.get("overfitting") or {}
    cal = ev_report.get("calibration") or {}

    if not ovf and not cal:
        return ReportSection(
            id="generalization", title=title, order=ORDER_GENERALIZATION,
            status="not_computed",
            not_computed_reason=(
                "Genelleme analizi bu çalışmada üretilmedi."
                if lang == "tr" else "Generalization analysis was not produced in this run."),
        )

    evidence: list[Evidence] = []
    if ovf:
        gap = ovf.get("gap")
        status_txt = _OVF_STATUS_LABEL.get(lang, _OVF_STATUS_LABEL["tr"]).get(
            ovf.get("status", ""), ovf.get("status", ""))
        label_disp = ovf.get("display_label", "CV")
        n_pos = ovf.get("n_test_positives")
        note = (f"{status_txt} Ölçülen fark ({label_disp}−Test): {gap}. "
                f"Bu fark, modelin test skorunun yeni veride ne kadar düşebileceğini "
                f"gösterir — SADECE buna bakılır, eğitim skoruna değil."
                if lang == "tr" else
                f"{status_txt} Measured gap ({label_disp}−Test): {gap}. This gap shows how "
                f"much the model's test score might drop on new data — read only this, not "
                f"the training score.")
        if ovf.get("status") == "inconclusive" and n_pos is not None:
            note += (f" (Test setinde yalnızca {n_pos} pozitif var.)"
                     if lang == "tr" else f" (Only {n_pos} positives in the test set.)")
        evidence.append(Evidence(
            label=("Yeni veriye genelleme (CV−Test farkı)" if lang == "tr"
                   else "Generalization to new data (CV−Test gap)"),
            value=gap, provenance="cv",
            # uncertainty KASITLI None: ovf.ci_lower/upper TEST metriğinin bootstrap CI'ıdır
            # (evaluator._analyze_overfitting'de inconclusive verdict'i için kullanılır) — GAP'in
            # CI'ı DEĞİL. Onu buraya "%95 aralık" diye basmak yanlış-etiketti (aralık gap değerini
            # içermiyordu bile). Test-ROC CI'ı zaten Model Performansı bölümünde doğru yerinde.
            # TODO: gap'in KENDİ CI'ı paired bootstrap (CV-fold vs test yeniden örnekleme) ile
            # hesaplanıp buraya eklenebilir; şimdilik yanlış CI göstermektense hiç gösterme.
            uncertainty=None,
            note=note,
        ))

    if cal and cal.get("brier_score") is not None:
        # #5 Brier baseline: taban-oranı tahmincisinin skoru (binary'de p(1-p)). Kazanım
        # mütevazı olabilir (churn: baseline 0.1621 → model 0.1448) — gösterilmesi gereken tam bu.
        brier_ref = ""
        if cal.get("brier_baseline") is not None:
            bb = cal["brier_baseline"]
            brier_ref = (
                f"Taban çizgisi {bb}: hep aynı taban oranını tahmin eden (hiçbir şey öğrenmeyen) "
                f"model bu Brier'i alırdı. Model bunun ALTINA ne kadar inerse o kadar iyi."
                if lang == "tr" else
                f"Baseline {bb}: always predicting the same base rate (learning nothing) gives "
                f"this Brier. The further BELOW it the model goes, the better.")
        evidence.append(Evidence(
            label=("Olasılık tahminlerinin gerçekçiliği (Brier)" if lang == "tr"
                   else "Realism of probability estimates (Brier)"),
            value=cal.get("brier_score"), provenance="test_set", reference=brier_ref,
            note=("0'a yakın olması iyidir: modelin '%70 eminim' dediğinde gerçekten "
                  "%70 civarı haklı çıktığı anlamına gelir."
                  if lang == "tr" else
                  "Close to 0 is good: it means when the model says '70% sure' it is right "
                  "about 70% of the time."),
        ))

    return ReportSection(
        id="generalization", title=title, order=ORDER_GENERALIZATION,
        evidence=evidence, status="ok",
    )


# ── Abartma dedektörü (Model Performansı) ────────────────────────────────────
# Bu bölümün riski nedensellik değil HÜKÜMDÜR: 0.92 ROC gören bir LLM "model
# üretime hazır" / "kusursuz çalışıyor" yazmaya meyleder. Status rozetinde tam da
# bu oldu (prompt "üretime hazır deme" diyordu, LLM yine yazdı) → yapısal karşılık.
#
# KAPSAM BİLEREK DAR: skoru niteleyen sıfatlar ("güçlü", "yüksek", hatta "mükemmel")
# YASAKLANMAZ — bunlar blob'da da serbest ve yasaklamak iki bölüm arasında tutarsızlık
# yaratırdı. Yakalanan şey KANITIN TAŞIMADIĞI İDDİA sınıfıdır: dağıtıma hazırlık,
# hatasızlık, garanti. Bunların hiçbiri test skorundan çıkarılamaz.
_OVERSTATEMENT_PATTERNS = [
    r"\büretime\s+hazır", r"\bcanlıya\s+alın\w*", r"\bdağıtıma\s+hazır",
    r"\bproduction[-\s]ready\b", r"\bready\s+for\s+production\b",
    r"\bkusursuz\w*", r"\bhatasız\w*", r"\byanılmaz\w*", r"\bher\s+zaman\s+doğru\b",
    r"\bflawless\w*", r"\bnever\s+wrong\b", r"\balways\s+correct\b",
    r"\bgaranti\w*", r"\bguarantee\w*",
    r"\b%\s?100\s+doğru\b", r"\b100%\s+accurate\b",
    # "güvenilir" bu projenin ölçülmüş mayını: uydurulan status rozeti tam olarak
    # "✅ Durum: Model güvenilir" idi ve blob prompt'u da koşulsuz "güvenilir/reliable"
    # demeyi zaten YASAKLIYOR. Gerçek LLM çıktısında yine çıktı ("Modelin Performansı
    # Yüksek ve Güvenilir") → dar kapsam yetmedi.
    # ÖLÇÜM KARARLILIĞI için ayrı kelime var ("kararlı", "dar band"); "güvenilir"
    # MODEL hakkında bir hükümdür ve test skoru onu kuramaz. \b eki "güvenilirlik"
    # (skor bölümü) ile karışmayı önler.
    r"\bgüvenilir\b", r"\bgüvenilirdir\b", r"\breliable\b", r"\btrustworthy\b",
]


def detect_overstatement(text: str) -> list[str]:
    """Kanıtın taşımadığı hüküm ifadelerini bulur. Boş liste = temiz.

    Olumsuzlanan kullanımlar TEMİZ: "model üretime hazır DEĞİLDİR" ve "bu skor
    kusursuz çalıştığını GÖSTERMEZ" istediğimiz çerçevenin ta kendisidir
    (bkz. _is_negated — hata analizi dedektöründe ölçülmüş yanlış alarm dersi).
    """
    import re
    hits: list[str] = []
    for pat in _OVERSTATEMENT_PATTERNS:
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
            if _is_negated(text, m.start(), m.end()):
                continue
            hits.append(m.group())
    return hits


# ── "Yeniden ifade edilmiş güçlü yön" dedektörü (Güçlü ve Zayıf Yönler) ───────
# Bu haftanın ÜÇÜNCÜ prompt-yetmez dersi (status rozeti, nedensel dil, "güvenilir"
# sonrası): LLM zayıflık listesine bir GÜÇLÜ yönü risk gibi yazıyor ("zayıf yön:
# model iyi genelliyor", "risk: overfit yok"). Prompt bunu açıkça yasaklıyordu ama
# yine kaçtı → yapısal karşılık. Nedensellik/abartma dedektörlerinin kardeşi.
#
# OLUMSUZLAMA KÖRLÜĞÜNE DÜŞME (bu hafta iki kez düştük — bkz. _is_negated, -mAmIş):
# "overfit riski VAR" MEŞRU bir zayıflıktır; "overfit YOK" zayıflık değildir. Bu
# yüzden pozitifi ZATEN OLUMLU biçimiyle eşleştiririz ("overfit yok/görülmedi"),
# çıplak "overfit" ile değil — gerçek zayıflık ("overfit var") asla eşleşmez.
#
# YANLIŞ ALARM KORUMASI iki katmanlı: pozitif ifade yalnızca (a) aynı cümlede
# OLUMSUZLANMAMIŞ bir zayıflık ÇERÇEVESİ ("zayıf yön", "risk") varken VE (b) o
# cümlede GERÇEK bir olumsuz bulgu (düşük, ince, kaçırıyor...) YOKKEN yakalanır.
# "Model dengeli ama azınlıkta zayıf" gibi ödünleme cümleleri (gerçek olumsuz var)
# cezalandırılmaz. "belirgin risk YOK" gibi olumsuzlanmış çerçeveler de tetiklemez.
_POSITIVE_STRENGTH_PATTERNS = [
    # genelleme İYİ. NOT: "genelle" DEĞİL "genell" — Türkçe ünlü düşmesi
    # (genelle→genelliyor) "genelle" substring'ini yok eder; ölçüldü, üç vaka patladı.
    r"\biyi\s+genell\w*", r"\bgenell\w*\s+(?:iyi|güçlü|başarılı|sağlam)",
    r"\b(?:sağlam|güçlü)\s+genell\w*",
    r"\bgeneraliz\w+\s+well\b", r"\b(?:good|strong|solid|sound)\s+generaliz\w*",
    # overfit YOK (olumlu) — yalnız olumsuzlanmış-kötü biçim (çıplak "overfit" DEĞİL)
    r"\b(?:overfit\w*|aşırı\s+öğrenme|ezber\w*)\s+(?:yok\w*|görülme\w+|bulunma\w+|saptanma\w+|tespit\s+edilme\w+)",
    r"\bno\s+(?:sign\s+of\s+)?overfit\w*", r"\bwithout\s+overfit\w*",
    r"\bnot\s+overfit\w*", r"\boverfit\w*\s+(?:is\s+)?absent\b",
    # performans YÜKSEK/GÜÇLÜ
    r"\byüksek\s+(?:performans\w*|skor\w*|doğruluk|başarı\w*|isabet|ayırt\w*|duyarlılık|kesinlik|ayrım)",
    r"\bgüçlü\s+(?:performans\w*|ayırt\w*|skor\w*|sonuç\w*|ayrım)",
    r"\bhigh\s+(?:performance|score|accuracy|precision|recall|discrimination)\b",
    r"\bstrong\s+(?:performance|discrimination|score|result\w*)\b",
    # dengeli / sağlam / kararlı ölçüm (olumlu özellikler)
    r"\bdengeli\b", r"\bsağlam\b", r"\bwell[-\s]balanced\b",
    r"\bbalanced\s+(?:class\w*|performance|result\w*)\b",
    r"\bkararlı\s+ölçüm\w*", r"\bölçüm\w*\s+kararlı\w*",
    r"\bstable\s+(?:measurement|performance|result\w*)\b", r"\brobust\b",
]

# "Bu bir ZAYIFLIK" çerçevesi (ETİKET). "zayıf yön/nokta/taraf" etikettir; çıplak
# "zayıf" YÜKLEMİ gerçek olumsuza girer.
_WEAKNESS_FRAMING_PATTERNS = [
    r"\bzayıf\s+(?:yön\w*|nokta\w*|taraf\w*)", r"\brisk\w*",
    r"\bsakınca\w*", r"\bdezavantaj\w*", r"\bendişe\w*", r"\bkaygı\w*",
    r"\bweakness\w*", r"\blimitation\w*", r"\bdrawback\w*", r"\bdownside\w*",
    r"\bconcern\w*", r"\bcaveat\w*", r"\bshortcoming\w*",
]

# GERÇEK olumsuz bulgu — cümlede varsa zayıflık gerçektir, pozitif ifade ödünlemedir
# (o cümleyi cezalandırma). "zayıf yön" ETİKETİ buraya GİRMEZ; çıplak olumsuz YÜKLEM girer.
# "overfit riski var" burada yakalanır → gerçek zayıflık sayılır, ASLA restated-strength değil.
_GENUINE_NEGATIVE_PATTERNS = [
    r"\bdüşük\w*", r"\byetersiz\w*", r"\beksik\w*", r"\bkaçır\w*", r"\bhata\w*",
    r"\bince\b", r"\bdengesiz\w*", r"\bgeniş\s+(?:aralık|band)", r"\bdar\s+(?:veri|band)",
    r"\baz\s+(?:pozitif|örnek|veri|sayı)", r"\bsızıntı\w*", r"\bgürültü\w*", r"\bbelirsiz\w*",
    r"\bzayıf\b(?!\s+(?:yön|nokta|taraf))",
    r"\b(?:overfit\w*|aşırı\s+öğrenme)\s+(?:var\w*|riski|mevcut|yüksek|tespit\s+edildi)",
    r"\blow\b", r"\bpoor\b", r"\bfew\b", r"\bmiss\w*", r"\berror\w*", r"\bthin\b",
    r"\bwide\s+(?:interval|band)", r"\bimbalanc\w*", r"\bleak\w*", r"\binsufficient\b",
    r"\bnoisy\b", r"\buncertain\b", r"\bweak\b(?!ness)",
]

# "RİSK YOK" beyanı — cümlede varsa hiçbir güçlü yön risk gibi SUNULMUYOR, tam tersine
# risk YOKLUĞU söyleniyor → cezalandırma. Marker'a dar-pencere olumsuzlama uygulamak
# yerine bu açık veto kullanılır: "risk: overfit görülmedi"deki '-medi' RİSK'i değil
# OVERFIT'i olumsuzlar; pencere ikisini ayıramadı (ölçüldü, iki vaka patladı). Veto
# yalnız olumsuzlama RİSK/ZAYIFLIK sözcüğüne BİTİŞİKKEN tetikler (araya kelime girerse
# değil) — böylece 'risk: overfit görülmedi' vetolanmaz ama 'risk tespit edilmedi' vetolanır.
_NO_RISK_DECLARATION = [
    r"\brisk\w*\s+(?:yok\w*|bulunmad\w*|saptanmad\w*|görülmed\w*|tespit\s+edilmed\w*|edilmed\w*)",
    r"\b(?:zayıf\s+(?:yön\w*|nokta\w*)|sakınca\w*|endişe\w*|kaygı\w*)\s+(?:yok\w*|bulunmad\w*)",
    r"\bno\s+(?:significant\s+|genuine\s+|major\s+|real\s+|notable\s+|meaningful\s+)?risk\w*",
    r"\brisk\w*\s+(?:\w+\s+){0,2}not\s+(?:detected|found|identified|present|there)",
    r"\bno\s+(?:significant\s+|genuine\s+|major\s+|real\s+)?(?:weakness|limitation|concern|drawback|shortcoming)\w*",
]


def _has_unnegated(text: str, patterns) -> bool:
    """Kalıplardan biri metinde OLUMSUZLANMAMIŞ biçimde geçiyor mu?"""
    import re
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            if not _is_negated(text, m.start(), m.end()):
                return True
    return False


def _first_marker_pos(text: str) -> Optional[int]:
    """Metindeki İLK zayıflık-çerçevesi etiketinin konumu (yoksa None)."""
    import re
    positions = [m.start() for pat in _WEAKNESS_FRAMING_PATTERNS
                 for m in re.finditer(pat, text, flags=re.IGNORECASE)]
    return min(positions) if positions else None


def detect_restated_strength(text: str) -> list[str]:
    """Zayıflık olarak sunulan GÜÇLÜ yönü yakalar. Boş liste = temiz.

    Kural (yüksek-kesinlik, yanlış-alarma karşı üç katman):
      1. "RİSK YOK" beyanı varsa cümle vetolanır ("risk tespit edilmedi", "no risk").
      2. Cümlede GERÇEK olumsuz bulgu (düşük, ince, kaçırıyor, "overfit riski var")
         varsa zayıflık gerçektir, pozitif ödünlemedir → dokunulmaz.
      3. Bir zayıflık ETİKETİ ("zayıf yön", "risk") pozitif ifadeden ÖNCE gelmeli —
         yani etiket pozitifi 'zayıflık' diye takdim ediyor olmalı. "Güçlü performans,
         riskler yönetilebilir" (pozitif ÖNCE, sonra risk) böylece cezalandırılmaz.

    Pozitifler ZATEN OLUMLU biçimde eşleştirilir ("overfit yok/görülmedi"), çıplak
    "overfit" ile değil — "overfit riski var" gerçek bir zayıflıktır ve asla eşleşmez.
    """
    import re
    hits: list[str] = []
    for seg in re.split(r"[.!?\n]+", text or ""):
        if not seg.strip():
            continue
        # (1) "risk yok" beyanı → güçlü yön risk gibi SUNULMUYOR, tam tersi
        if any(re.search(p, seg, flags=re.IGNORECASE) for p in _NO_RISK_DECLARATION):
            continue
        # (2) gerçek olumsuz bulgu varsa zayıflık gerçektir → dokunma
        if _has_unnegated(seg, _GENUINE_NEGATIVE_PATTERNS):
            continue
        # (3) zayıflık etiketi pozitiften ÖNCE gelmeli (etiket pozitifi takdim ediyor)
        marker_pos = _first_marker_pos(seg)
        if marker_pos is None:
            continue
        for pat in _POSITIVE_STRENGTH_PATTERNS:
            for m in re.finditer(pat, seg, flags=re.IGNORECASE):
                if m.start() > marker_pos:
                    hits.append(m.group())
    return hits


_PROSE_SYSTEM = """You write ONE section of a data-science report for a non-technical business reader (e.g. a bank manager).

You are given the section's EVIDENCE: numbers already computed by the system, each tagged with its source, sample size and uncertainty. You write only two things: a headline and an explanation.

ABSOLUTE RULES
- Use ONLY the numbers in the evidence. You have no other source. Never invent, estimate, round differently, or infer a number that is not there.
- Evidence marked "reliable: false" (thin sample / wide interval) MUST NOT be used to draw a conclusion. Say the measurement is too thin instead.
- Evidence with value null was NOT measured. Its note says why. Report it as not measured — never as zero, and never as good news.
- Do NOT deliver a verdict the evidence does not contain. Do not label anything as "biased", "fair", "high bias", "discriminatory" or "production-ready". Present what was measured and let the reader judge.
- Do NOT add flattering adjectives that qualify the DATA, MODEL or ANALYSIS unless that exact quality is in the evidence. Banned noun-phrases include "robust dataset", "comprehensive analysis", "rich/clean data", "powerful/state-of-the-art model" / "sağlam veri", "kapsamlı analiz", "zengin/temiz veri", "güçlü/mükemmel model". State what was measured; do not decorate it. (A statistical term like "robust standard errors" is fine — the ban is only the empty data/model compliment.)
- Explain every technical term in one plain sentence the first time it appears.
- NUMBER FORMATTING: each number in the evidence is already in its correct display form — rate metrics (accuracy, recall, precision, false-alarm/miss rate, selection/base rate) as percentages like "86,4%" / "%86,4"; scores (ROC-AUC, PR-AUC, MCC, correlations, Cohen's d, any "ratio"/DI) as plain decimals like "0.922". Reproduce each number in the SAME form shown. Never put % next to a plain 0-to-1 score, and never turn a shown percentage back into a raw decimal.
- Write in {language}. Every word of both fields must be in {language}.

FAIRNESS-SPECIFIC (if this is the fairness section)
- Two different things are measured and they are NOT the same:
  * false-alarm / miss rate differences  = PERFORMANCE parity (is the model equally accurate per group?)
  * selection-rate ratio (DI)            = SELECTION parity (how often does the model say yes per group?)
  A model can be equally accurate for every group and still select at very different rates — that is not a contradiction. If they diverge, say so plainly and explain that they answer different questions.
- MANDATORY — NEVER REPORT A SELECTION-RATE DIFFERENCE ON ITS OWN. A bare sentence like "the model says yes 50% of the time for group A but only 10% for group B" is misleading and is exactly what this report exists to prevent: on its own it implies the model is doing something to group B. Every single time you mention a selection rate, a selection-rate difference, or DI, you MUST in the SAME breath state what the data's own base rate ("verideki gerçek oran") is for those groups, and then say which of these two the evidence supports:
    (a) the model's selection ratio is close to the data's base-rate ratio (the "unexplained part" / "veriyle açıklanamayan kısım" is near 0) → the difference comes FROM THE DATA. The model is reflecting a real difference that already exists in the data; it is not creating it. Say this explicitly.
    (b) they diverge (the unexplained part is clearly away from 0) → the difference comes FROM THE MODEL and is worth examining. Say this explicitly, and give the size of the unexplained part.
  If you cannot tell because the evidence is unreliable, say that instead. A selection-rate sentence without its base-rate counterpart is a FAILED response.
- Never apply the 0.8-1.25 band as a verdict. It is context for the reader.

DATA-RELATIONSHIPS-SPECIFIC (if this is the "Veri İlişkileri" / "Data Relationships" section)
- These numbers measure CO-MOVEMENT (correlation) ONLY. They are NOT causal. This is the single most important rule for this section.
- BANNED — never write any of these, in any grammatical form: "causes", "leads to", "drives", "influences", "affects", "results in", "due to", "because of", "determines" / "neden olur", "sebep olur", "yol açar", "etkiler", "etkiliyor", "sürükler", "kaynaklanır", "belirler", "sayesinde", "yüzünden", "nedeniyle". A sentence like "income affects education" or "gelir eğitimi etkiliyor" is a FAILED response.
- REQUIRED phrasing instead: "move together", "rise together", "move in opposite directions", "is associated with" / "birlikte hareket ediyor", "birlikte artıyor", "ters yönde hareket ediyor", "ile ilişkili".
- You MUST include one sentence stating plainly that these are relationships, not causes, and that an unmeasured common driver could explain them.
- A strong relationship does NOT mean the field is important to the model, and a weak one does not mean it is useless. Do not rank "importance" here.

ERROR-ANALYSIS-SPECIFIC (if this is the "Hata Analizi" / "Error Analysis" section)
- This section reports WHERE errors concentrate. WHY the model errs was NOT measured and you have no evidence for it.
- BANNED — never explain or guess a reason: "because", "the reason is", "probably", "likely due to", "this suggests that", "presumably" / "çünkü", "sebebi", "nedeni", "muhtemelen", "büyük ihtimalle", "görünüşe göre", "olduğu için". A sentence like "the model errs on older customers because it has too few of them" is a FAILED response — that explanation was never measured.
- REQUIRED framing: state in which fields the wrong predictions differ from the correct ones and by how much, and say plainly that this shows where to look, not why the model is wrong.
- The separation size (Cohen's d) is unitless and comparable across fields; the raw averages ("42 vs 17") are in the field's own unit. Use the raw averages when they are given — they are what the reader understands. When an evidence note says there is no raw equivalent, do NOT present the scaled number as if it were a real-world value; just say that field separates by <d> and move on.
- "Confident but wrong" predictions are the operationally risky ones — worth stating plainly, without speculating why.

MOST-INFLUENTIAL-FEATURES-SPECIFIC (if this is the "En Etkili Özellikler" / "Most Influential Features" section)
- This section RANKS how much the model leaned on each field. It does NOT explain WHY a field matters, and importance is NOT causation.
- BANNED — never explain or assert a cause, in any grammatical form: "because", "the reason is", "determines", "drives", "influences", "causes", "leads to" / "çünkü", "sebebi", "nedeni", "belirliyor", "etkiliyor", "sürükler", "neden olur". A sentence like "education is the top feature because it determines income" / "eğitim en önemli çünkü geliri belirliyor" is a FAILED response.
- You MAY say a field carries weight / ranks high / contributes a large share ("etkili", "önemli", "yüksek katkı", "influential", "important"). That is a statement about the MODEL, and it is fine. The forbidden step is explaining WHY, or claiming the field acts on the target.
- DO NOT attach a "because" clause to a ranking. NEVER write "X is important because it has the highest contribution" / "X en etkili çünkü / X en yüksek katkıya sahip olduğu için önemli" — that is circular and BANNED (the word "olduğu için" / "sahip olduğu için" is forbidden here). Just state the rank and the number and STOP: "Model en çok X'e (%N), sonra Y'ye (%M) yaslanıyor." — a bare list, no justifying clause.
- Two different measures are supplied and they are NOT the same: SHAP contribution is measured on the TEST data; the model's internal importance is read from the model itself. If they disagree on the ranking, say so plainly — do not merge them into one list.
- A high rank does not make a field a cause; a low rank does not make it useless. The ranking is specific to this model.

MODEL-PERFORMANCE-SPECIFIC (if this is the "Model Performansı" / "Model Performance" section)
- SHOW, DO NOT JUDGE. Report what each number is and what it means in plain language. Do not inflate it into a verdict.
- BANNED — the evidence cannot support any of these, in any grammatical form: "production-ready", "ready to deploy", "flawless", "never wrong", "always correct", "guaranteed", "reliable", "trustworthy", "100% accurate" / "üretime hazır", "canlıya alınabilir", "kusursuz", "hatasız", "yanılmaz", "her zaman doğru", "garanti", "güvenilir". A test score measures past performance on one held-out sample; it cannot establish any of these.
- "reliable" / "güvenilir" is banned specifically as a claim about the MODEL. If you mean the MEASUREMENT is stable (a narrow 95% interval), say "kararlı" / "stable" / "dar band" / "the score barely moves across resamples" instead. These are different claims: a stable measurement of a mediocre model is still mediocre.
- Describing the SIZE of a score is fine and encouraged ("güçlü", "yüksek", "strong", "high") — it is the verdict about trustworthiness or deployment that the evidence cannot carry.
- The confusion table is rendered by the system directly beneath your text. Do NOT reproduce it, and do not restate every cell. You may refer to what it shows (e.g. false alarms vs missed cases) in one sentence.
- On imbalanced data an overall score hides per-class differences. If per-class evidence is supplied and a class is caught much less often than another, state that plainly with the counts given — this is a description, not a criticism.
- Translate rates into what they mean for the reader using ONLY the counts already in the evidence notes: "catches roughly half of the real cases" is good; inventing a business consequence ("so the bank loses money") is NOT — that was never measured.
- If a measurement is absent, say it was not produced. Never treat an absent metric as zero or as good news."""

_PROSE_HUMAN = """Section: {title}

EVIDENCE (the only numbers you may use):
{evidence}

Write the headline and explanation for this section."""


# ── Prose sayı biçimi (metrik tipine göre) ───────────────────────────────────
# Oran metrikleri (accuracy, recall, precision, FPR/FNR, seçim/taban oranı) prose'da
# YÜZDE okunur ("%86,4"); skorlar (ROC-AUC, PR-AUC, MCC, korelasyon, Cohen's d, DI,
# oran-oranı) ONDALIK kalır ("0.922"). Bu YALNIZCA LLM'in GÖRDÜĞÜ metni (digest +
# evidence-for-llm) biçimler → LLM formu kopyalar. KANIT KATMANI (render_evidence)
# DEĞİŞMEZ: orada kesin değer (0.8643) korunur (kullanıcının açık şartı).
_RATE_KW = (
    "accuracy", "doğruluk", "isabet", "recall", "duyarlılık", "precision", "kesinlik",
    "(fpr)", "(fnr)", "pozitif deme", "says positive", "actual rate",
    "verideki gerçek oran", "error rate", "hata oranı", "hatalı tahmin oranı",
)
# Bu işaretlerden biri varsa metrik oran DEĞİL skordur/orandır → ondalık kalır.
_RATE_EXCLUDE = (
    "ratio", "oranı oranı", "oran oranı", "gap", "fark", "(di)", "auc", "roc", "mcc",
    "correlation", "korelasyon", "cohen", "separation", "ayrışma", "impact",
)


def _is_rate_metric(label: str) -> bool:
    """Etiket bir ORAN metriği mi (yüzde okunmalı) — skor/oran-oranı DEĞİL?"""
    low = (label or "").lower()
    if any(x in low for x in _RATE_EXCLUDE):
        return False
    return any(k in low for k in _RATE_KW)


def _fmt_metric_for_llm(label, value, lang: str):
    """LLM'e gösterilecek sayı biçimi. Oran metriği + [0,1] değer → yüzde; aksi ham.

    KANIT KATMANINI ETKİLEMEZ — yalnız digest/evidence-for-llm metnini biçimler.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if _is_rate_metric(label) and 0.0 <= float(value) <= 1.0:
        s = f"{round(float(value) * 100, 1):g}"
        return f"%{s.replace('.', ',')}" if lang == "tr" else f"{s}%"
    return value


def _format_evidence_for_llm(section: ReportSection, lang: str) -> str:
    from src.core.report_schema import PROVENANCE_LABEL

    lines = []
    for e in section.evidence:
        prov = PROVENANCE_LABEL.get(lang, PROVENANCE_LABEL["tr"]).get(e.provenance, e.provenance)
        _v = _fmt_metric_for_llm(e.label, e.value, lang) if e.value is not None else "ÖLÇÜLMEDİ (null)"
        parts = [f"- {e.label}: {_v}"]
        parts.append(f"source={prov}")
        if e.sample_size is not None:
            parts.append(f"n={e.sample_size}")
        if e.uncertainty:
            parts.append(f"95% CI=[{e.uncertainty.get('lower')}, {e.uncertainty.get('upper')}]")
        parts.append(f"reliable={str(e.reliable).lower()}")
        line = " | ".join(parts)
        if e.note:
            line += f"\n    note: {e.note}"
        if e.reference:
            line += f"\n    reference: {e.reference}"
        lines.append(line)
    return "\n".join(lines)


_CAUSAL_RETRY_SUFFIX = """

⛔ YOUR PREVIOUS ATTEMPT WAS REJECTED. It contained causal language: {hits}.
These numbers measure co-movement only and CANNOT support a causal claim. Rewrite using
"birlikte hareket ediyor" / "birlikte artıyor" / "ters yönde hareket ediyor" / "ile ilişkili"
(or, in English, "move together" / "are associated with"). Do not use any word implying that
one field produces, drives, affects or explains another."""

_SPECULATION_RETRY_SUFFIX = """

⛔ YOUR PREVIOUS ATTEMPT WAS REJECTED. It contained causal or speculative language: {hits}.
This section reports WHERE the errors concentrate. WHY the model errs was NOT measured, so you
cannot state or guess a reason. Remove every "because" / "the reason" / "probably" / "çünkü" /
"sebebi" / "muhtemelen" construction. Describe only what was measured: in which fields the wrong
predictions differ from the correct ones, and by how much."""

# Bölüm başına prose denetimi: (dedektör, yeniden-deneme eki).
# Prompt'a bel bağlanmaz — yakalanırsa bir kez sertleştirilmiş talimatla denenir,
# yine kaçırırsa açıklama yayınlanmaz.
_FEATURES_RETRY_SUFFIX = """

⛔ YOUR PREVIOUS ATTEMPT WAS REJECTED. It contained a justifying / causal clause: {hits}.
The most common mistake is attaching a "because" clause to the ranking — e.g. "...en yüksek katkıya
sahip OLDUĞU İÇİN önemli" or "...ÇÜNKÜ...". That is circular and forbidden.
Rewrite as a BARE ranking with NO justifying clause. Use exactly this shape and nothing more:
"Model kararını verirken en çok <alan> (%<n>), ardından <alan> (%<n>) ve <alan> (%<n>) alanlarına
yaslanıyor. Bu sıralama bu modele özeldir." Do NOT add any "olduğu için" / "çünkü" / "sebebi" /
"belirliyor" / "etkiliyor" / "because" / "determines" clause. Just the fields, their percentages, full stop."""

_OVERSTATEMENT_RETRY_SUFFIX = """

⛔ YOUR PREVIOUS ATTEMPT WAS REJECTED. It contained claims the evidence cannot support: {hits}.
A test score measures how the model did on one held-out sample. It CANNOT establish that the model
is production-ready, flawless, guaranteed, trustworthy or reliable. Remove every such claim.
You MAY still describe how large a score is ("güçlü", "yüksek", "strong"). If you meant that the
MEASUREMENT is stable, use "kararlı" / "stable" / "dar band" instead — that is a claim about the
interval, not about the model. State what the numbers are and leave the verdict to the reader."""

_PROSE_GUARDS: dict = {
    "data_relations":           (detect_causal_language, _CAUSAL_RETRY_SUFFIX),
    "data_relations_all":       (detect_causal_language, _CAUSAL_RETRY_SUFFIX),
    "data_relations_redundant": (detect_causal_language, _CAUSAL_RETRY_SUFFIX),
    "error_analysis":           (detect_speculation, _SPECULATION_RETRY_SUFFIX),
    "error_analysis_all":       (detect_speculation, _SPECULATION_RETRY_SUFFIX),
    "error_analysis_raw":       (detect_speculation, _SPECULATION_RETRY_SUFFIX),
    "performance":             (detect_overstatement, _OVERSTATEMENT_RETRY_SUFFIX),
    "performance_per_class":   (detect_overstatement, _OVERSTATEMENT_RETRY_SUFFIX),
    # "X önemli çünkü hedefi belirliyor" → nedensellik dersinin türevi.
    "features":                (detect_speculation, _FEATURES_RETRY_SUFFIX),
    "features_model_internal": (detect_speculation, _FEATURES_RETRY_SUFFIX),
}


def build_section_prose(section: ReportSection, lang: str = "tr") -> ReportSection:
    """Bölümün headline/explanation'ını LLM ile yazdırır.

    Kanıt YAZILMAZ — LLM'in şemasında o alan yok (report_schema.LLM_SECTION_SCHEMA).
    Çağrı başarısız olursa bölüm kanıtlarıyla birlikte prose'suz kalır: rapor yine
    üretilir, kanıt yine görünür, yalnızca yorum cümlesi eksik olur.

    Nedensellik korumasının 3. katmanı burada: ilişki bölümlerinde çıktı nedensel dil
    içeriyorsa bir kez sertleştirilmiş talimatla YENİDEN denenir; yine kaçırırsa
    açıklama DÜŞÜRÜLÜR. Rapor nedensel iddia taşımaz — prompt'un tutmasına
    bel bağlanmaz (status rozetinde bağlandı, tutmadı).
    """
    if section.status != "ok" or not section.evidence:
        return section  # boş/uygulanmamış bölüme yorum yazılmaz

    from src.core.report_i18n import LANGUAGE_NAME
    from langchain_core.messages import HumanMessage, SystemMessage

    guard = _PROSE_GUARDS.get(section.id)
    system = _PROSE_SYSTEM.format(language=LANGUAGE_NAME[lang])
    human = _PROSE_HUMAN.format(
        title=section.title, evidence=_format_evidence_for_llm(section, lang)
    )

    try:
        llm = get_llm(model_name=settings.DEFAULT_LLM_MODEL, temperature=0.3)
        structured = llm.with_structured_output(LLM_SECTION_SCHEMA)

        for attempt in (1, 2):
            result = structured.invoke([
                SystemMessage(content=system), HumanMessage(content=human),
            ])
            if not isinstance(result, dict):
                break
            headline = (result.get("headline") or "").strip()
            explanation = (result.get("explanation") or "").strip()

            if guard is None:
                section.headline, section.explanation = headline, explanation
                break

            detector, retry_suffix = guard
            hits = detector(f"{headline} {explanation}")
            if not hits:
                section.headline, section.explanation = headline, explanation
                break

            logger.warning(
                f"[report_sections] '{section.id}' yasaklı dil içeriyor {hits} "
                f"(deneme {attempt}/2) — {'yeniden deneniyor' if attempt == 1 else 'açıklama düşürülüyor'}."
            )
            if attempt == 1:
                system = system + retry_suffix.format(hits=", ".join(sorted(set(hits))))
            else:
                # Son çare: kural dışı prose YAYINLANMAZ. Kanıt katmanı zaten kendi
                # damgasını taşıyor, bölüm okunur kalır.
                section.headline = ""
                section.explanation = ""
                logger.error(
                    f"[report_sections] '{section.id}': iki denemede de yasaklı dil — "
                    f"açıklama yayınlanmadı, kanıt katmanı korundu."
                )
    except Exception as e:
        logger.warning(
            f"[report_sections] '{section.id}' prose üretilemedi ({e}) — "
            f"bölüm kanıtlarıyla birlikte yorumsuz kalıyor."
        )
    return section


# ═══ SENTEZ BÖLÜMLERİ (Özet / Güçlü-Zayıf / Öneriler) ════════════════════════
#
# Bunların KENDİ KANITI YOKTUR — yukarıdaki kanıt bölümlerine dayanırlar. Risk:
# LLM burada üstte olmayan yeni bir sayı ya da iddia uydurabilir.
#
# YAPISAL PANZEHİR: sentez LLM'ine ham state DEĞİL, yalnızca kurulmuş bölümlerin
# DIGEST'i verilir. Ardından prose'daki her sayı digest'in izinli-sayı kümesinde
# mi diye KOD denetler (detect_fabrication). Nedensellik/abartma guard'larıyla
# aynı desen: "uydurma" yasak değil, yakalanır → sertleştirilmiş talimatla yeniden
# denenir → yine kaçırırsa açıklama düşürülür. Sentez üstteki kanıta bağlı kalır.


def _iter_sections(sections: list[ReportSection]):
    """Bölüm ağacını (alt bölümler dahil) düzleştirir."""
    for s in sections:
        yield s
        yield from _iter_sections(s.subsections)


_NUM_RE = None  # lazy


def _numbers_in(text: str) -> list[str]:
    import re
    global _NUM_RE
    if _NUM_RE is None:
        _NUM_RE = re.compile(r"\d[\d.,]*")
    return _NUM_RE.findall(text or "")


def _norm_num(s: str) -> str:
    """Sayıyı kanonik biçime çevirir. Türkçe ondalık VİRGÜLÜNÜ dürüstçe ele alır.

    Türkçe'de ondalık ayracı virgüldür ('92,2' = 92.2) ama binlikte de virgül
    kullanılır ('6,508' = 6508). Ayrım: virgülden sonra TAM 3 hane → binlik;
    1-2 hane → ondalık. Bu ayrım olmadan '92,2' → '922' olur ve doğru bir Türkçe
    yüzde, izinli kümede ('92.2') bulunamayıp UYDURMA sanılırdı.
    """
    s = s.strip()
    if "." in s and "," in s:
        # Karışık ('1,234.56'): İngilizce varsay — virgül binlik, nokta ondalık.
        s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        # '0,...' ASLA binlik değildir (kimse '0'ı binlik grup yazmaz) → ondalık.
        # Diğer durumda: virgülden sonra 1-2 hane ondalık, tam 3 hane binlik.
        if len(parts) == 2 and (parts[0] in ("0", "") or 1 <= len(parts[1]) <= 2):
            s = parts[0] + "." + parts[1]     # ondalık virgül
        else:
            s = s.replace(",", "")            # binlik virgül
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def collect_allowed_numbers(sections: list[ReportSection]) -> set[str]:
    """Kurulmuş bölümlerin TÜM sayılarını toplar — sentezin izinli kümesi.

    Değer, örneklem, belirsizlik, not, headline, explanation ve tablo hücreleri:
    üstte GÖRÜNEN her sayı. Ayrıca oranların yüzde formu (0.87 → 87) eklenir —
    LLM 0.87'yi '%87' yazarsa bu uydurma değil, meşru gösterimdir.
    """
    allowed: set[str] = set()

    def _add(v):
        if v is None:
            return
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            allowed.add(_norm_num(str(v)))
            if 0 <= float(v) <= 1:  # oran → yüzde formu da meşru (2/1/0 ondalık)
                pct = float(v) * 100
                allowed.add(_norm_num(f"{round(pct, 2)}"))
                allowed.add(_norm_num(f"{round(pct, 1)}"))
                allowed.add(_norm_num(f"{round(pct)}"))
        else:
            for tok in _numbers_in(str(v)):
                allowed.add(_norm_num(tok))

    for sec in _iter_sections(sections):
        for field in (sec.headline, sec.explanation, sec.not_computed_reason):
            _add(field)
        for e in sec.evidence:
            _add(e.value)
            _add(e.sample_size)
            _add(e.note)
            if e.uncertainty:
                _add(e.uncertainty.get("lower"))
                _add(e.uncertainty.get("upper"))
        if sec.table:
            for row in (sec.table.get("rows") or []):
                for cell in row:
                    _add(cell)
    # Küçük tam sayılar (sıra, "3 öneri" gibi) her zaman serbest — uydurma sayılmaz.
    allowed |= {str(i) for i in range(0, 11)}
    return allowed


def detect_fabrication(text: str, allowed: set[str]) -> list[str]:
    """Prose'da izinli kümede OLMAYAN sayıları bulur. Boş liste = temiz.

    Sentezin can damarı: bir sayı üstteki kanıt bölümlerinde geçmiyorsa, sentez
    onu UYDURMUŞTUR (ya da yeniden üretmiştir). İkisi de yasak.

    YUVARLAMA TOLERANSI: bir özetin "yaklaşık %87" demesi (üstteki 87.05'in
    yuvarlanması) uydurma DEĞİLDİR — mevcut sayının daha kaba gösterimidir. Bir aday,
    izinli bir değerin KENDİ hassasiyetine yuvarlanmışına eşitse temizdir. Ama "0.95"
    (üstteki 0.922'nin yuvarlaması değil) yakalanır: 0.922→0.92, 0.87→0.87, hiçbiri 0.95.
    """
    allowed_floats: list[float] = []
    for a in allowed:
        try:
            allowed_floats.append(float(a))
        except ValueError:
            pass

    hits: list[str] = []
    for tok in _numbers_in(text):
        n = _norm_num(tok)
        if n in allowed:
            continue
        try:
            c = float(tok.replace(",", ""))
        except ValueError:
            continue
        dec = len(n.split(".")[1]) if "." in n else 0
        # Üstteki bir değer, adayın hassasiyetine yuvarlandığında adaya eşit mi?
        if any(round(a, dec) == c for a in allowed_floats):
            continue
        hits.append(tok)
    return hits


# #11(a): öneri digest'ine taşınan nedensellik damgası. error_analysis öneri girdisinde
# KALIR (aksiyon alınabilir "nerede bakılmalı") ama LLM onu SEBEP diye kullanmasın diye
# çerçeve digest'e de basılır — bölümün explanation'ı digest'e taşınmadığından (yalnız
# headline+kanıt), damga aksi hâlde kaybolurdu.
_DIGEST_CAUSAL_NOTE = {
    "tr": "- ⚠️ Bu bölüm hatanın NEREDE olduğunu gösterir, NEDENİNİ değil — öneride 'sebep' olarak kullanma.",
    "en": "- ⚠️ This section shows WHERE errors occur, not WHY — do not use it as a cause in a recommendation.",
}


def _build_digest(
    sections: list[ReportSection],
    lang: str,
    *,
    exclude_ids: frozenset = frozenset(),
    causal_note_ids: frozenset = frozenset(),
) -> str:
    """Kurulmuş bölümlerin özetini sentez LLM'ine verilecek metne çevirir.

    Her bölümün başlığı + kurulmuş kanıtın ana sayıları. Sentez SADECE bunu görür;
    ham state'i görmez, dolayısıyla üstte olmayan bir şeyi 'hatırlayamaz'.

    exclude_ids: bu id'li bölümler (ve alt-bölümleri) digest'e HİÇ girmez. #11(a) için
        öneriler hedefinde {"features"} — özellik önemi katkıyı ölçer, aksiyon değildir;
        digest'te olması LLM'i "credit_score'u iyileştir" gibi aksiyonizasyona iter
        (nedensel FİİL yok → regex kaçırır). Kaynağı kesmek tek yapısal çözüm.
    causal_note_ids: bu bölümlerin başlığına _DIGEST_CAUSAL_NOTE eklenir (KALIR ama
        "sebep değil" damgalı).
    """
    from src.core.report_schema import PROVENANCE_LABEL
    prov_map = PROVENANCE_LABEL.get(lang, PROVENANCE_LABEL["tr"])
    _unmeasured = "ölçülmedi" if lang == "tr" else "not measured"
    _thin = " (ölçüm ince)" if lang == "tr" else " (thin measurement)"
    lines: list[str] = []
    for sec in sections:
        if sec.raw_markdown is not None:
            continue
        if sec.id in exclude_ids:
            continue   # #11(a): hedefli dışlama (öneriler için importance) — alt-bölümler de gider
        title = sec.title or sec.id
        if sec.status in ("not_applicable", "not_computed"):
            lines.append(f"### {title}: {sec.not_computed_reason}")
            continue
        lines.append(f"### {title}")
        if sec.id in causal_note_ids:
            lines.append(_DIGEST_CAUSAL_NOTE.get(lang, _DIGEST_CAUSAL_NOTE["tr"]))
        if sec.headline:
            lines.append(f"- {sec.headline}")
        for e in sec.evidence[:8]:
            val = _fmt_metric_for_llm(e.label, e.value, lang) if e.value is not None else _unmeasured
            prov = prov_map.get(e.provenance, e.provenance)
            line = f"- {e.label}: {val} [{prov}]"
            if not e.reliable:
                line += _thin
            # BASELINE'lar reference'ta yaşıyor (accuracy/PR-AUC/Brier taban çizgisi). Digest'e
            # taşınmazsa sentez "83.9% strong" der, taban çizgisini görmez → yanıltıcı özet.
            if e.reference:
                line += f"\n    baseline/ref: {e.reference}"
            lines.append(line)
        # Alt bölümler: başlık + headline + İLK BİRKAÇ kanıt satırı. Yalnız başlık
        # yetmiyordu — feature detayı (SHAP/ham-birim) subsection kanıtında yaşıyor;
        # digest onu taşımayınca sentez "özellikler detaylandırılmadı" diye SPEKÜLASYON
        # üretiyordu (gerçek koşuda ölçüldü). Subsection sayıları collect_allowed_numbers'ın
        # izinli kümesinde zaten var (_iter_sections recursive) → fabrication'a düşmez.
        for sub in sec.subsections:
            if sub.status in ("not_applicable", "not_computed"):
                continue
            if sub.headline:
                lines.append(f"  · {sub.title}: {sub.headline}")
            for e in sub.evidence[:3]:
                val = _fmt_metric_for_llm(e.label, e.value, lang) if e.value is not None else _unmeasured
                prov = prov_map.get(e.provenance, e.provenance)
                lines.append(f"    - {e.label}: {val} [{prov}]")
    return "\n".join(lines)


_SUMMARY_TITLE = {"tr": "Özet", "en": "Summary"}
_ASSESSMENT_TITLE = {"tr": "Güçlü ve Zayıf Yönler", "en": "Strengths & Weaknesses"}
_RECS_TITLE = {"tr": "Öneriler ve Sonraki Adımlar", "en": "Recommendations & Next Steps"}


def build_summary_section(state: dict, order: int = ORDER_SUMMARY) -> ReportSection:
    """Özet — sentez. Kullanıcının sorusunu yanıtlar + genel tablo. Kanıtı yok."""
    lang = normalize_lang_local(state)
    return ReportSection(
        id="summary", title=_SUMMARY_TITLE.get(lang, _SUMMARY_TITLE["tr"]),
        order=order, status="ok",
    )


_RISK_NAME = {
    "severe_imbalance":         {"tr": "Ciddi sınıf dengesizliği", "en": "Severe class imbalance"},
    "minority_recall":          {"tr": "Azınlık sınıf yakalama", "en": "Minority-class recall"},
    "confident_errors":         {"tr": "Emin-hatalar", "en": "Confident errors"},
    "brier_vs_baseline":        {"tr": "Olasılık kalitesi (Brier)", "en": "Probability quality (Brier)"},
    "overfitting":              {"tr": "Genelleme (CV−Test)", "en": "Generalization (CV−Test)"},
    "leakage":                  {"tr": "Sızıntı işaretleri", "en": "Leakage flags"},
    "near_duplicates":          {"tr": "Near-duplicate özellikler", "en": "Near-duplicate features"},
    "normalization_collisions": {"tr": "Kolon adı çakışması", "en": "Column-name collision"},
    "data_quality":             {"tr": "Veri kalitesi", "en": "Data quality"},
    "fairness":                 {"tr": "Adalet (fairness)", "en": "Fairness"},
    "action_concentration":     {"tr": "Aksiyon yoğunlaşması", "en": "Action concentration"},
}
_RISK_PROV = {"overfitting": "cv", "leakage": "model_internal",
              "normalization_collisions": "schema_semantic"}


def build_assessment_section(state: dict) -> ReportSection:
    """Güçlü/Zayıf — ZAYIF taraf DETERMİNİSTİK risk listesinden (kod üretir, ölçmez, toplar).
    LLM madde ekleyemez/çıkaramaz. "Temiz" ≠ "ölçülmedi": koşan/koşmayan kontroller ayrı."""
    lang = normalize_lang_local(state)
    from src.services.risk_assessment import build_risk_list
    rl = build_risk_list(state)
    risks, clean, not_run = rl["risks"], rl["clean"], rl["not_run"]
    sec = ReportSection(
        id="assessment", title=_ASSESSMENT_TITLE.get(lang, _ASSESSMENT_TITLE["tr"]),
        order=ORDER_ASSESSMENT, status="ok",
    )

    def _nm(cid: str) -> str:
        return _RISK_NAME.get(cid, {}).get(lang, cid)

    if risks:
        by_sev: dict = {}
        for r in risks:
            by_sev[r.severity] = by_sev.get(r.severity, 0) + 1
        sev_txt = ", ".join(f"{n} {s}" for s, n in
                            sorted(by_sev.items(), key=lambda kv: -{"info": 0, "warn": 1, "critical": 2}[kv[0]]))
        sec.headline = (f"{len(risks)} risk/uyarı bulundu ({sev_txt}). Bunlar 'nereye bakılmalı' der."
                        if lang == "tr" else
                        f"{len(risks)} risks/warnings found ({sev_txt}). These say 'where to look'.")
        for r in risks:
            sec.evidence.append(Evidence(
                label=f"[{r.severity}] {_nm(r.id)}", value=r.value,
                provenance=_RISK_PROV.get(r.id, "test_set"),
                note=r.detail.get(lang, r.detail.get("tr", "")),
            ))
        # Rule 3 (tam şeffaflık): risk VARKEN bile temiz koşan kontroller compact listelenir —
        # "sadece bunlara baktık" izlenimi olmasın; her kontrolün koştuğu görünsün.
        if clean:
            names = ", ".join(_nm(c.id) for c in clean)
            sec.evidence.append(Evidence(
                label=(f"Ayrıca temiz koşan {len(clean)} kontrol" if lang == "tr"
                       else f"{len(clean)} other checks ran clean"),
                value=None, provenance="test_set",
                note=(f"İşaret bulmayan kontroller: {names}." if lang == "tr"
                      else f"Checks that flagged nothing: {names}."),
            ))
    else:
        ran = ", ".join(_nm(c.id) for c in clean)
        sec.headline = (f"{len(clean)} risk kontrolü koştu, hiçbiri işaret bulmadı: {ran}."
                        if lang == "tr" else
                        f"{len(clean)} risk checks ran, none flagged anything: {ran}.")

    # Rule 3: KOŞMAYAN kontrol her zaman AÇIKÇA listelenir — "temiz" sayılmaz (sessiz boşluk yok).
    for c in not_run:
        sec.evidence.append(Evidence(
            label=(f"[koşmadı] {_nm(c.id)}" if lang == "tr" else f"[not run] {_nm(c.id)}"),
            value=None, provenance=_RISK_PROV.get(c.id, "test_set"),
            note=c.not_run_reason.get(lang, c.not_run_reason.get("tr", "")),
        ))
    return sec


def build_recommendations_section(state: dict) -> ReportSection:
    # ÖNERİLER HENÜZ KANITA-ÇAPALI DEĞİL (assessment'ın aksine). Bölüm boş kurulur;
    # prose'u build_synthesis_prose DIGEST'ten yazar. Nedensellik koruması İKİ katman:
    #   1) detect_speculation (⊇ detect_causal_language), fail-closed (kalıcı ihlal → düşürülür);
    #   2) #11(a): DIGEST'ten importance KESİLDİ → LLM'in causalize edeceği kaynak yok.
    #
    # TODO (#11-b, İLERİDE — kapatılmadı): TAM determinizm için öneriler serbest-sentez
    # olmaktan çıkıp risk listesinden (build_risk_list) TÜRETİLMELİ — her öneri bir bulguya
    # kod-tarafı çapalı, LLM yalnız anlatır (assessment/Grup B deseni). (a) kaynağı kesti ama
    # öneriler hâlâ kendi kanıtını taşımıyor; bu yüzden regex son savunma hattı olmaya devam
    # ediyor. "Neden öneriler assessment gibi kanıt-çapalı değil?" → burası cevabı.
    lang = normalize_lang_local(state)
    return ReportSection(
        id="recommendations", title=_RECS_TITLE.get(lang, _RECS_TITLE["tr"]),
        order=ORDER_RECOMMENDATIONS, status="ok",
    )


def normalize_lang_local(state: dict) -> str:
    l = (state.get("report_language") or "tr").strip().lower()
    return "en" if l.startswith("en") else "tr"


_SYNTHESIS_SYSTEM = """You write ONE synthesis section of a data-science report for a non-technical business reader (e.g. a bank manager).

This section has NO evidence of its own. You are given a DIGEST of the sections already written above (their findings and numbers). You synthesize — you do NOT introduce anything new.

ABSOLUTE RULES
- Use ONLY facts and numbers that appear in the DIGEST. NEVER introduce a number that is not in the digest. If a number is not in the digest, you may not state it — not even "roughly", not even rounded differently. This is the single most important rule.
- Do NOT re-derive or recompute anything. Refer to what the sections above established; do not produce a new figure.
- Every word in {language}. Explain any technical term in one plain sentence.
- Do NOT deliver a verdict the digest does not contain. NEVER call the model "reliable" / "trustworthy" / "production-ready" / "güvenilir" / "üretime hazır". A stable MEASUREMENT (a narrow gap/interval) is not the same as a reliable MODEL — if you mean the measurement is stable, say "kararlı" / "stable". Present what was found and let the reader judge.
- Be honest and calibrated: reflect both strengths and limitations. Do not overstate.
- Do NOT decorate the DATA, MODEL or ANALYSIS with empty compliments. Banned noun-phrases: "robust dataset", "comprehensive analysis", "rich/clean data", "powerful/state-of-the-art model" / "sağlam veri", "kapsamlı analiz", "zengin/temiz veri", "güçlü/mükemmel model". State findings; do not decorate them.
- NUMBER FORMATTING: every number in the digest is ALREADY written in its correct display form. Rate metrics (accuracy, recall, precision, false-alarm/miss rate, selection/base rate) appear as percentages like "86,4%" / "%86,4"; scores (ROC-AUC, PR-AUC, MCC, correlations, Cohen's d, any "ratio"/DI) appear as plain decimals like "0.922". REPRODUCE EACH NUMBER IN THE SAME FORM THE DIGEST SHOWS. Never attach a % sign to a plain 0-to-1 score (never "%0.922" — that reads as 0.9 percent), and never turn a shown percentage back into a raw decimal.
{section_specific}"""

_SUMMARY_SPECIFIC = {
    "tr": """
BU BÖLÜM: ÖZET
- 3-4 cümlelik yönetici özeti. Kullanıcının asıl sorusuna DOĞRUDAN yanıt ver (soru digest'te "Kullanıcı sorusu" olarak var).
- Hangi problem çözüldü, model ne kadar iyi (digest'teki performans sayısıyla), ve kullanıcının sorusuna kısa yanıt.
- Yeni sayı UYDURMA — yalnızca digest'tekiler.""",
    "en": """
THIS SECTION: SUMMARY
- A 3-4 sentence executive summary. Directly answer the user's original question (it is in the digest as "User question").
- What problem was solved, how good the model is (using the digest's performance number), and a short answer to the user's question.
- Do NOT invent numbers — only those in the digest.""",
}

_ASSESSMENT_SPECIFIC = {
    "tr": """
BU BÖLÜM: GÜÇLÜ VE ZAYIF YÖNLER
- İki kısa liste: güçlü yönler ve zayıf yönler/riskler. Her madde digest'teki BİR bulguya dayanmalı.
- "Güçlü yön: ayırt etme gücü yüksek (ROC 0.92)" derken o sayı digest'ten gelmeli, yeniden üretme.
- Genelleme için SADECE digest'teki CV−Test farkına bak; eğitim skorundan overfit çıkarma.
- "Model güvenilir" DEME. Ölçüm kararlıysa "kararlı" de. Güçlü yön = ne iyi ölçüldü; zayıf yön = nerede sınır var.
- ZAYIF YÖN LİSTESİ YALNIZCA GERÇEK RİSK/SINIR İÇERİR: düşük bir metrik, ince örneklem,
  belirsiz (geniş aralıklı) ölçüm, sızıntı şüphesi, gruplar arası fark gibi. Bir GÜÇLÜ yön
  (iyi genelleme, overfit YOK, yüksek performans, kararlı ölçüm) ASLA zayıf yön DEĞİLDİR —
  onu zayıflık listesine yazma, "risk" gibi yeniden ifade etme.
- Digest'te gerçek bir zayıflık/risk YOKSA: zayıf yön listesi TEK satırdır —
  "Bu çalışmada belirgin bir risk tespit edilmedi." Güçlü yönü tekrar etme, liste doldurma.""",
    "en": """
THIS SECTION: STRENGTHS & WEAKNESSES
- Two short lists: strengths and weaknesses/risks. Each item must rest on ONE finding in the digest.
- "Strength: strong discrimination (ROC 0.92)" — that number must come from the digest, do not re-derive.
- For generalization use ONLY the digest's CV−Test gap; never infer overfitting from a training score.
- Do NOT say the model is "reliable". If a measurement is stable, say "stable". Strength = what measured well; weakness = where the limits are.
- THE WEAKNESS LIST CONTAINS ONLY GENUINE RISKS/LIMITS: a low metric, a thin sample, an
  uncertain (wide-interval) measurement, suspected leakage, a between-group gap, etc. A STRENGTH
  (good generalization, NO overfitting, high performance, a stable measurement) is NEVER a
  weakness — do not put it in the weakness list, do not reword it as a "risk".
- If the digest holds no genuine weakness/risk: the weakness list is a SINGLE line —
  "No significant risk was detected in this run." Do not restate a strength, do not pad the list.""",
}

_RECS_SPECIFIC = {
    "tr": """
BU BÖLÜM: ÖNERİLER VE SONRAKİ ADIMLAR
- 3-5 somut, öncelikli öneri. Her öneri digest'teki BİR bulguya DAYANMALI — genel ML tavsiyesi UYDURMA.
- YASAK: "şunu yaparsanız model iyileşir" gibi ölçülmemiş nedensel iddialar. Bunun yerine kanıta bağla:
  "digest'te N sütun neredeyse aynı bilgiyi taşıyor → birini çıkarmayı değerlendirin."
  "test seti ince (az pozitif) → daha çok pozitif örnek toplayın."
- Bir öneri digest'teki bir bulguya bağlanamıyorsa, YAZMA.""",
    "en": """
THIS SECTION: RECOMMENDATIONS & NEXT STEPS
- 3-5 concrete, prioritized recommendations. Each MUST rest on ONE finding in the digest — do NOT invent generic ML advice.
- BANNED: unmeasured causal claims like "if you do X the model will improve". Instead tie to evidence:
  "the digest shows N columns carry nearly the same information → consider removing one."
  "the test set is thin (few positives) → collect more positive examples."
- If a recommendation cannot be tied to a digest finding, do NOT write it.""",
}

_SYNTHESIS_SPECIFIC = {
    "summary": _SUMMARY_SPECIFIC,
    "assessment": _ASSESSMENT_SPECIFIC,
    "recommendations": _RECS_SPECIFIC,
}

_SYNTHESIS_HUMAN = """User question: {user_query}

DIGEST — the sections already written above (the ONLY facts you may use):
{digest}

Write the headline and explanation for the "{title}" section, synthesizing ONLY from the digest."""

_FABRICATION_RETRY = """

⛔ YOUR PREVIOUS ATTEMPT WAS REJECTED. It contained numbers NOT present in the digest: {hits}.
Those numbers do not appear in the sections above, so you cannot state them. Rewrite using ONLY
numbers that are literally in the digest, or state the point qualitatively with no number at all."""

# Sentez için nedensellik retry mesajı: _CAUSAL_RETRY_SUFFIX ilişkiler bölümüne
# özeldir ("bu sayılar co-movement ölçer") — sentezde yanlış çerçeve. Sentez ölçülmüş
# bulguları özetler, hedefin NEDENİNİ açıklamaz.
_SYNTHESIS_CAUSAL_RETRY_SUFFIX = """

⛔ YOUR PREVIOUS ATTEMPT WAS REJECTED. It contained causal language about the outcome: {hits}.
This is a synthesis of what was MEASURED — associations and model contributions — not an explanation
of WHY the outcome happens. The analysis did NOT establish that any field produces, drives,
influences, explains or plays a role in the target. Rewrite without any such claim: state what was
found ("the strongest measured discriminator is X", "the model leans most on Y") and do not assert
cause. Do not speculate about factors "likely" at play."""

# Öneriler için spekülasyon retry mesajı: hata-analizi sürümü ("errors concentrate")
# öneri bölümünde yanlış çerçeve. Öneri digest'teki bir bulguya dayanmalı, sebebe değil.
_SYNTHESIS_SPECULATION_RETRY_SUFFIX = """

⛔ YOUR PREVIOUS ATTEMPT WAS REJECTED. It contained causal / speculative language: {hits}.
A recommendation must rest on a measured finding in the digest, not on a guess about cause. Do NOT
claim "if you do X the model will improve" or explain WHY the outcome happens. Rewrite each
recommendation as <finding in the digest> → <concrete next step>, and remove every
"because" / "the reason" / "likely" / "probably" / "plays a role" / "influences" construction."""

# Sentez bölümü guard'ları: fabrication HER ZAMAN (izinli-sayı kümesine dayanır, ayrı
# tutulur). Bunun ÜSTÜNE her bölüm kendi liste-guard'larını çalıştırır:
#   summary       → nedensellik (Özet, hedefin sebebini AÇIKLAMAZ — sadece özetler)
#   assessment    → nedensellik + abartma ("güvenilir" mayını burada patlar)
#   recommendations → spekülasyon (nedenselliği İÇERİR; öneri en sıkı yer, "yaparsan iyileşir" yasak)
# Her bölüm bir (dedektör, retry-eki) LİSTESİ taşır — birden çok guard aynı bölümde
# çalışabilsin diye (eskiden tek tuple'dı; Özet HİÇBİR dil-guard'ı almıyordu → gerçek
# koşuda "significantly influence churn" ve "likely ... play a crucial role" kaçtı).
# Güçlü/Zayıf bölümünde "güçlü yönü risk gibi yazma" retry'si (bkz. detect_restated_strength).
_RESTATED_STRENGTH_RETRY_SUFFIX = """

⛔ YOUR PREVIOUS ATTEMPT WAS REJECTED. You presented a STRENGTH as a weakness/risk: {hits}.
A good property — good generalization, no overfitting, a high score, balanced classes, a robust or
stable measurement — is NEVER a weakness. Remove it from the weakness list; do NOT reword it as a
"risk". A weakness is a GENUINE limitation: a low metric, a thin sample, a wide interval, suspected
leakage, or a between-group gap. If there is no genuine weakness, do NOT invent one."""

# Deterministik "risk yok" satırı — LLM'e BIRAKILMAZ (güçlü yönü tekrar etmeye meyilli).
# Gerçek risk yoksa kod bu tek satırı basar; LLM'e yalnız güçlü yönleri yazması söylenir.
_ASSESS_NO_RISK_ADDON = {
    "tr": ("\n\n- ZORUNLU: Bu koşuda DETERMİNİSTİK risk taraması gerçek bir risk BULMADI. "
           "Zayıf yön / risk listesi YAZMA — sistem 'belirgin risk tespit edilmedi' satırını "
           "kendisi ekleyecek. Yalnızca güçlü yönleri yaz, zayıflık UYDURMA."),
    "en": ("\n\n- MANDATORY: the deterministic risk scan found NO genuine risk in this run. "
           "Do NOT write a weakness / risk list — the system will append the 'no significant risk' "
           "line itself. Write only the strengths; do NOT invent a weakness."),
}
_ASSESS_NO_RISK_LINE = {
    "tr": "**Zayıf yönler / riskler:** Bu çalışmada belirgin bir risk tespit edilmedi.",
    "en": "**Weaknesses / risks:** No significant risk was detected in this run.",
}
# Risk VARSA: zayıflık listesi DETERMİNİSTİK risk taramasından gelir. LLM bu maddeleri
# anlatıya çevirir — EKLEYEMEZ/ÇIKARAMAZ (statü rozeti dersi). Sayılar zaten Evidence'ta.
_ASSESS_RISK_ADDON = {
    "tr": ("\n\n- ZORUNLU: Zayıf yön / risk listesi AŞAĞIDAKİ deterministik maddelerdir — "
           "bunları anlatıya çevir, HİÇBİRİNİ ekleme, HİÇBİRİNİ çıkarma, yeni risk UYDURMA:\n{items}\n"
           "Nedensellik dili KULLANMA ('neden' değil 'nereye bak'). Güçlü yönleri digest'ten yaz."),
    "en": ("\n\n- MANDATORY: the weakness / risk list is EXACTLY the deterministic items below — "
           "narrate them, ADD nothing, REMOVE nothing, INVENT no new risk:\n{items}\n"
           "Use NO causal language ('where to look', not 'why'). Write strengths from the digest."),
}


def _assessment_no_genuine_risk(built_sections: list[ReportSection], state: dict) -> bool:
    """Güçlü/Zayıf bölümü için: koşuda GERÇEK bir risk YOK mu? (deterministik).

    KASITEN MUHAFAZAKÂR — yanlış "risk yok" bir sessiz-sıfır hatasıdır (gerçek riski
    gizler), bu yüzden en ufak şüphede False döner ve zayıflık yazımı guard'lı LLM'e
    kalır. True yalnızca AÇIKÇA temiz koşularda: genelleme sağlam + hiç sızıntı yok +
    hiç ince/güvenilmez ölçüm yok + hiç 'warn' bölüm yok.
    """
    er = state.get("evaluation_report") or {}
    # (1) Genelleme AÇIKÇA sağlam olmalı. Eksik/mild/detected/underfit/inconclusive → temiz değil.
    if (er.get("overfitting") or {}).get("status") != "normal":
        return False
    # (2) Herhangi bir sızıntı bayrağı → risk.
    if (state.get("domain_leakage_report") or {}).get("concerns"):
        return False
    if er.get("target_lock_leakage"):
        return False
    if (er.get("feature_importance_summary") or {}).get("leakage_suspects"):
        return False
    # (3) Herhangi bir ölçülmüş-ama-ince (güvenilmez) kanıt ya da 'warn' bölüm → gerçek sınır.
    for sec in _iter_sections(built_sections):
        if sec.status == "warn":
            return False
        for e in sec.evidence:
            if e.value is not None and not e.reliable:
                return False
    return True


# Güçlü/Zayıf → nedensellik + abartma + "güçlü yönü risk gibi sunma".
_SYNTHESIS_GUARDS: dict = {
    "summary":         [(detect_causal_language, _SYNTHESIS_CAUSAL_RETRY_SUFFIX)],
    "assessment":      [(detect_causal_language, _SYNTHESIS_CAUSAL_RETRY_SUFFIX),
                        (detect_overstatement, _OVERSTATEMENT_RETRY_SUFFIX),
                        (detect_restated_strength, _RESTATED_STRENGTH_RETRY_SUFFIX)],
    "recommendations": [(detect_speculation, _SYNTHESIS_SPECULATION_RETRY_SUFFIX)],
}


def build_synthesis_prose(
    section: ReportSection,
    built_sections: list[ReportSection],
    state: dict,
    lang: str = "tr",
) -> ReportSection:
    """Sentez bölümünün prose'unu yazdırır — üstteki bölümlerin DIGEST'inden.

    Fabrication guard yapısaldır: prose'daki sayılar üstteki kanıtın izinli
    kümesinde değilse yakalanır, sertleştirilmiş talimatla yeniden denenir, yine
    kaçırırsa açıklama düşürülür. Böylece sentez üstte olmayan bir sayı taşıyamaz.
    """
    if section.status != "ok":
        return section

    from src.core.report_i18n import LANGUAGE_NAME
    from langchain_core.messages import HumanMessage, SystemMessage

    allowed = collect_allowed_numbers(built_sections)
    # #11(a) HEDEFLİ DIŞLAMA: yalnız ÖNERİLER importance'ı görmez (aksiyonizasyon kaynağı).
    # summary/assessment importance'ı görmeye devam eder ("en güçlü özellik" orada meşru
    # bir güçlü-yön ve ikisinin de kendi causal guard'ı var). error_analysis öneri
    # girdisinde KALIR ama nedensellik damgalı (aksiyon alınabilir "nerede bakılmalı").
    if section.id == "recommendations":
        digest = _build_digest(built_sections, lang,
                               exclude_ids=frozenset({"features"}),
                               causal_note_ids=frozenset({"error_analysis"}))
    else:
        digest = _build_digest(built_sections, lang)
    spec = _SYNTHESIS_SPECIFIC.get(section.id, {}).get(lang, "")
    # Güçlü/Zayıf: gerçek risk yoksa zayıflık satırını LLM'e bırakma — KOD basar
    # (LLM güçlü yönü tekrar etmeye meyilli; bkz. _assessment_no_genuine_risk).
    _no_risk_line = None
    if section.id == "assessment":
        # Zayıf yön/risk = DETERMİNİSTİK risk listesi (kod üretir, LLM anlatır — ekleyemez).
        from src.services.risk_assessment import build_risk_list
        _rl = build_risk_list(state)
        if _rl["risks"]:
            _items = "\n".join(f"  - [{r.severity}] {r.detail.get(lang, r.detail.get('tr', ''))}"
                               for r in _rl["risks"])
            spec = spec + _ASSESS_RISK_ADDON.get(lang, _ASSESS_RISK_ADDON["tr"]).format(items=_items)
        else:
            _no_risk_line = _ASSESS_NO_RISK_LINE.get(lang, _ASSESS_NO_RISK_LINE["tr"])
            spec = spec + _ASSESS_NO_RISK_ADDON.get(lang, _ASSESS_NO_RISK_ADDON["tr"])
    system = _SYNTHESIS_SYSTEM.format(language=LANGUAGE_NAME[lang], section_specific=spec)
    human = _SYNTHESIS_HUMAN.format(
        user_query=state.get("user_query", "—"), digest=digest, title=section.title)

    guards = _SYNTHESIS_GUARDS.get(section.id, [])

    try:
        llm = get_llm(model_name=settings.DEFAULT_LLM_MODEL, temperature=0.3)
        structured = llm.with_structured_output(LLM_SECTION_SCHEMA)

        for attempt in (1, 2):
            result = structured.invoke([
                SystemMessage(content=system), HumanMessage(content=human)])
            if not isinstance(result, dict):
                break
            headline = (result.get("headline") or "").strip()
            explanation = (result.get("explanation") or "").strip()
            prose = f"{headline} {explanation}"

            fab_hits = detect_fabrication(prose, allowed)
            # Her guard'ı ayrı çalıştır; hangisinin tetiklendiğini SUFFIX'iyle sakla
            # (retry'de yalnız tetikleyen guard'ın talimatını ekleyelim diye).
            guard_results = [(fn(prose), suffix) for fn, suffix in guards]
            extra_hits = [h for ghits, _ in guard_results for h in ghits]
            hits = fab_hits + extra_hits

            if not hits:
                section.headline, section.explanation = headline, explanation
                # Gerçek risk yoksa: zayıflık satırı KOD tarafından basılır (LLM'e
                # yalnız güçlü yönler söylendi) — deterministik, tekrar-güçlü-yön yok.
                if _no_risk_line:
                    section.explanation = (explanation + "\n\n" + _no_risk_line).strip()
                break

            logger.warning(
                f"[report_sections] sentez '{section.id}' sorunlu: "
                f"uydurma={fab_hits} yasak-dil={extra_hits} (deneme {attempt}/2)."
            )
            if attempt == 1:
                if fab_hits:
                    system = system + _FABRICATION_RETRY.format(hits=", ".join(sorted(set(fab_hits))))
                for ghits, suffix in guard_results:
                    if ghits:
                        system = system + suffix.format(hits=", ".join(sorted(set(ghits))))
            else:
                # SENTEZDE DÜŞME BOŞ BAŞLIK BIRAKAMAZ: kanıt bölümünün aksine sentezin
                # kendi kanıtı yok, prose düşerse geriye '## Özet' + hiçbir şey kalır
                # (boş-başlık artığı). O yüzden görünür sebeple not_computed'e çekilir.
                section.headline = ""
                section.explanation = ""
                section.status = "not_computed"
                section.not_computed_reason = (
                    "Bu özet otomatik üretilemedi (üretilen metin üstteki kanıtla "
                    "tutarsızdı ve yayınlanmadı)."
                    if lang == "tr" else
                    "This synthesis could not be produced automatically (the generated text "
                    "was inconsistent with the evidence above and was withheld).")
                logger.error(
                    f"[report_sections] sentez '{section.id}': iki denemede de sorunlu — "
                    f"açıklama yayınlanmadı (uydurma/yasak dil)."
                )
    except Exception as e:
        # LLM tümden çökerse de boş başlık bırakma — görünür sebep.
        section.status = "not_computed"
        section.not_computed_reason = (
            "Bu özet otomatik üretilemedi (dil modeli yanıtı alınamadı)."
            if lang == "tr" else
            "This synthesis could not be produced automatically (no model response).")
        logger.warning(
            f"[report_sections] sentez '{section.id}' üretilemedi ({e}) — görünür sebep bırakıldı.")
    return section
