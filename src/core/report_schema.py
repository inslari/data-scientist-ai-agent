from __future__ import annotations

"""
Report Schema (katmanlı rapor)
------------------------------
Raporun YAPISAL kaynağı. ``report: str`` bundan RENDER edilir — düz metin artık
birincil veri değil, çıktı biçimidir.

Üç katman (her bölüm aynı üçlüyü taşır):

    headline    → yönetici cümlesi   ("country grubunda performans farkı yok")
    explanation → iş dili açıklaması (ne kadar, hangi yön, ne anlama geliyor)
    evidence    → analist detayı     (sayı + kaynak + örneklem + belirsizlik)

Neden bu ayrım: "uzman analist detayı + herkesin anlayacağı sadelik" tek bir düz
metinde çelişir — ya sayıya boğulur ya da kanıtsız iddia olur. Katmanlar ayrıysa
aynı bölüm iki okuyucuya aynı anda hizmet eder ve her iddia kanıtına bağlıdır.

SESSİZ SIFIR YASAĞI (bu modülün varlık sebebi)
==============================================
Bir bölüm kanıtsız kalıyorsa NEDENİNİ taşımak ZORUNDADIR. ``status`` alanı
``not_computed``/``not_applicable`` ise ``not_computed_reason`` boş olamaz —
``ReportSection.validate()`` bunu zorlar.

Bu kural bir üslup tercihi değil; ölçülmüş bir hata sınıfının panzehiri.
Fairness'ta aynı sessizlikten DÖRT ayrı bug çıktı (bkz. src/services/fairness.py
başlığı): rapor olmayan bir anahtarı ``.get('fpr', 0)`` ile okuyup "FPR=0.000"
yazıyordu; ölçeklenmiş one-hot kolonda grup maskesi boşalıp grup sessizce
atlanıyordu; ``drop_first`` bir kategoriyi tümüyle görünmez yapıyordu; DI=0.0
(tam ayrımcılık) falsy kontrolü yüzünden "hesaplanmadı"ya düşüyordu. Hepsinin
ortak paydası: eksik veri 0/None'a düşüyor, düşüş hiçbir yere yazılmıyordu.

LLM SINIRI
==========
``evidence`` KOD tarafından state'ten deterministik kurulur. LLM'e verilen
structured-output şemasında ``evidence`` alanı YOKTUR (bkz. LLM_SECTION_SCHEMA) —
yani LLM sayı uyduramaz, çünkü uyduracağı alan mevcut değildir. Bu, prompt
disiplinine değil YAPIYA dayanan bir garantidir; ``analysis_engine.extract_plan_hints``
in "imzasında target parametresi yok" yaklaşımının aynısı: kuralı yasak olarak
değil, imkânsızlık olarak kur.

worker.py NOTU
==============
``api/core/worker.py`` step_data'yı sanitize ederken NaN/Inf → None çevirir ve
BÜYÜK listeleri [:5] keser. Kesme bir ALLOWLIST'e bağlıdır (sql_results, X_train,
...) — ``report_sections`` o listede olmadığı için olduğu gibi geçer. Ama NaN→None
dönüşümü geçerlidir: bu yüzden ``value=None`` "hesaplanmadı" ANLAMINA GELMEZ;
o bilgi ayrı alanlarda (``status`` + ``not_computed_reason``) taşınır.
"""

from typing import Any, Literal, Optional

from src.core.logging_config import get_logger

logger = get_logger(__name__)


# ── Provenance sözlüğü ───────────────────────────────────────────────────────
# analysis_engine.extract_plan_hints()'teki ``generated_from`` deseninin
# genellenmiş hâli. Her kanıt hangi satırlardan geldiğini SÖYLEMEK zorunda:
# tüm veriden hesaplanmış bir korelasyonu "gerçek ilişki" diye sunmak teknik
# sızıntı değil ama ÇERÇEVELEME hatasıdır — okuyucu neyin kanıtı olduğunu
# bilmeden yargıya varamaz.

Provenance = Literal[
    "test_set",              # holdout — model bu satırları hiç görmedi (en güçlü kanıt)
    "train_only",            # yalnızca eğitim partisyonu
    "cv",                    # çapraz doğrulama (out-of-fold)
    "full_data_descriptive", # TÜM veri — tanımlayıcı, nedensel/tahmine dönük DEĞİL
    "model_internal",        # modelin kendi iç yapısından (feature importance vb.)
    "schema_semantic",       # veriden DEĞİL — kolon adları/anlamları üzerinden LLM yargısı
]

PROVENANCE_LABEL: dict[str, dict[str, str]] = {
    "tr": {
        "test_set":              "test seti (model bu satırları görmedi)",
        "train_only":            "yalnızca eğitim verisi",
        "cv":                    "çapraz doğrulama (out-of-fold)",
        "full_data_descriptive": "tüm veri — tanımlayıcı",
        "model_internal":        "modelin iç yapısı",
        "schema_semantic":       "şema/anlam analizi (veriden değil)",
    },
    "en": {
        "test_set":              "test set (unseen by the model)",
        "train_only":            "training data only",
        "cv":                    "cross-validation (out-of-fold)",
        "full_data_descriptive": "full data — descriptive",
        "model_internal":        "model internals",
        "schema_semantic":       "schema/semantic analysis (not from data)",
    },
}

SectionStatus = Literal["ok", "warn", "not_applicable", "not_computed"]

# status → bölüm kanıt taşımak ZORUNDA mı?
_STATUS_REQUIRES_REASON = ("not_applicable", "not_computed")


# ── Montaj sırası (raporun TEK yerleşim planı) ───────────────────────────────
# Bölümler yazılma/taşınma sırasına göre değil BU sayılara göre dizilir. Sebep:
# katmanlı yapıya geçiş kademeli — her taşınan bölüm raporun sonuna eklenseydi
# okuma sırası migrasyon sırasına esir olurdu (fairness tam da böyle sonda kaldı).
#
# Aralıklar 10'ar: araya bölüm sokmak yeniden numaralandırma GEREKTİRMEZ.
# Yeni bölüm eklerken buraya bir sabit ekleyin; bölümün kendisi order'ı buradan alsın.
ORDER_HEADER          = 0    # deterministik H1 + meta satırı
ORDER_TIMESERIES      = 5    # zaman serisi özeti (yalnız modality=timeseries)
ORDER_STATUS          = 8    # quality_gate durum rozeti + Karar + FAIL/WARN
ORDER_SUMMARY         = 10
ORDER_DATA_QUALITY    = 20
ORDER_DATA_RELATIONS  = 30
ORDER_PERFORMANCE     = 40
ORDER_GENERALIZATION  = 45
ORDER_LEGACY_PROSE    = 50   # GEÇİCİ: henüz taşınmamış tek parça LLM blob'u
ORDER_ERROR_ANALYSIS  = 55
ORDER_FEATURES        = 60
ORDER_LEAKAGE_RISK    = 70
ORDER_FAIRNESS        = 80
ORDER_RECOMMENDATIONS = 90
ORDER_NOTES           = 95   # veri kalitesi/sampling/heavy-guard dipnotları

ORDER_DEFAULT = ORDER_LEGACY_PROSE


class Evidence:
    """Tek bir kanıt kalemi: sayı + nereden geldiği + ne kadar güvenilir olduğu.

    ``value`` None olabilir (NaN sanitizasyonu veya tanımsız metrik) — bu
    "hesaplanmadı" demek DEĞİLDİR; o bilgi bölüm düzeyinde ``status`` ile taşınır.
    Kalem düzeyinde tanımsızlık ``note`` ile açıklanır.
    """

    __slots__ = ("label", "value", "provenance", "sample_size", "uncertainty", "note", "reference")

    def __init__(
        self,
        label: str,
        value: Any,
        provenance: Provenance,
        *,
        sample_size: Optional[int] = None,
        uncertainty: Optional[dict] = None,
        note: str = "",
        reference: str = "",
    ) -> None:
        self.label = label
        self.value = value
        self.provenance = provenance
        self.sample_size = sample_size
        # {"lower": .., "upper": .., "method": "wilson"} | None
        self.uncertainty = uncertainty
        self.note = note
        # "referans eşik" gibi yorum ÇERÇEVESİ — hüküm değil (bkz. fairness DI)
        self.reference = reference

    @property
    def reliable(self) -> bool:
        """Kanıt tek başına iddiaya dayanak olur mu?

        Belirsizlik aralığı verilmişse GENİŞLİK belirler: aralık 0.30'dan genişse
        (örn. DI için [0.31, 0.94]) sayı bir yön göstermez, gürültüdür.
        Aralık yoksa örneklem boyutuna düşülür (MIN_RELIABLE_N).
        """
        if self.uncertainty:
            lo, hi = self.uncertainty.get("lower"), self.uncertainty.get("upper")
            if lo is not None and hi is not None:
                return (hi - lo) <= 0.30
        if self.sample_size is not None:
            return self.sample_size >= MIN_RELIABLE_N
        return True

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "provenance": self.provenance,
            "sample_size": self.sample_size,
            "uncertainty": self.uncertainty,
            "note": self.note,
            "reference": self.reference,
            "reliable": self.reliable,
        }


# Belirsizlik aralığı olmayan kanıtlar için "bu sayıya güvenilir mi" eşiği.
# 30: normal yaklaşımın kabaca tutmaya başladığı klasik sınır. Altındaki her şey
# raporda GÖRÜNÜR ama "ölçüm ince" damgasıyla — gizlemek de sessiz kayıp olurdu.
MIN_RELIABLE_N = 30


class ReportSection:
    """Raporun bir bölümü — üç katman + durum.

    ``headline``/``explanation`` LLM tarafından yazılır (veya fallback'te boş kalır);
    ``evidence``/``status``/``not_computed_reason`` KOD tarafından doldurulur ve
    LLM bunlara dokunamaz.

    ``order`` raporun MONTAJ sırasıdır (bkz. SECTION_ORDER). Bölümler yazılma
    sırasına göre değil, bu sayıya göre dizilir — yeni bir bölüm eklemek onu
    raporun sonuna itmez, ait olduğu yere koyar.

    ``raw_markdown`` GEÇİCİ bir migrasyon kaçışıdır: henüz katmanlı yapıya
    taşınmamış hazır markdown (tek parça LLM blob'u, deterministik başlık/zaman
    serisi bloğu) montaja sıralı biçimde katılabilsin diye. ``None`` DEĞİLSE bölüm
    bir "ham taşıyıcı"dır: diğer tüm alanları yok sayılır ve metin AYNEN basılır.
    Migrasyon bitince kaldırılmalıdır — yeni bölümler için KULLANMAYIN.

    DİKKAT — None vs "": ayrım kasıtlıdır. Boş string GEÇERLİ bir ham taşıyıcıdır
    ("bu bölüm bu çalışmada boş", ör. tabular veride zaman serisi özeti) ve hiçbir
    şey basmamalıdır. Truthy kontrolü (``if raw_markdown:``) kullanılırsa boş
    taşıyıcı normal bölüm gibi render edilir ve başlıksız bir '## ' artığı basar —
    bu hata bir kez yapıldı, ölçüldü ve testle sabitlendi
    (test_bos_ham_tasiyici_hic_bir_sey_basmaz).
    """

    __slots__ = ("id", "title", "headline", "explanation", "evidence",
                 "status", "not_computed_reason", "subsections", "order",
                 "raw_markdown", "disclaimer", "table")

    def __init__(
        self,
        id: str,
        title: str,
        *,
        headline: str = "",
        explanation: str = "",
        evidence: Optional[list[Evidence]] = None,
        status: SectionStatus = "ok",
        not_computed_reason: str = "",
        subsections: Optional[list["ReportSection"]] = None,
        order: int = ORDER_DEFAULT,
        raw_markdown: Optional[str] = None,
        disclaimer: str = "",
        table: Optional[dict] = None,
    ) -> None:
        self.id = id
        self.title = title
        self.headline = headline
        self.explanation = explanation
        # KOD tarafından kurulan tablo — bazı kanıtlar liste hâlinde OKUNMAZ.
        # Confusion matrix bunun tipik örneği: 2x2'yi dört ayrı kanıt satırına
        # dağıtmak "hangi sayı neydi" sorusunu okuyucuya bırakır; kxk çok sınıflı
        # matriste liste tümüyle çöker. LLM_SECTION_SCHEMA'da YOKTUR → uydurulamaz.
        # {"headers": [...], "rows": [[...], ...], "caption": "..."}
        self.table = table
        # KOD tarafından yazılan, LLM'in dokunamadığı uyarı. Başlığın hemen altına,
        # prose'dan ÖNCE basılır. Yorumun okuyucuyu yanlış çerçeveye sokabileceği
        # bölümlerde (ör. korelasyon → nedensellik) zorunlu koruma.
        self.disclaimer = disclaimer
        self.evidence = evidence or []
        self.status = status
        self.not_computed_reason = not_computed_reason
        self.subsections = subsections or []
        self.order = order
        self.raw_markdown = raw_markdown

    def validate(self) -> None:
        """Sessiz sıfır yasağını ZORLAR.

        Bir bölüm "hesaplanmadı/uygulanmadı" diyorsa nedenini söylemek zorunda.
        İhlal ProgrammingError değil, LOG + görünür iz: rapor üretimi bir uyarı
        yüzünden çökmemeli, ama sebep-siz bir boşluk da sessizce geçmemeli.
        """
        if self.status in _STATUS_REQUIRES_REASON and not self.not_computed_reason.strip():
            logger.error(
                f"[report_schema] '{self.id}' bölümü status='{self.status}' ama "
                f"not_computed_reason BOŞ — sessiz sıfır. Sebep zorunludur."
            )
            self.not_computed_reason = _MISSING_REASON_PLACEHOLDER
        for sub in self.subsections:
            sub.validate()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "order": self.order,
            "disclaimer": self.disclaimer,
            "headline": self.headline,
            "explanation": self.explanation,
            "table": self.table,
            "evidence": [e.to_dict() for e in self.evidence],
            "status": self.status,
            "not_computed_reason": self.not_computed_reason,
            "subsections": [s.to_dict() for s in self.subsections],
        }


_MISSING_REASON_PLACEHOLDER = (
    "(sebep kaydedilmedi — bu bir hata izidir, lütfen bildirin)"
)


# ── LLM structured-output şeması ─────────────────────────────────────────────
# DİKKAT: 'evidence' BİLEREK YOK. LLM'in doldurabileceği tek alanlar headline ve
# explanation. Sayı uydurma bu yüzden yasak değil, İMKÂNSIZ — şemada yeri yok.
# Bu şemayı genişletirken kanıt alanı EKLEMEYİN; eklerseniz anti-fabrication
# garantisi yapısal olmaktan çıkıp yine prompt disiplinine düşer.

LLM_SECTION_SCHEMA: dict = {
    "title": "ReportSectionProse",
    "description": (
        "Prose for one report section. You may ONLY write the headline and the "
        "explanation. All numbers are supplied to you as evidence and are rendered "
        "separately by the system — you cannot add, change or invent any number."
    ),
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": (
                "One executive sentence stating what this section found, in plain "
                "business language. No numbers unless they appear verbatim in the "
                "supplied evidence."
            ),
        },
        "explanation": {
            "type": "string",
            "description": (
                "2-4 sentences of business-language explanation: how much, in which "
                "direction, and what it means for the reader. Use ONLY the supplied "
                "evidence. If the evidence is marked unreliable (thin sample), say so "
                "rather than drawing a conclusion from it."
            ),
        },
    },
    "required": ["headline", "explanation"],
}


# ── Markdown render ──────────────────────────────────────────────────────────

def _fmt_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "evet" if v else "hayır"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def render_evidence(ev: Evidence, lang: str = "tr") -> str:
    """Tek kanıt satırı: değer · kaynak · örneklem · belirsizlik · referans."""
    parts = [f"**{ev.label}**: {_fmt_value(ev.value)}"]

    prov = PROVENANCE_LABEL.get(lang, PROVENANCE_LABEL["tr"]).get(ev.provenance, ev.provenance)
    parts.append(f"kaynak: {prov}" if lang == "tr" else f"source: {prov}")

    if ev.sample_size is not None:
        parts.append(f"n={ev.sample_size:,}")

    if ev.uncertainty:
        lo, hi = ev.uncertainty.get("lower"), ev.uncertainty.get("upper")
        meth = ev.uncertainty.get("method", "")
        if lo is not None and hi is not None:
            rng = f"[{_fmt_value(lo)} — {_fmt_value(hi)}]"
            parts.append(f"%95 aralık {rng} ({meth})" if lang == "tr"
                         else f"95% interval {rng} ({meth})")

    line = " · ".join(parts)

    if not ev.reliable:
        line += ("  \n  ⚠️ *Ölçüm ince — bu sayı tek başına bir yargıya dayanak olmaz.*"
                 if lang == "tr" else
                 "  \n  ⚠️ *Thin measurement — not a basis for a judgement on its own.*")
    if ev.reference:
        line += f"  \n  ↳ {ev.reference}"
    if ev.note:
        line += f"  \n  ↳ {ev.note}"
    return line


def render_table(tbl: dict) -> str:
    """Kod tarafından kurulan tabloyu markdown'a çevirir.

    Hücreler AYNEN basılır — biçimlendirme (binlik ayracı, ✓/✗ işaretleri, düz
    dilli hücre etiketleri) tabloyu kuran bölümün işidir; burada karar verilmez.
    """
    headers = [str(h) for h in (tbl.get("headers") or [])]
    rows = tbl.get("rows") or []
    if not headers or not rows:
        return ""
    out: list[str] = []
    if tbl.get("caption"):
        out.append(f"*{tbl['caption']}*")
        out.append("")
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_section(sec: ReportSection, lang: str = "tr", level: int = 2) -> str:
    """Bir bölümü markdown'a çevirir: başlık → headline → explanation → kanıt."""
    # Migrasyon kaçışı: ham taşıyıcı aynen basılır. `is not None` ŞART — boş
    # taşıyıcı ("" ) hiçbir şey basmamalı, normal bölüm gibi render edilip
    # başlıksız '## ' artığı üretmemeli (bkz. ReportSection.raw_markdown notu).
    if sec.raw_markdown is not None:
        return sec.raw_markdown

    sec.validate()
    h = "#" * level
    out: list[str] = [f"{h} {sec.title}"]

    if sec.status in _STATUS_REQUIRES_REASON:
        # Boş bölüm KURALI: doldurma yapma, tek satırda sebebi söyle.
        out.append(f"*{sec.not_computed_reason}*")
        out.append("")
        return "\n".join(out)

    # Deterministik uyarı — prose'dan ÖNCE, LLM'den bağımsız. Yorum yanlış
    # çerçeveleme yapsa bile okuyucu doğru çerçeveyi ilk burada görür.
    if sec.disclaimer:
        out.append(sec.disclaimer)
        out.append("")

    if sec.headline:
        out.append(f"**{sec.headline}**")
        out.append("")
    if sec.explanation:
        out.append(sec.explanation)
        out.append("")

    # Tablo: prose'dan SONRA (yorum önce okunur), kanıttan ÖNCE (kanıt katlanır,
    # tablo katlanmaz — okunabilirliğin kendisi tablonun varlık sebebi).
    if sec.table:
        rendered = render_table(sec.table)
        if rendered:
            out.append(rendered)
            out.append("")

    if sec.evidence:
        out.append(f"<details>\n<summary>{'Kanıt' if lang == 'tr' else 'Evidence'}</summary>\n")
        for ev in sec.evidence:
            out.append(f"- {render_evidence(ev, lang)}")
        out.append("\n</details>")
        out.append("")

    for sub in sec.subsections:
        out.append(render_section(sub, lang, level + 1))

    return "\n".join(out)


def render_report(sections: list[ReportSection], lang: str = "tr") -> str:
    """Bölümleri ORDER'a göre dizip tam markdown rapora çevirir.

    ``report: str`` bunun çıktısıdır — mevcut UI hiçbir değişiklik görmeden
    okumaya devam eder.

    Sıralama STABİL: aynı order'a sahip bölümler verildikleri sırayı korur.
    """
    ordered = sorted(sections, key=lambda s: s.order)
    parts = [render_section(s, lang) for s in ordered]
    return "\n".join(p for p in parts if p.strip()).rstrip() + "\n"


def sections_to_dicts(sections: list[ReportSection]) -> list[dict]:
    """state['report_sections'] için JSON-serileştirilebilir hâl.

    ORDER'a göre dizili döner — UI bölümleri aynı sırada görsün. Geçici
    raw_markdown bölümleri (henüz taşınmamış blob) yapısal veri taşımadığı için
    DIŞARIDA bırakılır: yapısal tüketici yarım bir bölüm görmemeli.
    """
    ordered = sorted((s for s in sections if s.raw_markdown is None), key=lambda s: s.order)
    for s in ordered:
        s.validate()
    return [s.to_dict() for s in ordered]
