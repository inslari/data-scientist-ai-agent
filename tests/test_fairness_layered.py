"""
Katmanlı rapor + fairness pilotu — DEĞİŞMEZ testleri.

Bu dosya bir "geçti/kaldı" listesi değil; düzeltilen dört sessiz hatanın GERİ
GELMESİNİ engelleyen çit. Her testin adı hangi bug'ı tuttuğunu söyler.

Çalıştırma:
    uv run pytest tests/test_fairness_layered.py -v
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.core.report_schema import (
    LLM_SECTION_SCHEMA,
    ORDER_DATA_RELATIONS,
    ORDER_FAIRNESS,
    ORDER_HEADER,
    ORDER_LEGACY_PROSE,
    Evidence,
    ReportSection,
    render_report,
    render_section,
    sections_to_dicts,
)
from src.services.fairness import analyze_fairness, wilson_interval


# ── Yardımcılar ──────────────────────────────────────────────────────────────

def _meta(columns, **kw):
    # Taşıyıcı TÜM ham kolonları getirir; fairness'ın kapsamı segment_columns'dır.
    base = {"segment_columns": list(columns), "carried_columns": list(columns),
            "mode": "narrow", "skipped": {}, "n_rows": 0, "reason": ""}
    base.update(kw)
    return base


# ═══════════════════════════════════════════════════════════════════════════
#  BUG 1 — "sessiz FPR=0": rapor olmayan anahtarı 0 diye yazıyordu
# ═══════════════════════════════════════════════════════════════════════════

def test_fpr_fnr_gercekten_hesaplaniyor():
    """FPR/FNR artık ÜRETİLİYOR — eskiden hiç üretilmiyor, rapor 0.000 yazıyordu."""
    # Her grupta HEM pozitif HEM negatif olsun (yoksa FPR/FNR tanımsızdır).
    # A: kusursuz. B: negatiflerinin bir kısmına yanlışlıkla pozitif diyor.
    y_true = np.array([1, 0] * 20)               # dönüşümlü → her grupta iki sınıf da var
    y_pred = y_true.copy()
    seg = pd.DataFrame({"grp": ["A"] * 20 + ["B"] * 20})
    b_neg = np.where((seg["grp"] == "B").to_numpy() & (y_true == 0))[0]
    y_pred[b_neg[:5]] = 1                        # B'de 5 yanlış alarm

    out = analyze_fairness(y_true, y_pred, seg, _meta(["grp"]), "classification")

    assert out["status"] == "ok"
    groups = out["segments"]["grp"]["groups"]
    assert groups["A"]["fpr"] == 0.0            # gerçekten ölçülmüş sıfır
    assert groups["B"]["fpr"] > 0.0             # ve gerçekten sıfırdan farklı
    # En kritiği: değerler None DEĞİL — yani "hesaplanmadı" ile "sıfır" ayrışıyor.
    for g in ("A", "B"):
        assert groups[g]["fpr"] is not None
        assert groups[g]["fnr"] is not None


def test_tanimsiz_fpr_none_doner_sifir_degil():
    """Grupta hiç negatif yoksa FPR TANIMSIZDIR → None + sebep. 0 DEĞİL.

    Bu tam olarak eski bug'ın kalbi: tanımsızı sıfıra düşürmek 'bu grupta hiç
    yanlış alarm yok' gibi okunur — oysa ölçülecek vaka hiç yoktur.
    """
    y_true = np.array([1] * 10 + [0] * 10)       # A grubu tamamen pozitif
    y_pred = np.array([1] * 10 + [0] * 10)
    seg = pd.DataFrame({"grp": ["A"] * 10 + ["B"] * 10})

    out = analyze_fairness(y_true, y_pred, seg, _meta(["grp"]), "classification")
    a = out["segments"]["grp"]["groups"]["A"]

    assert a["fpr"] is None, "tanımsız FPR sıfıra düşmemeli"
    assert a["fpr_undefined_reason"], "tanımsızlığın sebebi taşınmalı"
    # Sebep artık dil-nötr KOD ({code,params}); R_REASON rapor diline çevirir.
    from src.core.report_i18n import R_REASON
    assert "negatif" in R_REASON(a["fpr_undefined_reason"], "tr").lower()
    assert "negative" in R_REASON(a["fpr_undefined_reason"], "en").lower()


# ═══════════════════════════════════════════════════════════════════════════
#  BUG 4 — DI = 0.0 falsy yüzünden "hesaplanmadı"ya düşüyordu
# ═══════════════════════════════════════════════════════════════════════════

def test_di_sifir_gecerli_deger_olarak_donuyor():
    """DI=0.0 (gruba HİÇ pozitif verilmemiş) = tam ayrımcılık.

    Eski kod ``if disparate_impact else None`` yazıyordu → 0.0 falsy → "ölçülemedi".
    En uç vaka en sessiz kayıptı.
    """
    # Referans grup A (kalabalık): pozitif tahminler var.
    # Grup B: hiç pozitif tahmin YOK → DI = 0/x = 0.0
    y_true = np.array([1, 0] * 15 + [1, 0] * 5)
    y_pred = np.array([1, 0] * 15 + [0, 0] * 5)
    seg = pd.DataFrame({"grp": ["A"] * 30 + ["B"] * 10})

    out = analyze_fairness(y_true, y_pred, seg, _meta(["grp"]), "classification")
    cmp = out["segments"]["grp"]["comparisons"]["groups"]["B"]

    assert cmp["disparate_impact"] == 0.0, "DI=0.0 geçerli bir ölçümdür, None değil"
    assert cmp["disparate_impact"] is not None


# ═══════════════════════════════════════════════════════════════════════════
#  BUG 2/3 — atlanan grup sessizce kayboluyordu
# ═══════════════════════════════════════════════════════════════════════════

def test_ince_grup_sessizce_kaybolmuyor_gorunur_sebep_tasiyor():
    """min_size altındaki grup rapordan SİLİNMEZ — sebebiyle listelenir."""
    y_true = np.array([1, 0] * 20 + [1, 0])      # B grubu yalnızca 2 satır
    y_pred = np.array([1, 0] * 20 + [1, 0])
    seg = pd.DataFrame({"grp": ["A"] * 40 + ["B"] * 2})

    out = analyze_fairness(y_true, y_pred, seg, _meta(["grp"]), "classification")
    skipped = out["segments"]["grp"]["skipped_groups"]

    assert "B" in skipped, "ince grup görünür olmalı — sessizce kaybolmamalı"
    # Sebep dil-nötr KOD; n bilgisi params'ta taşınır, R_REASON metne döker.
    from src.core.report_i18n import R_REASON
    assert skipped["B"]["params"]["n"] == 2
    assert "n=2" in R_REASON(skipped["B"], "tr")
    assert R_REASON(skipped["B"], "en").strip(), "atlama sebebi BOŞ olamaz"
    assert "B" not in out["segments"]["grp"]["groups"], "ince gruba metrik atfedilmemeli"


def test_tasinmayan_kolonun_sebebi_rapora_gecer():
    """Segment olarak taşınmayan kolon (yüksek kardinalite vb.) görünür kalır."""
    y_true = np.array([1, 0] * 10)
    y_pred = np.array([1, 0] * 10)
    seg = pd.DataFrame({"grp": ["A"] * 20})
    meta = _meta(["grp"], skipped={"user_id": "çok fazla farklı değer (5000 > 10)"})

    out = analyze_fairness(y_true, y_pred, seg, meta, "classification")

    assert out["skipped_columns"]["user_id"]
    from src.services.report_sections import _fairness_skipped_section
    sec = _fairness_skipped_section(out, "tr")
    assert sec is not None
    labels = [e.label for e in sec.evidence]
    assert any("user_id" in l for l in labels), "taşınmayan kolon rapora yazılmalı"


def test_carrier_yoksa_gorunur_sebeple_atlanir():
    """Taşıyıcı yoksa fairness sessizce yok olmaz — sebep taşır."""
    out = analyze_fairness(
        np.array([1, 0]), np.array([1, 0]), None,
        _meta([], reason="veri setinde demografik/segment kolonu bulunamadı"),
        "classification",
    )
    assert out["status"] == "not_computed"
    assert out["reason"], "sebep ZORUNLU"


def test_hizalama_bozuksa_analiz_reddedilir():
    """Taşıyıcı ile test seti hizalanmıyorsa KIRPMA yok — analiz reddedilir.

    Kırpmak sessizce yanlış gruba metrik atfederdi; en tehlikeli sessiz hata bu olurdu.
    """
    y_true = np.array([1, 0, 1, 0])
    seg = pd.DataFrame({"grp": ["A", "B"]})      # 2 satır vs 4
    out = analyze_fairness(y_true, y_true, seg, _meta(["grp"]), "classification")

    assert out["status"] == "not_computed"
    from src.core.report_i18n import R_REASON
    assert "hizalan" in R_REASON(out["reason"], "tr")
    assert "align" in R_REASON(out["reason"], "en").lower()


# ═══════════════════════════════════════════════════════════════════════════
#  ÇELİŞKİ — DI ile performans farkı aynı şeyi ölçmez
# ═══════════════════════════════════════════════════════════════════════════

def test_di_dusukken_performans_farki_sifir_olabilir():
    """country_NL vakası: model KUSURSUZ ama DI düşük.

    Bu bir çelişki DEĞİL. Model her grupta hatasız (FPR/FNR farkı = 0) ama grubun
    GERÇEK taban oranı düşük olduğu için seçim oranı da düşük → DI düşük.
    base_rate_ratio bunu açığa çıkarır: DI ≈ base_rate_ratio ise fark VERİDEN gelir.
    """
    # A (referans): %50 pozitif. B: %10 pozitif. Model ikisinde de KUSURSUZ.
    y_true = np.array([1, 0] * 25 + [1] + [0] * 9 + [1] + [0] * 9)
    y_pred = y_true.copy()                        # kusursuz model
    seg = pd.DataFrame({"grp": ["A"] * 50 + ["B"] * 20})

    out = analyze_fairness(y_true, y_pred, seg, _meta(["grp"]), "classification")
    cmp = out["segments"]["grp"]["comparisons"]["groups"]["B"]

    # Performans paritesi: kusursuz model → fark yok
    assert cmp["fpr_diff"] == 0.0
    assert cmp["fnr_diff"] == 0.0
    # Seçim paritesi: DI belirgin şekilde 1'den uzak
    assert cmp["disparate_impact"] < 0.8
    # Ve kritik olan: fark VERİDEN geliyor → DI ≈ base_rate_ratio
    assert cmp["base_rate_ratio"] == pytest.approx(cmp["disparate_impact"], abs=1e-6)
    assert cmp["selection_gap_vs_base"] == pytest.approx(0.0, abs=1e-6), \
        "kusursuz modelde seçim farkının veriyle açıklanamayan kısmı SIFIR olmalı"


def test_model_kaynakli_secim_farki_ayirt_ediliyor():
    """Taban oranı AYNI ama model bir gruba daha az pozitif diyorsa → gap ≠ 0."""
    # İki grup da %50 pozitif; ama model B'de pozitiflerin yarısını kaçırıyor.
    y_true = np.array([1, 0] * 25 + [1, 0] * 25)
    y_pred = y_true.copy()
    b = np.arange(50, 100)
    b_pos = b[y_true[b] == 1]
    y_pred[b_pos[: len(b_pos) // 2]] = 0          # B'nin pozitiflerinin yarısı kaçırıldı
    seg = pd.DataFrame({"grp": ["A"] * 50 + ["B"] * 50})

    out = analyze_fairness(y_true, y_pred, seg, _meta(["grp"]), "classification")
    cmp = out["segments"]["grp"]["comparisons"]["groups"]["B"]

    assert cmp["base_rate_ratio"] == pytest.approx(1.0, abs=1e-6), "taban oranları eşit"
    assert cmp["disparate_impact"] < 1.0, "model B'ye daha az pozitif diyor"
    assert abs(cmp["selection_gap_vs_base"]) > 0.1, \
        "taban oranı eşitken seçim farkı MODELDEN gelir — gap bunu göstermeli"


def test_otomatik_high_bias_damgasi_yok():
    """Hüküm vermiyoruz, kanıt sunuyoruz: çıktıda etiket/damga alanı OLMAMALI."""
    y_true = np.array([1, 0] * 25 + [1] + [0] * 9 + [1] + [0] * 9)
    seg = pd.DataFrame({"grp": ["A"] * 50 + ["B"] * 20})
    out = analyze_fairness(y_true, y_true.copy(), seg, _meta(["grp"]), "classification")

    blob = repr(out).lower()
    for damga in ("high bias", "high_bias", "unfair", "biased", "fair\"", "verdict"):
        assert damga not in blob, f"otomatik damga sızmış: {damga}"
    # 4/5 bandı yalnızca REFERANS olarak taşınır
    assert out["di_reference_band"] == [0.8, 1.25]


# ═══════════════════════════════════════════════════════════════════════════
#  WILSON — ince örneklem disiplini
# ═══════════════════════════════════════════════════════════════════════════

def test_wilson_uc_oranlarda_cokmez():
    """p=0 veya p=1'de Wald sıfır genişlikte aralık verir ("kesin biliyoruz").
    Wilson vermez — fairness'ta gruplar tam da böyle olduğu için kritik."""
    ci = wilson_interval(0, 5)
    assert ci["upper"] > 0, "p=0'da bile üst sınır sıfırdan büyük olmalı"
    ci = wilson_interval(5, 5)
    assert ci["lower"] < 1, "p=1'de bile alt sınır 1'den küçük olmalı"


def test_ince_olcum_guvenilmez_isaretleniyor():
    """Geniş aralıklı kanıt reliable=False → LLM ondan sonuç çıkaramaz."""
    thin = Evidence("DI", 0.656, "test_set", sample_size=6,
                    uncertainty={"lower": 0.31, "upper": 0.94, "method": "wilson"})
    assert thin.reliable is False

    solid = Evidence("DI", 0.98, "test_set", sample_size=5000,
                     uncertainty={"lower": 0.95, "upper": 1.01, "method": "wilson"})
    assert solid.reliable is True

    # Aralık yoksa örnekleme düşülür
    assert Evidence("x", 1, "test_set", sample_size=5).reliable is False
    assert Evidence("x", 1, "test_set", sample_size=500).reliable is True


# ═══════════════════════════════════════════════════════════════════════════
#  ŞEMA — sessiz sıfır yasağı + LLM kanıt uyduramaz
# ═══════════════════════════════════════════════════════════════════════════

def test_llm_semasinda_evidence_alani_yok():
    """Anti-fabrication YAPISAL: LLM'in dolduracağı kanıt alanı MEVCUT DEĞİL.

    Bu test bir üslup kontrolü değil — şemaya evidence eklenirse garanti prompt
    disiplinine düşer ve sessizce kaybolur.
    """
    props = LLM_SECTION_SCHEMA["properties"]
    assert set(props) == {"headline", "explanation"}, \
        "LLM şeması SADECE prose alanları içermeli"
    for yasak in ("evidence", "value", "metric", "number", "status"):
        assert yasak not in props, f"LLM'e '{yasak}' yazma imkânı verilmiş — kanıt uydurabilir"


def test_sebepsiz_bos_bolum_gorunur_iz_birakir():
    """status=not_computed + sebep BOŞ → sessizce geçmez, görünür iz bırakır."""
    sec = ReportSection(id="x", title="X", status="not_computed", not_computed_reason="")
    sec.validate()
    assert sec.not_computed_reason, "sebep-siz boşluk sessiz kalamaz"
    assert "hata izi" in sec.not_computed_reason


def test_bos_bolum_doldurma_yapmaz():
    """Kanıtsız bölüm tek satırda sebebini söyler — sahte içerikle şişmez."""
    sec = ReportSection(id="fairness", title="Adalet", status="not_applicable",
                        not_computed_reason="kümeleme analizinde tanımlı değil")
    md = render_section(sec, "tr")
    assert "kümeleme analizinde tanımlı değil" in md
    assert "Kanıt" not in md, "boş bölüme kanıt bloğu açılmamalı"


def test_render_kaynak_etiketini_her_kanita_yazar():
    """Provenance şeffaflığı ZORUNLU: her kanıt nereden geldiğini söyler."""
    sec = ReportSection(
        id="s", title="T", headline="H", explanation="E",
        evidence=[
            Evidence("test metriği", 0.9, "test_set", sample_size=100),
            Evidence("tanımlayıcı korelasyon", 0.7, "full_data_descriptive"),
        ],
    )
    md = render_section(sec, "tr")
    assert "test seti (model bu satırları görmedi)" in md
    assert "tüm veri — tanımlayıcı" in md, \
        "tüm veriden gelen kanıt 'tanımlayıcı' diye işaretlenmeli (çerçeveleme)"


def test_report_str_render_ediliyor():
    """report: str hâlâ üretiliyor — mevcut UI kırılmadı."""
    secs = [ReportSection(id="a", title="Bölüm A", headline="Baslik", explanation="Aciklama",
                          evidence=[Evidence("m", 1.0, "test_set", sample_size=10)])]
    md = render_report(secs, "tr")
    assert isinstance(md, str) and md.strip()
    assert md.startswith("## Bölüm A")
    assert "**Baslik**" in md


def test_sections_json_serilestirilebilir():
    """report_sections step_data'ya yazılacak → JSON'a çevrilebilmeli."""
    import json
    secs = [ReportSection(id="a", title="A", evidence=[Evidence("m", 0.5, "test_set")])]
    d = sections_to_dicts(secs)
    json.dumps(d)          # patlamamalı
    assert d[0]["evidence"][0]["provenance"] == "test_set"
    assert d[0]["evidence"][0]["reliable"] is True


# ═══════════════════════════════════════════════════════════════════════════
#  MONTAJ — order'a göre dizilim
# ═══════════════════════════════════════════════════════════════════════════

def test_bolumler_yazilma_sirasina_gore_degil_order_a_gore_dizilir():
    """Yeni bölüm eklemek onu SONA itmez — ait olduğu yere koyar.

    Fairness tam da bu mekanizma olmadığı için raporun sonunda kalmıştı;
    Veri İlişkileri ise ortaya girmek zorunda.
    """
    secs = [
        ReportSection(id="fairness", title="Adalet", order=ORDER_FAIRNESS,
                      headline="F", evidence=[Evidence("m", 1, "test_set")]),
        ReportSection(id="header", title="", order=ORDER_HEADER, raw_markdown="# Rapor\n"),
        ReportSection(id="relations", title="Veri İlişkileri", order=ORDER_DATA_RELATIONS,
                      headline="R", evidence=[Evidence("m", 1, "train_only")]),
        ReportSection(id="legacy", title="", order=ORDER_LEGACY_PROSE, raw_markdown="## Blob\n"),
    ]
    md = render_report(secs, "tr")
    # Beklenen sıra: header(0) → relations(30) → legacy(50) → fairness(80)
    assert md.index("# Rapor") < md.index("Veri İlişkileri") < md.index("## Blob") < md.index("Adalet")


def test_raw_markdown_aynen_basilir():
    """Migrasyon kaçışı: taşınmamış blob başlık/kanıt eklenmeden aynen geçer."""
    sec = ReportSection(id="legacy", title="YOKSAYILMALI", order=ORDER_LEGACY_PROSE,
                        raw_markdown="## Ham İçerik\nmetin")
    out = render_section(sec, "tr")
    assert out == "## Ham İçerik\nmetin"
    assert "YOKSAYILMALI" not in out


def test_bos_ham_tasiyici_hic_bir_sey_basmaz():
    """BOŞ ham taşıyıcı ("") HİÇBİR ŞEY basmamalı — başlıksız '## ' artığı bile.

    Gerçek hata: `if sec.raw_markdown:` truthy kontrolüydü; boş taşıyıcı (tabular
    veride zaman serisi bölümü) guard'a takılmayıp NORMAL bölüm gibi render
    ediliyor ve title="" olduğu için '## ' basıyordu. Üstelik '## '.strip() truthy
    olduğu için boş-parça filtresi de onu elemiyordu → her raporda görünür artık.
    Ayrım artık None (taşıyıcı değil) vs "" (boş taşıyıcı).
    """
    secs = [
        ReportSection(id="header", title="", order=ORDER_HEADER, raw_markdown="# Rapor\n"),
        ReportSection(id="timeseries", title="", order=5, raw_markdown=""),   # tabular → boş
        ReportSection(id="legacy", title="", order=ORDER_LEGACY_PROSE, raw_markdown="## Blob"),
    ]
    md = render_report(secs, "tr")
    assert not re.search(r"^#+\s*$", md, flags=re.M), f"başlıksız başlık artığı basıldı:\n{md!r}"
    assert "\n\n\n" not in md, "boş bölüm ardışık boş satır bırakmamalı"
    assert md.index("# Rapor") < md.index("## Blob")


def test_bos_ham_tasiyici_normal_bolume_donusmez():
    """Boş taşıyıcı render_section'da da tek başına '' dönmeli."""
    assert render_section(ReportSection(id="x", title="", raw_markdown=""), "tr") == ""


def test_raw_markdown_bolumler_report_sections_a_sizmaz():
    """Yapısal tüketici (UI) yarım bölüm görmemeli — blob dışarıda kalır."""
    secs = [
        ReportSection(id="legacy", title="", order=ORDER_LEGACY_PROSE, raw_markdown="## Blob"),
        ReportSection(id="fairness", title="Adalet", order=ORDER_FAIRNESS,
                      evidence=[Evidence("m", 1, "test_set")]),
    ]
    d = sections_to_dicts(secs)
    assert [s["id"] for s in d] == ["fairness"]


def test_fairness_bolumu_kendi_order_ini_tasir():
    """Bölüm order'ını AÇIKÇA vermezse ORDER_DEFAULT'a (=blob) düşer ve blob'un
    içine karışır. Fairness'ta tam olarak bu oldu; her bölüm order'ını taşımalı."""
    from src.services.report_sections import build_fairness_section
    for state in (
        {"intent": "clustering"},                                   # not_applicable
        {"intent": "classification",
         "evaluation_report": {"fairness": {"status": "not_computed", "reason": "x"}}},
    ):
        assert build_fairness_section(state, "tr").order == ORDER_FAIRNESS

    y = np.array([1, 0] * 25 + [1] + [0] * 9 + [1] + [0] * 9)
    seg = pd.DataFrame({"grp": ["A"] * 50 + ["B"] * 20})
    fair = analyze_fairness(y, y.copy(), seg, _meta(["grp"]), "classification")
    sec = build_fairness_section(
        {"intent": "classification", "evaluation_report": {"fairness": fair}}, "tr")
    assert sec.order == ORDER_FAIRNESS
    assert sec.order != ORDER_LEGACY_PROSE, "fairness blob sırasına düşmemeli"


def test_sections_to_dicts_order_a_gore_sirali_doner():
    secs = [
        ReportSection(id="fairness", title="F", order=ORDER_FAIRNESS),
        ReportSection(id="relations", title="R", order=ORDER_DATA_RELATIONS),
    ]
    d = sections_to_dicts(secs)
    assert [s["id"] for s in d] == ["relations", "fairness"]
    assert d[0]["order"] == ORDER_DATA_RELATIONS


def test_prompt_ve_fallback_basliklarinda_numara_yok():
    """Numaralandırma KALDIRILDI: bölüm taşındıkça numaralar kayıyordu.

    (fb_h7 fairness'a taşındığında fallback 1,2,3,4,5,6,8,9 olmuştu — canlı kanıt.)
    """
    import re
    from pathlib import Path
    from src.core.report_i18n import _STRINGS

    for key, entry in _STRINGS.items():
        if key.startswith("fb_h"):
            for lang, text in entry.items():
                assert not re.search(r"##\s+\d+\.", text), f"{key}[{lang}] hâlâ numaralı: {text}"

    for p in Path("src/prompts").glob("report_system_*.md"):
        body = p.read_text(encoding="utf-8")
        assert not re.search(r"^##\s+\d+\.", body, flags=re.M), f"{p.name} hâlâ numaralı başlık içeriyor"


# ═══════════════════════════════════════════════════════════════════════════
#  ADIM 6 — SEGMENT ADAYLARINDA SAYISAL TÜREVLER YOK (gürültü ≠ demografik grup)
# ═══════════════════════════════════════════════════════════════════════════

def _seg_frame(n=2000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n).astype(float)
    return pd.DataFrame({
        "age": age,                                        # ham sürekli sayısal
        "age_log": np.log1p(age),                          # türev (LLM planı deseni)
        "ratio_age_credit_score": age / rng.uniform(300, 850, n),      # türev, 1719+ distinct
        "ratio_age_estimated_salary": age / rng.uniform(2e4, 15e4, n), # türev
        "is_zero_age": (age == age.min()).astype(int),     # türev bayrak (age keyword içerir!)
        "gender": rng.choice(["M", "F"], n),               # gerçek demografik, düşük-kard
        "geography": rng.choice([f"c{i}" for i in range(40)], n),  # yüksek-kard KATEGORİK demografik
    })


def _demo_intel(*cols):
    return {"columns": [{"column_name": c, "semantic_type": "demographic"} for c in cols]}


def test_sayisal_turevler_segment_adayi_degil():
    """age_log/ratio_age_*/is_zero_age 'ölçülemedi' listesine BİLE girmez —
    demografik grup değil, türetilmiş sayısaldır (kullanıcının açık şartı)."""
    from src.services.segment_carrier import select_segment_columns
    df = _seg_frame()
    sel, skip = select_segment_columns(df, _demo_intel("gender", "geography"))
    for noise in ("age_log", "ratio_age_credit_score", "ratio_age_estimated_salary", "is_zero_age"):
        assert noise not in sel, f"{noise} segment olarak seçilmemeli"
        assert noise not in skip, f"{noise} 'ölçülemedi' listesine bile girmemeli"


def test_yuksek_kard_surekli_sayisal_gorunmez_kategorik_gorunur():
    """age (yüksek-kard sürekli sayısal) 'ölçülemedi'ye girmez; geography
    (yüksek-kard KATEGORİK demografik) sebebiyle GÖRÜNÜR kalır (binleme ayrı iş)."""
    from src.services.segment_carrier import select_segment_columns
    df = _seg_frame()
    sel, skip = select_segment_columns(df, _demo_intel("gender", "geography"))
    assert "age" not in skip, "sürekli sayısal 'ölçülemedi' listesini kirletmemeli"
    assert skip.get("geography", {}).get("code") == "seg_skip_high_card", \
        "gerçek yüksek-kard kategorik demografik sebebiyle görünür kalmalı"
    assert "gender" in sel, "gerçek düşük-kard demografik seçilmeli"


def test_is_derived_feature_ham_demografige_yanlis_alarm_vermez():
    """Filtre yalnız TÜREVLERİ eler — ham demografik kolonları yanlışlıkla değil."""
    from src.core.feature_naming import is_derived_feature
    for turev in ("age_log", "ratio_a_b", "is_zero_x", "was_missing_income",
                  "count_tx", "grpstd_bal_by_geo", "log_amount", "x_isna"):
        assert is_derived_feature(turev), f"{turev} türev sayılmalı"
    for ham in ("age", "sex", "gender", "race", "marital_status", "country",
                "native_country", "estimated_salary", "balance", "geography"):
        assert not is_derived_feature(ham), f"{ham} ham kolon, türev sayılmamalı"
