"""
Veri İlişkileri bölümü — DEĞİŞMEZ testleri.

İki şeyi çitler:
  1. "train_only" etiketi YALAN SÖYLEMİYOR — hesap fiilen train partisyonundan geçiyor.
  2. Nedensellik koruması YAPISAL — prompt'a bel bağlamıyor.

Çalıştırma:
    uv run pytest tests/test_relations_layered.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.report_schema import render_section, ORDER_DATA_RELATIONS
from src.services.relations import compute_relations, fisher_z_interval
from src.services.report_sections import (
    RELATIONS_VISIBLE_TOP,
    build_relations_section,
    detect_causal_language,
)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    return pd.DataFrame({
        "x1": x1,
        "x2": x1 * 2 + rng.normal(scale=0.01, size=n),   # x1 ile neredeyse aynı
        "x3": rng.normal(size=n),                        # ilişkisiz
    })


# ═══════════════════════════════════════════════════════════════════════════
#  PROVENANCE — "train_only" yapısal olgu mu, iddia mı?
# ═══════════════════════════════════════════════════════════════════════════

def test_train_only_etiketi_gercekten_train_satirlarindan_geliyor():
    """Etiket 'train_only' diyorsa hesap SADECE verilen partisyondan geçmeli.

    Kanıt: test satırlarına KASITLI olarak ters bir ilişki koyuyoruz. Hesap tüm
    veriden geçseydi skor sulanırdı; yalnızca train'den geçtiği için ters ilişki
    sonuca HİÇ karışmamalı.
    """
    n = 400
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    X_train = pd.DataFrame({"x": x})
    y_train = x * 3 + rng.normal(scale=0.1, size=n)      # train: güçlü POZİTİF

    out = compute_relations(X_train, y_train, "hedef", generated_from="train_only")
    assert out["generated_from"] == "train_only"
    assert out["n_rows"] == n, "hesap tam olarak verilen partisyonun satır sayısını görmeli"
    score = out["target_relations"][0]["score"]
    assert score > 0.9, f"train'deki güçlü pozitif ilişki görünmeli, görülen: {score}"

    # Aynı X ama TERS y (test partisyonunu temsilen) → ayrı hesap ayrı sonuç verir.
    out_ters = compute_relations(X_train, -y_train, "hedef", generated_from="train_only")
    assert out_ters["target_relations"][0]["score"] < -0.9, \
        "hesap kendisine verilen y'den geçiyor — global bir yerden değil"


def test_provenance_etiketi_cagirandan_gelir_uydurulmaz():
    """Unsupervised dalda holdout yok → full_data_descriptive. Etiket hesabın
    fiilen geçtiği satırları söyler; fonksiyon kendi kafasına göre karar vermez."""
    df = _frame()
    out = compute_relations(df, None, "", generated_from="full_data_descriptive")
    assert out["generated_from"] == "full_data_descriptive"
    assert out["top_pairs"], "hedefsiz de olsa X-X ilişkileri hesaplanmalı"
    assert out["target_relations"] == [], "hedef yokken X-y ilişkisi olamaz"


def test_bolum_provenance_i_kanita_yaziyor():
    """Kanıt katmanında kaynak GÖRÜNÜR olmalı — çerçeveleme şeffaflığı zorunlu."""
    df = _frame()
    rel = compute_relations(df, df["x1"] * 2, "hedef", generated_from="train_only")
    sec = build_relations_section({"relations_report": rel}, "tr")
    assert all(e.provenance == "train_only" for e in sec.evidence)
    md = render_section(sec, "tr")
    assert "yalnızca eğitim verisi" in md.lower()
    assert "test satırları bu hesaba GİRMEDİ" in md


def test_full_data_kaniti_tanimlayici_diye_isaretlenir():
    """Tüm veriden gelen ilişki 'tanımlayıcı' damgası taşımalı — 'gerçek ilişki'
    diye sunmak çerçeveleme hatasıdır (kullanıcının açık şartı)."""
    df = _frame()
    rel = compute_relations(df, None, "", generated_from="full_data_descriptive")
    sec = build_relations_section({"relations_report": rel}, "tr")
    md = render_section(sec, "tr")
    assert "tüm veri — tanımlayıcı" in md
    assert "TANIMLAYICIDIR" in md


# ═══════════════════════════════════════════════════════════════════════════
#  NEDENSELLİK KORUMASI — üç katman
# ═══════════════════════════════════════════════════════════════════════════

def test_katman1_nedensellik_cercevesi_bolum_basinda_bir_kez():
    """BOILERPLATE TEKRARI YASAĞI: nedensellik çerçevesi + provenance açıklaması
    bölüm başında BİR KEZ (disclaimer), her kanıt satırında DEĞİL.

    Eskiden her satır aynı iki uzun cümleyi taşıyordu (30 ilişki × 2 = 60 satır
    aynı metin). Artık çerçeve disclaimer'da bir kez; satırda yalnız değişen bilgi."""
    df = _frame()
    rel = compute_relations(df, df["x1"] * 2, "hedef", generated_from="train_only")
    sec = build_relations_section({"relations_report": rel}, "tr")
    assert sec.evidence
    # Çerçeve bölüm başında BİR KEZ (nedensellik + provenance açıklaması):
    assert "sebebini açıklamaz" in sec.disclaimer
    assert "test satırları bu hesaba GİRMEDİ" in sec.disclaimer   # provenance açıklaması da
    # Satır başına TEKRAR YOK — boilerplate satırlardan temizlendi:
    for e in sec.evidence:
        assert "sebebini açıklamaz" not in (e.note or ""), f"satırda tekrar: {e.label}"
        assert "birlikte hareket" not in (e.note or "").lower(), f"satırda damga tekrarı: {e.label}"
        assert not (e.reference or ""), f"satırda provenance açıklaması tekrarı: {e.label}"


def test_katman2_bolum_uyarisi_llm_den_bagimsiz():
    """Bölüm uyarısı disclaimer alanında — LLM'in şemasında bu alan YOK."""
    from src.core.report_schema import LLM_SECTION_SCHEMA
    df = _frame()
    rel = compute_relations(df, df["x1"] * 2, "hedef", generated_from="train_only")
    sec = build_relations_section({"relations_report": rel}, "tr")
    assert "sebebini açıklamaz" in sec.disclaimer
    assert "disclaimer" not in LLM_SECTION_SCHEMA["properties"], \
        "LLM uyarıyı değiştirebilir hâle gelmiş"
    md = render_section(sec, "tr")
    assert md.index("ilişkiyi tanımlar") < md.index("Kanıt"), "uyarı kanıttan ÖNCE gelmeli"


def test_katman3_dedektor_nedensel_dili_yakaliyor():
    """Dedektör hem TR hem EN nedensel ifadeleri yakalamalı."""
    assert detect_causal_language("Gelir, eğitimi etkiliyor.")
    assert detect_causal_language("Yaş artışı gelire neden oluyor.")
    assert detect_causal_language("Bu durum düşük gelire yol açıyor.")
    assert detect_causal_language("Eğitim seviyesi geliri belirliyor.")
    assert detect_causal_language("Income causes higher education.")
    assert detect_causal_language("This leads to churn.")
    assert detect_causal_language("Tenure drives revenue.")


def test_katman3_dedektor_masum_dile_yanlis_alarm_vermiyor():
    """Doğru (ilişkisel) dil temiz geçmeli — yoksa dedektör açıklamayı boşuna düşürür."""
    for temiz in (
        "Gelir ile eğitim süresi birlikte artıyor.",
        "Bu iki alan ters yönde hareket ediyor.",
        "Çalışma saati, gelir ile ilişkili görünüyor.",
        "Income and education move together.",
        "These two fields are associated with each other.",
        "En etkili özellikler bölümünde ayrıca ele alınmıştır.",
    ):
        assert detect_causal_language(temiz) == [], f"yanlış alarm: {temiz!r}"


# ═══════════════════════════════════════════════════════════════════════════
#  KATMANLAMA — kırpma yok, katlama var
# ═══════════════════════════════════════════════════════════════════════════

def test_ilk_10_gorunur_kalani_katlanir_hicbiri_kirpilmaz():
    """Kullanıcı şartı: kırpma hedefe aykırı; derinlik isteğe bağlı olmalı."""
    n = 300
    rng = np.random.default_rng(2)
    cols = {f"x{i}": rng.normal(size=n) for i in range(25)}
    df = pd.DataFrame(cols)
    y = df["x0"] * 3 + rng.normal(scale=0.5, size=n)

    rel = compute_relations(df, y, "hedef", generated_from="train_only")
    toplam = len(rel["target_relations"])
    assert toplam > RELATIONS_VISIBLE_TOP, "test anlamlı olsun diye 10'dan fazla ilişki gerekli"

    sec = build_relations_section({"relations_report": rel}, "tr")
    gorunur = [e for e in sec.evidence if "↔ hedef" in e.label]
    assert len(gorunur) == RELATIONS_VISIBLE_TOP

    katlanan = next(s for s in sec.subsections if s.id == "data_relations_all")
    assert len(gorunur) + len(katlanan.evidence) == toplam, \
        "görünür + katlanan = TOPLAM olmalı; hiçbir ilişki kırpılmamalı"


def test_gereksiz_sutun_ciftleri_ayri_bolumde():
    df = _frame()
    rel = compute_relations(df, df["x1"] * 2, "hedef", generated_from="train_only")
    assert rel["redundant_pairs"], "x1≈x2 çifti yakalanmalı"
    sec = build_relations_section({"relations_report": rel}, "tr")
    assert any(s.id == "data_relations_redundant" for s in sec.subsections)


def test_bolum_kendi_order_ini_tasir():
    sec = build_relations_section({}, "tr")
    assert sec.order == ORDER_DATA_RELATIONS


def test_relations_yoksa_gorunur_sebep():
    sec = build_relations_section({}, "tr")
    assert sec.status == "not_computed"
    assert sec.not_computed_reason


# ═══════════════════════════════════════════════════════════════════════════
#  FISHER Z — korelasyon güven aralığı
# ═══════════════════════════════════════════════════════════════════════════

def test_fisher_z_sinirlar_icinde_kalir():
    """Düz normal yaklaşım |r|→1'de [-1,1] dışına taşar; Fisher z taşmaz."""
    ci = fisher_z_interval(0.98, 50)
    assert -1 <= ci["lower"] <= ci["upper"] <= 1
    assert ci["method"] == "fisher-z"


def test_fisher_z_kucuk_ornekte_genis_aralik():
    dar = fisher_z_interval(0.5, 5000)
    genis = fisher_z_interval(0.5, 10)
    assert (genis["upper"] - genis["lower"]) > (dar["upper"] - dar["lower"])


def test_fisher_z_tanimsizsa_none():
    """n<=3 → aralık yok. Sıfır genişlikte sahte kesinlik ÜRETİLMEZ."""
    assert fisher_z_interval(0.5, 3) is None
    assert fisher_z_interval(None, 100) is None


def test_kategorik_hedefte_fisher_z_uygulanmaz():
    """Kategorik hedefte skor korelasyon DEĞİL (ayrışma oranı) → CI iddia edilmez."""
    n = 200
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": rng.normal(size=n)})
    y = pd.Series(["evet" if v > 0 else "hayir" for v in df["x"]])
    out = compute_relations(df, y, "hedef", generated_from="train_only")
    assert out["target_type"] == "categorical"
    assert all(r["ci"] is None for r in out["target_relations"]), \
        "korelasyon olmayan skora Fisher z uygulanmış"


# ═══════════════════════════════════════════════════════════════════════════
#  ADIM 5 — ANLAMSIZ İLİŞKİLER ANA LİSTEDEN ELENİR (gürültü ≠ bilgi)
# ═══════════════════════════════════════════════════════════════════════════

def _derived_frame(n=2000, seed=42):
    """Ham + motorun türettiği kolonlar (dataset-agnostik desenler)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, n).astype(float)
    credit_card = rng.integers(0, 2, n)
    active_member = rng.integers(0, 2, n)
    return pd.DataFrame({
        "age": age,
        "credit_card": credit_card,
        "active_member": active_member,
        "balance": rng.normal(50000, 15000, n),
        "age_log": np.log1p(age),                    # age'in math türevi
        "is_zero_credit_card": 1 - credit_card,      # credit_card'ın ters kopyası
        "is_zero_active_member": 1 - active_member,  # active_member'ın ters kopyası
    }), active_member


def _has(pairs, x, y):
    s = {(p["a"], p["b"]) for p in pairs}
    return (x, y) in s or (y, x) in s


def test_turetilmis_kolon_kaynakla_iliskisi_ana_listeden_elenir():
    """age↔age_log, credit_card↔is_zero_credit_card gibi TANIM gereği yüksek çiftler
    ana listede DEĞİL — near-duplicate bölümünde kalır (kullanıcının açık şartı)."""
    df, active_member = _derived_frame()
    rel = compute_relations(df, active_member, "hedef", generated_from="train_only")

    assert not _has(rel["top_pairs"], "age", "age_log"), "türev↔kaynak ana listede kalmış"
    assert _has(rel["redundant_pairs"], "age", "age_log"), "türev↔kaynak near-dup'ta olmalı"
    assert not _has(rel["top_pairs"], "credit_card", "is_zero_credit_card")
    assert _has(rel["redundant_pairs"], "credit_card", "is_zero_credit_card")


def test_ayna_hedef_ciftleri_teke_iner():
    """active_member ve is_zero_active_member ikisi de hedefe karşı listelenmez —
    biri (ham kolon) yeterli; ayna kopyası bilgi eklemez."""
    df, _ = _derived_frame()
    # churn: active_member ile ilişkili → hem raw hem is_zero_ türevi skorlanır
    rng = np.random.default_rng(1)
    churn = ((df["active_member"] == 0).astype(int)
             + (rng.random(len(df)) < 0.3).astype(int) >= 1).astype(int)
    rel = compute_relations(df, churn, "churn", generated_from="train_only")
    feats = [r["feature"] for r in rel["target_relations"]]

    assert "active_member" in feats, "ham kolon korunmalı"
    assert "is_zero_active_member" not in feats, "ayna kopyası elenmeli — biri yeterli"


def test_gercek_iliskiler_yanlislikla_elenmez():
    """Farklı kaynaklardan gelen GERÇEK ilişkiler (age↔balance) korunur —
    filtre yalnız türev↔kaynak/ayna gürültüsünü eler, bilgiyi değil."""
    from src.services.relations import _is_definitional_pair
    assert not _is_definitional_pair("age", "balance")
    assert not _is_definitional_pair("credit_score", "estimated_salary")
    # ratio'nun kaynağı DEĞİL bir kolon yanlışlıkla türev sayılmaz
    assert not _is_definitional_pair("estimated_salary", "ratio_age_credit_score")
