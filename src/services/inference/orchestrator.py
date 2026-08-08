from __future__ import annotations

"""Katman 1 orkestratör İSKELESİ — mevcut adımları gerçek ``build_features`` sırasıyla replay eder.

İki giriş, tek adım-sırası:
  * ``fit_pipeline(df, y, ctx)``       → (X_train dönüştürülmüş, Artifact). Her adımın
    ``fit_transform``'ını çağırır (train; TargetEncode cross-fitted). Manifest ``order``'ı
    FİİLİ fit-döngüsü indeksinden TÜRETİR (elle atanmaz).
  * ``transform_pipeline(raw, artifact)`` → X_ready. Manifest'i ``order``'a göre replay eder,
    YALNIZ ``step.transform`` çağırır → ``fit_transform`` pipeline düzeyinde ULAŞILAMAZ
    (E invariant'ı: cross-fitting serving'e sızamaz). route/skip kalemlerini atlar.

Sıra doğruluğu iki katman (§handoff A):
  1. Manifest ``order`` ← fit-döngüsü yürütme indeksinden (elle atama YOK) → manifest-sadakat invariant'ı.
  2. ``PIPELINE`` sırası ← elle ama ``legacy_ref`` KAYNAK-DRİFT testiyle sürekli doğrulanır: her adımın
     ``legacy_ref``'i ``build_features``'ta o adımın karşılık geldiği eski fonksiyon; test kaynaktaki
     göreli sıranın PIPELINE ile eşleştiğini assert eder. Koşullu/henüz-yazılmamış adımlar (ts-lag,
     aggregate, groupby, select) PIPELINE'da temsil EDİLMEZ → drift testi yalnız temsil edilenlerin
     GÖRELİ sırasını kontrol eder ("temsil ettiklerimiz doğru sırada", "her şeyi temsil ediyoruz" değil).

Route'lar (encode, scaler): TransformStep DEĞİL — "fit-öncesi karar". ``fit_pipeline`` route'u
X_train'den hesaplar, manifest'e ``frozen_choice`` karar kalemi yazar, executor'larını (config'i
route'tan) sırayla fit eder. Serving'de route kalemi ATLANIR (executor params'ı donmuş kolonları
zaten taşır). İki route AYNI ``RouteSpec`` deseniyle ele alınır (şekil-tutarlılığı → tek kod yolu).
"""

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from src.core.logging_config import get_logger
from src.services.inference.protocol import Artifact
from src.services.inference.steps import (
    CountEncodeStep, DateFeaturesStep, DropColumnsStep, GroupbyDeviationStep, ImputeStep,
    InfImputeStep, MissingIndicatorStep,
    NormalizeStep, OneHotStep, PlanDerivedStep, RatioConstructStep, RatioFinalizeStep,
    RobustScaleStep,
    SelectDropStep, StandardScaleStep, TargetEncodeStep, ZeroFlagStep, ZScoreStep,
    decide_encode_route, decide_scaler_route,
)

logger = get_logger(__name__)


class MissingFeatureProduction(RuntimeError):
    """``final_features``'ta olup HİÇBİR adımın üretmediği kolon(lar) var → FAIL-CLOSED.

    Terminal ``reindex(fill_value=0)`` sessiz bir tehlike taşır: portlanmamış bir adımın
    üretmesi gereken kolon, hata vermeden 0 dolar → model yanlış girdiyle skorlar ve
    hiçbir iz kalmaz. Bu tam olarak Katman 1'in önlemek için var olduğu hata sınıfı.

    Bu yüzden reindex artık YALNIZ *politikası tanımlı* eksikleri doldurur (bkz.
    ``zero_fill_policy``); tanımsız eksik → bu istisna, skorlama HİÇ yapılmaz. Yanlış
    skor yerine hata: sessiz bozulmadan iyidir.

    ENVANTER ARACI: istisnanın ``missing`` listesi, "orkestratörün ham girdiden
    üretemediği kolonlar"ın TAM listesidir — elle çıkarılmış bir envanterin aksine
    kodun kendisi söyler.

    İKİ FARKLI BAŞARISIZLIK MODU (karıştırılmamalı — sebepleri farklı):
      * ``never_produced``        — replay boyunca bu kolon HİÇ var olmadı → adım
        portlanmamış / girdi sözleşmesi eksik. **Kapsam (port) borcu.**
      * ``produced_then_dropped`` — kolon bir adım tarafından üretildi ama sonraki bir
        adım (tipik olarak ``select_drop``) onu düşürdü → manifest ile ``final_features``
        TUTARSIZ. **Artifact bütünlüğü hatası** (farklı bir fit'in feature listesiyle
        eşleştirme, bayat artifact vb.). Port etmek bunu ÇÖZMEZ.
    """

    def __init__(self, never_produced: list, produced_then_dropped: list, policy_filled: list):
        self.never_produced = sorted(never_produced)
        self.produced_then_dropped = sorted(produced_then_dropped)
        self.missing = sorted(set(self.never_produced) | set(self.produced_then_dropped))
        self.policy_filled = sorted(policy_filled)
        parts = [
            f"final_features'ta {len(self.missing)} kolon üretilemedi "
            f"(sessiz 0-doldurma ENGELLENDİ, skorlama yapılmadı)."
        ]
        if self.never_produced:
            parts.append(f"HİÇ ÜRETİLMEDİ ({len(self.never_produced)}, port borcu): {self.never_produced}")
        if self.produced_then_dropped:
            parts.append(f"ÜRETİLDİ SONRA DÜŞÜRÜLDÜ ({len(self.produced_then_dropped)}, "
                         f"artifact tutarsızlığı): {self.produced_then_dropped}")
        if self.policy_filled:
            parts.append(f"politikayla doldurulan (izinli): {self.policy_filled}")
        super().__init__(" | ".join(parts))


class FrameShapeDivergence(RuntimeError):
    """Serving ara çerçevesi, EĞİTİMDEKİ ara çerçeveden ayrıldı → FAIL-CLOSED.

    ``MissingFeatureProduction`` ASİMETRİKTİ: yalnız *eksik üretim* soruyordu, *fazla
    mevcudiyet* sormuyordu. Bu asimetri 4. kökü (hedef/id drop'unun portlanmamış olması)
    SAKLADI — eksik bir kaldırma "kolon üretilmedi" göstermez, sessizce downstream SEÇİM
    SAPMASI üretir. Bu sınıf o boşluğu kapatır: her adım sınırında eğitim kolon-seti ile
    serving kolon-seti İKİ YÖNDE karşılaştırılır.

    KAPSAM DÜRÜSTLÜĞÜ (abartma yok): bu kontrol **serving↔eğitim** sapmasını yakalar. 4.
    kökün kendisi bir **fit-tarafı** sorunuydu (orkestratörün fit girdisi ``build_features``'ınkinden
    farklıydı) ve bu kontrol onu YAKALAMAZDI — fit ve transform aynı fazla kolonu görüyordu.
    Fit tarafının karşılığı ``StructuralPreconditionViolation`` + kapsam envanteri scripti.

    ``extra`` neden hata: §2.4 "fazla girdi kolonu tolere edilir" diyordu, ama bu oturumun
    dersi sessiz sapmanın düşman olduğu. Uzlaşma: tolerans MÜMKÜN ama **beyan edilmeli**
    (``Artifact.tolerated_extra_columns``); beyansız fazla kolon = hata.
    """

    def __init__(self, step_id: str, missing: list, extra: list):
        self.step_id, self.missing, self.extra = step_id, sorted(missing), sorted(extra)
        parts = [f"'{step_id}' adımından sonra serving çerçevesi eğitimden AYRILDI"]
        if self.missing:
            parts.append(f"EĞİTİMDE VAR, SERVING'DE YOK ({len(self.missing)}): {self.missing}")
        if self.extra:
            parts.append(f"SERVING'DE FAZLA, EĞİTİMDE YOK ({len(self.extra)}): {self.extra} "
                         f"— beyan edilmemiş fazla kolon (bkz. tolerated_extra_columns)")
        super().__init__(" | ".join(parts))


class StepDeltaDivergence(RuntimeError):
    """Bir adımın KOLON SÖZLEŞMESİ serving'de eğitimdekinden ayrıldı → FAIL-CLOSED.

    ``FrameShapeDivergence``'ın (mutlak kolon seti) yerini alır. Fark neden önemli: mutlak set
    GLOBAL bir invariant'tı — girdideki tek bir meşru sapma sonraki HER adımda alarm üretiyordu ve
    pre-split adımların incremental çerçevesi fully-built X_test'le hiç uyuşmadığı için **5 adım
    guard'dan muaf** tutulmak zorunda kalmıştı. Delta YEREL'dir: "bu adım şu kolonları ekledi/çıkardı"
    adımın kendi sözleşmesidir, çerçevenin geri kalanına bakmaz → muafiyet gerekmez ve sapma
    ADIMDA lokalize olur.

    Dört mod ayrı raporlanır (sebepleri farklı):
      * ``eksik_ekleme`` — eğitimde üretilen kolon serving'de doğmadı (girdi kolonu eksik olabilir)
      * ``fazla_ekleme`` — serving'de fazladan kolon doğdu (config/param sapması)
      * ``eksik_silme``  — eğitimde düşen kolon serving'de DURUYOR (drop listesi uygulanmadı)
      * ``fazla_silme``  — serving'de fazladan kolon düştü
    """

    def __init__(self, step_id, beklenen, fiili, eksik_ekleme, fazla_ekleme, eksik_silme, fazla_silme):
        self.step_id, self.beklenen, self.fiili = step_id, beklenen, fiili
        self.eksik_ekleme, self.fazla_ekleme = eksik_ekleme, fazla_ekleme
        self.eksik_silme, self.fazla_silme = eksik_silme, fazla_silme
        parts = [f"'{step_id}' adımının kolon sözleşmesi SERVING'de değişti"]
        for ad, v in (("EKSİK EKLEME", eksik_ekleme), ("FAZLA EKLEME", fazla_ekleme),
                      ("EKSİK SİLME", eksik_silme), ("FAZLA SİLME", fazla_silme)):
            if v:
                parts.append(f"{ad} ({len(v)}): {sorted(v)}")
        super().__init__(" | ".join(parts))


class StructuralPreconditionViolation(RuntimeError):
    """Eski yolun **YAPI** kaynaklı (adım kaynaklı DEĞİL) bir garantisi sağlanmadı → artifact KURULMAZ.

    DERS (2026-07-23, ölçülerek): ``build_features``'ın bazı garantileri hiçbir fonksiyonda
    yazılı değildi — pipeline'ın SIRASINDAN/YAPISINDAN geliyordu. Adımlara çevirirken o
    garantiler sessizce KAYBOLDU ve eski yolda İMKÂNSIZ olan sızıntı yüzeyleri açıldı:

      * **Hedef çerçevede kalırsa** groupby/count HEDEFTEN feature türetir (``dev_<target>_by_X``).
      * **ID çerçevede kalırsa** (ÖLÇÜLDÜ, daha ağır): benzersiz string ID ``encode_route``
        eşiğini aşıp **TargetEncoder**'a gider → serving çıktısı ``y``'nin BİREBİR kendisidir
        (ölçüm: 60 satırda ``allclose(encoded, y) == True``). Yani kolon target'ın kopyası olur.
        Integer ID ise groupby'ın ``-norm_var`` sıralamasına girip gerçek bir kolonu top-N'den atar.
      * **Dedup edilmemiş çerçeve** (ÖLÇÜLDÜ): tüm ``fitted_stat``'lar farklı dağılımdan öğrenilir
        (impute medyanı 3.0 vs 2.0). ``drop_duplicates`` "skip" diye işaretlenmişti ama masum DEĞİL —
        satır üyeliğini değiştirdiği için ondan SONRAKİ her istatistiği değiştirir.

    Bu yüzden yapısal garantiler artık AÇIK ÖN-KOŞUL: beyan edilir ve fit-anında DENETLENİR.
    """

    def __init__(self, violations: list):
        self.violations = list(violations)
        super().__init__(
            f"Yapısal ön-koşul ihlali ({len(self.violations)}) — artifact kurulmadı: "
            + "; ".join(self.violations)
        )


class UnsupportedPipelineScope(RuntimeError):
    """Bu koşuda Katman 1'in DESTEKLEMEDİĞİ bir yol tetiklendi → artifact KURULMAZ.

    Kapsam envanteri (``scripts/layer1_coverage_inventory.py``) gösterdi ki guard yalnız
    GÖRDÜĞÜNÜ kanıtlar; tetiklenmemiş yollar hakkında hiçbir şey söylemez. Doğru cevap o
    yolları TAHMİNEN portlamak değil — **desteklemediğimizi söylemek**. Ölçemediğimiz bir
    yolu desteklediğimizi iddia etmektense reddetmek hem dürüst hem çok daha ucuz.

    FAIL-CLOSED KONUMU: serving'de değil **artifact-build anında** (``fit_pipeline``).
    Sebep: serving'de patlamak "eğitim bitti, model var, ama skorlanamıyor" demektir —
    hatayı en geç yerde bulmak. Build anında patlamak "bu koşu deploy edilemez" der ve
    eksik artifact hiç doğmaz.

    KAPSAM GENİŞLETME (her yeni modalite için sıra): port et → guard'ı koş (eksik-üretim
    0 olmalı) → ``_SCOPE_RULES``'a destekleniyor diye ekle. Üçü birden yapılmadan kapsam
    genişlemez.
    """

    def __init__(self, violations: list):
        self.violations = list(violations)
        detail = "; ".join(f"{v['fact']}={v['value']!r} → {v['reason']}" for v in self.violations)
        super().__init__(
            f"Katman 1 kapsamı DIŞI ({len(self.violations)} koşul) — artifact kurulmadı, "
            f"bu koşu deploy EDİLEMEZ: {detail}"
        )


# Kapsam kuralları — TEK doğruluk kaynağı. (fact, ihlal mi?, gerekçe)
# Bir yol PORTLANIP guard'da 0 eksik verdiğinde kuralı buradan kaldır = kapsam genişledi.
_SCOPE_RULES: list = [
    ("modality",       lambda v: bool(v) and str(v).lower() != "tabular",
     "yalnız tabular destekleniyor (ts-lag katmanı portlanmadı)"),
    ("aggregated",     lambda v: bool(v),
     "aggregation yolu (_aggregate_if_needed) portlanmadı — satır grameri değişiyor"),
    ("timeseries_lag", lambda v: bool(v),
     "time-series lag/rolling feature'ları portlanmadı"),
    ("domain_features", lambda v: bool(v),
     "domain pack feature'ları (_apply_feature_specs/_apply_domain_recipes) portlanmadı"),
    # ("date_features", ...) KALDIRILDI — §2.19 KAPSAM PORT'U: timestamp türevleri (_year/_month/
    # _dayofweek) artık DateFeaturesStep (rowwise, durumsuz) ile üretiliyor. Port et → guard 0 →
    # kuralı kaldır (6-op portundaki `unported_ops` kaldırımıyla aynı disiplin). Ham timestamp
    # kolonu imzada `required`; adım drop_columns'tan ÖNCE çalışır, serving replay'inde üretilir.
    ("column_renames", lambda v: bool(v),
     "_normalize_column_names yeniden-adlandırması portlanmadı (ham ad → eğitim adı eşlemesi yok)"),
    # ("unported_ops", ...) KALDIRILDI — 6-op portu sonrası palet 8/8 PORTLU.
    # Kapsam genişletme kuralı uygulandı: port et → guard 0 → kuralı kaldır.
    # §2.19'un 6. ÖRNEĞİ: eski yol spec'leri TEK df üzerinde PLAN SIRASINDA işliyordu →
    # bir spec ÖNCEKİNİN çıktısını girdi olarak kullanabiliyordu (zincir, 1657 ile çalışıyor).
    # Orkestratörde spec'ler op-kovalarına bölünür (parametric ↔ rowwise) ve sıra PIPELINE'dan
    # gelir → zincir SESSİZCE kırılırdı (girdi o an yok → spec atlanır → kolon hiç doğmaz).
    # Garanti YAPIDAN geliyordu; açık beyana çevrildi. Şema zinciri artık AÇIKÇA yasaklıyor
    # (birinci savunma); bu kural son savunma. Zincir sıklığı ÖLÇÜLEMEDİ (alan-sonrası üretim
    # verisi yok) → ölçemediğimiz riski kabul etmek yerine REDDEDİYORUZ (§2.17 ilkesi).
    ("chained_derived_features", lambda v: bool(v),
     "plan'da ZİNCİRLEME türev feature var (bir spec'in girdisi başka bir spec'in çıktısı) — "
     "orkestratör spec'leri op-kovalarına böldüğü için zincir sessizce kırılır"),
    # YAPISAL ÖN-KOŞUL (adım değil, YAPI kaynaklı garanti — ölçüldü: dedup'sız çerçevede
    # impute medyanı 3.0 yerine 2.0 çıkıyor → ondan sonraki HER fitted_stat kayar).
    ("deduplicated",   lambda v: not v,
     "girdi çerçevesi dedup EDİLMEMİŞ — _clean_data'nın drop_duplicates'i portlanmadı; "
     "satır üyeliği farklı → tüm fitted_stat'lar başka dağılımdan öğrenilir"),
    # T5-(2) — §2.19'un BEŞİNCİ örneği. `_clean_data`'nın currency bloğu (regex + to_numeric)
    # portlanmadı: serving girdisinin ZATEN numerik geldiği SESSİZCE varsayılıyordu. Artifact
    # bugüne dek "currency'yi hallederim" iddiasını beyansız taşıyordu. Yön güvenli: currency
    # tetikleyen kullanıcı bugün SESSİZCE eksik artifact alıyor; olgu beyan edilince
    # deployable=False olur — eğitim/rapor etkilenmez, yalnız DEPLOYMENT reddedilir.
    # **SESSİZ YANLIŞTAN DÜRÜST REDDE.** Port edilince (backlog) bu kural da kalkar.
    ("currency_normalized", lambda v: not v,
     "girdide metin-para (`$`/`€`/`£`) kolonları var ve dönüşüm portlanmadı — serving ham satırı "
     "string gelirse hiçbir adım onu numeriğe çevirmez (eğitimde _clean_data çeviriyordu)"),
]


_SIG_MAX_VOCAB = 50        # bu kardinalitenin üstü "yüksek-kardinalite" — vocab saklanmaz


def build_raw_input_signature(X: pd.DataFrame, datetime_cols: "list | None" = None) -> dict:
    """Serving girdisinin **DOMAIN-FARKINDA** imzası (§1.1 şeması, ilk fiili dolum).

    NEDEN DOMAIN-FARKINDA (yalnız dtype YETMEZ): depolama tipi `float64` olan bir kolon hem
    `HasCrCard∈{0,1}` hem `income∈[0,10^6]` olabilir. Serving'de `HasCrCard=0.7` gelirse dtype
    kontrolü GEÇER, model saçmalar. §1.1'in kararı: **binary/kategorik için DAR tolerans +
    DEĞER-DOMAIN kontrolü, sürekli için GENİŞ tolerans + aralık UYARISI.**

    **KRİTİK AYRIM — kategorikte görülmemiş değer HATA DEĞİLDİR:** encoder'lar onu zaten ilkeli
    biçimde karşılıyor (onehot→all_zero ≡ referans kategori, target→prior, count→0, groupby→global).
    Kategorikte katı domain kontrolü uygulamak MEŞRU serving'i kırardı. Bu yüzden:
      * ``binary``      → **STRICT**: bilinen iki değerin dışı = ABORT (HasCrCard=0.7 vakası).
      * ``categorical`` → görülmemiş değer İZİNLİ ama SAYILIR/loglanır (Katman 2 veri-kalite sinyali).
      * ``continuous``  → yalnız aralık UYARISI (dağılım kayması PSI'ın işi, imzanın değil).
      * ``datetime``    → parse + gözlenen tarih-aralığı UYARISI (abort DEĞİL). ``datetime_cols``
        (pipeline'ın meta.timestamp'i) AUTHORITATIVE işaretler — dtype string de olsa (ham CSV
        timestamp'i object gelir). §2.19 date portu sonrası ham ts kolonu imzada ``required``.
    Yüksek kardinalite (> ``_SIG_MAX_VOCAB``) → vocab saklanmaz (artifact şişmesin), yalnız sayım.
    """
    _dt_cols = {str(c) for c in (datetime_cols or [])}
    cols: dict = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            aile = "bool"
        elif pd.api.types.is_numeric_dtype(s):
            aile = "number"
        elif pd.api.types.is_datetime64_any_dtype(s):
            aile = "datetime"
        else:
            aile = "object"
        nun = int(s.nunique(dropna=True))
        kayit: dict = {"dtype_family": aile, "nunique": nun,
                       "null_ratio": round(float(s.isna().mean()), 6)}
        if str(c) in _dt_cols:
            # DATETIME — pipeline AUTHORITATIVE tipledi (dtype object/datetime64 fark etmez).
            # range = gözlenen [min,max] tarih (ISO); parse_rate serving'de parse-uyarısı için.
            kayit["kind"] = "datetime"
            _parsed = pd.to_datetime(s, errors="coerce")
            _valid = _parsed.dropna()
            kayit["range"] = ([_valid.min().isoformat(), _valid.max().isoformat()]
                              if len(_valid) else None)
            kayit["parse_rate"] = round(float(_parsed.notna().mean()), 6)
        elif nun <= 2 and aile in ("number", "bool"):
            kayit["kind"] = "binary"
            kayit["allowed_values"] = sorted(str(v) for v in s.dropna().unique())
        elif aile in ("object", "bool") or (aile == "number" and nun <= _SIG_MAX_VOCAB):
            kayit["kind"] = "categorical"
            if nun <= _SIG_MAX_VOCAB:
                kayit["known_values"] = sorted(str(v) for v in s.dropna().unique())
            else:
                kayit["known_values"] = None          # yüksek kardinalite: vocab saklanmaz
        elif aile == "number":
            kayit["kind"] = "continuous"
            kayit["range"] = [float(s.min()), float(s.max())] if len(s.dropna()) else None
        else:
            kayit["kind"] = "other"
        cols[str(c)] = kayit
    return {
        "columns": cols,
        "required": sorted(str(c) for c in X.columns),
        "policy": {"binary": "strict_domain_abort", "categorical": "unseen_allowed_counted",
                   "continuous": "range_warn_only", "datetime": "parse_and_range_warn",
                   "missing_required": "abort", "undeclared_extra": "abort_unless_tolerated"},
        "signature_version": "1.0",
    }


class RawInputSignatureViolation(RuntimeError):
    """Serving girdisi artifact'in beyan ettiği imzaya UYMUYOR → skorlama HİÇ yapılmaz.

    Delta-guard adım-başına kolon SÖZLEŞMESİNİ doğrular; ama "hiçbir adımın dokunmadığı fazladan
    kolon" onun kör noktasıdır (hiçbir delta bozulmaz, kolon sonda ``reindex`` ile SESSİZCE düşer).
    O kontrol ait olduğu yere — girişe — taşındı. Ayrıca bu, dataset-imza guard'ının (§5 task#7)
    ilk fiili parçasıdır: yanlış dataset'i doğru artifact'e vermek burada yakalanır."""

    def __init__(self, missing: list, extra: list, domain: list):
        self.missing, self.extra, self.domain = sorted(missing), sorted(extra), list(domain)
        parts = ["serving girdisi raw_input_signature ile UYUŞMUYOR (skorlama yapılmadı)"]
        if self.missing:
            parts.append(f"EKSİK zorunlu kolon ({len(self.missing)}): {self.missing}")
        if self.extra:
            parts.append(f"BEYANSIZ fazla kolon ({len(self.extra)}): {self.extra} "
                         f"(bkz. tolerated_extra_columns)")
        if self.domain:
            parts.append(f"DEĞER-DOMAIN ihlali ({len(self.domain)}): {self.domain}")
        super().__init__(" | ".join(parts))


def check_raw_input(raw: pd.DataFrame, signature: dict, tolerated: "list | None" = None) -> dict:
    """Giriş imzası denetimi. Fail-closed: eksik/fazla/binary-domain ihlali → istisna.
    Görülmemiş kategori ve aralık-dışı sürekli değer SAYILIR, engellemez (bkz. build_raw_input_signature)."""
    if not signature:
        return {"checked": False, "reason": "imza yok (eski artifact)"}
    cols = signature.get("columns") or {}
    tol = set(tolerated or [])
    gelen = {str(c) for c in raw.columns}
    beklenen = set(signature.get("required") or [])
    missing = sorted(beklenen - gelen)
    extra = sorted(gelen - beklenen - tol)
    domain_ihlal, unseen, aralik_disi = [], {}, {}
    for c, k in cols.items():
        if c not in raw.columns:
            continue
        s = raw[c]
        if k.get("kind") == "binary":
            izin = set(k.get("allowed_values") or [])
            gorulen = {str(v) for v in s.dropna().unique()}
            disi = sorted(gorulen - izin)
            if disi:
                domain_ihlal.append({"column": c, "allowed": sorted(izin), "seen": disi})
        elif k.get("kind") == "categorical" and k.get("known_values") is not None:
            bilinen = set(k["known_values"])
            yeni = sorted({str(v) for v in s.dropna().unique()} - bilinen)
            if yeni:
                unseen[c] = len(yeni)                  # İZİNLİ — yalnız sayılır (Katman 2 sinyali)
        elif k.get("kind") == "continuous" and k.get("range"):
            lo, hi = k["range"]
            n = int(((pd.to_numeric(s, errors="coerce") < lo) |
                     (pd.to_numeric(s, errors="coerce") > hi)).sum())
            if n:
                aralik_disi[c] = n                     # yalnız UYARI (dağılım kayması PSI'ın işi)
        elif k.get("kind") == "datetime":
            # PARSE + ARALIK UYARISI (abort DEĞİL — kötü tarih → date türevleri NaN, degrade ama fatal değil).
            parsed = pd.to_datetime(s, errors="coerce")
            bad = int((parsed.isna() & s.notna()).sum())            # parse edilemeyen (zaten-null hariç)
            rng = k.get("range")
            if rng:
                lo, hi = pd.to_datetime(rng[0]), pd.to_datetime(rng[1])
                bad += int(((parsed < lo) | (parsed > hi)).sum())   # gözlenen tarih-aralığı dışı
            if bad:
                aralik_disi[c] = bad                   # continuous ile aynı kova (parse+aralık uyarısı)
    if missing or extra or domain_ihlal:
        raise RawInputSignatureViolation(missing, extra, domain_ihlal)
    if unseen:
        logger.info(f"[orchestrator] imza: görülmemiş kategori (izinli, sayıldı): {unseen}")
    if aralik_disi:
        logger.info(f"[orchestrator] imza: aralık-dışı sürekli değer (uyarı): {aralik_disi}")

    # ── PER-ROW kalite bayrakları (Katman 2 veri kalitesi) ──
    # Yukarıdaki `unseen`/`aralik_disi` BATCH-agregatı (unseen=distinct sayı, aralık=satır sayısı) →
    # pencerede toplanamaz (scoring her satıra KOPYALIYOR). Per-row bool bayrak satır-bazlı toplanır
    # → doğru pencere oranı. AYRI döngü + try/except: çekirdek imza kontrolünü ETKİLEMEZ, patlarsa
    # None (scoring fail-open'ı korur — "bayrak hesabı patlarsa skor yine döner"). Küçük tekrar (isin/
    # range) izolasyon bedeli; hot-path maliyeti ölçüldü (ihmal edilebilir).
    row_flags = None
    try:
        has_unseen = pd.Series(False, index=raw.index)
        has_oor = pd.Series(False, index=raw.index)
        for c, k in cols.items():
            if c not in raw.columns:
                continue
            s = raw[c]
            if k.get("kind") == "categorical" and k.get("known_values") is not None:
                yeni = {str(v) for v in s.dropna().unique()} - set(k["known_values"])
                if yeni:
                    has_unseen |= s.astype(str).isin(yeni)
            elif k.get("kind") == "continuous" and k.get("range"):
                lo, hi = k["range"]
                sn = pd.to_numeric(s, errors="coerce")
                has_oor |= ((sn < lo) | (sn > hi)).fillna(False)
            elif k.get("kind") == "datetime":
                parsed = pd.to_datetime(s, errors="coerce")
                bad = (parsed.isna() & s.notna())                   # parse edilemeyen
                rng = k.get("range")
                if rng:
                    lo, hi = pd.to_datetime(rng[0]), pd.to_datetime(rng[1])
                    bad = bad | ((parsed < lo) | (parsed > hi)).fillna(False)
                has_oor |= bad.fillna(False)                        # datetime kalite → out_of_range kovası
        row_flags = {"has_unseen": [bool(x) for x in has_unseen],
                     "has_out_of_range": [bool(x) for x in has_oor]}
    except Exception as _e:                                  # noqa: BLE001 — bayrak asla skoru bozmaz
        logger.info(f"[orchestrator] per-row kalite bayrağı hesaplanamadı (fail-safe, None): {_e}")
        row_flags = None

    return {"checked": True, "unseen_categories": unseen, "out_of_range": aralik_disi,
            "row_flags": row_flags}


def check_structural_preconditions(X: pd.DataFrame, ctx: dict) -> list:
    """Eski yolun YAPI kaynaklı garantilerini fiili çerçeve üzerinde denetle (fit-anı).

    ``_SCOPE_RULES`` ctx'teki OLGULARA bakar; bu ise ÇERÇEVENİN KENDİSİNE bakar — çünkü
    "hedef kolon hâlâ burada mı" sorusu ancak veriye bakarak cevaplanır. Drop adımından
    SONRA çağrılır (drop'un işini yapıp yapmadığını da doğrular)."""
    ctx, violations = ctx or {}, []
    cols = set(X.columns)

    target = ctx.get("target_col")
    if target and target in cols:
        violations.append(
            f"hedef kolon '{target}' çerçevede — groupby/count/target-encode HEDEFTEN feature "
            f"türetir (dev_<target>_by_X); eski yolda hedef split'ten ÖNCE düşüyordu")

    kalan = [c for c in (ctx.get("drop_columns") or []) if c in cols]
    if kalan:
        violations.append(
            f"drop listesindeki kolon(lar) hâlâ çerçevede: {sorted(kalan)} — id/timestamp "
            f"kalırsa benzersiz ID target-encode'a gider (çıktı = y'nin kopyası) veya "
            f"groupby'ın seçimini saptırır")
    return violations

# ``_FEATURE_OPS``'tan protokole portlanmış op'lar. Yenisi portlandıkça buraya eklenir.
# 6-OP PORTU sonrası palet TAM: parametric ikili (fitted-state) + rowwise altılı.
PORTED_PARAMETRIC_OPS: set = {"zscore", "normalize"}            # ZScoreStep / NormalizeStep
PORTED_ROWWISE_OPS: set = {"ratio", "product", "difference", "log", "power", "weighted_sum"}
PORTED_FEATURE_OPS: set = PORTED_PARAMETRIC_OPS | PORTED_ROWWISE_OPS


def check_pipeline_scope(ctx: dict) -> dict:
    """ctx'te BEYAN EDİLEN olgulara göre kapsam kararı. Beyan edilmeyen olgu 'bilinmiyor'dur
    (ihlal değil) ama artifact'i **deploy edilemez** yapar — sessiz "destekleniyor" varsayımı yok."""
    ctx = ctx or {}
    violations, declared, undeclared = [], {}, []
    for fact, is_violation, reason in _SCOPE_RULES:
        if fact not in ctx:
            undeclared.append(fact)
            continue
        declared[fact] = ctx[fact]
        if is_violation(ctx[fact]):
            violations.append({"fact": fact, "value": ctx[fact], "reason": reason})
    return {
        "declared": declared,
        "undeclared": undeclared,
        "violations": violations,
        # Deploy ancak TÜM olgular beyan edilmiş VE hiçbiri ihlal değilse mümkün.
        "deployable": not violations and not undeclared,
        "ported_feature_ops": sorted(PORTED_FEATURE_OPS),
    }


# ── PIPELINE tanımı — TEK doğruluk kaynağı (legacy_ref drift testi doğrular) ──


@dataclass
class StepSpec:
    """Tek TransformStep/TrainOnlyStep. ``config_from``: ctx → construction config (rowwise/param).

    ``pre_split`` (T3a-1): bu adım ``build_features``'ın split'inin ÖNCESİNDE (tüm-df, incremental
    çerçeve) fit edilir. Sonucu — ``transform_pipeline`` bugün FULLY-BUILT X_test'e karşı doğrulama
    yaptığından — simetrik guard'dan MUAF tutulur (bkz. ``_record`` ``columns_out_skipped``). **DAR
    KAPI:** yalnız burada ``True`` beyan edilen adım muaftır; post-split adım ASLA muaf olamaz
    (``test_only_declared_pre_split_steps_skip_guard`` kilitler)."""
    step_cls: type
    legacy_ref: str
    config_from: Callable[[dict], dict] = field(default=lambda ctx: {})
    pre_split: bool = False


@dataclass
class RouteSpec:
    """Fit-öncesi route kararı + executor'ları. ``args_from``: ctx → ``decide_fn`` kwargs."""
    decide_fn: Callable
    legacy_ref: str
    executors: list                    # [(route_key, executor_cls), ...] — sıralı
    args_from: Callable[[dict], dict] = field(default=lambda ctx: {})


PIPELINE: list = [
    # SIRA KİLİDİ: eksiklik bayrağı fillna'dan ÖNCE (kaynak: _add_missing_indicators@2130 <
    # _clean_data@2133). Sonra çalışsaydı null kalmaz, bayrak hep 0 olurdu (sinyal sessizce silinir).
    StepSpec(MissingIndicatorStep, "_add_missing_indicators",
             config_from=lambda c: {"columns": c.get("missing_indicator_cols", [])},
             pre_split=True),                            # T3a-1: split ÖNCESİ (impute'tan da önce)
    StepSpec(ImputeStep, "_clean_data", pre_split=True), # T3a-1: sahiplik _clean_data'dan buraya
    # DATE FEATURES PORTU (§2.19): timestamp → _year/_month/_dayofweek. KONUM: impute SONRASI /
    # drop_columns ÖNCESİ (ham ts kolonu hâlâ çerçevede; türev sonrası drop düşürür). Rowwise/durumsuz;
    # config kaynağı meta.timestamp (feature_builder ``date_source_cols``). Eski inline blok bu adıma
    # devredildi → ``date_features`` KAPSAM KURALINDAN ÇIKARILDI (artifact deploy edilebilir).
    StepSpec(DateFeaturesStep, "_apply_date_features",
             config_from=lambda c: {"columns": c.get("date_source_cols", [])},
             pre_split=True),                            # T-date: sahiplik build_features inline'dan buraya
    # KALDIRMA adımı (guard'ın kör noktası — eksik kaldırma "eksik üretim" göstermez, downstream
    # SEÇİM SAPMASI üretir; ayrıca hedef çerçevede kalırsa groupby/count HEDEFTEN feature türetir).
    # Kaynak: build_features:2233-2255; legacy_ref o blokta fiilen çağrılan fonksiyon (@2251).
    StepSpec(DropColumnsStep, "_find_related_target_cols",
             config_from=lambda c: {"columns": c.get("drop_columns", [])},
             pre_split=True),                            # T4a: sahiplik build_features'tan buraya
    # #5 FIX: plan-güdümlü zscore/normalize — train-fit (eski full-df/leaky yerine). Kaynak sırası
    # _apply_plan_derived_features@2260 (_clean_data@2133 ↔ _build_ratio_features@2289 arası). specs boşsa no-op.
    StepSpec(ZScoreStep, "_apply_plan_derived_features",
             config_from=lambda c: {"specs": c.get("zscore_specs", [])},
             pre_split=True),                            # T4b: eski op emekli, sahiplik burada
    StepSpec(NormalizeStep, "_apply_plan_derived_features",
             config_from=lambda c: {"specs": c.get("normalize_specs", [])},
             pre_split=True),                            # T4b
    # 6-OP PORTU: plan'ın ROWWISE türev op'ları (ratio/product/difference/log/power/weighted_sum).
    # Aynı legacy_ref (_apply_plan_derived_features) → kaynak-drift testi sadık kalır.
    StepSpec(PlanDerivedStep, "_apply_plan_derived_features",
             config_from=lambda c: {"specs": c.get("derived_specs", [])},
             pre_split=True),
    StepSpec(RatioConstructStep, "_build_ratio_features",
             config_from=lambda c: {"pairs": c.get("ratio_pairs", [])}),
    # is_zero: kaynakta _build_zero_missing_flags@2291 — _build_ratio_features@2289 SONRASI
    # (oran motoru sıfır-payda kolonlarını bildirir → force_zero invariant'ı), split ÖNCESİ.
    StepSpec(ZeroFlagStep, "_build_zero_missing_flags",
             config_from=lambda c: {"columns": c.get("zero_flag_cols", [])}),
    StepSpec(RatioFinalizeStep, "_finalize_ratio_features_tt",
             config_from=lambda c: {"columns": [n for n, _, _ in c.get("ratio_pairs", [])]}),
    StepSpec(CountEncodeStep, "_count_encode_tt",
             config_from=lambda c: {"exclude": c.get("encode_exclude", [])}),
    StepSpec(GroupbyDeviationStep, "_groupby_deviation_tt",   # encode ÖNCESİ (ham cat = grup anahtarı)
             config_from=lambda c: {"exclude": c.get("encode_exclude", [])}),
    RouteSpec(decide_encode_route, "_encode_categorical_tt",
              executors=[("onehot", OneHotStep), ("target", TargetEncodeStep)],
              args_from=lambda c: {"exclude": c.get("encode_exclude", [])}),
    StepSpec(InfImputeStep, "_scale_features_tt"),
    RouteSpec(decide_scaler_route, "_scale_features_tt",
              executors=[("robust", RobustScaleStep), ("standard", StandardScaleStep)],
              args_from=lambda c: {"outlier_cols": c.get("outlier_cols", [])}),
    StepSpec(SelectDropStep, "_select_features_tt",          # TERMINUS (#3) — EN SON, sonrasında fitted_stat YOK
             config_from=lambda c: {"intent": c.get("intent", ""), "meta": c.get("meta"),
                                    "domain_leakage_report": c.get("domain_leakage_report"),
                                    "protected_high_card": c.get("protected_high_card"),
                                    # protected beyan EDİLMEMİŞSE adım bunu oran adlarından
                                    # türetir (T2 — ts_protected_cols denkliği, bkz. SelectDropStep.fit)
                                    "ratio_names": [n for n, _, _ in c.get("ratio_pairs", [])]}),
]


STEP_REGISTRY: dict = {
    cls.id: cls for cls in (
        MissingIndicatorStep, ImputeStep, DateFeaturesStep, DropColumnsStep, ZScoreStep,
        NormalizeStep, PlanDerivedStep, RatioConstructStep,
        ZeroFlagStep, RatioFinalizeStep, CountEncodeStep, GroupbyDeviationStep, OneHotStep,
        TargetEncodeStep, InfImputeStep, RobustScaleStep, StandardScaleStep, SelectDropStep,
    )
}


def _notify(on_step, step_id, X: pd.DataFrame) -> None:
    """Adım sınırı gözlemcisini çağır — **YAPISAL olarak READ-ONLY**.

    Gözlemciye çerçevenin KOPYASI geçilir. Assert değil, yapı: gözlemci ne yaparsa
    yapsın pipeline'ın çerçevesine ULAŞAMAZ. Aksi halde pipeline'ın DIŞINDA gizli bir
    dönüşüm yolu doğardı ("observer içinde bir kolon düzelt") — A refactor'unun
    imkânsız kılmak için var olduğu şeyin ta kendisi: eğitimde çalışan ama serving
    replay'inde OLMAYAN bir adım. Kilit: ``test_on_step_observer_cannot_mutate_frame``.

    Maliyeti yalnız gözlemci VARKEN ödenir (``on_step=None`` → kopya yok).
    """
    if on_step is not None:
        on_step(step_id, X.copy())


def _frozen_choice_of(step, params) -> "dict | None":
    """Adım donmuş bir YAPISAL seçim beyan ediyorsa manifest ``extra``'sına çevir (opt-in)."""
    fn = getattr(step, "frozen_selection", None)
    if not callable(fn):
        return None
    fc = fn(params)
    return {"frozen_choice": fc} if fc else None


def _param_footprint_of(step, params) -> "dict | None":
    """Adım öğrendiği param'ın BOYUTUNU beyan ediyorsa manifest ``extra``'sına yaz (opt-in).

    NEDEN (T3b-gate): ``ImputeStep``'in ``isnull().any()`` kapısı kalkınca **çıktı**
    byte-nötr kaldı ama **öğrenilen param sayısı büyüdü** (null'suz kolonlar da dolgu
    kazandı). Kullanılmayan bir dolgu da olsa bu bir davranış değişikliğidir — serving'de
    o kolona null gelirse artık medyan uygulanacak, eskiden terminal ``fillna(0)``
    uygulanırdı. ``param_store`` artifact'in KENDİSİ olduğuna göre boyutunun değişmesi
    sessiz kalmamalı: manifest kalemi ``param_footprint`` taşır, ölçüm/denetim onu okur.

    ``frozen_choice`` ile aynı disiplin: kaynak hâlâ ``param_store``, bu bir GÖRÜNÜRLÜK
    katmanı (kopya değil türev)."""
    fn = getattr(step, "param_footprint", None)
    if not callable(fn):
        return None
    fp = fn(params)
    return {"param_footprint": fp} if fp else None


def _step_extras(step, params) -> "dict | None":
    """Adımın opt-in manifest beyanlarını (donmuş seçim + param ayak izi) TEK sözlükte topla."""
    extras: dict = {}
    for part in (_frozen_choice_of(step, params), _param_footprint_of(step, params)):
        if part:
            extras.update(part)
    return extras or None


def _record(manifest: list, order: int, step, *, type_: str, legacy_ref: str,
            params_ref: "str | None", config: "dict | None", extra: "dict | None" = None,
            delta: "dict | None" = None, pre_split: bool = False) -> None:
    """Manifest kalemi. ``frozen_choice`` (varsa) TEK şekildedir: ``{kind, selection, ...}`` —
    route kararı da adım-içi seçim de (groupby çiftleri) aynı alandan okunur, böylece Katman 2
    "seçim nerede" sorusunu adım tipine göre DEĞİŞTİRMEDEN cevaplar.

    **``delta`` (T5-(2), MUTLAK ``columns_out``'un YERİNE):** ``{"added": [...], "removed": [...]}``
    = adımın kolon sözleşmesi. Eskiden adım-sonrası MUTLAK kolon seti kaydediliyordu; o, eğitimdeki
    çerçevenin TAMAMINA bağlı GLOBAL bir invariant'tı ve iki bedeli vardı: (1) pre-split adımların
    incremental çerçevesi fully-built X_test'le uyuşmadığı için **5 adım guard'dan MUAF** tutulmak
    zorunda kaldı; (2) girdideki tek bir meşru fark (beyan edilmiş fazla kolon) SONRAKİ HER adımda
    uyuşmazlık üretiyordu. Delta **YEREL**: adımın kendi sözleşmesi, çerçevenin geri kalanından
    bağımsız → incremental ve fully-built çerçevede AYNI çalışır → **muafiyet kavramı ortadan kalkar**
    (``columns_out_skipped`` kaydı da kalktı; gerekçesi kalmayan kayıt yanıltıcı artık olurdu).

    Delta'nın GÖREMEDİĞİ tek şey — "hiçbir adımın dokunmadığı fazladan kolon" — kontrolü ait olduğu
    yere, **giriş noktasına** taşındı (``check_raw_input`` + ``Artifact.raw_input_signature``)."""
    entry = {
        "id": getattr(step, "id", None) if not isinstance(step, str) else step,
        "order": order,                          # FİİLİ yürütme indeksi (elle atama YOK)
        # Adımın KOLON SÖZLEŞMESİ (yerel invariant) — serving replay'inde birebir doğrulanır.
        "delta": delta or {"added": [], "removed": []},
        "type": type_,
        "param_kind": getattr(step, "param_kind", None) if not isinstance(step, str) else "frozen_choice",
        "legacy_ref": legacy_ref,
        "params_ref": params_ref,
        "config": config,
    }
    if pre_split:
        entry["pre_split"] = True               # yalnız BİLGİ (fit sırası); guard anlamı YOK
    if extra:
        entry.update(extra)
    manifest.append(entry)


def _delta_of(before, after) -> dict:
    """Adımın kolon deltası — sıra DEĞİL küme (sıra sonda ``final_features`` reindex'iyle sabitlenir)."""
    b, a = set(before), set(after)
    return {"added": sorted(a - b), "removed": sorted(b - a)}


@dataclass
class ArtifactBuilder:
    """Adımları TEK ``order`` uzayında fit+kaydet — split boyunca YAŞAYAN state.

    ``fit_pipeline`` (aşağıda) bunun ince sarmalayıcısıdır. Ayrı olma sebebi: T3+ pre-split
    adımları (impute vb.) ``build_features``'ın KENDİ split'inin İKİ yanında, TEK sayaçla fit
    edilebilsin. **İki ``fit_pipeline`` çağrısı (iki artifact, iki order uzayı) REDDİNİ İHLAL
    ETMEZ (imza düzeyi):** TEK builder instance ``self.order``'ı sahiplenir → pre-split ve
    post-split fazları AYNI manifest'e monoton yazar → tek ``finalize()`` → **tek Artifact**.
    Red, İKİ instance'a karşıydı; bu, red'in işaret ettiği "paylaşılan builder" alternatifi.
    """
    manifest: list = field(default_factory=list)
    param_store: dict = field(default_factory=dict)
    raw_input_signature: dict = field(default_factory=dict)   # ilk fit_step'te yakalanır
    order: int = 0
    scope: dict = field(default_factory=dict)

    def declare_raw_input(self, raw: pd.DataFrame, datetime_cols: "list | None" = None) -> None:
        """SERVING SÖZLEŞMESİNİ açıkça beyan et — beyan, yedek yolun türettiğini EZER.

        Neden beyan, neden çıkarım DEĞİL: ``build_features``'ın order-0 girdisi, ikiz borç
        (§2.25 #2) yüzünden inline üretilmiş kolonları (``*_isna`` vb.) ZATEN taşır. O çerçeveden
        türetilen imza, serving'de asla gelmeyecek kolonları "zorunlu" ilan ederdi — ve ilk
        duman testinde tam olarak bu oldu (`total_bedrooms_isna` EKSİK hatası). Serving girdisini
        yalnız çağıran bilir; orkestratör tahmin etmez.

        ``datetime_cols``: pipeline'ın timestamp tiplediği kolonlar (meta.timestamp) — imzada
        ``kind="datetime"`` işaretlenir (date portu sonrası ham ts kolonu ``required``)."""
        self.raw_input_signature = build_raw_input_signature(raw, datetime_cols=datetime_cols)

    def check_scope(self, ctx: "dict | None") -> None:
        """KAPSAM KAPISI — eski ``fit_pipeline`` 381-386'sıyla BİREBİR (ilk adımdan ÖNCE çağrılır)."""
        self.scope = check_pipeline_scope(ctx or {})
        if self.scope["violations"]:
            raise UnsupportedPipelineScope(self.scope["violations"])
        if not self.scope["deployable"]:
            logger.info(f"[orchestrator] artifact DEPLOY EDİLEMEZ olarak kurulacak — "
                        f"beyan edilmemiş kapsam olguları: {self.scope['undeclared']}")

    def fit_step(self, spec, X: pd.DataFrame, y=None, ctx: "dict | None" = None,
                 *, fit_frame: "pd.DataFrame | None" = None, on_step=None) -> pd.DataFrame:
        """Tek spec'i fit+transform, ``self.order`` ile manifest'e kaydet.

        Dispatch eski ``fit_pipeline`` gövdesinin BİREBİR aynısı. **TEK ek:** parametric dalda
        ``fit_frame`` ayrıştırılabilir — ``None`` → ``fit_transform`` (mevcut davranış, byte-nötr);
        verilirse ``fit(fit_frame)+transform(X)`` (pre-split adımın train-only fit'i, T3b). Bu ikinci
        yol yalnız ``DefaultFitTransform`` adımları için kullanılır (cross-fitting yapan TargetEncode
        RouteSpec executor'ıdır, orada ``fit_transform`` korunur → E invariant'ı sağlam).
        """
        ctx = ctx or {}
        # GİRİŞ İMZASI — yedek yol: açıkça beyan edilmediyse pipeline'ın gördüğü İLK çerçeveden
        # türetilir (``fit_pipeline`` gibi tek-girişli kullanımlar için doğru). **Üretim yolunda
        # `declare_raw_input` ile AÇIKÇA beyan edilir** — çünkü ``build_features``'ın order-0
        # girdisi inline üretilmiş kolonları (ikiz borç, §2.25 #2) ZATEN taşır ve o çerçeve
        # serving sözleşmesi DEĞİLDİR. Tahmin etmiyoruz, beyan alıyoruz.
        if not self.raw_input_signature:
            self.raw_input_signature = build_raw_input_signature(X)
        _before = list(X.columns)
        if isinstance(spec, RouteSpec):
            route = spec.decide_fn(X, **spec.args_from(ctx))        # fit-öncesi karar
            _record(self.manifest, self.order, f"route:{spec.decide_fn.__name__}", type_="route",
                    legacy_ref=spec.legacy_ref, params_ref=None, config=None,
                    delta={"added": [], "removed": []},             # route çerçeveye DOKUNMAZ
                    extra={"frozen_choice": {                       # TEK şekil (bkz. _record)
                        "kind": "route", "selection": route,
                        "executors": [k for k, _ in spec.executors]}})
            self.order += 1
            for key, exec_cls in spec.executors:                    # executor'lar sıralı
                _b = list(X.columns)
                step = exec_cls(config={"columns": route[key]})
                X, params = step.fit_transform(X, y)
                self.param_store[step.id] = params
                _record(self.manifest, self.order, step, type_=step.type, legacy_ref=spec.legacy_ref,
                        params_ref=step.id, config=None, extra=_step_extras(step, params),
                        delta=_delta_of(_b, X.columns))
                _notify(on_step, step.id, X)
                self.order += 1
            return X

        _pre = getattr(spec, "pre_split", False)                     # yalnız BİLGİ (fit sırası)
        cfg = spec.config_from(ctx)
        step = spec.step_cls(config=cfg) if cfg else spec.step_cls()
        stype = getattr(step, "type", "parametric")
        if stype == "skip":                                          # TrainOnlyStep
            X = step.apply_train(X)
            _record(self.manifest, self.order, step, type_="skip", legacy_ref=spec.legacy_ref,
                    params_ref=None, config=None, delta=_delta_of(_before, X.columns), pre_split=_pre)
        elif stype == "rowwise":                                     # config manifest'te taşınır
            X, _ = step.fit_transform(X, y)                          # fit None; transform config'ten
            _record(self.manifest, self.order, step, type_="rowwise", legacy_ref=spec.legacy_ref,
                    params_ref=None, config=cfg, extra=_step_extras(step, None),
                    delta=_delta_of(_before, X.columns), pre_split=_pre)
        else:                                                        # parametric
            if fit_frame is None:
                X, params = step.fit_transform(X, y)                 # mevcut yol (byte-nötr)
            else:
                params = step.fit(fit_frame, y)                      # frame-ayrık (T3b tek satırı)
                X = step.transform(X, params)
            self.param_store[step.id] = params
            _record(self.manifest, self.order, step, type_="parametric", legacy_ref=spec.legacy_ref,
                    params_ref=step.id, config=None, extra=_step_extras(step, params),
                    delta=_delta_of(_before, X.columns), pre_split=_pre)
        # YAPISAL ÖN-KOŞUL: drop adımı işini yaptı mı + hedef/id çerçeveden çıktı mı?
        if getattr(step, "id", None) == "drop_columns":
            ihlaller = check_structural_preconditions(X, ctx)
            if ihlaller:
                raise StructuralPreconditionViolation(ihlaller)
        _notify(on_step, getattr(step, "id", None), X)
        self.order += 1
        return X

    def finalize(self, final_features) -> Artifact:
        logger.info(f"[orchestrator] finalize: {self.order} manifest kalemi, {len(self.param_store)} param, "
                    f"imza {len((self.raw_input_signature or {}).get('columns') or {})} kolon, "
                    f"deployable={self.scope.get('deployable')}")
        return Artifact(manifest=self.manifest, param_store=self.param_store,
                        raw_input_signature=self.raw_input_signature,
                        final_features=list(final_features), supported_scope=self.scope)


def fit_pipeline(df: pd.DataFrame, y: "pd.Series | None" = None,
                 ctx: "dict | None" = None, pipeline: list = PIPELINE,
                 on_step=None) -> "tuple[pd.DataFrame, Artifact]":
    """(İnce sarmalayıcı) ``ArtifactBuilder``'ı PIPELINE üstünde koştur — davranış BİREBİR korunur.

    ``on_step(step_id, X)``: opsiyonel adım-sınırı gözlemcisi (READ-ONLY, bkz. ``_notify``).
    T2'de ``build_features``'ın zincir-ORTASI tüketicileri (fairness segment taşıyıcısı ve
    Veri İlişkileri — ikisi de ``ratio_finalize`` sonrası, encode ÖNCESİ çerçeveyi ister)
    bununla beslenir. **Tek girişten tek order uzayı** (builder tek instance).
    """
    ctx = ctx or {}
    b = ArtifactBuilder()
    b.check_scope(ctx)                         # KAPSAM KAPISI — adım koşmadan ÖNCE
    X = df.copy()
    for spec in pipeline:
        X = b.fit_step(spec, X, y, ctx, on_step=on_step)
    return X, b.finalize(X.columns)


def transform_pipeline(raw: pd.DataFrame, artifact: Artifact, on_step=None,
                       verify_input: bool = True, verify_delta: bool = True) -> pd.DataFrame:
    """Manifest'i order'a göre replay — YALNIZ step.transform (E: fit_transform ulaşılamaz).
    route/skip kalemleri atlanır; sonda ÜRETİM-KANITI guard'ı + final_features'a reindex.

    ÜRETİM-KANITI (fail-closed): reindex'ten ÖNCE, ``final_features``'ın her kolonu için
    onu fiilen üreten bir adım çalıştı mı diye bakılır. Eksikler ikiye ayrılır:
      * politikası tanımlı (``zero_fill_policy``) → 0 doldurmak DOĞRU cevap (one-hot unseen
        dummy'leri gibi: all_zero ≡ referans kategori, §2.8 D) → izinli;
      * tanımsız → ``MissingFeatureProduction``, skorlama HİÇ yapılmaz.
    Böylece terminal reindex "her boşluğu sessizce 0'la" olmaktan çıkıp yalnız beyan
    edilmiş boşlukları dolduran bir mekanizmaya iner.
    """
    X = raw.copy()
    policy_filled: set = set()
    ever_produced: set = set(X.columns)          # replay boyunca BİR KEZ bile görülen kolonlar
    tolerated = list(artifact.tolerated_extra_columns or [])
    # ── GİRİŞ KAPISI: imza denetimi (delta-guard'ın kör noktası burada kapanır) ──
    if verify_input:
        check_raw_input(X, artifact.raw_input_signature, tolerated)
    else:
        # ARA DURUM (T5-(2)): çağıran, girdinin HAM OLMADIĞINI biliyor. Kontrol sessizce
        # zayıflatılmıyor — çağrı yerinde AÇIKÇA devre dışı bırakılıyor ve gerekçesi orada
        # yazılı. Ham-satır yolu ikiz borç (§2.25 #2) kapanınca bu kapı hep açık olacak.
        logger.info("[orchestrator] giriş imzası denetimi ATLANDI (verify_input=False, "
                    "girdi ham değil — bkz. çağrı yeri gerekçesi)")
    for entry in sorted(artifact.manifest, key=lambda e: e["order"]):
        etype = entry["type"]
        if etype in ("route", "skip"):
            continue                                                 # B: route compile-time; C: skip serving-dışı
        step_cls = STEP_REGISTRY[entry["id"]]
        _before = list(X.columns)
        if etype == "rowwise":
            step = step_cls(config=entry.get("config") or {})
            X = step.transform(X, None)                              # rowwise: params yok, config taşır
        else:                                                        # parametric
            step = step_cls()
            params = artifact.param_store[entry["params_ref"]]
            X = step.transform(X, params)
            # Adım, "bu kolonlar eksikse 0 doldurmak TANIMLI politikamdır" diyebilir (opt-in).
            policy = getattr(step, "zero_fill_policy", None)
            if callable(policy):
                policy_filled |= set(policy(params))
        ever_produced |= set(X.columns)

        # ── DELTA-GUARD: adımın KOLON SÖZLEŞMESİ eğitimdekiyle aynı mı? ──
        # Mutlak kolon setinin (eski `columns_out`) yerine YEREL invariant: "bu adım şu kolonları
        # ekledi/çıkardı". Çerçevenin geri kalanından bağımsız olduğu için incremental (pre-split)
        # ve fully-built çerçevede AYNI çalışır → guard-muafiyeti gerekmez, sapma adımda lokalize.
        beklenen = entry.get("delta") if verify_delta else None
        if beklenen is not None:
            fiili = _delta_of(_before, X.columns)
            eksik_ekleme = [c for c in beklenen.get("added", [])
                            if c not in fiili["added"] and c not in policy_filled]
            fazla_ekleme = [c for c in fiili["added"] if c not in beklenen.get("added", [])]
            # Kolon _before'da VARDI ama silinmediyse GERÇEK ihlal (ör. hedef sızabilir). Kolon girdide
            # ZATEN YOKSA (serving satırı onu hiç göndermedi — ör. id imzadan çıkarıldı), "silme"nin invariant'ı
            # ("bu kolon çıktıda yok") trivial sağlanır → ihlal DEĞİL. Bu gevşetme yalnız yokluk için; mevcut
            # ama silinmemiş kolon HÂLÂ yakalanır (sızıntı koruması bozulmaz).
            eksik_silme = [c for c in beklenen.get("removed", [])
                           if c in _before and c not in fiili["removed"]]
            fazla_silme = [c for c in fiili["removed"] if c not in beklenen.get("removed", [])]
            if eksik_ekleme or fazla_ekleme or eksik_silme or fazla_silme:
                raise StepDeltaDivergence(entry["id"], beklenen, fiili,
                                          eksik_ekleme, fazla_ekleme, eksik_silme, fazla_silme)
        _notify(on_step, entry["id"], X)          # READ-ONLY (kopya) — bkz. _notify

    produced = set(X.columns)
    missing = [c for c in artifact.final_features if c not in produced and c not in policy_filled]
    if missing:
        raise MissingFeatureProduction(
            never_produced=[c for c in missing if c not in ever_produced],
            produced_then_dropped=[c for c in missing if c in ever_produced],
            policy_filled=[c for c in artifact.final_features
                           if c not in produced and c in policy_filled],
        )
    filled = [c for c in artifact.final_features if c not in produced]
    if filled:
        logger.info(f"[orchestrator] reindex politika-dolgusu: {len(filled)} kolon (izinli)")
    return X.reindex(columns=artifact.final_features, fill_value=0)
