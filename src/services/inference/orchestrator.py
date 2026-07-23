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
    CountEncodeStep, DropColumnsStep, GroupbyDeviationStep, ImputeStep, InfImputeStep,
    MissingIndicatorStep,
    NormalizeStep, OneHotStep, RatioConstructStep, RatioFinalizeStep, RobustScaleStep,
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
    ("date_features",  lambda v: bool(v),
     "timestamp türevleri (_year/_month/_dayofweek) portlanmadı"),
    ("column_renames", lambda v: bool(v),
     "_normalize_column_names yeniden-adlandırması portlanmadı (ham ad → eğitim adı eşlemesi yok)"),
    ("unported_ops",   lambda v: bool(v),
     "plan-türevli op'lardan bazıları portlanmadı (portlu: zscore, normalize)"),
    # YAPISAL ÖN-KOŞUL (adım değil, YAPI kaynaklı garanti — ölçüldü: dedup'sız çerçevede
    # impute medyanı 3.0 yerine 2.0 çıkıyor → ondan sonraki HER fitted_stat kayar).
    ("deduplicated",   lambda v: not v,
     "girdi çerçevesi dedup EDİLMEMİŞ — _clean_data'nın drop_duplicates'i portlanmadı; "
     "satır üyeliği farklı → tüm fitted_stat'lar başka dağılımdan öğrenilir"),
]


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
PORTED_FEATURE_OPS: set = {"zscore", "normalize"}


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
    """Tek TransformStep/TrainOnlyStep. ``config_from``: ctx → construction config (rowwise/param)."""
    step_cls: type
    legacy_ref: str
    config_from: Callable[[dict], dict] = field(default=lambda ctx: {})


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
             config_from=lambda c: {"columns": c.get("missing_indicator_cols", [])}),
    StepSpec(ImputeStep, "_clean_data"),
    # KALDIRMA adımı (guard'ın kör noktası — eksik kaldırma "eksik üretim" göstermez, downstream
    # SEÇİM SAPMASI üretir; ayrıca hedef çerçevede kalırsa groupby/count HEDEFTEN feature türetir).
    # Kaynak: build_features:2233-2255; legacy_ref o blokta fiilen çağrılan fonksiyon (@2251).
    StepSpec(DropColumnsStep, "_find_related_target_cols",
             config_from=lambda c: {"columns": c.get("drop_columns", [])}),
    # #5 FIX: plan-güdümlü zscore/normalize — train-fit (eski full-df/leaky yerine). Kaynak sırası
    # _apply_plan_derived_features@2260 (_clean_data@2133 ↔ _build_ratio_features@2289 arası). specs boşsa no-op.
    StepSpec(ZScoreStep, "_apply_plan_derived_features",
             config_from=lambda c: {"specs": c.get("zscore_specs", [])}),
    StepSpec(NormalizeStep, "_apply_plan_derived_features",
             config_from=lambda c: {"specs": c.get("normalize_specs", [])}),
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
                                    "protected_high_card": c.get("protected_high_card")}),
]


STEP_REGISTRY: dict = {
    cls.id: cls for cls in (
        MissingIndicatorStep, ImputeStep, DropColumnsStep, ZScoreStep, NormalizeStep, RatioConstructStep,
        ZeroFlagStep, RatioFinalizeStep, CountEncodeStep, GroupbyDeviationStep, OneHotStep,
        TargetEncodeStep, InfImputeStep, RobustScaleStep, StandardScaleStep, SelectDropStep,
    )
}


def _frozen_choice_of(step, params) -> "dict | None":
    """Adım donmuş bir YAPISAL seçim beyan ediyorsa manifest ``extra``'sına çevir (opt-in)."""
    fn = getattr(step, "frozen_selection", None)
    if not callable(fn):
        return None
    fc = fn(params)
    return {"frozen_choice": fc} if fc else None


def _record(manifest: list, order: int, step, *, type_: str, legacy_ref: str,
            params_ref: "str | None", config: "dict | None", extra: "dict | None" = None,
            columns_out: "list | None" = None) -> None:
    """Manifest kalemi. ``frozen_choice`` (varsa) TEK şekildedir: ``{kind, selection, ...}`` —
    route kararı da adım-içi seçim de (groupby çiftleri) aynı alandan okunur, böylece Katman 2
    "seçim nerede" sorusunu adım tipine göre DEĞİŞTİRMEDEN cevaplar."""
    entry = {
        "id": getattr(step, "id", None) if not isinstance(step, str) else step,
        "order": order,                          # FİİLİ yürütme indeksi (elle atama YOK)
        # Bu adımdan SONRAKİ eğitim kolon-seti — simetrik guard'ın referansı (iki yön).
        "columns_out": None,
        "type": type_,
        "param_kind": getattr(step, "param_kind", None) if not isinstance(step, str) else "frozen_choice",
        "legacy_ref": legacy_ref,
        "params_ref": params_ref,
        "config": config,
    }
    if columns_out is not None:
        entry["columns_out"] = sorted(columns_out)
    if extra:
        entry.update(extra)
    manifest.append(entry)


def fit_pipeline(df: pd.DataFrame, y: "pd.Series | None" = None,
                 ctx: "dict | None" = None, pipeline: list = PIPELINE) -> "tuple[pd.DataFrame, Artifact]":
    """Adımları PIPELINE sırasıyla fit+transform; manifest order'ı fiili döngüden türet."""
    ctx = ctx or {}
    scope = check_pipeline_scope(ctx)          # KAPSAM KAPISI — adım koşmadan ÖNCE
    if scope["violations"]:
        raise UnsupportedPipelineScope(scope["violations"])
    if not scope["deployable"]:
        logger.info(f"[orchestrator] artifact DEPLOY EDİLEMEZ olarak kurulacak — "
                    f"beyan edilmemiş kapsam olguları: {scope['undeclared']}")
    X = df.copy()
    manifest: list = []
    param_store: dict = {}
    order = 0

    for spec in pipeline:
        if isinstance(spec, RouteSpec):
            route = spec.decide_fn(X, **spec.args_from(ctx))        # fit-öncesi karar
            _record(manifest, order, f"route:{spec.decide_fn.__name__}", type_="route",
                    legacy_ref=spec.legacy_ref, params_ref=None, config=None, columns_out=X.columns,
                    extra={"frozen_choice": {                       # TEK şekil (bkz. _record)
                        "kind": "route", "selection": route,
                        "executors": [k for k, _ in spec.executors]}})
            order += 1
            for key, exec_cls in spec.executors:                    # executor'lar sıralı
                step = exec_cls(config={"columns": route[key]})
                X, params = step.fit_transform(X, y)
                param_store[step.id] = params
                _record(manifest, order, step, type_=step.type, legacy_ref=spec.legacy_ref,
                        params_ref=step.id, config=None, extra=_frozen_choice_of(step, params),
                        columns_out=X.columns)
                order += 1
            continue

        cfg = spec.config_from(ctx)
        step = spec.step_cls(config=cfg) if cfg else spec.step_cls()
        stype = getattr(step, "type", "parametric")
        if stype == "skip":                                          # TrainOnlyStep
            X = step.apply_train(X)
            _record(manifest, order, step, type_="skip", legacy_ref=spec.legacy_ref,
                    params_ref=None, config=None, columns_out=X.columns)
        elif stype == "rowwise":                                     # config manifest'te taşınır
            X, _ = step.fit_transform(X, y)                          # fit None; transform config'ten
            _record(manifest, order, step, type_="rowwise", legacy_ref=spec.legacy_ref,
                    params_ref=None, config=cfg, extra=_frozen_choice_of(step, None),
                    columns_out=X.columns)
        else:                                                        # parametric
            X, params = step.fit_transform(X, y)
            param_store[step.id] = params
            _record(manifest, order, step, type_="parametric", legacy_ref=spec.legacy_ref,
                    params_ref=step.id, config=None, extra=_frozen_choice_of(step, params),
                    columns_out=X.columns)
        # YAPISAL ÖN-KOŞUL: drop adımı işini yaptı mı + hedef/id çerçeveden çıktı mı?
        if getattr(step, "id", None) == "drop_columns":
            ihlaller = check_structural_preconditions(X, ctx)
            if ihlaller:
                raise StructuralPreconditionViolation(ihlaller)
        order += 1

    logger.info(f"[orchestrator] fit_pipeline: {order} manifest kalemi, {len(param_store)} param, "
                f"deployable={scope['deployable']}")
    return X, Artifact(manifest=manifest, param_store=param_store,
                       final_features=list(X.columns), supported_scope=scope)


def transform_pipeline(raw: pd.DataFrame, artifact: Artifact) -> pd.DataFrame:
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
    tolerated = set(artifact.tolerated_extra_columns or [])
    for entry in sorted(artifact.manifest, key=lambda e: e["order"]):
        etype = entry["type"]
        if etype in ("route", "skip"):
            continue                                                 # B: route compile-time; C: skip serving-dışı
        step_cls = STEP_REGISTRY[entry["id"]]
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

        # ── SİMETRİK GUARD: adım sınırında eğitim çerçevesiyle İKİ YÖNDE karşılaştır ──
        # Asimetri (yalnız "eksik üretim" sorma) 4. kökü saklamıştı; fazla mevcudiyet de
        # sessizce downstream seçim sapması üretebiliyor. İkisi de burada görünür olur.
        egitim = entry.get("columns_out")
        if egitim is not None:
            simdi = set(X.columns)
            eksik = [c for c in egitim if c not in simdi and c not in policy_filled]
            fazla = [c for c in simdi if c not in set(egitim) and c not in tolerated]
            if eksik or fazla:
                raise FrameShapeDivergence(entry["id"], eksik, fazla)

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
