from __future__ import annotations

"""Transform adımlarının fit/transform sözleşmesi + Artifact taşıyıcısı (imza katmanı).

Üç adım-tipi (manifest ``type`` alanı):
  * ``parametric`` — inference train-türevi state yükler. İki alt-tip (``param_kind``):
      - ``fitted_stat``   : dağılımdan öğrenilen sayı (median, scaler, winsor, freq-map,
                            zscore mean/std). LEAKAGE-HASSAS → train-only olmak ZORUNDA.
      - ``frozen_choice`` : train-türevi yapısal karar (rename-map, dummy-vocab, drop-liste,
                            ratio-çift listesi). Sayı değil karar; yine train'den gelir.
  * ``rowwise``    — saf per-satır aritmetik; train-türevi state YOK. Yalnız statik
                     ``config`` taşır (girdi kolonları, op). ``fit`` None döner.
  * ``skip``       — train-only (drop_duplicates). Inference'ta HİÇ çağrılmaz; ayrı
                     ``TrainOnlyStep`` protokolü (``transform`` yok → skew riski yok).

Skew-imkânsızlığının imza-düzeyi kanıtı: ``fit`` yalnız X_train alır, ``transform``
yalnız ``params`` alır. İkisinin arasında "tüm-df'den istatistik hesaplayıp uygulayan"
bir imza YOKTUR → eski sızıntılı (split-öncesi tüm-df) çağrı temsil edilemez.

──────────────────────────────────────────────────────────────────────────────
SELF-CONTAINED invariant'ının KAPSAMI (daraltıldı — bilinçli, kayıtlı)
──────────────────────────────────────────────────────────────────────────────
Başlangıçtaki kural "her Step SELF-CONTAINED'dir, ``feature_builder`` import ETMEZ"di
(gerekçe: inference runtime'ı yalın tutmak). Bu kural artık **``transform`` yolu için**
geçerlidir, ``fit`` için değil:

  * ``transform`` (serving + eğitim, İKİSİ de) — ``feature_builder`` import'u YASAK.
    Prodüksiyonda fiilen çalışan yol budur; yalın kalmak zorundadır ve eski koda
    bağlanırsa "iki kod yolu" riski geri gelir.
  * ``fit`` (YALNIZ eğitim-anı) — ``feature_builder``'a lazy+lokal delege edebilir.
    Eğitim zaten ``feature_builder``'ın çalıştığı bağlamdır; orada mevcut, test edilmiş
    bir mantığı yeniden yazmak sadakat kazandırmaz, yalnız DRIFT riski ekler.

Bugün tek kullanıcısı ``SelectDropStep`` (selection dizisi ~270 satır, 122'si
``_detect_leakage``; serving'de HİÇ çalışmaz → reimplement salt drift riski olurdu).
Bu bir "invariant ihlali" DEĞİL, invariant'ın kapsamının açıkça daraltılmasıdır; yeni
bir adım aynı gerekçeyi öne sürecekse ölçüt nettir: **transform'u delegesiz yazılabiliyor
mu?** Yazılamıyorsa adım yanlış bölünmüştür.

──────────────────────────────────────────────────────────────────────────────
``fit_transform`` — train-only regülarizasyon için kasıtlı ikinci yol (E invariant'ı)
──────────────────────────────────────────────────────────────────────────────
Çoğu adım için ``fit_transform(X_train) == transform(X_train, fit(X_train))`` (davranış
aynı; ``DefaultFitTransform`` mixin'i bunu verir). Ama sklearn ``TargetEncoder`` için
**öyle DEĞİL**: ``fit_transform`` bir *cross-fitting* şeması uygular (train satırlarını
kendi hedefine sızdırmadan kodlar), ``transform`` düz tam-veri encoding'i uygular. sklearn
dokümanı birebir: "``fit(X, y).transform(X)`` does not equal ``fit_transform(X, y)``".

Bu iki KOD YOLU (train: ``fit_transform`` cross-fitted / serving: ``transform`` düz)
arasındaki fark **kasıtlı** (train-only anti-leakage regülarizasyonu), skew DEĞİL. Ama
"kasıtlı fark"ı "kazayla-fark"tan ayıran şey bir invariant olmalı, yoksa biri ileride
inference'ta ``fit_transform`` çağırır → cross-fitting'i prodüksiyona sızdırır → sessiz
patlar. INVARIANT (imza + fail-closed ile bağlanır):

  * Serving yolu YALNIZ ``transform(X, params)`` çağırır. ``transform`` imzasında **``y``
    YOK** → inference bir hedef geçiremez → cross-fitting (matematiksel olarak ``y``
    gerektirir) ``transform``'dan ULAŞILAMAZ. Bu, imza-düzeyi kanıt.
  * ``fit_transform`` train-only'dir. Cross-fitting yapan adım (TargetEncode) ``y`` None
    ise ``RuntimeError`` atar (fail-closed) → serving (hedefi olmayan bağlam) onu
    çağıramaz. transform_pipeline (inference) ``fit_transform``'u ASLA çağırmaz; yalnız
    fit_pipeline, yalnız train çerçevesine.

Böylece serving-parity (``transform`` == ``transform``) ve train-parity (``fit_transform``
== ``fit_transform``) İKİ AYRI iddiadır; ikisi de korunur, karışmaz.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, Union, runtime_checkable

import pandas as pd

# ── Tip sözlüğü (§1.1 param_store'a birebir) ─────────────────
StepConfig = dict                                  # inşa-anı yapısal seçim (op/inputs/columns)
FittedStatParams = Union[dict, Any]                # inline dict VEYA pickle'lı sklearn estimator
FrozenChoiceParams = Union[list, dict]             # kolon-listeleri, rename-map, çift-listesi, vocab
StepParams = Optional[Union[FittedStatParams, FrozenChoiceParams]]   # rowwise → None

ParamKind = Literal["fitted_stat", "frozen_choice"]
StepType = Literal["parametric", "rowwise", "skip"]


@runtime_checkable
class TransformStep(Protocol):
    """parametric + rowwise adımların ORTAK sözleşmesi.

    Adım, config'iyle inşa edilir (sklearn deseni: hiperparam constructor'da, öğrenilen
    param ``fit``'te). ``fit`` yalnız X_train görür → parametre öğrenir (rowwise → None).
    ``transform`` yalnız ``params`` alır → uygular; istatistik HESAPLAMAZ. Aynı
    ``transform`` hem eğitimde (fit sonrası) hem inference'ta çağrılır → skew imkânsız.
    """

    id: str
    type: StepType
    param_kind: ParamKind
    config: StepConfig

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> StepParams:
        ...

    def transform(self, X: pd.DataFrame, params: StepParams) -> pd.DataFrame:
        # DİKKAT: imzada ``y`` YOK. Serving yolunun hedef geçirememesi = E invariant'ının
        # imza-düzeyi kanıtı (cross-fitting buradan ulaşılamaz).
        ...

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> "tuple[pd.DataFrame, StepParams]":
        # fit_pipeline (train) bunu çağırır → (dönüştürülmüş train çerçevesi, params).
        # DEFAULT (DefaultFitTransform): fit sonra transform → transform ile birebir aynı.
        # OVERRIDE (TargetEncode): sklearn cross-fitting → transform'dan FARKLI (kasıtlı).
        # transform_pipeline (serving) bunu ASLA çağırmaz (E invariant'ı).
        ...


class DefaultFitTransform:
    """``fit_transform`` = ``fit`` sonra ``transform`` (train == serving davranışı).

    ``TransformStep`` sözleşmesini uygulayan adımlar bu mixin'i miras alarak orkestratöre
    tek-tip ``fit_transform`` sunar. Cross-fitting yapmayan TÜM adımlar (impute, ratio,
    count, onehot, scaler ...) bunu kullanır → ``fit_transform``'un çıktısı
    ``transform``'unkiyle BİREBİR aynıdır. Yalnız cross-fitting yapan adım (TargetEncode)
    bu mixin'i KULLANMAZ, kendi ``fit_transform``'unu tanımlar → farkı kasıtlı ve tekil."""

    def fit_transform(self, X_train, y_train=None):
        params = self.fit(X_train, y_train)
        return self.transform(X_train, params), params


@runtime_checkable
class TrainOnlyStep(Protocol):
    """``skip`` adımların sözleşmesi (drop_duplicates gibi satır-üyeliği değiştiren,
    feature değerini değiştirmeyen işlemler). Yalnız ``fit_pipeline`` çağırır;
    ``transform_pipeline`` bu tipi manifest'ten atlar → inference'a yapısal olarak
    giremez, dolayısıyla skew riski taşımaz. ``transform`` bilinçli olarak YOKTUR."""

    id: str
    type: StepType

    def apply_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass
class Artifact:
    """Taşınabilir inference artifact'i — §1.1 şemasının in-memory karşılığı.

    Bu ilk parçada yalnız ``manifest`` + ``param_store`` fiilen doldurulur (ImputeStep).
    Kalan alanlar sonraki parçalarda dolar; şema şekli şimdiden sabit (ileri-uyum)."""

    manifest: list[dict] = field(default_factory=list)          # sıralı adım kayıtları
    param_store: dict[str, StepParams] = field(default_factory=dict)   # {params_ref: fitted state}
    raw_input_signature: dict = field(default_factory=dict)     # domain-aware guard (sonraki parça)
    # Bu artifact'in HANGİ koşullar altında geçerli olduğu + deploy edilebilirlik kararı.
    # ``{declared, undeclared, violations, deployable, ported_feature_ops}`` — orkestratörün
    # ``check_pipeline_scope``'u doldurur; ihlal varsa artifact ZATEN kurulmaz (fail-closed).
    # ``deployable=False`` = "kurulabilir ama deploy edilemez" (kapsam olguları beyan edilmemiş).
    supported_scope: dict = field(default_factory=dict)
    # Simetrik guard'ın TOLERE edeceği fazla kolonlar. §2.4 "fazla girdi kolonu tolere edilir"
    # diyordu; bu oturumun dersi ise sessiz sapmanın düşman olduğu. Uzlaşma: tolerans mümkün
    # ama **beyan edilmeli** — boş liste = hiçbir fazla kolon sessizce geçmez (fail-closed).
    tolerated_extra_columns: list[str] = field(default_factory=list)
    unseen_category_policy: dict = field(default_factory=dict)  # görülmemiş-kategori (sonraki parça)
    training_reference: dict = field(default_factory=dict)      # PSI/CBPE referansı (sonraki parça)
    model_ref: dict = field(default_factory=dict)               # model + eşik (sonraki parça)
    final_features: list[str] = field(default_factory=list)     # selection sonrası hayatta kalanlar
    artifact_schema_version: str = "1.1"
