from __future__ import annotations

"""Katman 1 transform adımları — her biri ``TransformStep`` sözleşmesini uygular.

Bu dosya adım adım büyür (~12 adım). İlk adım: ImputeStep (#4 fix). Sonraki adımlar
(ratio finalize, count-encode, scaler, encode, select ...) aynı fit/transform kalıbıyla
buraya eklenecek; parity testi her eklemede genişleyecek.
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import RobustScaler, StandardScaler, TargetEncoder

from src.core.config import settings
from src.core.logging_config import get_logger
from src.services.inference.pickle_io import load_pickle_ref, make_pickle_ref
from src.services.inference.protocol import DefaultFitTransform, StepConfig

logger = get_logger(__name__)

_RATIO_WINSOR_Q = 0.01   # feature_builder.py:545 ile aynı — self-contained (import YOK)
_FEATURE_SPEC_EPS = 1e-9  # feature_builder.py:1548 ile aynı — zscore/normalize payda koruması


class ImputeStep(DefaultFitTransform):
    """Eksik-değer impute — parametric / fitted_stat.

    #4 FIX: median/mode YALNIZ ``fit``'e verilen çerçeveden öğrenilir (eğitimde X_train).
    Eski ``_clean_data`` bunu tüm-df'den, split-ÖNCESİ hesaplıyordu → test dağılımı
    eğitime sızıyordu. ``transform`` istatistik HESAPLAMAZ; yalnız öğrenilmiş dolgu
    değerini uygular (fitting-frame görmez) → sızıntılı çağrı imza düzeyinde imkânsız.

    Davranışsal olarak ``_clean_data`` (feature_builder.py:339-350) ile EŞDEĞER — aynı
    çerçeve verildiğinde birebir aynı çıktı:
      * numeric : median; ``is_integer_dtype`` ise ``int(round(median))``
      * object  : ``mode()[0]`` (mode boş değilse)
    İki fark: (a) #4 — ``fit``'e X_train verilir, tüm-df değil; (b) T3b-gate — dolgu
    ÖĞRENİLEN kolon kümesi daha geniş (aşağıda). (b) çıktıda görünmez (byte-nötr),
    yalnız ``param_store``'u büyütür; eşdeğerlik iddiası ÇIKTI düzeyinde durur.

    Parametre yapısı (``fit`` dönüşü):
      * ``numeric_median``     : {kolon: uygulanan dolgu}  — int kolonda YUVARLANMIŞ (transform bunu uygular)
      * ``numeric_median_raw`` : {kolon: yuvarlanmamış float median} — #4 delta BAĞLAMI (transform KULLANMAZ)
      * ``integer_cols``       : round+int uygulanan kolonlar (İncelik 2: ham vs yuvarlanmış ayrımı)
      * ``object_mode``        : {kolon: mode[0]}

    **KAPI YOK (T3b-gate) — dolgu, UYGUN HER kolon için öğrenilir.** Önceki sürüm
    ``_clean_data``'nın ``isnull().any()`` kapısını replika ediyordu: yalnız *fit
    çerçevesinde* null taşıyan kolonlar dolgu öğrenirdi. O kapı kaldırıldı, çünkü:

      **train'de null olmayan ama serving'de null gelebilecek kolonlar için dolgu
      ÖĞRENİLMELİDİR; aksi halde terminal ``fillna(0)`` devreye girer ve dağılım-DIŞI
      bir değer üretir.** (0, çoğu ölçek için meşru bir değer bile değildir.)

    Somut vaka — housing/``total_bedrooms``: 207 null'un 207'si de ``random_state=42``
    split'inde TEST tarafına düşüyor → fit çerçevesi (train) null'suz → eski kapı kolonu
    ATLARDI → o 207 hücre medyan (437.0) yerine ``0.0`` olurdu. Yani kapı, #4 fiksinin
    (fit çerçevesinin full→train daralması) tam da faydasını REGRESYONA çeviriyordu.
    Kapının varlığı yapısal bir sınırdı, ölçülmüş bir tercih değil.

    Kapının kaldırılması ``fit_frame == transform çerçevesi`` iken **byte-nötr**: yeni
    öğrenilen kolonlar tanımı gereği o çerçevede null'suzdur, ``fillna`` onlarda no-op'tur.
    Değişen tek şey ``param_store``'un BOYUTU (daha çok kolon dolgu taşır) — bu sessiz
    kalmasın diye manifest'e ``param_footprint`` yazılır (bkz. ``param_footprint``).

    Medyanı NaN çıkan kolon (boş çerçeve / tümü-null) atlanır: NaN dolgu zaten hiçbir
    hücreyi doldurmaz, ama int kolonda ``int(round(nan))`` patlardı."""

    id = "impute_missing"
    type = "parametric"
    param_kind = "fitted_stat"

    def __init__(self, config: StepConfig | None = None):
        # config["columns"]: opsiyonel kısıtlama; None → fit'te UYGUN TÜM kolonlar (kapı yok)
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        only = self.config.get("columns")   # None → otomatik
        numeric_median: dict = {}
        numeric_median_raw: dict = {}
        integer_cols: list = []
        object_mode: dict = {}
        # Fit çerçevesinde FİİLEN null taşıyan kolonlar = ESKİ kapının öğreneceği kümenin ta
        # kendisi. Kapı kalktığı için artık bir filtre DEĞİL, yalnız kayıt: "kaç kolon büyüdük"
        # sorusu manifest'ten (param_footprint) cevaplanabilsin. ``transform`` bunu OKUMAZ.
        null_columns: list = []

        for col in X_train.select_dtypes(include="number").columns:
            if only is not None and col not in only:
                continue
            if X_train[col].isnull().any():                   # KAYIT (kapı DEĞİL)
                null_columns.append(col)
            med = X_train[col].median()
            if pd.isna(med):                 # boş çerçeve / tümü-null → dolgu yok
                continue                     # (float(pd.NA) ve int(round(nan)) ikisi de patlar)
            raw = float(med)
            numeric_median_raw[col] = raw
            if pd.api.types.is_integer_dtype(X_train[col]):
                numeric_median[col] = int(round(raw))
                integer_cols.append(col)
            else:
                numeric_median[col] = raw

        for col in X_train.select_dtypes(include="object").columns:
            if only is not None and col not in only:
                continue
            if X_train[col].isnull().any():
                null_columns.append(col)
            mode = X_train[col].mode()
            if not mode.empty:
                object_mode[col] = mode[0]

        if numeric_median or object_mode:
            logger.info(
                f"[impute] fit: {len(numeric_median)} numeric + {len(object_mode)} object kolon "
                f"dolgu ÖĞRENDİ (fit çerçevesinde fiilen null taşıyan: {len(null_columns)}) — "
                f"isnull() kapısı YOK (T3b-gate: serving'de null gelebilir)"
            )
        return {
            "numeric_median": numeric_median,
            "numeric_median_raw": numeric_median_raw,
            "integer_cols": integer_cols,
            "object_mode": object_mode,
            "null_columns": null_columns,      # transform OKUMAZ — ayak izi/denetim kaydı
        }

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        # params UYGULAR — X[col].median() YOK. Öğrenilmiş dolgu değeri neyse o.
        X = X.copy()
        for col, fill in (params.get("numeric_median") or {}).items():
            if col in X.columns:
                X[col] = X[col].fillna(fill)
        for col, fill in (params.get("object_mode") or {}).items():
            if col in X.columns:
                X[col] = X[col].fillna(fill)
        return X

    def param_footprint(self, params: dict) -> dict:
        """Kaç kolon için dolgu öğrenildiği — manifest'e opt-in beyan (bkz. ``_param_footprint_of``).

        T3b-gate'in ÇIKTI'sı byte-nötr ama ``param_store``'u DEĞİL: kapı kalkınca dolgu
        öğrenilen kolon sayısı büyüdü. Kullanılmayan bir dolgu da olsa bu bir davranış
        değişikliğidir (serving'de o kolona null gelirse ARTIK medyan uygulanacak) →
        manifest'te görünür olmalı, ölçüm sessizce kaymamalı.

        ``gate_growth`` BÜYÜMENİN KENDİSİDİR: ``fit_null_columns`` eski ``isnull().any()``
        kapısının öğreneceği küme, ``*_fill_learned`` ise şimdi öğrenilen. Fark = kapının
        kaldırılmasının artifact üzerindeki tam etkisi (çıktıda görünmez, burada görünür)."""
        num = params.get("numeric_median") or {}
        obj = params.get("object_mode") or {}
        nulls = params.get("null_columns") or []
        return {"numeric_fill_learned": len(num), "object_fill_learned": len(obj),
                "fill_columns": sorted([*num, *obj]),
                "fit_null_columns": len(nulls),                     # eski kapının kümesi
                "gate_growth": len(num) + len(obj) - len(nulls)}    # T3b-gate'in param deltası


_POWER_EXP_BOUNDS = (-3.0, 3.0)   # feature_builder.py:1556 ile aynı — YALNIZ belgeleme amaçlı
                                  # (clamp BEYAN anında uygulanır; bu adım YENİDEN clamp ETMEZ)


def _prod_series(cols: list) -> "pd.Series":
    out = cols[0]
    for c in cols[1:]:
        out = out * c
    return out


def _weighted_sum_series(cols: list, params: list) -> "pd.Series":
    out = None
    for w, c in zip(params, cols):
        term = float(w) * c
        out = term if out is None else out + term
    return out


# ``_FEATURE_OPS``'un ROWWISE altısının SELF-CONTAINED replikası (feature_builder.py:1578-1587).
# Neden kopya: SELF-CONTAINED invariant'ı — ``transform`` (serving+eğitim) ``feature_builder``
# import ETMEZ (protocol.py'deki gerekçe). Parametric ikili (zscore/normalize) BURADA YOK:
# onlar fitted-state taşır ve ayrı adımlardır (ZScoreStep/NormalizeStep, #5 fiksi).
_PLAN_ROWWISE_OPS = {
    "ratio":        lambda cols, p: cols[0] / (_prod_series(cols[1:]).abs() + _FEATURE_SPEC_EPS),
    "product":      lambda cols, p: _prod_series(cols),
    "difference":   lambda cols, p: cols[0] - cols[1],
    "log":          lambda cols, p: np.log1p(cols[0].clip(lower=0)),
    "power":        lambda cols, p: cols[0] ** float(p[0]),
    "weighted_sum": lambda cols, p: _weighted_sum_series(cols, p),
}


class PlanDerivedStep(DefaultFitTransform):
    """Plan-güdümlü ROWWISE türev feature'lar (6 op) — rowwise / frozen config.

    ``_FEATURE_OPS``'un fitted-state TAŞIMAYAN altı op'u: ``ratio · product · difference ·
    log · power · weighted_sum``. Hepsi girdi kolonlarının + sabit parametrelerin element-bazlı
    fonksiyonu → train'den öğrenilen istatistik YOK → **sızıntı yüzeyi YOK** (ölçüldü: paletteki
    tek iki parametric op zaten portlu olan zscore/normalize).

    Çıktı: ``series.replace([inf,-inf], 0).fillna(0)`` — ``_run_feature_specs``:1636 ile BİREBİR.

    **KABUL KURALLARI BURADA DEĞİL, BEYANDA:** şekil doğrulaması, leakage denylist'i ve ``power``
    clamp'i ``_admit_spec`` içinde uygulanır; config donmuş/temiz spec taşır. Adım yalnız iki
    savunma yapar: adı zaten varsa üretme (idempotans), girdi kolonu yoksa atla.

    Config: ``{"specs": [{"name","op","inputs","params"}, ...]}`` (sıra korunur).
    """

    id = "plan_derived"
    type = "rowwise"
    param_kind = None

    def __init__(self, config: StepConfig | None = None):
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> None:
        return None            # rowwise: train'den öğrenilen state YOK

    def frozen_selection(self, params=None) -> dict:
        specs = self.config.get("specs") or []
        return {"kind": "plan_ops", "source": "feature_plan.derived_features (rowwise op'lar)",
                "selection": [{"output": s["name"], "op": s["op"], "inputs": s["inputs"],
                               "params": s.get("params") or [],
                               **({"params_clamped": True,
                                   "params_original": s["params_original"]}
                                  if s.get("params_clamped") else {})}
                              for s in specs]} if specs else {}

    def transform(self, X: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
        X = X.copy()
        for spec in (self.config.get("specs") or []):
            name, op = spec.get("name"), spec.get("op")
            inputs, prm = spec.get("inputs") or [], spec.get("params") or []
            if op not in _PLAN_ROWWISE_OPS or not name or name in X.columns:
                continue
            if any(c not in X.columns for c in inputs):
                continue
            cols = [pd.to_numeric(X[c], errors="coerce") for c in inputs]
            seri = _PLAN_ROWWISE_OPS[op](cols, prm)
            X[name] = seri.replace([np.inf, -np.inf], 0).fillna(0)   # 1636 ile birebir
        return X


class DateFeaturesStep(DefaultFitTransform):
    """Timestamp kolonlarından DURUMSUZ rowwise türevler: ``_year`` · ``_month`` · ``_dayofweek``.

    Eski inline blok (feature_builder date-extraction: ``pd.to_datetime`` + ``.dt``) SELF-CONTAINED
    port'u — fitted-state YOK → sızıntı yüzeyi YOK (element-bazlı; train'den öğrenilen istatistik yok).
    **§2.19 KAPSAM PORT'U:** eski ``date_features`` kapsam olgusu (kapsam kapısı reddederdi) ADIMA
    çevrildi → ``_SCOPE_RULES``'tan çıkarıldı, timestamp taşıyan dataset artık orchestrator modda
    deploy EDİLEBİLİR.

    KONUM (pre-split): impute'tan SONRA, ``drop_columns``'tan ÖNCE — ham timestamp kolonu (imzada
    ``required``) hâlâ çerçevede; türevler üretildikten sonra ham kolonu ``drop_columns`` düşürür.
    İmpute SONRASI çalışır → NaT→NaN türev hücreleri burada DOLDURULMAZ (eski yol da doldurmuyordu;
    downstream ``inf_impute``/select yakalar). Serving replay'inde ham satırdan aynen üretilir.

    DİNAMİK: kolon adı GÖMÜLÜ DEĞİL. ``config["columns"]`` = genişletilecek timestamp kolonları
    (``meta.get_by_type("timestamp")`` − leaky). Savunmalar: kolon yoksa atla; ``{col}_year`` zaten
    varsa üretme (idempotans — shadow'da girdi çerçevesi çoktan genişletilmiş olabilir)."""

    id = "date_features"
    type = "rowwise"
    param_kind = None

    def __init__(self, config: StepConfig | None = None):
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> None:
        return None                # rowwise: train'den öğrenilen state YOK

    def frozen_selection(self, params=None) -> dict:
        cols = self.config.get("columns") or []
        return {"kind": "date_expand", "source": "meta.timestamp (non-leaky)",
                "selection": [{"source": c,
                               "emits": [f"{c}_year", f"{c}_month", f"{c}_dayofweek"]}
                              for c in cols]} if cols else {}

    def transform(self, X: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
        X = X.copy()
        for col in (self.config.get("columns") or []):
            if col not in X.columns or f"{col}_year" in X.columns:
                continue               # kolon yok → atla; türev zaten var → idempotans
            try:                       # eski inline blok da try/except pass (parse patlarsa atla)
                dt = pd.to_datetime(X[col], errors="coerce")
                X[f"{col}_year"]      = dt.dt.year
                X[f"{col}_month"]     = dt.dt.month
                X[f"{col}_dayofweek"] = dt.dt.dayofweek
            except Exception:
                continue
        return X


class ZScoreStep(DefaultFitTransform):
    """Plan-güdümlü ``zscore`` derived-feature — parametric / fitted_stat. **#5 FIX.**

    LLM feature-plan'i ``zscore`` op'unu önerdiğinde (``derived_features``): ``name = (x − μ)/(σ+eps)``.
    Eski ``_FEATURE_OPS["zscore"]`` (feature_builder.py:1577) ``μ``/``σ``'yı ``cols[0]`` = TÜM df'den,
    **split ÖNCESİ** hesaplıyordu → test dağılımı eğitime sızıyordu (#5). Bu adım μ/σ'yı YALNIZ ``fit``'e
    verilen çerçeveden (eğitimde X_train) öğrenir; ``transform`` donmuş μ/σ'yı uygular. Sızıntı imza
    düzeyinde imkânsız (fit yalnız X_train, transform yalnız params) — #4 impute ile AYNI disiplin.

    **Eski ile parity YOK — bilerek sapar** (eski full-df/leaky, yeni train-fit/doğru). Doğrulama iki ayrı
    iddia: (a) FORMÜL-sadakati — ``transform``'un ürettiği değer, μ/σ elle X_train'den hesaplanınca birebir;
    (b) SIZINTI-delta — full-df μ/σ ≠ train-only μ/σ (train-dist ≠ full-dist olduğunda) → düzeltmenin
    kaldırdığı sızıntının deterministik kanıtı. ROC A/B DEĞİL (leakage sorusu deterministik çözülür).

    DİNAMİK: kolon adı GÖMÜLÜ DEĞİL. ``config["specs"]`` = plandan gelen ``[(çıktı_adı, girdi_kolonu), ...]``
    (LLM ``spec.name``/``spec.inputs[0]``). Herhangi bir dataset; hangi kolonun zscore'lanacağı plan kararı.
    ``σ`` = ``std(ddof=0)`` (op ile birebir). Çıktı: ``replace([inf,-inf], 0).fillna(0)`` (op:1631 ile aynı).

    Parametre yapısı (``fit`` dönüşü / fitted_stat): ``{çıktı_adı: {"input": girdi, "mean": μ, "std": σ}}``.
    """

    id = "zscore_feature"
    type = "parametric"
    param_kind = "fitted_stat"

    def __init__(self, config: StepConfig | None = None):
        # config["specs"] = plandan donmuş [(çıktı_adı, girdi_kolonu), ...] (zscore op'lu FeatureSpec'ler)
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        params: dict = {}
        for name, src in (self.config.get("specs") or []):
            if src not in X_train.columns or name in X_train.columns or name in params:
                continue
            s = pd.to_numeric(X_train[src], errors="coerce")
            params[name] = {"input": src, "mean": float(s.mean()), "std": float(s.std(ddof=0))}
        if params:
            logger.info(f"[zscore] fit (train-only, #5): {len(params)} kolon → {list(params)}")
        return params

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        X = X.copy()
        for name, spec in (params or {}).items():
            src = spec["input"]
            if src not in X.columns:
                continue
            s = pd.to_numeric(X[src], errors="coerce")
            out = (s - spec["mean"]) / (spec["std"] + _FEATURE_SPEC_EPS)      # μ/σ TRAIN'den donmuş
            X[name] = out.replace([np.inf, -np.inf], 0).fillna(0)             # op:1631 ile birebir
        return X


class NormalizeStep(DefaultFitTransform):
    """Plan-güdümlü ``normalize`` derived-feature — parametric / fitted_stat. **#5 FIX.**

    ``ZScoreStep``'in min-max ikizi: ``name = (x − min)/((max − min)+eps)``. Eski ``_FEATURE_OPS["normalize"]``
    (feature_builder.py:1581) min/max'ı TÜM df'den (split öncesi) alıyordu → #5 sızıntısı. Bu adım min/max'ı
    YALNIZ X_train'den öğrenir, ``transform`` donmuş min/max'ı uygular. Gerekçe/doğrulama/dinamizm ZScoreStep
    ile birebir aynı (bkz. yukarı) — tek fark öğrenilen istatistik (μ/σ değil min/max).

    Parametre yapısı (``fit`` dönüşü / fitted_stat): ``{çıktı_adı: {"input": girdi, "min": lo, "max": hi}}``.
    """

    id = "normalize_feature"
    type = "parametric"
    param_kind = "fitted_stat"

    def __init__(self, config: StepConfig | None = None):
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        params: dict = {}
        for name, src in (self.config.get("specs") or []):
            if src not in X_train.columns or name in X_train.columns or name in params:
                continue
            s = pd.to_numeric(X_train[src], errors="coerce")
            params[name] = {"input": src, "min": float(s.min()), "max": float(s.max())}
        if params:
            logger.info(f"[normalize] fit (train-only, #5): {len(params)} kolon → {list(params)}")
        return params

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        X = X.copy()
        for name, spec in (params or {}).items():
            src = spec["input"]
            if src not in X.columns:
                continue
            s = pd.to_numeric(X[src], errors="coerce")
            out = (s - spec["min"]) / ((spec["max"] - spec["min"]) + _FEATURE_SPEC_EPS)   # min/max TRAIN'den
            X[name] = out.replace([np.inf, -np.inf], 0).fillna(0)                         # op:1631 ile birebir
        return X


class DropColumnsStep(DefaultFitTransform):
    """identifier / timestamp / TARGET kolonlarını düşür — rowwise (config donmuş liste).

    ``build_features``:2233-2255'in protokol karşılığı (semantic_type id/timestamp/target +
    ``_find_related_target_cols`` hedef-takma-adları). Bu bir KALDIRMA adımıdır ve neden
    portlanması gerektiği ÖLÇÜLEREK bulundu (2026-07-23):

    **Guard'ın KÖR NOKTASI:** eksik bir kaldırma "kolon üretilmedi" olarak GÖRÜNMEZ — guard
    yalnız *eksik üretim* görür, *fazladan mevcudiyet* görmez. Eksik kaldırma bunun yerine
    **downstream SEÇİM SAPMASI** üretir: fazladan kalan kolon aday havuzunda yarışır ve gerçek
    bir kolonu sıralamadan düşürür. Ölçülen vaka: credit'te hedef ``loan_status`` çerçevede
    kalınca ``GroupbyDeviation``'ın ``_norm_var`` sıralamasında BİRİNCİ sıraya oturdu
    (binary p≈0.22 → normalize varyans yüksek) ve ``cb_person_cred_hist_length``'i top-3'ten
    attı → ``grpstd_cb_person_cred_hist_length_by_*`` üretilemedi. Bu, "kaskad" sanılan
    sapmanın GERÇEK kökü çıktı (kaskad hipotezi ÇÜRÜDÜ: flag portları sapmayı çözmedi).

    **AYRICA SIZINTI YÜZEYİ:** hedef çerçevede kalırsa ``GroupbyDeviation``/``CountEncode``
    HEDEFTEN türetilmiş feature üretir (``dev_<target>_by_X`` = hedefin grup-ortalamasından
    sapması) → doğrudan target leakage. ``build_features``'ta imkânsız (hedef split'ten önce
    düşüyor); orkestratörde hiçbir şey engellemiyordu. Bu adım o yüzeyi kapatır.

    Config: ``{"columns": [düşürülecek...]}`` — semantic_type kararı upstream'de (meta),
    burada yalnız donmuş liste infaz edilir (``re_derive:false``, diğer frozen_choice'larla aynı)."""

    id = "drop_columns"
    type = "rowwise"
    param_kind = None

    def __init__(self, config: StepConfig | None = None):
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> None:
        return None

    def frozen_selection(self, params=None) -> dict:
        return {
            "kind": "columns",
            "source": "build_features:2233-2255 (semantic_type id/timestamp/target + hedef takma adları)",
            "selection": [{"source": c, "output": None} for c in (self.config.get("columns") or [])],
        }

    def transform(self, X: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
        cols = [c for c in (self.config.get("columns") or []) if c in X.columns]
        return X.drop(columns=cols, errors="ignore") if cols else X.copy()


class MissingIndicatorStep(DefaultFitTransform):
    """``<col>_isna`` (0/1) eksiklik bayrakları — rowwise (train-türevi state YOK, config donmuş liste).

    ``_add_missing_indicators`` (feature_builder.py:1392-1414) satır-düzeyi çekirdeğinin protokol
    karşılığı: ``df[c].isnull().astype("int8")``. Kolon SEÇİMİ (null taşıyanlar, null-oranı desc,
    ``_MISSING_IND_MAX_COLS`` cap, ``exclude`` own-goal koruması) bu adımda YENİDEN TÜRETİLMEZ.

    SIRA KİLİDİ — **impute'tan ÖNCE**: eksiklik bayrağı fillna ÖNCESİ hesaplanmalıdır, yoksa null
    kalmaz ve bayrak hep 0 olur (sinyal sessizce silinir). Kaynakta da böyledir
    (``_add_missing_indicators``@2130 < ``_clean_data``@2133) → PIPELINE'da ImputeStep'ten ÖNCE
    gelir ve ``legacy_ref`` drift testi bu sırayı sürekli doğrular.

    SEÇİMİN KAYNAĞI — **(a) SADAKAT-ÖNCE kararı** (ratio ile aynı kategori: "pre-split sabit,
    artifact'e frozen yaz"): seçim eski yolda nerede yapılıyorsa orada kalır, biz kararı taşırız.
    Alternatif (b) "seçimi ``fit(X_train)``'e al → selection-peek kapansın" REDDEDİLDİ (şimdilik):
    bu turun amacı KAPSAM, sızıntı düzeltme değil; peek'ler aynı sınıf olduğu için terminus sonrası
    TEK turda toplu ele alınacak (parça parça ıraksama parity'yi gereksiz yere bulandırır).
    **PEEK BORCU AÇIK, kayıtlı:** null-oranı sıralaması + top-N hâlâ tüm df'ten.

    Config: ``{"columns": [kaynak_kolon, ...]}`` → üretilen ad ``f"{kaynak}_isna"``."""

    id = "missing_indicator"
    type = "rowwise"
    param_kind = None

    def __init__(self, config: StepConfig | None = None):
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> None:
        return None            # rowwise: train'den öğrenilen state YOK

    def frozen_selection(self, params=None) -> dict:
        return {
            "kind": "columns",
            "source": "_add_missing_indicators (pre-split sabit karar)",
            "selection": [{"source": c, "output": f"{c}_isna"}
                          for c in (self.config.get("columns") or [])],
        }

    def transform(self, X: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
        X = X.copy()
        for col in (self.config.get("columns") or []):
            name = f"{col}_isna"
            if col not in X.columns or name in X.columns:
                continue
            X[name] = X[col].isnull().astype("int8")     # 1400 ile birebir (dtype dahil)
        return X


class ZeroFlagStep(DefaultFitTransform):
    """``is_zero_<col>`` (0/1) sıfır bayrakları — rowwise (train-türevi state YOK, config donmuş liste).

    ``_build_zero_missing_flags`` (feature_builder.py:1756-1858) satır-düzeyi çekirdeğinin protokol
    karşılığı: ``(pd.to_numeric(col) == 0).astype(int)``. Kolon SEÇİMİ bu adımda YENİDEN TÜRETİLMEZ.

    SEÇİM NEDEN TAŞINIR, TÜRETİLMEZ (ölçülmüş gerekçe — ratio'dakinden DAHA güçlü): eski seçim saf
    bir prevalence kapısı DEĞİL; ``force_zero_cols`` üzerinden **oran motoruna bağlı bir INVARIANT**
    taşıyor — oranın payda olarak kullandığı ve sıfır içeren kolonlar prevalence kapısından VE global
    cap'ten MUAF olarak garanti flag alır (sebep: oran sentinel'den arıtılırken sıfır-payda bilgisi
    flag'de DURMALI, yoksa bilgi tamamen silinir). Seçimi ``fit(X_train)``'de yeniden türetmek, bu
    motorlar-arası kuplajı inference paketinin içinde YENİDEN KURMAK demekti → tam da kaçındığımız
    drift riski. Donmuş listeyi taşımak invariant'ı bedavaya korur. **(a) SADAKAT-ÖNCE**; peek borcu
    (prevalence + cap tüm df'ten) AÇIK ve kayıtlı, terminus sonrası toplu turda ele alınacak.

    Ayrıca eski yolun iki kapısı seçimde zaten uygulanmıştır ve burada TEKRARLANMAZ: binary kolonda
    is_zero üretilmez (#10 — corr −1.0 birebir kopya) ve hiç/hep sıfır kolonlar elenir.

    Config: ``{"columns": [kaynak_kolon, ...]}`` → üretilen ad ``f"is_zero_{kaynak}"``."""

    id = "zero_flag"
    type = "rowwise"
    param_kind = None

    def __init__(self, config: StepConfig | None = None):
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> None:
        return None

    def frozen_selection(self, params=None) -> dict:
        return {
            "kind": "columns",
            "source": "_build_zero_missing_flags (pre-split sabit karar; force_zero invariant'ı dahil)",
            "selection": [{"source": c, "output": f"is_zero_{c}"}
                          for c in (self.config.get("columns") or [])],
        }

    def transform(self, X: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
        X = X.copy()
        for col in (self.config.get("columns") or []):
            name = f"is_zero_{col}"
            if col not in X.columns or name in X.columns:
                continue
            num = pd.to_numeric(X[col], errors="coerce")
            X[name] = (num.to_numpy() == 0).astype(int)   # 1812 ile birebir
        return X


class RatioConstructStep(DefaultFitTransform):
    """Ham oran (a/b) kolonlarını kur — rowwise (train-türevi state YOK, config donmuş çiftler).

    ``_build_ratio_features``'ın (feature_builder.py:1934-1936) SATIR-DÜZEYİ çekirdeğinin protokol
    karşılığı. Eski geçici ``_build_ratio_stand_in`` (test iskelesi) buraya FORMELLEŞTİ → stand-in
    emekliye ayrılır (işaretli-geçici kod gerçek adıma döndü, kalıcı ikiz olmadı).

    KAPSAM: yalnız row-wise BÖLME. Çift SEÇİMİ (hangi a/b — eligible/zero_frac/sign_mass/spread/cap)
    bu adımda YENİDEN TÜRETİLMEZ; ``config["pairs"]``'ten donmuş gelir. Bu yüzden ``rowwise``:
    ``fit`` None döner (train-türevi state yok), ``transform`` config'teki donmuş çiftlere böler.

    ÇİFT SEÇİMİNİN KAYNAĞI (8.1 kararı — "port et" DEĞİL, **"pre-split sabit, artifact'e frozen yaz"**):
    ``_build_ratio_features`` seçimi zaten yapıp ``feature_report["ratio_feature_meta"]``'ya
    ``{ratio_kolonu: (pay, payda)}`` olarak yazıyor (feature_builder.py:2491). O tablo **X_train'e
    göre FİLTRELENMEZ** — selection'da sonradan düşen oranlar da içinde (Faz 1 #4/#5'te bilinçle
    lookup tablosuna çevrilmişti). Yani KURULAN tüm çiftlerin tam kaydıdır ve seçim mantığını
    yeniden yazmaya gerek yoktur: karar zaten donmuş, biz onu OKURUZ. (``ratio_features`` anahtarı
    ise hayatta-kalanlar → frozen kaynağı olarak YANLIŞ olurdu; select_drop'un düşürdüklerini
    kaybederdik.) Sıra: build_features seçer → ctx["ratio_pairs"] → manifest ``frozen_choice`` →
    serving aynı çiftleri kurar.

    NOT (kapanmayan, kayıtlı): seçim hâlâ **split ÖNCESİ tüm df'ten** hesaplanıyor (§2.3 selection-peek
    sınıfı; değere test satırı gömmez). Bu adım onu KAPATMAZ, yalnız kararı sadakatle taşır. Kapatmak
    ayrı iş: seçimi fit(X_train)'e almak (InfImpute'un peek kapatması gibi, deterministik kanıtla).

    KİLİT — inf→NaN devri (Faz 0 sentinel'inin KÖKÜ): ``(num/den).replace([inf,-inf], nan)``. Payda-sıfır
    → inf → NaN (ham 0.0 sentinel DEĞİL); RatioFinalizeStep'in winsor+train-median impute'u tam bu
    NaN'lı çıktıya bağlı. Porlamada bu davranış eski row-wise kısımla BİREBİR (ayrı parity testi).

    Config: ``{"pairs": [(ratio_adı, pay_kolonu, payda_kolonu), ...]}``. Serving'de manifest kaleminden
    ``config`` geri yüklenir (rowwise → param_store yerine manifest'te config taşınır)."""

    id = "ratio_construct"
    type = "rowwise"
    param_kind = None          # rowwise: fitted_stat/frozen_choice değil

    def __init__(self, config: StepConfig | None = None):
        self.config: StepConfig = config or {}   # config["pairs"] = donmuş (ad, pay, payda) üçlüleri

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> None:
        return None            # rowwise: train'den öğrenilen state YOK

    def frozen_selection(self, params=None) -> dict:
        """Donmuş çift seçimini manifest'e kanonik ``{kind, selection}`` şekliyle beyan et
        (route/groupby ile AYNI alan → Katman 2 tek desenle okur). rowwise olduğu için
        kaynak ``params`` değil ``config``."""
        return {
            "kind": "pairs",
            "source": "feature_report.ratio_feature_meta",   # pre-split sabit karar (bkz. docstring)
            "selection": [{"output": n, "numerator": a, "denominator": b}
                          for n, a, b in (self.config.get("pairs") or [])],
        }

    def transform(self, X: pd.DataFrame, params: "dict | None" = None) -> pd.DataFrame:
        X = X.copy()
        for name, num, den in (self.config.get("pairs") or []):
            if num not in X.columns or den not in X.columns:
                continue
            n = pd.to_numeric(X[num], errors="coerce")
            d = pd.to_numeric(X[den], errors="coerce")
            X[name] = (n / d).replace([np.inf, -np.inf], np.nan)   # inf→NaN (KİLİT — 1936 ile birebir)
        return X


class RatioFinalizeStep(DefaultFitTransform):
    """Oran kolonlarını winsor[q01,q99] + train-median impute — parametric / fitted_stat.

    ``_fit_ratio_stats``/``_apply_ratio_stats`` (feature_builder.py:1978-2014) mantığının
    protokol karşılığı. YENİ FIX YOK: winsor zaten ``_finalize_ratio_features_tt``'de
    split-SONRASI train-fit'ti (Faz 0). Bu adım o doğru davranışı yalnız fit/transform
    imzasına taşır. Payda-sıfır ∪ kaynak-missing NaN'ları AYNI dağılım-içi median ile
    dolar → gizli 0.0 sentinel yok (Faz 0 bug'ının kökü buydu).

    SELF-CONTAINED: ``feature_builder`` import ETMEZ (inference runtime yalın kalmalı,
    2525-satırlık FE modülünü + sklearn/LLM bağımlılıklarını çekmemeli). Mantık
    ``_fit_ratio_stats``/``_apply_ratio_stats``'ı REPLİKA eder; parity testi ikisini
    birebir kıyaslar → X'i X'e değil, iki implementasyonu.

    EŞİK (replika eskime riski): eski yol REFERANS. Eski ``_fit_ratio_stats`` mantığı
    değişirse bu replika + parity BİRLİKTE güncellenmeli (aksi halde parity yeşil kalır
    ama pipeline farklı davranır). Risk şimdilik kabul (eski yol donmuş, denetim bitti);
    tüm adımlar bitip eski yol emekliye ayrılınca mantık yalnız burada yaşar → risk kalkar.

    Parametre yapısı (``fit`` dönüşü): ``{col: [lo, hi, med]}`` (JSON-able liste), veya
    train'de finite değer yoksa ``{col: None}`` (drop sinyali — transform kolonu düşürür,
    eski ``_apply_ratio_stats`` ile aynı). None-drop donmuş params'tan sürülür → inference
    ile eğitim birebir; orkestratör ``final_features`` reindex'i onu geri EKLEYEMEZ (kolon
    zaten final_features'ta yok) → iki mekanizma çatışmaz, reindex yalnız güvence."""

    id = "ratio_finalize"
    type = "parametric"
    param_kind = "fitted_stat"

    def __init__(self, config: StepConfig | None = None):
        # config["columns"]: winsor'lanacak oran kolonları; config["winsor_q"]: kuantil (vars. 0.01)
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        cols = self.config.get("columns") or []
        q = self.config.get("winsor_q", _RATIO_WINSOR_Q)
        stats: dict = {}
        for c in cols:
            if c not in X_train.columns:
                continue
            s = pd.to_numeric(X_train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = s[np.isfinite(s)]
            if finite.empty:
                stats[c] = None    # train'de finite yok → uygulanamaz (drop sinyali)
                continue
            lo = float(finite.quantile(q))
            hi = float(finite.quantile(1.0 - q))
            med = float(finite.clip(lo, hi).median())   # dağılım-içi impute değeri
            stats[c] = [lo, hi, med]
        if stats:
            logger.info(f"[ratio_finalize] fit (train-only): {len(stats)} oran kolonu")
        return stats

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        X = X.copy()
        params = params or {}
        drop_cols = [c for c, st in params.items() if st is None and c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        for c, st in params.items():
            if st is None or c not in X.columns:
                continue
            lo, hi, med = st
            s = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X[c] = s.clip(lo, hi).fillna(med)
        return X


_COUNT_ENCODE_MIN_CARD = 2      # feature_builder.py:1364 ile aynı — self-contained (import YOK)
_COUNT_ENCODE_MAX_CARD = 200    # feature_builder.py:1363

_GROUPBY_MAX_CARD       = 50    # feature_builder.py:1365 ile aynı — grup kolonu en fazla bu kart
_GROUPBY_MAX_NUM_COLS   = 3     # feature_builder.py:1366 — patlama korumasi (N sayisal × M kategorik)
_GROUPBY_MAX_CAT_COLS   = 3     # feature_builder.py:1367
_GROUPBY_MIN_GROUP_SIZE = 30    # feature_builder.py:1368 — grup ≥ bu → guvenilir; kucuk grup → global fallback
_GROUPBY_MAX_FEATURES   = 30    # feature_builder.py:1369 — dev+zdev+grpstd sert cap


# ── KANONİK unseen kaydı (TÜM encoder'larda AYNI şekil) ───────
# Katman 2 monitoring bunu TEK yerden okur; her encoder'a ayrı el sokmaz:
#   {"known": [...], "strategy": "<ad>", "monitor": "count_rows_with_unseen",
#    "known_limitation": "<opsiyonel — varsa OOD sentinel uyarısı>"}
# strateji adları: count→"map_to_zero", onehot→"all_zero", target→"prior", groupby→"global".
# Encoder adımları yazıldıkça hepsi bu şekle uyar (onehot/target/groupby sonraki paketler).


class CountEncodeStep(DefaultFitTransform):
    """Kategorik kolonları TRAIN frekansıyla count-encode — parametric / fitted_stat.

    Her uygun kategorik kolon için ``count_<c>`` = o kategorinin TRAIN frekansı. Frekanslar
    YALNIZ ``fit``'e verilen çerçeveden (eğitimde X_train) öğrenilir; ``transform`` yalnız
    haritalar. Görülmemiş kategori → 0 (``map`` ıskası → NaN → ``fillna(0)``).

    ``_count_encode_tt`` (feature_builder.py:1417-1441) mantığının REPLİKASI. SELF-CONTAINED:
    ``feature_builder`` import ETMEZ. YENİ FIX YOK: zaten train-fit → test-map'ti; yalnız imza.
    EŞİK (replika eskime): eski yol REFERANS — değişirse replika + parity birlikte güncellenmeli
    (bkz. RatioFinalizeStep aynı not; Katman 1 terminus'ta risk kalkar).

    KAYITLI KISITLAMA (Faz 0 sentinel'i, DÜZELTİLMEZ — davranış miras, sadık kopya): eğitimde
    count'lar >=1 iken görülmemiş → 0 dağılım-altı bir sentinel; model bu değeri hiç görmemiş
    olabilir (örn. churn'de country test-unseen yok → hiç öğrenmemiş). ``unseen.known_limitation``
    olarak artifact'e yazılır ve unseen-monitor'a bağlanır. Aday gelecek iyileştirme (unseen→
    min_count/mean) A/B'li AYRI iş — port DEĞİL.

    Parametre yapısı (``fit`` dönüşü): ``{f"count_{c}": {source, freq, unseen}}``. ``unseen``
    kanonik şekil (yukarı bkz.) — inference aynı kararı verir, monitoring tek yerden sayar."""

    id = "count_encode"
    type = "parametric"
    param_kind = "fitted_stat"

    def __init__(self, config: StepConfig | None = None):
        # config: {"exclude": [leakage/target], "min_card", "max_card"}
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        exclude = set(self.config.get("exclude") or [])
        min_card = self.config.get("min_card", _COUNT_ENCODE_MIN_CARD)
        max_card = self.config.get("max_card", _COUNT_ENCODE_MAX_CARD)
        params: dict = {}
        for c in X_train.select_dtypes(include="object").columns:
            if c in exclude:
                continue
            card = int(X_train[c].nunique())
            if card < min_card or card > max_card:
                continue
            name = f"count_{c}"
            if name in X_train.columns:
                continue
            freq = X_train[c].value_counts()
            freq_map = {k: int(v) for k, v in freq.items()}
            params[name] = {
                "source": c,
                "freq": freq_map,
                "unseen": {
                    "known": list(freq_map.keys()),      # bilinen değerler seti (= freq anahtarları)
                    "strategy": "map_to_zero",           # görülmemiş → 0
                    "monitor": "count_rows_with_unseen",
                    "known_limitation": (
                        "unseen -> out-of-distribution 0: egitim count'lari >=1, 0 dagilim-alti "
                        "sentinel; model bu degeri gormemis olabilir. Davranis miras (Faz 0), "
                        "sadik kopya. Aday iyilestirme: unseen->min_count (A/B'li ayri is, port degil)."
                    ),
                },
            }
        if params:
            logger.info(f"[count_encode] fit (train-only): {len(params)} kategorik kolon")
        return params

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        X = X.copy()
        for name, spec in (params or {}).items():
            c = spec["source"]
            if c not in X.columns:
                continue
            # map ıskası (görülmemiş kategori) → NaN → 0 (unseen.strategy = map_to_zero)
            X[name] = X[c].map(spec["freq"]).fillna(0).astype("int32")
        return X


class GroupbyDeviationStep(DefaultFitTransform):
    """Grup-istatistik sapma aileleri (dev / zdev / grpstd) — parametric / fitted_stat.

    Her seçili (cat, num) çifti için üç feature (TRAIN-fit → map):
      * ``dev_<num>_by_<cat>``    = num − grup ortalaması (satırın grup-normundan sapması)
      * ``zdev_<num>_by_<cat>``   = (num − grup ort.) / grup std → ölçek-bağımsız sapma (σ=0 → 0)
      * ``grpstd_<num>_by_<cat>`` = grubun std'i (kategorinin iç değişkenliği)

    ``_groupby_deviation_tt`` (feature_builder.py:1444-1544) mantığının REPLİKASI. SELF-CONTAINED:
    ``feature_builder`` import ETMEZ. Encode ÖNCESİ çalışır (ham kategorikleri grup anahtarı yapar);
    PIPELINE'da ``CountEncodeStep`` ile ``decide_encode_route`` arasına düşer. YENİ FIX YOK: zaten
    train-fit → test-map'ti (leakage-safe, target kullanmaz); yalnız fit/transform imzasına porlandı.
    EŞİK/SABİT (replika eskime): eski yol REFERANS — ``_GROUPBY_*`` sabitleri değişirse replika +
    parity birlikte güncellenmeli (bkz. CountEncodeStep aynı not; Katman 1 terminus'ta risk kalkar).

    İKİ-BOYUTLU DONMA (fitted_stat baskın, ama seçim de donar): (1) *seçim* — hangi cat/num, hangi
    çiftler (nunique/norm-var sıralaması + ``MAX_CAT/NUM_COLS`` + ``MAX_FEATURES`` cap + güvenilir-grup
    kapısı) YALNIZ fit-anı hesaplanır ve ``records`` LİSTESİNE gömülür (kayıt varsa çift seçilmiş
    demektir — ayrı ``frozen_choice`` boyutu gerekmez); (2) *istatistik* — ``grp_mean``/``grp_std``/
    ``global_*`` train'den öğrenilir. INVARIANT: inference'ta bir cat'in kardinalitesi/varyansı şişse
    bile seçim DEĞİŞMEZ — ``transform`` yeniden seçmez/ölçmez, yalnız donmuş ``records``'ı replay eder.

    GÜVENİLİR-GRUP KAPISI (``_GROUPBY_MIN_GROUP_SIZE``): grup ≥ N satır → istatistiği güvenilir;
    küçük gruplar (ve görülmeyen kategoriler) GLOBAL istatistiğe düşer. ``grp_mean.where(reliable)``
    unguvenilir grupları fit'te global'e sabitler → ``records``'a öyle donar; ``transform``'da
    ``map(...).fillna(global)`` hem unguvenilir hem unseen'i tek yolda global'e taşır (birebir).

    KAYITLI KISITLAMA (unseen → GLOBAL — count'un ``map_to_zero``'sundan AYRI): görülmemiş kategori
    dağılım-altı bir 0 sentinel'i DEĞİL, o feature'ın train-global ortalaması/std'i olur → nötr,
    OOD-değil. Kanonik unseen şeklinde ``strategy="global"`` (Katman 2 monitör tek yerden sayar).

    Parametre yapısı (``fit`` dönüşü / sıralı ``records`` — çift = seçim kanıtı):
      ``[{cat, num, grp_mean:{katVal:float}, grp_std:{katVal:float}, global_mean, global_std,
          emits:[[family, kolon_adı], ...], unseen:{strategy:"global", monitor, known_limitation}}]``
    """

    id = "groupby_deviation"
    type = "parametric"
    param_kind = "fitted_stat"

    def __init__(self, config: StepConfig | None = None):
        # config: {"exclude": [leakage/target]} — _auto_exclude (count_encode ile AYNI küme)
        self.config: StepConfig = config or {}

    def _select(self, X_train: pd.DataFrame) -> "tuple[list, list]":
        """cat + num seçimi (feature_builder.py:1466-1488 birebir). fit-anı, donar."""
        exclude = set(self.config.get("exclude") or [])
        cat_cols = [c for c in X_train.select_dtypes(include="object").columns
                    if c not in exclude and _GROUPBY_MAX_CARD >= X_train[c].nunique() >= 2]
        num_cols = [c for c in X_train.select_dtypes(include="number").columns
                    if c not in exclude
                    and not c.startswith(("count_", "dev_", "zdev_", "grpstd_", "is_zero_", "ratio_"))
                    and not c.endswith("_isna")]
        if not cat_cols or not num_cols:
            return [], []
        # cat: düşük kardinalite önce (grup başına daha çok satır → stabil ortalama)
        cat_cols = sorted(cat_cols, key=lambda c: X_train[c].nunique())[:_GROUPBY_MAX_CAT_COLS]

        def _norm_var(c: str) -> float:                       # min-max normalize varyans (ölçek-bağımsız)
            s = pd.to_numeric(X_train[c], errors="coerce")
            lo, hi = s.min(), s.max()
            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                return 0.0
            return float(((s - lo) / (hi - lo)).var())
        num_cols = sorted(num_cols, key=lambda c: (-_norm_var(c), c))[:_GROUPBY_MAX_NUM_COLS]
        return cat_cols, num_cols

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> list:
        cat_cols, num_cols = self._select(X_train)
        if not cat_cols or not num_cols:
            return []
        records: list = []
        present = set(X_train.columns)      # _emit çakışma-guard'ı (kolon zaten varsa aile atlanır)
        emitted = 0                         # _GROUPBY_MAX_FEATURES sert cap (eski `added` sayacı)
        unseen = {                          # kanonik unseen — groupby → "global"
            "strategy": "global",
            "monitor": "count_rows_with_unseen",
            "known_limitation": (
                "unseen -> train-global mean/std: dagilim-alti sentinel DEGIL, notr global "
                "istatistik (count'un map_to_zero OOD-sentinel'inden farkli). unguvenilir grup "
                "(<min_size) de ayni global'e duser. Davranis miras (Faz 0), sadik kopya."
            ),
        }
        for cat in cat_cols:
            if emitted >= _GROUPBY_MAX_FEATURES:
                break
            sizes = X_train[cat].value_counts()
            reliable = set(sizes[sizes >= _GROUPBY_MIN_GROUP_SIZE].index)
            if len(reliable) < 2:           # >=2 guvenilir kontrast grubu yoksa feature dejenere → atla
                continue
            cat_tr = X_train[cat]
            for num in num_cols:
                if emitted >= _GROUPBY_MAX_FEATURES:   # cap YALNIZ döngü tepesinde (çift içinde 3 aile beraber)
                    break
                xtr = pd.to_numeric(X_train[num], errors="coerce")
                g = xtr.groupby(cat_tr)
                grp_mean = g.mean()
                grp_std  = g.std(ddof=0)    # ddof=0 → tek-değerli grup NaN değil 0
                global_mean = float(xtr.mean())
                global_std  = float(xtr.std(ddof=0))
                if pd.isna(global_mean):
                    global_mean = 0.0
                if pd.isna(global_std):
                    global_std = 0.0
                # unguvenilir gruplari global'e sabitle (index bazli where) → records'a öyle donar
                grp_mean = grp_mean.where(grp_mean.index.isin(reliable), global_mean)
                grp_std  = grp_std.where(grp_std.index.isin(reliable), global_std)

                emits: list = []
                for fam in ("dev", "zdev", "grpstd"):        # _emit sırası; cap döngü-tepesinde
                    name = f"{fam}_{num}_by_{cat}"
                    if name in present:                       # kolon zaten var → aile atla (_emit guard)
                        continue
                    emits.append([fam, name])
                    present.add(name)
                    emitted += 1
                if not emits:
                    continue
                records.append({
                    "cat": cat, "num": num,
                    "grp_mean": {k: float(v) for k, v in grp_mean.items()},
                    "grp_std":  {k: float(v) for k, v in grp_std.items()},
                    "global_mean": global_mean, "global_std": global_std,
                    "emits": emits, "unseen": unseen,
                })
        if records:
            logger.info(f"[groupby_deviation] fit (train-only): {len(records)} çift → {emitted} feature")
        return records

    def frozen_selection(self, params: list) -> dict:
        """Manifest'e BEYAN: bu adımın dondurduğu YAPISAL seçim (route'larla AYNI şekil).

        Seçim istatistikle birlikte ``records``'a gömülü olduğu için Katman 2'nin
        "bu koşuda hangi çiftler seçildi" sorusunu ancak param_store blob'unu açarak
        cevaplaması gerekirdi; route'larda ise seçim manifest kaleminde duruyordu →
        "seçim nerede" sorusunun cevabı adıma göre DEĞİŞİYORDU. Bu beyan o tutarsızlığı
        kapatır: her donmuş seçim, tipi ne olursa olsun, manifest kaleminin
        ``frozen_choice`` alanında AYNI ``{kind, selection}`` şekliyle görünür
        (``param_store`` yine kaynak; bu yalnız görünürlük katmanı, kopya değil türev)."""
        return {
            "kind": "pairs",
            "selection": [{"group": r["cat"], "value": r["num"],
                           "emits": [name for _, name in r["emits"]]} for r in (params or [])],
        }

    def transform(self, X: pd.DataFrame, params: list) -> pd.DataFrame:
        X = X.copy()
        for rec in (params or []):
            cat, num = rec["cat"], rec["num"]
            if cat not in X.columns or num not in X.columns:   # serving kolonu yoksa atla (reindex 0'lar)
                continue
            xt = pd.to_numeric(X[num], errors="coerce")
            cat_s = X[cat]
            grp_mean = pd.Series(rec["grp_mean"], dtype="float64")
            grp_std  = pd.Series(rec["grp_std"],  dtype="float64")
            gm, gs = rec["global_mean"], rec["global_std"]
            mean_x = cat_s.map(grp_mean).fillna(gm)            # unseen/unguvenilir → global (map ıskası)
            std_x  = cat_s.map(grp_std).fillna(gs)
            for fam, name in rec["emits"]:
                if fam == "dev":
                    X[name] = (xt - mean_x).fillna(0.0)
                elif fam == "zdev":
                    X[name] = (((xt - mean_x) / std_x.replace(0.0, np.nan))
                               .replace([np.inf, -np.inf], np.nan).fillna(0.0))
                else:                                          # grpstd
                    X[name] = std_x.fillna(0.0)
        return X


# ── kardinalite-dallanma kararı (frozen_choice — TRANSFORM DEĞİL) ─────────────
_ENCODE_CARD_THRESHOLD = 10   # feature_builder.py:1204-1205 ile aynı — self-contained (import YOK)


def decide_encode_route(
    X_train: pd.DataFrame,
    exclude: "set | list | None" = None,
    threshold: int = _ENCODE_CARD_THRESHOLD,
) -> dict:
    """Hangi kategorik kolon hangi encoder'a — TRAIN kardinalitesinden bir kez, sonra DONDUR.

    Bu bir TransformStep DEĞİL (veriyi dönüştürmez); yalnızca yapısal bir karar üretir. Eski
    ``_encode_categorical_tt`` (feature_builder.py:1204-1205) bu partisyonu her çağrıda
    ``nunique() <= 10`` ile kardinaliteden HESAPLIYORDU. Burada ``nunique()`` **YALNIZ fit-anı**
    çağrılır; sonuç ``frozen_choice`` olarak artifact'e (``encode_route``) yazılır ve iki
    encoder'ın ``config["columns"]``'una enjekte edilir (RatioFinalizeStep'in
    ``config["columns"]`` desenıyle aynı — yeni mekanizma değil).

    INVARIANT A (donmuş karar): İnference'ta bir kolonun kardinalitesi şişse bile (Geography
    3→8) karar DEĞİŞMEZ — OneHotStep/TargetEncodeStep ``transform``'unda ``nunique`` çağrılmaz;
    kolon donmuş listeden, vocab/encoder donmuş params'tan sürülür. "Karar bir kez, dondur"
    (Faz 0.5). Kardinalite prodüksiyonda yeniden ÖLÇÜLMEZ.

    Dönüş (audit-dostu — "Surname neden target'a gitti" tek yerden görünür):
      ``{"onehot": [...], "target": [...], "threshold": int, "cardinalities": {col: n}}``
    """
    exclude = set(exclude or ())
    cat = [c for c in X_train.select_dtypes(include="object").columns if c not in exclude]
    cardinalities = {c: int(X_train[c].nunique()) for c in cat}   # nunique YALNIZ burada
    onehot = [c for c in cat if cardinalities[c] <= threshold]
    target = [c for c in cat if cardinalities[c] > threshold]
    if cat:
        logger.info(
            f"[encode_route] fit (train-only, donduruldu): onehot={onehot} target={target} "
            f"(threshold={threshold})"
        )
    return {
        "onehot": onehot,
        "target": target,
        "threshold": threshold,
        "cardinalities": cardinalities,
    }


class OneHotStep(DefaultFitTransform):
    """Düşük-kardinalite kategorikleri one-hot (dummy) kodla — parametric / frozen_choice.

    ``_encode_categorical_tt``'nin DÜŞÜK dalının (feature_builder.py:1206-1212) protokol
    karşılığı: ``pd.get_dummies(columns, drop_first=True, dtype=int)`` + eğitimde üretilen
    dummy kolon setini DONDUR; ``transform``'da o vocab'a ``reindex(fill_value=0)``. Bu,
    reindex mekanizmasının ilk gerçek adımı. Lokal reindex = YALNIZ dummy vocab; orkestratörün
    ``final_features`` reindex'i (§2.4 ``run_all_then_reindex``) tüm-çerçeve güvencesidir →
    iki katman (RatioFinalizeStep'teki "reindex yalnız güvence" notuyla tutarlı).

    SELF-CONTAINED: ``feature_builder`` import ETMEZ. Kardinalite kararı BURADA verilmez —
    ``decide_encode_route`` bir kez verir, ``config["columns"]`` ile enjekte edilir; bu adım
    yalnız o donmuş listeyi infaz eder (INVARIANT A). EŞİK (replika eskime): eski yol REFERANS
    — değişirse replika + parity birlikte güncellenmeli (bkz. diğer adımların aynı notu;
    Katman 1 terminus'ta risk kalkar).

    KAYITLI KISITLAMA (D — count'unkinden AYRI, karıştırma): görülmemiş kategori → tüm dummy 0.
    ``drop_first=True`` olduğundan bu, düşürülen REFERANS kategoriyle AYNI kodlama (unseen ≡
    baseline seviye). count'un ``unseen→0``'ı tek boyutta dağılım-altı bir OOD sentinel iken,
    one-hot'un ``all_zero``'ı bir sınıfın YOKLUĞU — OOD sayı değil, temiz yokluk-kodlaması.
    Yani one-hot unseen davranışı count'tan DAHA sağlam; tek sınırı unseen'in referans
    kategoriyle ayırt-edilemezliği (iyi huylu). ``known_limitation`` bunu böyle ayırır.

    Parametre yapısı (``fit`` dönüşü / frozen_choice):
      ``{"columns": [src...], "dummy_vocab": [sıralı dummy kolon adları], "drop_first": True,
         "unseen": {"known": {src: [train kategorileri]}, "strategy": "all_zero",
                    "monitor": "count_rows_with_unseen", "known_limitation": "..."}}``
    """

    id = "onehot_encode"
    type = "parametric"
    param_kind = "frozen_choice"

    def __init__(self, config: StepConfig | None = None):
        # config["columns"] = decide_encode_route(...)["onehot"] (donmuş düşük-kardinalite küme)
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        cols = [c for c in (self.config.get("columns") or []) if c in X_train.columns]
        if not cols:
            return {"columns": [], "dummy_vocab": [], "drop_first": True, "unseen": {}}
        before = list(X_train.columns)
        Xd = pd.get_dummies(X_train, columns=cols, drop_first=True, dtype=int)
        # dummy vocab = üretilen yeni kolonlar (kaynak kolonlar Xd'de artık YOK), sıra korunur
        dummy_vocab = [c for c in Xd.columns if c not in before]
        known = {c: [str(v) for v in X_train[c].dropna().unique().tolist()] for c in cols}
        logger.info(f"[onehot] fit (train-only): {len(cols)} kolon → {len(dummy_vocab)} dummy")
        return {
            "columns": cols,
            "dummy_vocab": dummy_vocab,
            "drop_first": True,
            "unseen": {
                "known": known,                       # per-kaynak-kolon (adım çok-kolonlu)
                "strategy": "all_zero",
                "monitor": "count_rows_with_unseen",
                "known_limitation": (
                    "unseen -> tum dummy 0; drop_first=True ile dusurulen REFERANS kategoriyle "
                    "AYNI kodlama (unseen == baseline seviye). count'un unseen->0'indan FARKLI: "
                    "OOD dagilim-alti sentinel degil, bir sinifin yoklugu (temiz yokluk-kodlamasi). "
                    "Tek sinir: unseen'in referans kategoriyle ayirt-edilemezligi (iyi huylu)."
                ),
            },
        }

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        params = params or {}
        cols = [c for c in (params.get("columns") or []) if c in X.columns]
        vocab = params.get("dummy_vocab") or []
        if not cols:
            return X.copy()
        drop_first = params.get("drop_first", True)
        Xd = pd.get_dummies(X, columns=cols, drop_first=drop_first, dtype=int)
        # Xd = (kaynak-DIŞI taşınan orijinaller) + (üretilen dummy'ler). Kaynak kolonlar Xd'de YOK.
        preserved = [c for c in Xd.columns if c in X.columns]     # dummy-dışı orijinaller
        # dummy'leri donmuş vocab'a hizala: vocab'da olmayan dummy (unseen kategori) DÜŞER;
        # train'de olup burada eksik dummy 0 ile eklenir → unseen ≡ tüm-dummy-0 (all_zero).
        aligned = Xd.reindex(columns=vocab, fill_value=0).astype(int)
        return pd.concat([Xd[preserved], aligned], axis=1)

    def zero_fill_policy(self, params: dict) -> set:
        """Orkestratörün ÜRETİM-KANITI guard'ına beyan: bu kolonlar eksikse 0 DOĞRU cevaptır.

        Donmuş ``dummy_vocab``'ın tamamı: bir dummy serving çerçevesinde yoksa, o kategori
        o partide hiç görülmemiş demektir → all_zero kodlaması TANIMLI politikadır
        (drop_first ile referans kategoriyle aynı kodlama, yukarıdaki KAYITLI KISITLAMA).
        Bu adım zaten kendi ``transform``'unda vocab'a reindex ettiği için pratikte guard'a
        eksik düşmez; beyan SAVUNMA-DERİNLİĞİdir (lokal reindex ileride kalkarsa guard
        haksız yere patlamasın — ve "hangi sıfır meşru" sorusu koda yazılı kalsın)."""
        return set((params or {}).get("dummy_vocab") or [])


class TargetEncodeStep:
    """Yüksek-kardinalite kategorikleri target-encode — parametric / fitted_stat (İLK PICKLE).

    ``_encode_categorical_tt``'nin YÜKSEK dalının (feature_builder.py:1213-1222) protokol
    karşılığı. Katman 1'in İLK sklearn-pickle adımı. Fitted estimator ``PickleRef`` olarak
    saklanır; serving'de ``load_pickle_ref`` TEK sürüm-kapısından geçirir (``pickle_io`` —
    RobustScale/StandardScale ile paylaşılan concern). SELF-CONTAINED (``feature_builder``
    import ETMEZ; sklearn zaten inference bağımlılığı — model de sklearn). Kardinalite kararı
    BURADA verilmez (``decide_encode_route`` bir kez verir, ``config["columns"]`` =
    ``route["target"]``; INVARIANT A).

    ════════════════ E — fit_transform ≠ transform (cross-fitting) ════════════════
    sklearn ``TargetEncoder.fit_transform`` bir CROSS-FITTING şeması uygular (train satırını
    kendi hedefine sızdırmadan kodlar); ``transform`` düz tam-veri encoding'i uygular. İkisi
    EŞİT DEĞİL (sklearn dokümanı birebir). Bu yüzden:
      * ``fit_transform`` (train-only, OVERRIDE — ``DefaultFitTransform`` mixin'i KULLANILMAZ):
        cross-fitted train değerlerini üretir. ``y`` ZORUNLU; None ise ``RuntimeError``
        (fail-closed) → hedefi olmayan serving bağlamı bunu ÇAĞIRAMAZ (E invariant'ı).
      * ``transform`` (serving): düz ``.transform``, görülmemiş → prior (hedef ortalaması).
        İmzada ``y`` YOK → cross-fitting ULAŞILAMAZ. Serving her zaman bu yol.
    Serving-parity (``transform``==``transform``) ve train-parity (``fit_transform``==
    ``fit_transform``) İKİ AYRI iddia — karışmaz.

    Not: ``fit_transform`` sonrası saklanan ``encoder``'ın ``encodings_``'i tam-veriyle
    öğrenilir (cross-fitting yalnız DÖNEN değerleri etkiler) → ``encoder.transform`` serving'de
    ``fit(X,y).transform`` ile birebir aynı. Bu yüzden serving-parity fit/fit_transform yolundan
    bağımsız korunur.

    İKİ FARKLI BAŞARISIZLIK MODU (karıştırma):
      * sklearn fit/transform RUNTIME hatası → graceful DROP (eski yol 1219-1222 ile parity;
        kolon final_features'a hiç girmez). Beklenen, yumuşak.
      * sürüm-uyumsuzluğu (pickle) → KASITLI hard-abort (``load_pickle_ref``, fail-closed).
        Yeni, sert. Bu ikisi AYRI.

    KAYITLI KISITLAMA (D — count'tan farklı, iyi huylu): unseen → prior (ilkeli, hedef
    ortalaması). count'un OOD-sentinel'i DEĞİL.

    Parametre yapısı (``fit``/``fit_transform`` dönüşü): ``{"columns":[...], "ref": PickleRef|None,
    "dropped": bool, "unseen": {"strategy":"prior", ...}}``."""

    id = "target_encode"
    type = "parametric"
    param_kind = "fitted_stat"

    def __init__(self, config: StepConfig | None = None):
        # config["columns"] = decide_encode_route(...)["target"] (donmuş yüksek-kardinalite küme)
        self.config: StepConfig = config or {}

    def _cols(self, X: pd.DataFrame) -> list:
        return [c for c in (self.config.get("columns") or []) if c in X.columns]

    def _empty(self, cols) -> dict:
        return {"columns": cols, "ref": None, "dropped": False, "unseen": {}}

    def _unseen_record(self) -> dict:
        return {
            "known": "sklearn TargetEncoder.categories_ (fitted)",
            "strategy": "prior",
            "monitor": "count_rows_with_unseen",
            "known_limitation": (
                "unseen -> prior (hedef ortalamasi); ilkeli, OOD sentinel DEGIL. count'un "
                "unseen->0 dagilim-alti sentinel'inden farkli (daha guvenli)."
            ),
        }

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        """Encoder'ı düz fit eder (params/serving için). Cross-fitted train DÖNMEZ —
        train çerçevesi için ``fit_transform`` kullanılmalı (orkestratörün train girişi)."""
        cols = self._cols(X_train)
        if not cols or y_train is None:
            return self._empty(cols)
        try:
            enc = TargetEncoder(random_state=settings.RANDOM_SEED).fit(X_train[cols], y_train)
        except Exception as e:   # eski yol 1219-1222 ile parity: sklearn patlarsa kolonları drop
            logger.error(f"[target_encode] fit hatasi → kolonlar drop: {cols}: {e}")
            return {"columns": cols, "ref": None, "dropped": True, "unseen": {}}
        logger.info(f"[target_encode] fit (train-only): {len(cols)} kolon, sklearn {sklearn.__version__}")
        return {"columns": cols, "ref": make_pickle_ref(enc),
                "dropped": False, "unseen": self._unseen_record()}

    def fit_transform(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None):
        """TRAIN-ONLY (E invariant'ı): cross-fitting. ``y`` ZORUNLU — serving (hedefsiz) çağıramaz."""
        if y_train is None:
            raise RuntimeError(
                "TargetEncodeStep.fit_transform ``y`` ZORUNLU (cross-fitting, train-only). "
                "Serving yolu hedef GEÇEMEZ → bu metodu cagiramaz; serving yalniz transform(X, "
                "params) cagirir (E invariant'i: cross-fitting'i produksiyona sizdirma)."
            )
        cols = self._cols(X_train)
        if not cols:
            return X_train.copy(), self._empty(cols)
        X = X_train.copy()
        try:
            enc = TargetEncoder(random_state=settings.RANDOM_SEED)
            X[cols] = enc.fit_transform(X_train[cols], y_train)   # CROSS-FITTED (transform'dan farklı)
        except Exception as e:   # eski yol parity: sklearn patlarsa high kolonları drop
            logger.error(f"[target_encode] fit_transform hatasi → kolonlar drop: {cols}: {e}")
            X = X_train.drop(columns=cols, errors="ignore")
            return X, {"columns": cols, "ref": None, "dropped": True, "unseen": {}}
        logger.info(f"[target_encode] fit_transform (train, cross-fitted): {len(cols)} kolon")
        return X, {"columns": cols, "ref": make_pickle_ref(enc),
                   "dropped": False, "unseen": self._unseen_record()}

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        """SERVING (düz .transform). ``load_pickle_ref`` baştan sürüm-kapısı (fail-closed):
        uyumsuzsa estimator dönmez → hiç skorlamaz."""
        params = params or {}
        cols = [c for c in (params.get("columns") or []) if c in X.columns]
        # graceful-drop yolu (fit'te sklearn patladı) VEYA ref yok → kolonları düşür (eski parity)
        if params.get("dropped") or params.get("ref") is None:
            return X.drop(columns=cols, errors="ignore")
        est = load_pickle_ref(params["ref"])   # abort → SklearnVersionMismatch (skorlamadan önce)
        X = X.copy()
        X[cols] = est.transform(X[cols])   # unseen → prior
        return X


# ═══════════════════ SCALER AİLESİ (inf-impute + route + Robust + Standard) ═══════════════════
# ``_scale_features_tt`` (feature_builder.py:1226-1257) → 4 mantıksal parça (encoder üçlüsüyle
# simetrik: inf-impute önde, sonra route + iki executor). Zincir sırası: inf_impute → robust →
# standard (route ikisini böler). Üçü de DefaultFitTransform (scaler'lar cross-fit YAPMAZ →
# fit_transform == fit+transform; yalnız TargetEncode override eder — mixin/override ayrımı anlamlı).


class InfImputeStep(DefaultFitTransform):
    """Sonsuz (±inf) değerleri train-median ile temizle — parametric / fitted_stat (inline).

    ``_scale_features_tt``'nin inf-temizlik bölümünün (feature_builder.py:1232-1244) protokol
    karşılığı. DEĞER zaten train-fit'ti (``med = X_train[inf_cols].median()``) → #4-tipi
    değer-sızıntısı DEĞİL; sadık-yeniden-yazım, A/B YOK.

    TEK FARK — inf_cols SEÇİM peek'i KAPATILDI (selection-sınıfı, §2.3). Eski yol ``inf_cols``'u
    ``isinf(X_train) OR isinf(X_test)`` ile seçiyordu (hangi kolonların temizleneceği test'e
    bakıyordu). ``fit`` yalnız X_train gördüğü için burada ``inf_cols`` YALNIZ train'den → peek
    imza düzeyinde temsil edilemez (impute #4'ün aynı disiplini). KASITLI IRAKSAMA: inf yalnız
    prod'da gelen "öğrenilmemiş" kolon eski yolda temizlenirdi, burada median öğrenilmez.

    INFERENCE SONUCU TANIMLI (peek kapatmanın postcondition'ı): ``transform`` çıkışında HİÇBİR
    kolonda inf KALMAZ. Öğrenilmiş inf_cols → inf→nan→train-median→fillna(0.0). Öğrenilmemiş
    kolonda prod-inf → 0.0 (yalnız inf; nan'lara dokunulmaz — onlar ImputeStep'in işi). Böylece
    ham inf downstream scaler'a geçip modeli PATLATMAZ. (Tüm-kolon median öğrenmek — daha akıllı
    fallback — ileri sağlamlaştırma; ImputeStep'teki aynı kapsam notu.)

    Parametre yapısı (``fit`` dönüşü): ``{"inf_cols": [train'de inf'li], "median": {col: train_med}}``.
    ``train_med`` NaN olabilir (train'de kolon tümü inf) → transform'da fillna(0.0) yakalar."""

    id = "inf_impute"
    type = "parametric"
    param_kind = "fitted_stat"

    def __init__(self, config: StepConfig | None = None):
        self.config: StepConfig = config or {}

    @staticmethod
    def _has_inf(s: pd.Series) -> bool:
        return bool(np.isinf(pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")).any())

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        num = X_train.select_dtypes(include="number").columns
        inf_cols = [c for c in num if self._has_inf(X_train[c])]     # YALNIZ train (peek kapalı)
        median: dict = {}
        for c in inf_cols:
            s = X_train[c].replace([np.inf, -np.inf], np.nan)
            median[c] = float(s.median())    # train-only; all-inf kolonda NaN → transform 0.0'lar
        if inf_cols:
            logger.info(f"[inf_impute] fit (train-only): {len(inf_cols)} inf kolonu")
        return {"inf_cols": inf_cols, "median": median}

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        X = X.copy()
        params = params or {}
        inf_cols = params.get("inf_cols") or []
        median = params.get("median") or {}
        learned = set(inf_cols)
        # öğrenilmiş inf_cols: inf→nan→train-median→0.0 (eski yol 1239-1243 birebir)
        for c in inf_cols:
            if c not in X.columns:
                continue
            s = X[c].replace([np.inf, -np.inf], np.nan)
            m = median.get(c)
            if m is not None and not np.isnan(m):
                s = s.fillna(m)
            X[c] = s.fillna(0.0)
        # öğrenilmemiş kolonlarda prod-inf → 0.0 (finite postcondition; nan'a DOKUNMA)
        for c in X.select_dtypes(include="number").columns:
            if c in learned:
                continue
            if self._has_inf(X[c]):
                X[c] = X[c].replace([np.inf, -np.inf], 0.0)
        return X


def decide_scaler_route(X_train: pd.DataFrame, outlier_cols: "set | list | None" = None) -> dict:
    """Hangi numerik kolon hangi scaler'a — TRAIN'de bir kez, sonra DONDUR (``decide_encode_route``
    kardeşi). ``outlier_cols``'li kolonlar RobustScaler, diğerleri StandardScaler.

    ROUTE TUTARLILIĞI (orkestratöre köprü): ``decide_encode_route`` ile AYNI frozen_choice deseni —
    executor-anahtarlı kolon listeleri (``robust``/``standard`` ~ ``onehot``/``target``) + audit alanı
    (``outlier_cols`` ~ ``cardinalities``). Executor'lar ``config["columns"] = route[<kendi anahtarı>]``
    alır. Orkestratör iki route'u TEK desenle ele alabilsin diye şekil bilinçli olarak paralel.

    INVARIANT (encode_route'un aynısı): ``num`` ve outlier profili YALNIZ fit-anı ölçülür; sonuç
    donar. Inference'ta bir kolonun outlier profili değişse bile scaler seçimi DEĞİŞMEZ (executor
    transform'unda outlier yeniden ölçülmez). NOT: ``outlier_cols`` kendisi tüm-df profilinden gelir
    (üst-katman §2.3 selection-peek, bu adımda düzeltilmez); ama route KARARI dondurulunca
    deployment-doğru."""
    outlier_set = set(outlier_cols or ())
    num = X_train.select_dtypes(include="number").columns.tolist()   # YALNIZ fit-anı
    robust = [c for c in num if c in outlier_set]
    standard = [c for c in num if c not in outlier_set]
    if num:
        logger.info(f"[scaler_route] fit (train-only, donduruldu): robust={len(robust)} standard={len(standard)}")
    return {"robust": robust, "standard": standard, "outlier_cols": list(outlier_set)}


class _ScalerStepBase(DefaultFitTransform):
    """Robust/Standard ortak gövdesi — sklearn scaler'ı ``PickleRef`` olarak sakla, serving'de
    ``load_pickle_ref`` tek kapıdan geç. Alt sınıf yalnız ``id`` + ``_scaler_cls``'i belirler
    (iki AYRI Step; ortak gövde DRY için). Kardinalite/outlier kararı BURADA değil — route bir
    kez verir, ``config["columns"]`` enjekte edilir (INVARIANT: donmuş atama uygulanır)."""

    type = "parametric"
    param_kind = "fitted_stat"
    _scaler_cls: type = None   # alt sınıf doldurur

    def __init__(self, config: StepConfig | None = None):
        # config["columns"] = decide_scaler_route(...)["robust" | "standard"]
        self.config: StepConfig = config or {}

    def _cols(self, X: pd.DataFrame) -> list:
        return [c for c in (self.config.get("columns") or []) if c in X.columns]

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        cols = self._cols(X_train)
        if not cols:
            return {"columns": [], "ref": None}
        sc = self._scaler_cls().fit(X_train[cols])         # train-fit
        logger.info(f"[{self.id}] fit (train-only): {len(cols)} kolon, sklearn {sklearn.__version__}")
        return {"columns": cols, "ref": make_pickle_ref(sc)}

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        params = params or {}
        cols = [c for c in (params.get("columns") or []) if c in X.columns]
        if not cols or params.get("ref") is None:
            return X.copy()
        est = load_pickle_ref(params["ref"])   # abort → SklearnVersionMismatch (skorlamadan önce)
        X = X.copy()
        X[cols] = est.transform(X[cols])
        return X


class RobustScaleStep(_ScalerStepBase):
    """Outlier'lı kolonları RobustScaler ile ölçekle — parametric / fitted_stat (PICKLE #2).
    ``_scale_features_tt`` 1247-1251 replikası. Kolonlar ``route["robust"]``'tan (donmuş)."""

    id = "robust_scale"
    _scaler_cls = RobustScaler


class StandardScaleStep(_ScalerStepBase):
    """Outlier'sız kolonları StandardScaler ile ölçekle — parametric / fitted_stat (PICKLE #3).
    ``_scale_features_tt`` 1252-1256 replikası. Kolonlar ``route["standard"]``'tan (donmuş)."""

    id = "standard_scale"
    _scaler_cls = StandardScaler


class SelectDropStep(DefaultFitTransform):
    """Feature seçimi/leakage → TEK donmuş drop-liste — parametric / frozen_choice. TERMINUS (#3).

    Katman 1'de dört selection alt-operasyonu (low-var/high-corr + supervised univariate + leakage
    tespiti + domain-leakage) serving'de AYNI şeye indirger: donmuş bir "şu kolonları düşür" listesi.
    Hiçbir tespit mantığı inference'ta çalışmaz — hepsi fit-anı (eğitim). Bu adım o gerçeği kodlar:

      * ``fit`` (YALNIZ eğitim) — mevcut ``feature_builder`` selection dizisini (build_features:2433-2447)
        X_train üzerinde SIRAYLA çalıştırır (LAZY import; onaylı DELEGE — replika DEĞİL → drift-sıfır,
        aynı kodu çağırır, parity birebir kesin). Hayatta-kalan set çıkar → ``drop_cols`` DONAR.
      * ``transform`` (eğitim + serving AYNI) — donmuş ``drop_cols``'u düşürür + terminal numeric
        cleanup (``to_numeric``→``fillna(0)``, build_features:2435). SELF-CONTAINED: ``feature_builder``
        import ETMEZ → inference runtime yalın kalır (delege yalnız ``fit``'te, eğitimde).

    DELEGE gerekçesi (§handoff select_drop çatalı, onaylı): transform pür-drop → learned numeric state
    YOK, yalnız kolon-listesi gerekir; "yalın inference runtime" hedefi transform yolu içindir, fit için
    değil. 122-satırlık ``_detect_leakage`` sezgisini serving'de HİÇ çalışmadığı halde birebir yeniden-yazmak
    salt drift riski. Terminus kararı zaten "eski ``_tt``'ler delege → tek implementasyon" diyordu; bu onu
    selection için erken uygular (uzlaştırılacak ikiz yok).

    INVARIANT KAPSAMI (dikkat — bu adım tek istisna DEĞİL, kuralın DARALTILMASIDIR): "her Step
    SELF-CONTAINED" kuralı artık "her **transform** SELF-CONTAINED" olarak okunur; ``fit``
    eğitim-anı olduğu için delege edebilir. Tam gerekçe + yeni adımlar için ölçüt:
    ``protocol.py`` → "SELF-CONTAINED invariant'ının KAPSAMI". Buradaki delege o kapsamın
    İÇİNDEdir; panik gerektirmez, ama başka bir adımda bahane olarak da kullanılamaz
    (ölçüt: transform'u delegesiz yazabiliyor musun?).

    TERMINUS INVARIANT (#3): PIPELINE'da EN SON — hiçbir fitted_stat ondan SONRA gelemez, aksi halde
    o adımın öğrendiği state seçimden geçmemiş kolonlara bağlanırdı. ``final_features`` = bu adımın çıktısı.
    ``re_derive:false`` — seçim serving'de YENİDEN hesaplanmaz (bir kolon inference'ta düşük-varyanslı/
    leaky görünse bile karar donmuş; count/scaler'daki "karar bir kez, dondur" ile aynı ilke).

    Parametre yapısı (``fit`` dönüşü / frozen_choice):
      ``{"drop_cols": [orijinal sırada düşenler], "provenance": {kolon: alt-detektör},
         "surviving": [hayatta kalanlar]}``. ``provenance`` audit-dostu ("X neden düştü" tek yerden).
    """

    id = "select_drop"
    type = "parametric"
    param_kind = "frozen_choice"

    def __init__(self, config: StepConfig | None = None):
        # config: {"intent", "meta", "domain_leakage_report", "protected_high_card"} — ctx'ten
        self.config: StepConfig = config or {}

    def fit(self, X_train: pd.DataFrame, y_train: "pd.Series | None" = None) -> dict:
        # LAZY + LOKAL import: steps.py modül-yüklemesi feature_builder'ı ÇEKMEZ (inference import yalın);
        # yalnız eğitimde fit çağrılınca yüklenir. DELEGE — build_features:2433-2447 dizisi birebir.
        from src.services.feature_builder import (
            _apply_domain_leakage_drops_tt, _detect_leakage_tt, _select_features_tt,
            _select_univariate_tt,
        )
        cfg = self.config
        intent = (cfg.get("intent") or "").strip().lower()
        meta = cfg.get("meta")
        domain_report = cfg.get("domain_leakage_report") or {"dropped": [], "suspected": []}
        protected = cfg.get("protected_high_card")
        if protected is None:
            # T2 — `ts_protected_cols` TÜRETİMİ (plumbing yerine yapısal denklik).
            # build_features'ta bu küme = ratio_finalize sonrası HAYATTA KALAN oran kolonları
            # ∪ ts-lag kolonları. ts-lag `_SCOPE_RULES` tarafından REDDEDİLİYOR → desteklenen
            # kapsamda küme = oran adları ∩ mevcut çerçeve. Türetim fit GİRİŞİNDE yapılır
            # (aşağıdaki drop'lardan ÖNCE) — legacy'nin hesap anıyla aynı çerçeve içeriği.
            # Kilit: `test_protected_high_card_derivation_matches_legacy`.
            protected = {n for n in (cfg.get("ratio_names") or []) if n in X_train.columns}

        Xtr = X_train.copy()
        prov: dict = {}
        # Zincirin yan-değerleri: build_features `feature_report`'a yazmak için bunlara İHTİYAÇ
        # DUYAR (leakage_dropped/leakage_suspected/univariate_dropped). Eskiden burada sessizce
        # atılıyorlardı → T2'de zincir devralınınca rapor alanları boşalırdı. `transform`
        # DEĞİŞMEZ (yalnız `drop_cols` okur) → davranış-nötr ekleme.
        _uni: list = []
        _rep: dict = {"dropped": [], "suspected": []}

        def _diff(prev_cols: set, tag: str) -> None:
            for c in prev_cols - set(Xtr.columns):
                prov.setdefault(c, tag)

        # 1. low-var + high-corr (unsupervised) — build_features:2433
        prev = set(Xtr.columns)
        Xtr, _ = _select_features_tt(Xtr, Xtr.copy())
        _diff(prev, "low_var_or_high_corr")
        # numeric cleanup (ANOVA NaN kabul etmez) — build_features:2435
        Xtr = Xtr.apply(pd.to_numeric, errors="coerce").fillna(0)
        # 2. supervised univariate (yalnız cls/reg + y var) — build_features:2439-2441
        if intent in ("classification", "regression") and y_train is not None:
            prev = set(Xtr.columns)
            Xtr, _, _uni = _select_univariate_tt(Xtr, Xtr.copy(), y_train, intent)
            _diff(prev, "univariate")
        # 3. leakage tespiti — build_features:2443. y guard: build_features'ta selection tail'i
        # YALNIZ supervised dalda (y hep var); y=None (unsupervised) → leakage atla (sadık + sağlam).
        if y_train is not None:
            prev = set(Xtr.columns)
            Xtr, _, _rep = _detect_leakage_tt(Xtr, Xtr.copy(), y_train, meta, protected_high_card=protected)
            _diff(prev, "leakage")
        # 4. domain-leakage drop'ları — build_features:2445
        prev = set(Xtr.columns)
        Xtr, _, _dom = _apply_domain_leakage_drops_tt(Xtr, Xtr.copy(), domain_report)
        _diff(prev, "domain_leakage")

        surviving = set(Xtr.columns)
        drop_cols = [c for c in X_train.columns if c not in surviving]   # orijinal sıra (deterministik)
        if drop_cols:
            logger.info(f"[select_drop] fit (train-only, DONDU): {len(drop_cols)} kolon düştü → {prov}")
        return {"drop_cols": drop_cols, "provenance": prov, "surviving": list(Xtr.columns),
                # ── zincir yan-değerleri (rapor girdisi; transform bunları OKUMAZ) ──
                "leakage_report": _rep, "univariate_dropped": _uni, "domain_dropped": _dom,
                "protected_high_card": sorted(protected or ())}

    def transform(self, X: pd.DataFrame, params: dict) -> pd.DataFrame:
        params = params or {}
        X = X.copy()
        drop = [c for c in (params.get("drop_cols") or []) if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
        # terminal numeric cleanup (build_features:2435) — model tam-numeric fillna(0) girdi ister;
        # idempotent → fit çıktısı (dizide fillna edilmiş) ile birebir (serving == training).
        return X.apply(pd.to_numeric, errors="coerce").fillna(0)
