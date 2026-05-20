from __future__ import annotations

"""
Feature Engineering Node — Tamamen Dinamik
-------------------------------------------
Hiçbir analiz tipi, kolon ismi veya domain hardcode edilmemiştir.
Tüm kararlar şu iki kaynaktan gelir:

1. feature_plan  → LLM'in ürettiği plan (ne yapılacak)
2. db_intelligence → intelligence_engine metadata'sı (kolonlar ne anlama geliyor)

Desteklenen senaryolar:
- Hedef değişken mevcut (income_binary gibi) → direkt kullan
- Hedef değişken türetilecek (churn gibi) → time_based / threshold_based türet
- Aggregation gerekli (müşteri bazında) → group_by ile aggregate et
- Aggregation gereksiz (tek tablo, her satır bağımsız) → ham veri kullan
- Tarih kolonu var → temporal feature'lar üret
- Tarih kolonu yok → atla
- Kategorik kolonlar → kardinaliteye göre one-hot / target encoding
- Sayısal kolonlar → outlier durumuna göre RobustScaler / StandardScaler

Pipeline:
    1. feature_plan'ı oku → ne yapılacak?
    2. db_intelligence'dan kolon metadata'sını oku → kolonlar ne?
    3. Veri temizleme (duplicate, imputation)
    4. Hedef değişkeni belirle veya türet (plan'a göre)
    5. Aggregation (plan gerekli diyorsa)
    6. Encoding (intelligence_engine semantic_type'a göre)
    7. Scaling (profiling outlier bilgisine göre)
    8. Feature selection (variance + correlation)
    9. Train / test split
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, TargetEncoder

from src.state import AgentState

warnings.filterwarnings("ignore")

CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 0.01
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────
# INTELLIGENCE ENGINE METADATA READER
# ─────────────────────────────────────────────────────────────────

class ColumnMetadata:
    """
    db_intelligence + profiling_report'u birleştirip
    kolon bazında metadata sağlar.
    """

    def __init__(self, db_intelligence: dict, profiling_report: dict):
        self.columns: dict[str, dict] = {}
        self._build(db_intelligence, profiling_report)

    def _build(self, db_intelligence: dict, profiling_report: dict) -> None:
        profiling_cols = profiling_report.get("columns", {})
        for col_info in db_intelligence.get("columns", []):
            col_name = col_info.get("column_name", "")
            if not col_name:
                continue
            profile = profiling_cols.get(col_name, {})
            self.columns[col_name] = {
                "semantic_type": col_info.get("semantic_type", "unknown"),
                "importance_score": col_info.get("importance_score", 0.5),
                "has_outliers": profile.get("has_outliers", False),
                "null_rate": profile.get("null_rate", 0.0),
                "unique_count": profile.get("unique_count", 0),
                "dtype": profile.get("dtype", "object"),
            }

    def get_by_type(self, semantic_type: str) -> list[str]:
        return [c for c, i in self.columns.items() if i.get("semantic_type") == semantic_type]

    def get_outlier_cols(self) -> list[str]:
        return [c for c, i in self.columns.items() if i.get("has_outliers")]

    def summary(self) -> dict:
        return {
            "identifiers": self.get_by_type("identifier"),
            "timestamps": self.get_by_type("timestamp"),
            "metrics": self.get_by_type("metric"),
            "categories": self.get_by_type("category"),
            "targets": self.get_by_type("target"),
        }


# ─────────────────────────────────────────────────────────────────
# STEP 1 — Veri temizleme
# ─────────────────────────────────────────────────────────────────

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[feature_eng] {dropped} duplicate temizlendi")

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])

    null_total = df.isnull().sum().sum()
    print(f"[feature_eng] Temizlik sonrası: {len(df)} satır | kalan null: {null_total}")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Hedef değişken belirleme / türetme
# ─────────────────────────────────────────────────────────────────

def _resolve_target(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    feature_plan'a göre hedef değişkeni belirler veya türetir.
    Hiçbir analiz tipi hardcode değildir.

    Returns:
        df: hedef kolonu çıkarılmış DataFrame
        y: hedef değişken Series
        target_name: hedef kolonun adı
    """
    target = plan.get("target", {})
    target_name = target.get("column_name", "")
    is_derived = target.get("is_derived", False)
    derivation_type = target.get("derivation_type", "none")

    # ── Hedef mevcut → direkt kullan ─────────────────────────────
    if not is_derived and target_name in df.columns:
        y = df[target_name].copy()

        # String label'ı binary'ye çevir (gerekirse)
        if y.dtype == object:
            unique_vals = y.unique()
            if len(unique_vals) == 2:
                print(f"[feature_eng] Hedef '{target_name}' binary encode ediliyor: {unique_vals}")
                y = pd.Categorical(y).codes
            else:
                y = pd.Categorical(y).codes

        # Aynı bilgiyi taşıyan kolonları drop et
        cols_to_drop = _find_related_target_cols(df, target_name)
        df = df.drop(columns=cols_to_drop, errors="ignore")

        print(f"[feature_eng] Hedef: '{target_name}' (mevcut) | "
              f"sınıf dağılımı: {pd.Series(y).value_counts().to_dict()}")
        return df, pd.Series(y, name=target_name), target_name

    # ── Hedef türetilecek → plan'a göre türet ────────────────────
    if is_derived and derivation_type == "time_based":
        return _derive_time_based_target(df, target, plan)

    if is_derived and derivation_type == "threshold_based":
        return _derive_threshold_target(df, target)

    # Fallback — hedef bulunamadı, en iyi adayı bul
    print(f"[feature_eng] UYARI: '{target_name}' bulunamadı, aday aranıyor...")
    return _find_best_target_candidate(df)


def _find_related_target_cols(df: pd.DataFrame, target_name: str) -> list[str]:
    """
    Hedef değişkenle aynı bilgiyi taşıyan kolonları bulur.
    Örn: income ve income_binary ikisi de aynı şey → birini drop et.
    """
    cols_to_drop = [target_name]
    target_stem = target_name.replace("_binary", "").replace("_label", "").replace("_target", "")

    for col in df.columns:
        if col == target_name:
            continue
        col_stem = col.replace("_binary", "").replace("_label", "").replace("_target", "")
        if col_stem == target_stem:
            cols_to_drop.append(col)

    return cols_to_drop


def _derive_time_based_target(
    df: pd.DataFrame,
    target: dict,
    plan: dict,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Tarih kolonundan time-based hedef değişken türetir.
    Örn: Churn — son N günde sipariş vermemiş müşteri.
    """
    target_name = target.get("column_name", "derived_target")
    date_col = target.get("date_column", "")
    threshold_days = target.get("threshold_days", 0)
    group_by = plan.get("aggregation", {}).get("group_by_column", "")

    if date_col not in df.columns:
        print(f"[feature_eng] UYARI: Tarih kolonu '{date_col}' bulunamadı")
        return _find_best_target_candidate(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if plan.get("use_data_split_for_label", False) or threshold_days == 0:
        reference_date = df[date_col].quantile(0.80)
    else:
        reference_date = df[date_col].max()
        reference_date = reference_date - pd.Timedelta(days=threshold_days)

    if group_by and group_by in df.columns:
        last_activity = df.groupby(group_by)[date_col].max().reset_index()
        last_activity[target_name] = (last_activity[date_col] < reference_date).astype(int)
        df = df.merge(last_activity[[group_by, target_name]], on=group_by, how="left")
    else:
        df[target_name] = (df[date_col] < reference_date).astype(int)

    y = df[target_name].copy()
    df = df.drop(columns=[target_name, date_col], errors="ignore")

    print(f"[feature_eng] Hedef: '{target_name}' (time_based türetildi) | "
          f"dağılım: {y.value_counts().to_dict()}")
    return df, y, target_name


def _derive_threshold_target(
    df: pd.DataFrame,
    target: dict,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Sayısal eşikten binary hedef değişken türetir.
    Örn: revenue > 1000 → high_revenue = 1
    """
    target_name = target.get("column_name", "derived_target")
    threshold_value = target.get("threshold_value", 0.0)

    # threshold uygulanacak kolonu bul
    source_col = target_name.replace("_binary", "").replace("_label", "")
    if source_col not in df.columns:
        return _find_best_target_candidate(df)

    y = (df[source_col] > threshold_value).astype(int)
    df = df.drop(columns=[source_col], errors="ignore")

    print(f"[feature_eng] Hedef: '{target_name}' (threshold_based türetildi) | "
          f"eşik: {threshold_value} | dağılım: {y.value_counts().to_dict()}")
    return df, pd.Series(y, name=target_name), target_name


def _find_best_target_candidate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Fallback: binary veya low-cardinality sayısal kolon arar.
    """
    for col in df.columns:
        if df[col].nunique() == 2 and df[col].dtype in [np.int64, np.float64, int, float]:
            y = df[col].copy()
            df = df.drop(columns=[col])
            print(f"[feature_eng] Fallback hedef: '{col}' (binary kolon bulundu)")
            return df, y, col

    raise ValueError("Hedef değişken bulunamadı ve türetilemedi.")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — Aggregation (plan gerekli diyorsa)
# ─────────────────────────────────────────────────────────────────

def _aggregate_if_needed(
    df: pd.DataFrame,
    y: pd.Series,
    plan: dict,
    meta: ColumnMetadata,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    feature_plan aggregation=True diyorsa müşteri/ürün bazında aggregate eder.
    aggregation=False ise hiçbir şey yapmaz.
    """
    agg_config = plan.get("aggregation", {})
    if not agg_config.get("required", False):
        print(f"[feature_eng] Aggregation gerekmez — ham veri kullanılıyor")
        return df, y

    group_col = agg_config.get("group_by_column", "")
    if not group_col or group_col not in df.columns:
        print(f"[feature_eng] UYARI: group_by kolonu '{group_col}' bulunamadı, aggregation atlanıyor")
        return df, y

    level = agg_config.get("level", "none")
    feature_groups = plan.get("feature_groups", [])

    print(f"[feature_eng] Aggregation: {level} bazında, group_by='{group_col}'")

    # Identifier kolonlarını bul
    id_cols = meta.get_by_type("identifier")
    metric_cols = [c for c in meta.get_by_type("metric") if c in df.columns and c != group_col]
    timestamp_cols = [c for c in meta.get_by_type("timestamp") if c in df.columns]
    category_cols = [c for c in meta.get_by_type("category") if c in df.columns]

    # Tarih parse
    reference_date = None
    date_col = timestamp_cols[0] if timestamp_cols else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        reference_date = df[date_col].quantile(0.80) if plan.get("use_data_split_for_label") else df[date_col].max()

    # Data leakage: geçmiş veriyi kullan
    if date_col and reference_date and plan.get("use_data_split_for_label"):
        agg_df = df[df[date_col] < reference_date].copy()
    else:
        agg_df = df.copy()

    # Temel aggregation
    agg_dict: dict[str, Any] = {}

    # Metric kolonları
    for col in metric_cols:
        if col in agg_df.columns:
            agg_dict[f"sum_{col}"] = (col, "sum")
            agg_dict[f"avg_{col}"] = (col, "mean")
            agg_dict[f"max_{col}"] = (col, "max")
            agg_dict[f"std_{col}"] = (col, "std")

    # ID kolonları → count (frequency)
    for col in id_cols:
        if col in agg_df.columns and col != group_col:
            agg_dict[f"count_{col}"] = (col, "nunique")

    customer_df = agg_df.groupby(group_col).agg(**agg_dict).reset_index()

    # RFM feature'lar (plan'da istendiyse)
    if "rfm" in feature_groups and date_col and date_col in agg_df.columns:
        recency = agg_df.groupby(group_col)[date_col].max().reset_index()
        recency["recency_days"] = (reference_date - recency[date_col]).dt.days
        customer_df = customer_df.merge(recency[[group_col, "recency_days"]], on=group_col, how="left")

    # Temporal trend (plan'da istendiyse)
    if "temporal_trend" in feature_groups and date_col and metric_cols:
        amount_col = metric_cols[0]
        mid_point = agg_df[date_col].quantile(0.50)
        recent = agg_df[agg_df[date_col] >= mid_point].groupby(group_col)[amount_col].sum().reset_index().rename(columns={amount_col: "spend_recent"})
        old = agg_df[agg_df[date_col] < mid_point].groupby(group_col)[amount_col].sum().reset_index().rename(columns={amount_col: "spend_old"})
        customer_df = customer_df.merge(recent, on=group_col, how="left")
        customer_df = customer_df.merge(old, on=group_col, how="left")
        customer_df["spend_recent"] = customer_df["spend_recent"].fillna(0)
        customer_df["spend_old"] = customer_df["spend_old"].fillna(0)
        customer_df["spend_trend"] = np.where(
            customer_df["spend_old"] > 0,
            customer_df["spend_recent"] / customer_df["spend_old"], 0.0
        )

    # Behavioral (plan'da istendiyse)
    if "behavioral" in feature_groups and category_cols:
        for cat_col in category_cols[:2]:
            if cat_col in agg_df.columns:
                diversity = agg_df.groupby(group_col)[cat_col].nunique().reset_index().rename(columns={cat_col: f"{cat_col}_diversity"})
                fav = agg_df.groupby(group_col)[cat_col].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index().rename(columns={cat_col: f"fav_{cat_col}"})
                customer_df = customer_df.merge(diversity, on=group_col, how="left")
                customer_df = customer_df.merge(fav, on=group_col, how="left")

    # y'yi group_by bazında al (ilk değer)
    y_df = pd.DataFrame({"__y": y, group_col: df[group_col]})
    y_grouped = y_df.groupby(group_col)["__y"].first().reset_index()
    customer_df = customer_df.merge(y_grouped, on=group_col, how="left")

    y_out = customer_df["__y"].copy()
    customer_df = customer_df.drop(columns=["__y", group_col], errors="ignore")
    customer_df = customer_df.fillna(0)

    print(f"[feature_eng] Aggregation sonucu: {len(customer_df)} satır | {len(customer_df.columns)} kolon")
    return customer_df, y_out


# ─────────────────────────────────────────────────────────────────
# STEP 4 — Drop identifier ve timestamp kolonları
# ─────────────────────────────────────────────────────────────────

def _drop_non_feature_cols(df: pd.DataFrame, meta: ColumnMetadata) -> pd.DataFrame:
    """Identifier ve timestamp kolonları model için gereksiz — drop et."""
    drop_cols = []
    for col in df.columns:
        info = meta.columns.get(col, {})
        sem = info.get("semantic_type", "")
        if sem in ("identifier", "timestamp"):
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        print(f"[feature_eng] Drop (identifier/timestamp): {drop_cols}")

    return df


# ─────────────────────────────────────────────────────────────────
# STEP 5 — Encoding
# ─────────────────────────────────────────────────────────────────

def _encode_categorical(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Object/string kolonlarını encode eder.
    Kardinaliteye göre otomatik karar:
    - ≤ 10 unique → one-hot
    - > 10 unique → target encoding (sklearn, leakage-safe)
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not cat_cols:
        return df

    low_card = [c for c in cat_cols if df[c].nunique() <= 10]
    high_card = [c for c in cat_cols if df[c].nunique() > 10]

    if low_card:
        df = pd.get_dummies(df, columns=low_card, drop_first=True, dtype=int)
        print(f"[feature_eng] One-hot encoding ({len(low_card)} kolon): {low_card}")

    if high_card:
        try:
            te = TargetEncoder(random_state=RANDOM_STATE)
            df[high_card] = te.fit_transform(df[high_card], y)
            print(f"[feature_eng] Target encoding ({len(high_card)} kolon): {high_card}")
        except Exception as e:
            print(f"[feature_eng] Target encoding hatası, one-hot'a geçiliyor: {e}")
            df = pd.get_dummies(df, columns=high_card, drop_first=True, dtype=int)

    return df


# ─────────────────────────────────────────────────────────────────
# STEP 6 — Scaling
# ─────────────────────────────────────────────────────────────────

def _scale_features(df: pd.DataFrame, outlier_cols: list[str]) -> pd.DataFrame:
    """
    Sayısal kolonları ölçekler.
    Outlier olan → RobustScaler, diğerleri → StandardScaler
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df

    robust = [c for c in outlier_cols if c in numeric_cols]
    standard = [c for c in numeric_cols if c not in robust]

    if robust:
        df[robust] = RobustScaler().fit_transform(df[robust])
        print(f"[feature_eng] RobustScaler: {len(robust)} kolon")

    if standard:
        df[standard] = StandardScaler().fit_transform(df[standard])
        print(f"[feature_eng] StandardScaler: {len(standard)} kolon")

    return df


# ─────────────────────────────────────────────────────────────────
# STEP 7 — Feature selection
# ─────────────────────────────────────────────────────────────────

def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df

    # Variance threshold
    low_var = df[numeric_cols].var()[lambda s: s < VARIANCE_THRESHOLD].index.tolist()
    if low_var:
        df = df.drop(columns=low_var)
        numeric_cols = [c for c in numeric_cols if c not in low_var]
        print(f"[feature_eng] Düşük varyans drop: {low_var}")

    # Correlation threshold
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
        if high_corr:
            df = df.drop(columns=high_corr)
            print(f"[feature_eng] Yüksek korelasyon drop: {high_corr}")

    return df


# ─────────────────────────────────────────────────────────────────
# MAIN NODE
# ─────────────────────────────────────────────────────────────────

def feature_engineering_node(state: AgentState) -> AgentState:
    """
    Node 5: Tamamen dinamik feature engineering.

    Hiçbir analiz tipi, kolon ismi veya domain hardcode edilmemiştir.
    Tüm kararlar feature_plan (LLM) ve db_intelligence (metadata) tarafından yönetilir.

    State girdileri:
        sql_results      : Execution node'dan gelen ham veri
        profiling_report : Kolon istatistikleri
        db_intelligence  : intelligence_engine metadata'sı
        feature_plan     : LLM'in ürettiği analiz planı

    State çıktıları:
        feature_report   : Yapılan dönüşümlerin özeti
        X_train, X_test  : Eğitim ve test feature'ları
        y_train, y_test  : Eğitim ve test label'ları
    """
    sql_results = state.get("sql_results", [])
    profiling_report = state.get("profiling_report", {})
    db_intelligence = state.get("db_intelligence", {})
    feature_plan = state.get("feature_plan", {})

    if not sql_results:
        return {**state, "error": "Veri yok.", "current_step": "feature_engineering_node"}
    if not feature_plan:
        return {**state, "error": "Feature plan yok — feature_plan_node çalıştı mı?",
                "current_step": "feature_engineering_node"}

    print(f"\n{'='*60}")
    print(f"[feature_eng] BAŞLIYOR — Dinamik Feature Engineering")
    print(f"{'='*60}")
    print(f"[feature_eng] Analiz: {feature_plan.get('analysis_summary', '?')}")
    print(f"[feature_eng] Veri: {len(sql_results)} satır")

    meta = ColumnMetadata(db_intelligence, profiling_report)
    summary = meta.summary()
    print(f"[feature_eng] Metadata: {summary}")

    df = pd.DataFrame(sql_results)

    # 1. Temizleme
    df = _clean_data(df)

    # 2. Hedef değişken
    df, y, target_name = _resolve_target(df, feature_plan)

    # 3. Aggregation (plan gerekli diyorsa)
    df, y = _aggregate_if_needed(df, y, feature_plan, meta)

    # 4. Identifier ve timestamp kolonlarını drop et
    df = _drop_non_feature_cols(df, meta)

    # 5. Encoding
    df = _encode_categorical(df, y)

    # 6. Scaling
    outlier_cols = meta.get_outlier_cols()
    df = _scale_features(df, outlier_cols)

    # 7. Feature selection
    df = _select_features(df)

    # 8. NaN temizliği (son kontrol)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 9. Train / test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        # Stratify başarısız olursa (çok az sınıf) stratify olmadan split
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    print(f"\n[feature_eng] TAMAMLANDI:")
    print(f"  Hedef değişken   : {target_name}")
    print(f"  X_train          : {X_train.shape}")
    print(f"  X_test           : {X_test.shape}")
    if len(np.unique(y_train_arr)) <= 10:
        print(f"  y_train dağılımı : {pd.Series(y_train_arr).value_counts().to_dict()}")
        print(f"  y_test dağılımı  : {pd.Series(y_test_arr).value_counts().to_dict()}")

    feature_report = {
        "target_name": target_name,
        "analysis_summary": feature_plan.get("analysis_summary", ""),
        "final_features": X_train.columns.tolist(),
        "n_features": len(X_train.columns),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "target_distribution_train": pd.Series(y_train_arr).value_counts().to_dict(),
        "target_distribution_test": pd.Series(y_test_arr).value_counts().to_dict(),
        "intelligence_metadata": summary,
    }

    return {
        **state,
        "feature_report": feature_report,
        "X_train": X_train.to_dict(orient="records"),
        "y_train": y_train.tolist(),
        "X_test": X_test.to_dict(orient="records"),
        "y_test": y_test.tolist(),
        "current_step": "feature_engineering_node",
    }