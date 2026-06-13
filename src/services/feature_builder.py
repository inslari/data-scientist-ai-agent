from __future__ import annotations



import warnings
from pathlib import Path
from typing import Any
import re
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, TargetEncoder
from src.core.config import settings
from typing import Any
import re
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, TargetEncoder

warnings.filterwarnings("ignore")

CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD    = 0.01
TEST_SIZE             = 0.2
ML_SAMPLE_ROWS        = 100_000   


# Column metadata
class ColumnMetadata:
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
                "semantic_type":   col_info.get("semantic_type", "unknown"),
                "importance_score":col_info.get("importance_score", 0.5),
                "has_outliers":    profile.get("has_outliers", False),
                "null_rate":       profile.get("null_rate", 0.0),
                "unique_count":    profile.get("unique_count", 0),
                "dtype":           profile.get("dtype", "object"),
            }

    def get_by_type(self, semantic_type: str) -> list[str]:
        return [c for c, i in self.columns.items() if i.get("semantic_type") == semantic_type]

    def get_outlier_cols(self) -> list[str]:
        return [c for c, i in self.columns.items() if i.get("has_outliers")]

    def summary(self) -> dict:
        return {
            "identifiers": self.get_by_type("identifier"),
            "timestamps":  self.get_by_type("timestamp"),
            "metrics":     self.get_by_type("metric"),
            "categories":  self.get_by_type("category"),
            "targets":     self.get_by_type("target"),
        }


def _load_data(state: dict) -> pd.DataFrame:
    """
    Parquet'ten akıllı sampling ile veri yükler.

    Sampling stratejisi:
    1. Veri küçükse (≤ ML_SAMPLE_ROWS) → hepsini al
    2. Target confirmed ise:
       - Imbalanced (ratio < 0.3): minority TAMAMINI al + majority orantılı
       - Balanced: stratified sample
    3. Target yok veya confirmed değil: random sample (güvenli fallback)

    Target sırasıyla şuralardan okunur:
      1. state["target_definition"]  (target_discovery_node + feature_plan onayı)
      2. state["feature_plan"]["target"]  (fallback)
    """
    data_path    = state.get("data_path", "")
    sql_results  = state.get("sql_results", [])
    target_def   = state.get("target_definition", {}) or {}
    feature_plan = state.get("feature_plan", {})

    # Target column name
    target_col = target_def.get("column_name")
    if not target_col and feature_plan:
        target_col = feature_plan.get("target", {}).get("column_name")
    confirmed = target_def.get("confirmed", False)

    # No Parquet path fallback
    if not data_path or not Path(data_path).exists():
        if sql_results:
            print(f"[feature_eng] UYARI: data_path yok, sql_results preview ({len(sql_results)} satır)")
            return pd.DataFrame(sql_results)
        raise ValueError("Veri yok: data_path ve sql_results ikisi de boş.")

    # Load from Parquet
    conn = duckdb.connect(":memory:")
    pq   = f"read_parquet('{data_path}')"
    res  = conn.execute(f"SELECT COUNT(*) FROM {pq}").fetchone()
    total = res[0] if res is not None else 0
    print(f"[feature_eng] Parquet: {total:,} satir → '{data_path}'")

    # Target kolon suffix'li olabilir (multi-table join sonrası)
    cols_df       = conn.execute(f"SELECT * FROM {pq} LIMIT 0").df()
    actual_cols   = cols_df.columns.tolist()
    actual_target = None
    if target_col:
        if target_col in actual_cols:
            actual_target = target_col
        else:
            # Suffix'li versiyonu ara
            for col in actual_cols:
                if col.startswith(target_col + "_") or col == target_col:
                    actual_target = col
                    print(f"[feature_eng] Target kolon suffix'li bulundu: '{target_col}' → '{actual_target}'")
                    break

    # Small data: take all
    if total <= ML_SAMPLE_ROWS:
        df = conn.execute(f"SELECT * FROM {pq}").df()
        print(f"[feature_eng] Tum veri yuklendi: {len(df):,} satır")
        conn.close()
        df = df.convert_dtypes(convert_integer=False)
        return df

    # Big data + target confirmed -> smart sampling
    if actual_target and confirmed:
        try:
            dist = conn.execute(
                f'SELECT "{actual_target}", COUNT(*) as cnt FROM {pq} '
                f'WHERE "{actual_target}" IS NOT NULL '
                f'GROUP BY "{actual_target}" ORDER BY cnt'
            ).fetchall()

            if len(dist) >= 2:
                minority_class = dist[0][0]
                minority_count = dist[0][1]
                majority_class = dist[-1][0]
                majority_count = dist[-1][1]
                ratio = minority_count / max(majority_count, 1)

                print(f"[feature_eng] Target dagilimi: "
                      f"minority({minority_class})={minority_count:,}, "
                      f"majority({majority_class})={majority_count:,}, "
                      f"ratio={ratio:.4f}")

                if ratio < 0.3:
                    # Imbalanced → minority hepsini al + majority orantılı
                    minority_limit = min(minority_count, ML_SAMPLE_ROWS // 3)
                    majority_limit = min(majority_count, ML_SAMPLE_ROWS - minority_limit)

                    df_min = conn.execute(
                        f'SELECT * FROM {pq} WHERE "{actual_target}" = $1 '
                        f'USING SAMPLE {minority_limit}', [minority_class]
                    ).df()
                    df_maj = conn.execute(
                        f'SELECT * FROM {pq} WHERE "{actual_target}" = $1 '
                        f'USING SAMPLE {majority_limit}', [majority_class]
                    ).df()

                    df = pd.concat([df_min, df_maj], ignore_index=True)
                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

                    print(f"[feature_eng] Imbalance-aware sampling: "
                          f"minority={len(df_min):,} + majority={len(df_maj):,} = {len(df):,}")
                    conn.close()
                    df = df.convert_dtypes(convert_integer=False)
                    return df
                else:
                    # Balanced → stratified
                    frames = []
                    for cls_val, cls_cnt in dist:
                        cls_n = max(1, int(ML_SAMPLE_ROWS * cls_cnt / total))
                        df_cls = conn.execute(
                            f'SELECT * FROM {pq} WHERE "{actual_target}" = $1 '
                            f'USING SAMPLE {cls_n}', [cls_val]
                        ).df()
                        frames.append(df_cls)
                    df = pd.concat(frames, ignore_index=True)
                    print(f"[feature_eng] Stratified sample: {len(df):,}/{total:,} satır")
                    conn.close()
                    df = df.convert_dtypes(convert_integer=False)
                    return df

        except Exception as e:
            print(f"[feature_eng] Akilli sampling hatasi: {e} → random fallback")

    # Fallback: random sample
    df = conn.execute(f"SELECT * FROM {pq} USING SAMPLE {ML_SAMPLE_ROWS}").df()
    print(f"[feature_eng] Random sample: {len(df):,}/{total:,} satır "
          f"(target {'confirmed' if confirmed else 'unconfirmed'})")
    conn.close()
    df = df.convert_dtypes(convert_integer=False)
    return df
    
# Column suffix normalization
def _normalize_column_names(
    df: "pd.DataFrame",
    table_names: list[str] | None = None,
) -> "pd.DataFrame":
    """
    Multi-table join sonrası oluşan suffix'leri temizler.

    Örnek: is_fraud_transactions → is_fraud
            merchant_name_merchants → merchant_name

    Mantık:
    - Suffix'siz kolon mevcutsa → suffix'li kolonu sil (duplicate)
    - Suffix'siz kolon yoksa   → suffix'i kaldır
    """
    if table_names:
        suffixes = [f"_{t}" for t in table_names]
    else:
        # Fallback: detect suffix automatically
        import re
        suffixes_set: set[str] = set()
        for col in df.columns:
            parts = col.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) >= 3:
                suffixes_set.add(f"_{parts[1]}")
        suffixes = list(suffixes_set)

    if not suffixes:
        return df

    rename_map: dict[str, str] = {}
    drop_cols: list[str] = []
    existing = set(df.columns)

    for col in df.columns:
        for suffix in sorted(suffixes, key=len, reverse=True):
            if col.endswith(suffix) and len(col) > len(suffix):
                clean = col[: -len(suffix)]
                if clean in existing:
                    # delete duplicate
                    drop_cols.append(col)
                elif clean not in rename_map.values():
                    rename_map[col] = clean
                break

    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"[feature_eng] Duplicate suffix kolonlar silindi: "
              f"{drop_cols[:5]}{'...' if len(drop_cols) > 5 else ''}")
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[feature_eng] {len(rename_map)} kolon suffix temizlendi "
              f"(örnek: {list(rename_map.items())[:3]})")

    return df


# Cleaning
def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    if before - len(df) > 0:
        print(f"[feature_eng] {before-len(df)} duplicate temizlendi")

    # String olarak gelen fiyat/sayı kolonlarını temizle (örn: '9.99$')
    for col in df.select_dtypes(include="object").columns:
        # Hız için tüm kolona değil, rastgele seçilmiş 1000 satıra bakıyoruz.
        non_nulls = df[col].dropna()
        n_sample = min(1000, len(non_nulls))
        sample = non_nulls.sample(n=n_sample, random_state=42).astype(str) if n_sample > 0 else pd.Series(dtype=str)
        if len(sample) > 0 and sample.str.contains(r'[\$€£]').mean() > 0.05:
            df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
            # Eğer sadece nokta kalmışsa NaN yap
            df[col] = df[col].replace('.', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"[feature_eng] '{col}' kolonu metin bazli currency algilandi, numerik formata cevrildi.")

    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            med_val = df[col].median()
            if pd.api.types.is_integer_dtype(df[col]):
                med_val = int(round(med_val))
            df[col] = df[col].fillna(med_val)

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode[0])

    print(f"[feature_eng] Temizlik sonrasi: {len(df):,} satır | null: {df.isnull().sum().sum()}")
    return df


# Target variable
def _resolve_target(
    df: pd.DataFrame, plan: dict, intent: str = ""
) -> tuple[pd.DataFrame, pd.Series, str]:
    target          = plan.get("target", {})
    target_name     = target.get("column_name", "")
    is_derived      = target.get("is_derived", False)
    derivation_type = target.get("derivation_type", "none")

    if not is_derived and target_name in df.columns:
        y = df[target_name].copy()

        # String/object ama sayısal mı? ($, €, virgül temizle)
        # pandas StringDtype VE object dtype ikisini de yakala
        is_string_like = (
            y.dtype == object
            or pd.api.types.is_string_dtype(y)
            or str(y.dtype) == "string"
        )
        if is_string_like:
            is_categorical = y.nunique() < 20
            if not is_categorical:
                import re
                cleaned = y.astype(str).str.replace(r'[^\d.\-]', '', regex=True)
                numeric_y = pd.to_numeric(cleaned, errors='coerce')
                if numeric_y.notna().mean() > 0.8:
                    y = numeric_y
                    print(f"[feature_eng] Hedef '{target_name}' string→numeric cevrildi (para/sayi)")
                else:
                    y = pd.Categorical(y).codes
                    print(f"[feature_eng] Hedef '{target_name}' kategorik → codes")
            else:
                y = pd.Categorical(y).codes
                print(f"[feature_eng] Hedef '{target_name}' kategorik (unique < 20) → codes")

        cols_drop = _find_related_target_cols(df, target_name)
        df = df.drop(columns=cols_drop, errors="ignore")
        print(f"[feature_eng] Hedef: '{target_name}' | dtype: {y.dtype}")
        return df, pd.Series(y, name=target_name), target_name

    if is_derived and derivation_type == "time_based":
        return _derive_time_based_target(df, target, plan)

    if is_derived and derivation_type == "threshold_based":
        return _derive_threshold_target(df, target)

    print(f"[feature_eng] UYARI: '{target_name}' bulunamadi, aday araniyor...")
    return _find_best_target_candidate(df)


def _find_related_target_cols(df: pd.DataFrame, target_name: str) -> list[str]:
    cols = [target_name]
    stem = target_name.replace("_binary","").replace("_label","").replace("_target","")
    for col in df.columns:
        if col == target_name: continue
        if col.replace("_binary","").replace("_label","").replace("_target","") == stem:
            cols.append(col)
    return cols


def _derive_time_based_target(
    df: pd.DataFrame, target: dict, plan: dict
) -> tuple[pd.DataFrame, pd.Series, str]:
    target_name   = target.get("column_name","derived_target")
    date_col      = target.get("date_column","")
    threshold_days= target.get("threshold_days",0)
    group_by      = plan.get("aggregation",{}).get("group_by_column","")

    if date_col not in df.columns:
        return _find_best_target_candidate(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if plan.get("use_data_split_for_label") or threshold_days == 0:
        reference_date = df[date_col].quantile(0.80)
    else:
        reference_date = df[date_col].max() - pd.Timedelta(days=threshold_days)

    if group_by and group_by in df.columns:
        last = df.groupby(group_by)[date_col].max().reset_index()
        last[target_name] = (last[date_col] < reference_date).astype(int)
        df = df.merge(last[[group_by, target_name]], on=group_by, how="left")
    else:
        df[target_name] = (df[date_col] < reference_date).astype(int)

    y = df[target_name].copy()
    df = df.drop(columns=[target_name, date_col], errors="ignore")
    print(f"[feature_eng] Hedef: '{target_name}' (time_based) | {y.value_counts().to_dict()}")
    return df, y, target_name


def _derive_threshold_target(
    df: pd.DataFrame, target: dict
) -> tuple[pd.DataFrame, pd.Series, str]:
    target_name     = target.get("column_name","derived_target")
    threshold_value = target.get("threshold_value",0.0)
    source_col      = target_name.replace("_binary","").replace("_label","")
    if source_col not in df.columns:
        return _find_best_target_candidate(df)
    y = (df[source_col] > threshold_value).astype(int)
    df = df.drop(columns=[source_col], errors="ignore")
    print(f"[feature_eng] Hedef: '{target_name}' (threshold={threshold_value}) | {y.value_counts().to_dict()}")
    return df, pd.Series(y, name=target_name), target_name


def _find_best_target_candidate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str]:
    for col in df.columns:
        if df[col].nunique() == 2 and df[col].dtype in [np.int64, np.float64, int, float]:
            y = df[col].copy()
            df = df.drop(columns=[col])
            print(f"[feature_eng] Fallback hedef: '{col}'")
            return df, y, col
    raise ValueError("Hedef değişken bulunamadı.")


# ── Aggregation ──────────────────────────────────────────────
def _aggregate_if_needed(
    df: pd.DataFrame, y: pd.Series, plan: dict, meta: ColumnMetadata,
    entity_resolution: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    # Entity resolution zaten aggregate ettiyse, tekrar aggregate ETME
    if entity_resolution and entity_resolution.get("action") == "aggregate":
        print("[feature_eng] Entity resolution zaten aggregate etti → "
              "feature_builder aggregation atlanıyor (çift aggregation önlendi)")
        return df, y

    agg_config = plan.get("aggregation", {})
    if not agg_config.get("required", False):
        return df, y

    group_col = agg_config.get("group_by_column","")
    if not group_col or group_col not in df.columns:
        print(f"[feature_eng] group_by '{group_col}' bulunamadi, aggregation atlandi")
        return df, y

    level          = agg_config.get("level","none")
    feature_groups = plan.get("feature_groups",[])
    metric_cols    = [c for c in meta.get_by_type("metric") if c in df.columns and c != group_col]
    id_cols        = meta.get_by_type("identifier")
    timestamp_cols = [c for c in meta.get_by_type("timestamp") if c in df.columns]
    category_cols  = [c for c in meta.get_by_type("category") if c in df.columns]
    date_col       = timestamp_cols[0] if timestamp_cols else None

    reference_date = None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        reference_date = df[date_col].quantile(0.80) if plan.get("use_data_split_for_label") else df[date_col].max()
    agg_df = df[df[date_col] < reference_date].copy() if (date_col and reference_date and plan.get("use_data_split_for_label")) else df.copy()

    agg_dict: dict[str, Any] = {}
    for col in metric_cols:
        if col in agg_df.columns:
            agg_dict[f"sum_{col}"]=(col,"sum"); agg_dict[f"avg_{col}"]=(col,"mean")
            agg_dict[f"max_{col}"]=(col,"max"); agg_dict[f"std_{col}"]=(col,"std")
    for col in id_cols:
        if col in agg_df.columns and col != group_col:
            agg_dict[f"count_{col}"]=(col,"nunique")

    customer_df = agg_df.groupby(group_col).agg(**agg_dict).reset_index()

    if "rfm" in feature_groups and date_col:
        recency = agg_df.groupby(group_col)[date_col].max().reset_index()
        recency["recency_days"] = (reference_date - recency[date_col]).dt.days
        customer_df = customer_df.merge(recency[[group_col,"recency_days"]], on=group_col, how="left")

    if "temporal_trend" in feature_groups and date_col and metric_cols:
        ac = metric_cols[0]; mid = agg_df[date_col].quantile(0.50)
        rec = agg_df[agg_df[date_col]>=mid].groupby(group_col)[ac].sum().reset_index().rename(columns={ac:"spend_recent"})
        old = agg_df[agg_df[date_col]<mid].groupby(group_col)[ac].sum().reset_index().rename(columns={ac:"spend_old"})
        customer_df = customer_df.merge(rec, on=group_col, how="left").merge(old, on=group_col, how="left")
        customer_df[["spend_recent","spend_old"]] = customer_df[["spend_recent","spend_old"]].fillna(0)
        customer_df["spend_trend"] = np.where(customer_df["spend_old"]>0,customer_df["spend_recent"]/customer_df["spend_old"],0.0)

    if "behavioral" in feature_groups and category_cols:
        for cat_col in category_cols[:2]:
            if cat_col in agg_df.columns:
                div = agg_df.groupby(group_col)[cat_col].nunique().reset_index().rename(columns={cat_col:f"{cat_col}_diversity"})
                fav = agg_df.groupby(group_col)[cat_col].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index().rename(columns={cat_col:f"fav_{cat_col}"})
                customer_df = customer_df.merge(div,on=group_col,how="left").merge(fav,on=group_col,how="left")

    y_df = pd.DataFrame({"__y":y, group_col:df[group_col]})
    y_grouped = y_df.groupby(group_col)["__y"].first().reset_index()
    customer_df = customer_df.merge(y_grouped, on=group_col, how="left")
    y_out = customer_df["__y"].copy()
    customer_df = customer_df.drop(columns=["__y",group_col], errors="ignore").fillna(0)

    print(f"[feature_eng] Aggregation: {len(customer_df):,} satır × {len(customer_df.columns)} kolon")
    return customer_df, y_out


# ── Encoding ─────────────────────────────────────────────────
def _encode_categorical(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols: return df
    low  = [c for c in cat_cols if df[c].nunique() <= 10]
    high = [c for c in cat_cols if df[c].nunique() >  10]
    if low:
        df = pd.get_dummies(df, columns=low, drop_first=True, dtype=int)
        print(f"[feature_eng] One-hot ({len(low)}): {low}")
    if high:
        try:
            # sklearn 1.3+ TargetEncoder expects 2D array for X and 1D for y
            df[high] = TargetEncoder(random_state=RANDOM_STATE).fit_transform(df[high], y)
            print(f"[feature_eng] Target encoding ({len(high)}): {high}")
        except Exception as e:
            print(f"[feature_eng] Target encoding hatasi → kolonlar drop ediliyor: {e}")
            df = df.drop(columns=high, errors="ignore")
    return df


# ── Scaling ──────────────────────────────────────────────────
def _scale_features(df: pd.DataFrame, outlier_cols: list[str]) -> pd.DataFrame:
    num = df.select_dtypes(include="number").columns.tolist()
    if not num: return df
    robust   = [c for c in outlier_cols if c in num]
    standard = [c for c in num if c not in robust]
    if robust:
        df[robust]   = RobustScaler().fit_transform(df[robust])
        print(f"[feature_eng] RobustScaler: {len(robust)} kolon")
    if standard:
        df[standard] = StandardScaler().fit_transform(df[standard])
        print(f"[feature_eng] StandardScaler: {len(standard)} kolon")
    return df


# ── Feature selection ────────────────────────────────────────
def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number").columns.tolist()
    if not num: return df
    low_var = df[num].var()[lambda s: s < VARIANCE_THRESHOLD].index.tolist()
    if low_var:
        df = df.drop(columns=low_var)
        num = [c for c in num if c not in low_var]
        print(f"[feature_eng] Low variance drop: {low_var}")
    if len(num) > 1:
        corr  = df[num].corr().abs()
        mask  = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.where(mask)  # type: ignore
        hc    = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
        if hc:
            df = df.drop(columns=hc)
            print(f"[feature_eng] High corr drop: {hc}")
    return df


def _apply_domain_leakage_drops(
    df: pd.DataFrame,
    domain_leakage_report: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Domain leakage detector'ın high-confidence suspect'lerini drop eder.

    Güvenlik şartları:
    - Sadece confidence='high' olanlar drop edilir
    - Drop sonrası en az 2 feature kalmalı (boş feature set'i önle)
    - Drop edilenler audit için raporlanır
    - Ana kolon df'de olmasa bile türev kolonları (_year, _month vb.) drop edilir

    Dönüş:
        df              : drop sonrası dataframe
        domain_dropped  : drop edilen feature'ların listesi (dict)
    """
    domain_dropped = []

    if not domain_leakage_report:
        return df, domain_dropped

    concerns = domain_leakage_report.get("concerns", []) or []
    high_confidence = [c for c in concerns if c.get("confidence") == "high"]

    if not high_confidence:
        return df, domain_dropped

    leaky_col_names = {c.get("column_name") for c in high_confidence}

    # Türev kolonları da tespit et (örn: ref_date → ref_date_year, ref_date_month, ref_date_dayofweek)
    derived_suffixes = ("_year", "_month", "_dayofweek", "_day", "_hour", "_week", "_quarter")
    derived_to_drop = [
        col for col in df.columns
        if any(
            col == f"{leaky}{suf}"
            for leaky in leaky_col_names
            for suf in derived_suffixes
        )
    ]

    # Ana kolonlar + türevler: df'de olanları al
    all_cols_to_drop = list((leaky_col_names | set(derived_to_drop)) & set(df.columns))

    if not all_cols_to_drop:
        return df, domain_dropped

    # Güvenlik: drop sonrası en az 2 feature kalmalı
    remaining_after_drop = len(df.columns) - len(all_cols_to_drop)
    if remaining_after_drop < 2:
        print(f"[leakage] Domain drop iptal: drop sonrasi {remaining_after_drop} feature kalacak (min 2 gerekli)")
        return df, domain_dropped

    # Drop uygula
    df = df.drop(columns=all_cols_to_drop, errors="ignore")

    for c in high_confidence:
        col_name = c.get("column_name", "")
        domain_dropped.append({
            "column": col_name,
            "reason": f"domain leakage (high confidence): {c.get('reasoning', '')[:120]}",
            "relationship_type": c.get("relationship_type", "unknown"),
        })
        print(f"[leakage] DOMAIN DROP: '{col_name}' "
              f"({c.get('relationship_type')}, high) - "
              f"{c.get('reasoning', '')[:80]}")

    for derived_col in derived_to_drop:
        domain_dropped.append({
            "column": derived_col,
            "reason": "domain leakage (türev — parent kolon high confidence leaky)",
            "relationship_type": "temporal_derived",
        })
        print(f"[leakage] DOMAIN DROP (türev): '{derived_col}' — leaky parent: {leaky_col_names}")

    return df, domain_dropped

# ── Data leakage detection ───────────────────────────────────
# ── Data leakage detection (Adım 1: identifier + kesin korelasyon) ──
def _detect_leakage(
    df: pd.DataFrame,
    y: pd.Series,
    meta=None,  # ColumnMetadata, identifier tespiti için
) -> tuple[pd.DataFrame, dict]:
    """
    Leakage tespiti — iki seviyeli:
      KESİN  → otomatik drop (identifier, ~birebir korelasyon)
      ŞÜPHELİ → drop etme, uyarı listesine ekle

    Dönüş:
      df              : temizlenmiş dataframe
      leakage_report  : {"dropped": [...], "suspected": [...]}
    """
    dropped = []      # otomatik atılanlar (kesin)
    suspected = []    # şüpheli, kullanıcıya raporlanacak
    n_rows = len(df)
    print(f"\n[leakage] === BASLIYOR ===")
    print(f"[leakage] df shape: {df.shape}, y dtype: {y.dtype}")

    # y'yi numeric'e çevir (korelasyon için), başarısızsa korelasyon atla
    try:
        y_num = pd.to_numeric(y, errors="coerce").astype(float)
        y_valid = y_num.notna().mean() > 0.8
    except Exception:
        y_valid = False
        y_num = None

    for col in list(df.columns):
        # ── KESİN 1: Identifier semantic_type ──
        if meta is not None:
            sem = meta.columns.get(col, {}).get("semantic_type", "")
            if sem == "identifier":
                df = df.drop(columns=[col], errors="ignore")
                dropped.append({"column": col, "reason": "identifier (semantic_type)"})
                print(f"[leakage] KESIN drop: '{col}' (identifier)")
                continue

        # ── KESİN 2: Aşırı yüksek kardinalite (ID-benzeri) ──
        nunique = df[col].nunique()
        if n_rows > 0 and nunique / n_rows > 0.95:
            df = df.drop(columns=[col], errors="ignore")
            dropped.append({"column": col, "reason": f"asiri kardinalite ({nunique}/{n_rows})"})
            print(f"[leakage] KESIN drop: '{col}' (kardinalite orani {nunique/n_rows:.2f})")
            continue

    # ── Korelasyon tabanlı (yalnızca numeric + y numeric ise) ──
    if y_valid:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        print(f"[leakage DEBUG] numeric cols: {num_cols}")
        print(f"[leakage DEBUG] y_num NaN orani: {y_num.isna().mean():.2f}")
        for col in num_cols:
            if df[col].nunique() <= 1:
                continue
            try:
                col_vals = pd.to_numeric(df[col], errors="coerce").astype(float)
                corr = abs(col_vals.corr(y_num))
                print(f"[leakage DEBUG] '{col}' corr = {corr}")
            except Exception as e:
                print(f"[leakage DEBUG] '{col}' corr hatasi: {e}")
                continue
            if pd.isna(corr):
                continue

            if corr > 0.999:
                # KESİN: birebir kopya (örn. target'in kendisi/duplikası)
                df = df.drop(columns=[col], errors="ignore")
                dropped.append({"column": col, "reason": f"birebir korelasyon ({corr:.4f})"})
                print(f"[leakage] KESIN drop: '{col}' (corr={corr:.4f})")
            elif corr > 0.85:
                # ŞÜPHELİ: yüksek ama kesin değil → drop etme, uyar
                suspected.append({"column": col, "reason": f"yuksek korelasyon ({corr:.3f})"})
                print(f"[leakage] SUPHELI: '{col}' (corr={corr:.3f}) — droplanmadi, uyari verildi")

    return df, {"dropped": dropped, "suspected": suspected}


# ── MAIN NODE ────────────────────────────────────────────────
def build_features(state: dict) -> dict:
    """
    Feature Engineering Service — BigData Edition

    Girdiler:
        data_path        : Parquet dosya yolu (execution_node'dan)
        sql_results      : Preview (fallback)
        profiling_report : DuckDB profiling çıktısı
        db_intelligence  : intelligence_engine metadata
        feature_plan     : LLM planı
        critic_feedback  : Critic node'dan drop direktifleri (retry)

    State çıktıları:
        feature_report, X_train, X_test, y_train, y_test
    """
    intent           = state.get("intent","")
    profiling_report = state.get("profiling_report",{})
    db_intelligence  = state.get("db_intelligence",{})
    feature_plan     = state.get("feature_plan",{})
    
    if not feature_plan:
        return {"error":"Feature plan yok."}

    print(f"\n{'='*60}")
    print(f"[feature_eng] BASLIYOR — BigData Feature Engineering")
    print(f"{'='*60}")
    print(f"[feature_eng] Analiz: {feature_plan.get('analysis_summary','?')}")

    meta = ColumnMetadata(db_intelligence, profiling_report)

    try:
        df = _load_data(state)
    except Exception as e:
        import traceback
        print(f"\n[feature_eng] HATA: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}

    if df.empty or len(df) < 5:
        return {"error": f"Veri seti yüklenemedi veya çok az veri var ({len(df)} satır)."}

    print(f"[feature_eng] Veri: {len(df):,} satır × {len(df.columns)} kolon")

    # Multi-table join suffix'lerini temizle
    all_tables = [t["table_name"] for t in db_intelligence.get("tables", [])]
    df = _normalize_column_names(df, table_names=all_tables if all_tables else None)
    print(f"[feature_eng] Normalizasyon sonrasi: {len(df.columns)} kolon")

    # Critic'ten gelen drop_features uygula
    drop_features = feature_plan.get("drop_features", [])
    if drop_features:
        existing_drops = [c for c in drop_features if c in df.columns]
        if existing_drops:
            df = df.drop(columns=existing_drops, errors="ignore")
            print(f"[feature_eng] Critic drop_features: {existing_drops}")

    # Temizlik
    df = _clean_data(df)

    # Hedef değişken
    df, y, target_name = _resolve_target(df, feature_plan, intent)

    # Aggregation
    entity_resolution = state.get("entity_resolution") or {}
    df, y = _aggregate_if_needed(df, y, feature_plan, meta, entity_resolution=entity_resolution)

    if df.empty or len(df) < 5:
        return {"error": f"İşlemler sonrası yetersiz veri sayısı ({len(df)} satır). Model kurulamıyor."}

    # Domain leakage raporunu erken oku — timestamp türevi üretmeden önce lazım
    domain_leakage_report = state.get("domain_leakage_report") or {}
    leaky_cols_high = {
        c.get("column_name")
        for c in (domain_leakage_report.get("concerns") or [])
        if c.get("confidence") == "high"
    }
    if leaky_cols_high:
        print(f"[feature_eng] High-confidence leaky kolonlar (türev üretilmeyecek): {leaky_cols_high}")

    # Date feature extraction — leaky olan timestamp kolonları atlanır
    for col in meta.get_by_type("timestamp"):
        if col in df.columns:
            if col in leaky_cols_high:
                print(f"[feature_eng] SKIP türev: '{col}' domain leakage (high) — year/month/dayofweek üretilmiyor")
                continue
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"]      = dt.dt.year
                df[f"{col}_month"]     = dt.dt.month
                df[f"{col}_dayofweek"] = dt.dt.dayofweek
            except Exception:
                pass

    # Drop identifier / timestamp / target kolonları
    target_col = feature_plan.get("target",{}).get("column_name","")
    cols_drop  = set()
    for col, info in meta.columns.items():
        if col not in df.columns: continue
        sem = info.get("semantic_type","")
        if sem in ("identifier","timestamp"):
            cols_drop.add(col)
        elif sem == "target":
            cols_drop.add(col)
    
    if target_name and target_name in df.columns and intent not in ("clustering", "anomaly_detection"):
        cols_drop.add(target_name)
    related = _find_related_target_cols(df, target_col)
    cols_drop.update([c for c in related if c in df.columns and c != target_col])
    existing = [c for c in cols_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing, errors="ignore")
        print(f"[feature_eng] Drop (id/ts/target): {existing}")

    # Math interactions
    for expr in feature_plan.get("math_interactions", []):
        try:
            # Noktalı kolon isimlerini backtick ile sar
            safe_expr = expr
            for col in df.columns:
                if "." in col and col in expr:
                    safe_expr = safe_expr.replace(col, f"`{col}`")
            safe_name = "math_" + "".join([c if c.isalnum() else "_" for c in expr])[:30]
            df[safe_name] = df.eval(safe_expr)
        except Exception as e:
            print(f"[feature_eng] Math interaction hatasi ({expr}): {e}")

    # Encoding
    df = _encode_categorical(df, y)

    # Scaling
    df = _scale_features(df, meta.get_outlier_cols())

    # Feature selection
    df = _select_features(df)

    # Final NaN temizliği
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Leakage detection
    df, leakage_report = _detect_leakage(df, y, meta)

    df, domain_dropped = _apply_domain_leakage_drops(df, domain_leakage_report)

    # Birleştir: domain dropped'ı leakage_report'a ekle
    leakage_report["dropped"].extend(domain_dropped)

    # 14. Train/test split
    if intent in ("clustering","anomaly_detection"):
        X_train = X_test = df.copy()
        y_train = y_test = pd.Series([], dtype=int)
        print(f"[feature_eng] Unsupervised — split atlandi")
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df, y, test_size=TEST_SIZE, random_state=settings.RANDOM_SEED, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                df, y, test_size=TEST_SIZE, random_state=settings.RANDOM_SEED
            )

    y_train_arr = np.array(y_train)
    y_test_arr  = np.array(y_test)

    print(f"\n[feature_eng] TAMAMLANDI:")
    print(f"  Hedef   : {target_name}")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    if len(np.unique(y_train_arr)) <= 10:
        print(f"  Train dist: {pd.Series(y_train_arr).value_counts().to_dict()}")  # type: ignore

    feature_report = {
        "target_name":              target_name,
        "analysis_summary":         feature_plan.get("analysis_summary",""),
        "final_features":           X_train.columns.tolist(),
        "n_features":               len(X_train.columns),
        "n_train":                  len(X_train),
        "n_test":                   len(X_test),
        "target_distribution_train": pd.Series(y_train_arr).value_counts().to_dict(),  # type: ignore
        "target_distribution_test": pd.Series(y_test_arr).value_counts().to_dict(),  # type: ignore
        "intelligence_metadata":    meta.summary(),
        "data_source":              "parquet" if state.get("data_path") else "sql_results_preview",
        "total_rows_available":     None,  # execution_node profiling'den alınabilir
        "leakage_dropped":          leakage_report["dropped"],
        "leakage_suspected":        leakage_report["suspected"],
    }

    return {
        "feature_report": feature_report,
        "X_train":        X_train.to_dict(orient="records"),
        "y_train":        y_train.tolist(),
        "X_test":         X_test.to_dict(orient="records"),
        "y_test":         y_test.tolist(),
        "error":          None
    }