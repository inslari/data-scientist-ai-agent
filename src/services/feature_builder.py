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
from src.core.logging_config import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD    = 0.01
TEST_SIZE             = 0.2
ML_SAMPLE_ROWS        = 100_000
RANDOM_STATE          = 42   # projenin genel seed konvansiyonu (settings.RANDOM_SEED / ml_trainer.RANDOM_STATE ile tutarli)

# Denetimli univariate seçim (ANOVA / f_regression): hedefle istatistiksel ilişkisi
# anlamsız feature'ları eler. p-value eşiği keyfi skor eşiğinden (örn. "score<50")
# prensiplidir; küçük feature setlerini bozmamak için kapılıdır.
UNIVARIATE_PVALUE     = 0.05      # bu p-value'dan büyük (anlamsız) feature aday-drop
UNIVARIATE_MIN_KEEP   = 5         # seçim sonrası asla bu sayının altına inme
UNIVARIATE_MIN_COLS   = 12        # bu kadardan az sayısal feature varsa hiç uygulama


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
            logger.warning(f"[feature_eng] UYARI: data_path yok, sql_results preview ({len(sql_results)} satır)")
            return pd.DataFrame(sql_results)
        raise ValueError("Veri yok: data_path ve sql_results ikisi de boş.")

    # Load from Parquet
    conn = duckdb.connect(":memory:")
    pq   = f"read_parquet('{data_path}')"
    res  = conn.execute(f"SELECT COUNT(*) FROM {pq}").fetchone()
    total = res[0] if res is not None else 0
    logger.info(f"[feature_eng] Parquet: {total:,} satir → '{data_path}'")

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
                    logger.info(f"[feature_eng] Target kolon suffix'li bulundu: '{target_col}' → '{actual_target}'")
                    break

    # Small data: take all
    if total <= ML_SAMPLE_ROWS:
        df = conn.execute(f"SELECT * FROM {pq}").df()
        logger.info(f"[feature_eng] Tum veri yuklendi: {len(df):,} satır")
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

                logger.info(f"[feature_eng] Target dagilimi: "
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

                    logger.info(f"[feature_eng] Imbalance-aware sampling: "
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
                    logger.info(f"[feature_eng] Stratified sample: {len(df):,}/{total:,} satır")
                    conn.close()
                    df = df.convert_dtypes(convert_integer=False)
                    return df

        except Exception as e:
            logger.error(f"[feature_eng] Akilli sampling hatasi: {e} → random fallback")

    # Fallback: random sample
    df = conn.execute(f"SELECT * FROM {pq} USING SAMPLE {ML_SAMPLE_ROWS}").df()
    logger.info(f"[feature_eng] Random sample: {len(df):,}/{total:,} satır "
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
        logger.info(f"[feature_eng] Duplicate suffix kolonlar silindi: "
              f"{drop_cols[:5]}{'...' if len(drop_cols) > 5 else ''}")
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"[feature_eng] {len(rename_map)} kolon suffix temizlendi "
              f"(örnek: {list(rename_map.items())[:3]})")

    return df


# Cleaning
def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # _load_data'daki convert_dtypes() kategorikleri pandas StringDtype yapar; ama tum
    # encoding/feature kodu select_dtypes(include="object") kullanir ve StringDtype'i
    # YAKALAMAZ → kategorikler encode edilmeden to_numeric ile 0'a coerce olur (olu
    # feature). Tek noktada object'e indir: encode/count/deviation/currency hepsi calissin.
    str_cols = df.select_dtypes(include="string").columns
    if len(str_cols):
        df[str_cols] = df[str_cols].astype(object)
        logger.info(f"[feature_eng] StringDtype → object ({len(str_cols)} kolon): encode icin")

    before = len(df)
    df = df.drop_duplicates()
    if before - len(df) > 0:
        logger.info(f"[feature_eng] {before-len(df)} duplicate temizlendi")

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
            logger.info(f"[feature_eng] '{col}' kolonu metin bazli currency algilandi, numerik formata cevrildi.")

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

    logger.info(f"[feature_eng] Temizlik sonrasi: {len(df):,} satır | null: {df.isnull().sum().sum()}")
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
                    logger.info(f"[feature_eng] Hedef '{target_name}' string→numeric cevrildi (para/sayi)")
                else:
                    y = pd.Categorical(y).codes
                    logger.info(f"[feature_eng] Hedef '{target_name}' kategorik → codes")
            else:
                y = pd.Categorical(y).codes
                logger.info(f"[feature_eng] Hedef '{target_name}' kategorik (unique < 20) → codes")

        cols_drop = _find_related_target_cols(df, target_name)
        df = df.drop(columns=cols_drop, errors="ignore")
        logger.info(f"[feature_eng] Hedef: '{target_name}' | dtype: {y.dtype}")
        return df, pd.Series(y, name=target_name), target_name

    if is_derived and derivation_type == "time_based":
        return _derive_time_based_target(df, target, plan)

    if is_derived and derivation_type == "threshold_based":
        return _derive_threshold_target(df, target)

    logger.warning(f"[feature_eng] UYARI: '{target_name}' bulunamadi, aday araniyor...")
    return _find_best_target_candidate(df, planned_target=target_name)


def _find_related_target_cols(df: pd.DataFrame, target_name: str) -> list[str]:
    cols = [target_name]
    stem = target_name.replace("_binary","").replace("_label","").replace("_target","")
    for col in df.columns:
        if col == target_name: continue
        if col.replace("_binary","").replace("_label","").replace("_target","") == stem:
            cols.append(col)
    return cols


def _resolve_time_col(df: pd.DataFrame, time_col: str) -> str | None:
    """Kullanıcının UI'dan seçtiği ham time_col adını df'teki gerçek kolona eşle.
    _normalize_column_names sonrası adlar değişmiş olabilir → exact, sonra
    case-insensitive/trimmed eşleme dene. Bulunamazsa None."""
    if not time_col:
        return None
    if time_col in df.columns:
        return time_col
    lookup = {str(c).lower().strip(): c for c in df.columns}
    return lookup.get(time_col.lower().strip())


def _chronological_order(time_sort_key: pd.Series, index) -> np.ndarray:
    """time_sort_key'i verilen satır index'ine hizala ve eski→yeni KARARLI pozisyonel
    sıra döndür (np.argsort mergesort = stabil, deterministik). Parse edilemeyen az
    sayıda satır (NaT) en eski tarihe atanır → train tarafında kalır, test'i kirletmez."""
    key = time_sort_key.reindex(index)
    key = key.fillna(key.min())
    return np.argsort(key.to_numpy(), kind="mergesort")


def _entity_aware_split_positions(
    time_sort_key: pd.Series, entity_split_key: pd.Series, index, test_size: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Her entity'nin KENDI son %test_size'i test. df.index'e hizali calisir, POZISYONEL
    (df.iloc ile kullanilir) train/test index dizileri dondurur → df ve y ayni permutasyonla
    hizali kalir (FAZ 1 disiplini). Boylece hicbir entity tamamen bir tarafa dusmez.
    Donus: (train_pos, test_pos, imbalanced_entity_sayisi)."""
    ent = entity_split_key.reindex(index).to_numpy()
    tk = time_sort_key.reindex(index)
    tk = tk.fillna(tk.min()).to_numpy()
    helper = pd.DataFrame({"pos": np.arange(len(index)), "ent": ent, "tk": tk})
    helper = helper.sort_values(["ent", "tk"], kind="mergesort")   # [entity, zaman] stabil
    train_pos: list[int] = []
    test_pos: list[int] = []
    imbalanced = 0
    for _, g in helper.groupby("ent", sort=False):
        m = len(g)
        n_test_g = max(1, int(round(m * test_size)))   # her entity >=1 test
        gp = g["pos"].to_numpy()
        cut = m - n_test_g                              # cut<=0 → train bos (cok kucuk entity)
        if cut <= 0:
            imbalanced += 1
        train_pos.extend(gp[:max(cut, 0)].tolist())
        test_pos.extend(gp[max(cut, 0):].tolist())
    return np.array(train_pos, dtype=int), np.array(test_pos, dtype=int), imbalanced


def _derive_time_based_target(
    df: pd.DataFrame, target: dict, plan: dict
) -> tuple[pd.DataFrame, pd.Series, str]:
    target_name   = target.get("column_name","derived_target")
    date_col      = target.get("date_column","")
    threshold_days= target.get("threshold_days",0)
    group_by      = plan.get("aggregation",{}).get("group_by_column","")

    if date_col not in df.columns:
        logger.warning(f"[feature_eng] UYARI: time_based date_column '{date_col}' yok, aday araniyor...")
        return _find_best_target_candidate(df, planned_target=target_name)

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
    logger.info(f"[feature_eng] Hedef: '{target_name}' (time_based) | {y.value_counts().to_dict()}")
    return df, y, target_name


def _derive_threshold_target(
    df: pd.DataFrame, target: dict
) -> tuple[pd.DataFrame, pd.Series, str]:
    target_name     = target.get("column_name","derived_target")
    threshold_value = target.get("threshold_value",0.0)
    source_col      = target_name.replace("_binary","").replace("_label","")
    if source_col not in df.columns:
        logger.warning(f"[feature_eng] UYARI: threshold source kolonu '{source_col}' yok, aday araniyor...")
        return _find_best_target_candidate(df, planned_target=target_name)
    y = (df[source_col] > threshold_value).astype(int)
    df = df.drop(columns=[source_col], errors="ignore")
    logger.info(f"[feature_eng] Hedef: '{target_name}' (threshold={threshold_value}) | {y.value_counts().to_dict()}")
    return df, pd.Series(y, name=target_name), target_name


def _find_best_target_candidate(df: pd.DataFrame, planned_target: str = "") -> tuple[pd.DataFrame, pd.Series, str]:
    for col in df.columns:
        if df[col].nunique() == 2 and df[col].dtype in [np.int64, np.float64, int, float]:
            y = df[col].copy()
            df = df.drop(columns=[col])
            logger.info(f"[feature_eng] Fallback hedef: '{col}'")
            return df, y, col
    # Tanılayıcı: planlanan hedef ile gerçekte yüklenen kolonları göster ki
    # uyumsuzluğun nedeni (SQL hedefi seçmedi / ad normalize oldu / vb.) anlaşılsın.
    available = ", ".join(f"{c}({df[c].dtype}, n={df[c].nunique()})" for c in df.columns) or "<bos>"
    raise ValueError(
        "Hedef değişken bulunamadı. "
        f"Planlanan hedef: '{planned_target or '?'}' — yüklenen veride yok ve türetilemiyor; "
        "binary numeric fallback aday da bulunamadı. "
        f"Mevcut kolonlar: [{available}]. "
        "Olası neden: SQL hedef kolonu seçmedi, kolon adı normalize edilince değişti, "
        "ya da unsupervised bir intent için target aranıyor."
    )


# ── Time-series lag features (FAZ A: cekirdek) ───────────────
_TS_MAX_FEATURE_COLS = 5   # hedef disinda laglenecek en fazla sayisal kolon (feature patlamasi korumasi)

# ── Deterministik sifir/eksik flag motoru (Adim 1) ───────────
_DERIVED_FLAG_PREFIX = "__wm__"   # was_missing maskesi icin DAHILI tasiyici prefix (slot'ta temizlenir)
_FLAG_MIN_PREVALENCE = 0.01       # yayginlik kapisi alt sinir (hep-ayni flag uretme)
_FLAG_MAX_PREVALENCE = 0.99       # yayginlik kapisi ust sinir
_MAX_DERIVED_FLAGS   = 30         # global cap (_TS_MAX_FEATURE_COLS muadili — feature patlamasi korumasi)
_FLAG_INTENTS        = {"classification", "regression", "clustering"}  # kapsam (anomaly bu yola girmez)
# TODO(iyilestirme): scale skip-list — flag'leri 0/1 tutup yorumlanabilirligi koru (su an scale z-skora ceviriyor)

# ── Deterministik oran (ratio) motoru (Adim 2) ───────────────
_MAX_RATIO_FEATURES   = 20      # global cap (feature patlamasi — _MAX_DERIVED_FLAGS muadili)
_RATIO_MIN_UNIQUE     = 3       # binary/sabit kolonlari hariç tut (nunique >= 3)
_RATIO_DENOM_MAX_ZERO = 0.05    # payda kolonunda izinli max sifir orani (sifir-bolme stabilizasyonu)
_RATIO_WINSOR_Q       = 0.01    # robust clip kuantili (q01..q99) — uc deger nötr
_RATIO_MAX_CARD_RATIO = 0.95    # ID-benzeri kaynak kapisi: nunique/n_rows > 0.95 → orana sokma
                                # (oranlar protected_high_card ile kardinaliteden MUAF; bu on-filtre
                                #  yanlis-etiketli ID'nin orana sizmasini ENGELLER — LLM metadata hatasina dayaniklilik)


def _detect_frequency(parsed: pd.Series, group_key: "pd.Series | None" = None) -> tuple[str, list[int]]:
    """Ardisik tarih farklarinin medyanindan veri sikligini ve dinamik lag pencerelerini turet.

    ``group_key`` verilirse (multi-entity, FAZ B2) fark GRUP-ICI alinir: panelde tum
    entity'ler ayni tarihleri paylasinca havuzlanmis GLOBAL diff 0'a coker (ayni gun
    tekrari) → within-group diff bunu onler. group_key None ise GLOBAL diff (FAZ A/B1).
    """
    if group_key is not None:
        tmp = pd.DataFrame({"t": parsed, "g": group_key}).dropna(subset=["t"])
        deltas = tmp.sort_values(["g", "t"]).groupby("g")["t"].diff().dropna()
    else:
        deltas = parsed.dropna().sort_values().diff().dropna()
    if len(deltas) == 0:
        logger.info("[feature_eng] timeseries: frekans icin yeterli tarih farki yok → varsayilan [1]")
        return "unknown", [1]
    days = deltas.median() / pd.Timedelta(days=1)
    if 0.5 <= days <= 1.5:
        return "daily", [1, 7, 30]
    if 5 <= days <= 9:
        return "weekly", [1, 4, 12]
    if 26 <= days <= 32:
        return "monthly", [1, 3, 12]
    logger.info(f"[feature_eng] timeseries: taninmayan frekans (medyan ~{days:.1f} gun) → varsayilan [1] + WARN")
    return "unknown", [1]


def _build_timeseries_features(
    df: pd.DataFrame, feature_plan: dict, resolved_time_col: str, meta: "ColumnMetadata",
    entity_col: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Zaman anahtarina gore sirala; frekanstan dinamik pencere turet; hedef + secili
    sayisal feature'larin LAG (FAZ A) ve ROLLING mean/std (FAZ B1) feature'larini uret;
    tum NaN-lead satirlari (lag+rolling birlikte) TEK seferde at — fillna'dan ONCE.
    Hem lag hem rolling YALNIZCA gecmise bakar (lag: shift(k>0); rolling: shift(1).rolling(w))
    → simdiki/gelecek deger sizmaz. y henuz ayrilmadi → tek df maskesi, hizalama derdi yok.

    ``entity_col`` (FAZ B2-a): BOS → GLOBAL yol (FAZ A/B1 birebir). DOLU → multi-entity
    panel: sort/lag/rolling/frekans GRUP ICINDE hesaplanir (bir entity'nin satiri
    digerinin penceresine sizmaz). Split bu adimda hala GLOBAL (B2-b duzeltecek)."""
    parsed = pd.to_datetime(df[resolved_time_col], errors="coerce")
    freq, windows = _detect_frequency(parsed, group_key=df[entity_col] if entity_col else None)

    if entity_col:
        logger.info(f"[feature_eng] timeseries: GRUP-ICI mod, entity_col='{entity_col}' "
              f"({df[entity_col].nunique()} entity) — lag/rolling/frekans entity icinde hesaplanir")
    else:
        # GLOBAL mod: identifier varsa panel-leakage WARN (entity_col secilmedi).
        id_cols = [c for c, info in meta.columns.items()
                   if info.get("semantic_type") == "identifier" and c in df.columns]
        if id_cols:
            ex = id_cols[0]
            try:
                n_ent = int(df[ex].nunique())
            except Exception:
                n_ent = -1
            logger.info(f"[feature_eng] WARN timeseries lag: identifier kolon(lar) {id_cols} "
                  f"(orn. '{ex}' → {n_ent} entity) ama entity_col SECILMEDI → GLOBAL shift; "
                  f"entity sinirinda sizinti olabilir. entity_col secersen grup-ici hesaplanir.")

    # Kronolojik sirala (kararli/stabil, eski→yeni). NaT'lar en eskiye.
    # entity_col dolu → COK-ANAHTARLI [entity, zaman]: her entity bitisik VE kendi icinde
    # sirali olur ki groupby.shift "grup icinde zamanca onceki" anlamina gelsin.
    if entity_col:
        df = df.assign(__parsed=parsed.fillna(parsed.min()).values)
        df = df.sort_values([entity_col, "__parsed"], kind="mergesort").reset_index(drop=True)
        df = df.drop(columns="__parsed")
    else:
        order = np.argsort(parsed.fillna(parsed.min()).to_numpy(), kind="mergesort")
        df = df.iloc[order].reset_index(drop=True)

    n = len(df)
    windows = [w for w in windows if w < n]   # veri boyutunu asan pencereyi ele (hepsini dropdan kacin)
    if not windows:
        logger.info(f"[feature_eng] timeseries: veri ({n} satir) pencereler icin cok kucuk → lag/rolling atlandi")
        return df, {"frequency": freq, "windows": [], "roll_windows": [], "lag_columns": [],
                    "rolling_columns": [], "lag_base_columns": [], "dropped_lead_rows": 0,
                    "entity_col": entity_col or None, "dropped_entities": 0,
                    "dropped_rows_from_small_entities": 0}

    target_name = feature_plan.get("target", {}).get("column_name", "")
    skip = {resolved_time_col}
    if entity_col:
        skip.add(entity_col)        # entity id'sini feature olarak laglemeyiz
    for col, info in meta.columns.items():
        if info.get("semantic_type") in ("identifier", "timestamp"):
            skip.add(col)

    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in skip and c != target_name
                    and not c.startswith(_DERIVED_FLAG_PREFIX)]   # was_missing tasiyicilari laglenmez

    # Kolon SECIMI (cok sayisal kolon varsa feature patlamasini onle): deterministik,
    # VERIYE DAYALI kriter = min-max normalize edilmis varyans (olcek-bagimsiz yayilim).
    # Model feature importance bu asamada YOK; IE importance_score'a guvenmiyoruz (cogu
    # zaman 0.5'e duser → sessizce "ilk 5 kolon" secimi). Yayilimi yuksek kolonlar
    # (lag'lemeye deger sinyal) once gelir; esitlikte kolon adina gore stabil siralanir.
    if len(numeric_cols) > _TS_MAX_FEATURE_COLS:
        def _norm_var(c: str) -> float:
            s = pd.to_numeric(df[c], errors="coerce")
            lo, hi = s.min(), s.max()
            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                return 0.0
            return float(((s - lo) / (hi - lo)).var())
        ranked = sorted(numeric_cols, key=lambda c: (-_norm_var(c), c))
        lag_base_cols = ranked[:_TS_MAX_FEATURE_COLS]
        logger.info(f"[feature_eng] timeseries lag: {len(numeric_cols)} sayisal kolondan "
              f"normalize-varyans (yayilim) ile {_TS_MAX_FEATURE_COLS} secildi: {lag_base_cols}")
    else:
        lag_base_cols = list(numeric_cols)

    # Hedef sayisalsa onu da ekle: gecmis target (lag_k_target) mesru feature
    if target_name in df.columns and pd.api.types.is_numeric_dtype(df[target_name]):
        lag_base_cols = [target_name] + lag_base_cols

    lag_cols_created: list[str] = []
    for col in lag_base_cols:
        for k in windows:
            new_col = f"lag_{k}_{col}"
            if new_col in df.columns:
                continue
            if entity_col:
                df[new_col] = df.groupby(entity_col)[col].shift(k)   # grup-ici gecmis
            else:
                df[new_col] = df[col].shift(k)   # shift>0 = gecmis; gelecek sizmaz
            lag_cols_created.append(new_col)

    # ── ROLLING (FAZ B1): trend (mean) + oynaklik (std) ──────────
    # Pencereler lag pencerelerinden gelir AMA 1-birim ATLANIR (1-birim rolling =
    # degerin kendisi, anlamsiz). KRITIK: shift(1).rolling(w) → pencere yalnizca
    # GECMISI (t-1..t-w) gorur, simdiki satiri (t) ASLA icermez → gelecek/simdi sizmaz.
    # Grup-ici modda transform(lambda s: s.shift(1).rolling(w)...) ZORUNLU: groupby.shift
    # sonrasi duz .rolling grup sinirini UNUTUR (bir entity'nin ilki oncekinin sonuyla
    # karisir = leakage); transform kaydirma+pencereyi grup ICINDE tutar.
    roll_windows = [w for w in windows if w >= 2]
    rolling_cols_created: list[str] = []
    for col in lag_base_cols:
        for w in roll_windows:
            m, s = f"roll{w}_mean_{col}", f"roll{w}_std_{col}"
            if m not in df.columns:
                if entity_col:
                    df[m] = df.groupby(entity_col)[col].transform(
                        lambda x, w=w: x.shift(1).rolling(w).mean())
                else:
                    df[m] = df[col].shift(1).rolling(w).mean()
                rolling_cols_created.append(m)
            if s not in df.columns:
                if entity_col:
                    df[s] = df.groupby(entity_col)[col].transform(
                        lambda x, w=w: x.shift(1).rolling(w).std())
                else:
                    df[s] = df[col].shift(1).rolling(w).std()
                rolling_cols_created.append(s)

    # NaN-lead drop: lag + rolling kolonlarinin NaN urettigi ilk satirlari TEK seferde at
    # — fillna/to_numeric'ten ONCE (yoksa lead satirlar sahte 0 ile modele girer).
    # Grup-ici modda bu dropna otomatik entity-aware: her entity'nin ilk max-pencere satiri
    # NaN olur ve duser. y henuz df icinde → hizalama derdi yok.
    before = len(df)
    counts_before = df[entity_col].value_counts() if entity_col else None
    all_new = lag_cols_created + rolling_cols_created
    if all_new:
        df = df.dropna(subset=all_new).reset_index(drop=True)
    dropped = before - len(df)

    # Kucuk-entity raporu (TUZAK 3): penceresinden kisa entity'nin HER satiri NaN → komple
    # duser. Yari-doldurma (min_periods) YOK; ama sessiz kalmasin → say + raporla.
    dropped_entities = 0
    dropped_rows_small = 0
    if entity_col and counts_before is not None:
        gone = set(counts_before.index) - set(df[entity_col].unique())
        dropped_entities = len(gone)
        dropped_rows_small = int(counts_before.loc[list(gone)].sum()) if gone else 0
        if dropped_entities:
            logger.info(f"[feature_eng] WARN grup-ici: {dropped_entities} entity penceresinden kisa → "
                  f"KOMPLE dustu ({dropped_rows_small} satir). Kucuk entity'ler atilir (yari-doldurma YOK).")

    info = {
        "frequency": freq, "windows": windows, "roll_windows": roll_windows,
        "lag_columns": lag_cols_created, "rolling_columns": rolling_cols_created,
        "lag_base_columns": lag_base_cols, "dropped_lead_rows": dropped,
        "entity_col": entity_col or None,
        "dropped_entities": dropped_entities,
        "dropped_rows_from_small_entities": dropped_rows_small,
    }
    logger.info(f"[feature_eng] timeseries: frekans={freq}, lag pencereleri={windows}, "
          f"rolling pencereleri={roll_windows}, {len(lag_cols_created)} lag + "
          f"{len(rolling_cols_created)} rolling kolonu uretildi, {dropped} lead satir atildi"
          + (f", {dropped_entities} kucuk entity dustu" if entity_col else ""))
    return df, info


# ── Aggregation ──────────────────────────────────────────────
def _aggregate_if_needed(
    df: pd.DataFrame, y: pd.Series, plan: dict, meta: ColumnMetadata,
    entity_resolution: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    # Entity resolution zaten aggregate ettiyse, tekrar aggregate ETME
    if entity_resolution and entity_resolution.get("action") == "aggregate":
        logger.info("[feature_eng] Entity resolution zaten aggregate etti → "
              "feature_builder aggregation atlanıyor (çift aggregation önlendi)")
        return df, y

    agg_config = plan.get("aggregation", {})
    if not agg_config.get("required", False):
        return df, y

    group_col = agg_config.get("group_by_column","")
    if not group_col or group_col not in df.columns:
        logger.info(f"[feature_eng] group_by '{group_col}' bulunamadi, aggregation atlandi")
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

    logger.info(f"[feature_eng] Aggregation: {len(customer_df):,} satır × {len(customer_df.columns)} kolon")
    return customer_df, y_out


# ── Encoding ─────────────────────────────────────────────────
def _encode_categorical(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if not cat_cols: return df
    low  = [c for c in cat_cols if df[c].nunique() <= 10]
    high = [c for c in cat_cols if df[c].nunique() >  10]
    if low:
        df = pd.get_dummies(df, columns=low, drop_first=True, dtype=int)
        logger.info(f"[feature_eng] One-hot ({len(low)}): {low}")
    if high:
        try:
            # sklearn 1.3+ TargetEncoder expects 2D array for X and 1D for y
            df[high] = TargetEncoder(random_state=settings.RANDOM_SEED).fit_transform(df[high], y)
            logger.info(f"[feature_eng] Target encoding ({len(high)}): {high}")
        except Exception as e:
            logger.error(f"[feature_eng] Target encoding hatasi → kolonlar drop ediliyor: {e}")
            df = df.drop(columns=high, errors="ignore")
    return df


# ── Scaling ──────────────────────────────────────────────────
def _scale_features(df: pd.DataFrame, outlier_cols: list[str]) -> pd.DataFrame:
    num = df.select_dtypes(include="number").columns.tolist()
    if not num: return df
    # inf temizliği: ratio/derived_feature kolonları 0'a bölünmeden inf üretebilir.
    # Scaler inf kabul etmez → inf'leri NaN'a, sonra kolon medyanına (yoksa 0) çevir.
    # Not: kolonlar nullable/object dtype olabilir (Int64, Float64, ratio çıktıları),
    # bu yüzden isinf öncesi her kolonu güvenle float'a coerce ediyoruz.
    inf_cols = [
        c for c in num
        if np.isinf(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")).any()
    ]
    if inf_cols:
        df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
        df[inf_cols] = df[inf_cols].fillna(df[inf_cols].median()).fillna(0.0)
        logger.info(f"[feature_eng] inf temizlendi (scaling oncesi): {inf_cols}")
    robust   = [c for c in outlier_cols if c in num]
    standard = [c for c in num if c not in robust]
    if robust:
        df[robust]   = RobustScaler().fit_transform(df[robust])
        logger.info(f"[feature_eng] RobustScaler: {len(robust)} kolon")
    if standard:
        df[standard] = StandardScaler().fit_transform(df[standard])
        logger.info(f"[feature_eng] StandardScaler: {len(standard)} kolon")
    return df


# ── Feature selection ────────────────────────────────────────
def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number").columns.tolist()
    if not num: return df
    low_var = df[num].var()[lambda s: s < VARIANCE_THRESHOLD].index.tolist()
    if low_var:
        df = df.drop(columns=low_var)
        num = [c for c in num if c not in low_var]
        logger.info(f"[feature_eng] Low variance drop: {low_var}")
    if len(num) > 1:
        corr  = df[num].corr().abs()
        mask  = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.where(mask)  # type: ignore
        hc    = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
        if hc:
            df = df.drop(columns=hc)
            logger.info(f"[feature_eng] High corr drop: {hc}")
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
        logger.info(f"[leakage] Domain drop iptal: drop sonrasi {remaining_after_drop} feature kalacak (min 2 gerekli)")
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
        logger.info(f"[leakage] DOMAIN DROP: '{col_name}' "
              f"({c.get('relationship_type')}, high) - "
              f"{c.get('reasoning', '')[:80]}")

    for derived_col in derived_to_drop:
        domain_dropped.append({
            "column": derived_col,
            "reason": "domain leakage (türev — parent kolon high confidence leaky)",
            "relationship_type": "temporal_derived",
        })
        logger.info(f"[leakage] DOMAIN DROP (türev): '{derived_col}' — leaky parent: {leaky_col_names}")

    return df, domain_dropped

# ── Data leakage detection ───────────────────────────────────
# ── Data leakage detection (Adım 1: identifier + kesin korelasyon) ──
def _binned_max_purity(s: pd.Series, yv: pd.Series, n_bins: int = 10,
                       min_frac: float = 0.03) -> float:
    """Sayisal feature'i binle; yeterince buyuk bir bolmede target tek sinifa
    kilitleniyorsa (saflik) bu safliklarin MAX'ini dondurur. Dusuk kardinalite
    (kategorik/indicator) → her deger ayri grup; yuksek → qcut. Multiclass icin
    AUC'ye gerek kalmadan calisir (herhangi sayida sinif)."""
    n = len(yv)
    d = pd.DataFrame({"s": s.values, "y": yv.values}).dropna()
    if len(d) < 30:
        return 0.0
    if d["s"].nunique() <= n_bins:
        groups = d.groupby("s", observed=True)["y"]
    else:
        try:
            bins = pd.qcut(d["s"], q=n_bins, duplicates="drop")
        except Exception:
            return 0.0
        groups = d.groupby(bins, observed=True)["y"]
    floor = max(10, min_frac * n)
    best = 0.0
    for _, g in groups:
        if len(g) < floor:
            continue
        purity = float(g.value_counts(normalize=True).max())
        if purity > best:
            best = purity
    return best


def _single_feature_target_lock(
    df: pd.DataFrame, y: pd.Series,
    *, auc_drop: float = 0.998, auc_suspect: float = 0.97,
    reg_drop: float = 0.999, reg_suspect: float = 0.95,
    pur_drop: float = 0.99, pur_suspect: float = 0.97,
) -> tuple[list[dict], list[dict]]:
    """Tek-feature 'hedef kilidi' tespiti — DOĞRUSAL OLMAYAN sızıntıları yakalar.

    Lineer korelasyon, ``ALERT_ID`` gibi 'bir eşikten target'ı birebir ayıran ama lineer
    korelasyonu düşük' kolonları kaçırır (−1 vs pozitif id → ağaç tek split'le ayırır,
    corr ~orta). Burada her sayısal feature'ın TEK BAŞINA target'ı ne kadar ayırdığını
    ölçeriz: binary → tek-feature AUC (yöne duyarsız, max(auc,1−auc)); multiclass →
    bin-bazlı max sınıf saflığı (kategorik/indicator dahil); regresyon → |Spearman|.
    Eşik üstü = sızıntı.

    df'i DEĞİŞTİRMEZ; (drop_listesi, suspect_listesi) döndürür — {"column","reason"}."""
    drop: list[dict] = []
    suspect: list[dict] = []
    if len(y) < 30:
        return drop, suspect
    yv = pd.to_numeric(y, errors="coerce")
    if yv.isna().mean() > 0.2:
        return drop, suspect
    n_classes = len(yv.dropna().unique())
    if n_classes < 2:
        return drop, suspect
    mode = "binary" if n_classes == 2 else ("multiclass" if n_classes <= 20 else "regression")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.nunique() <= 1:
            continue
        mask = s.notna() & yv.notna()
        if int(mask.sum()) < 30:
            continue
        try:
            if mode == "binary":
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(yv[mask], s[mask]))
                score = max(auc, 1.0 - auc)
                if score >= auc_drop:
                    drop.append({"column": col, "reason": f"tek-feature AUC={score:.4f} → hedef kilidi (leakage)"})
                elif score >= auc_suspect:
                    suspect.append({"column": col, "reason": f"tek-feature AUC={score:.3f} (asiri ayirt edici)"})
            elif mode == "multiclass":
                purity = _binned_max_purity(s[mask], yv[mask])
                if purity >= pur_drop:
                    drop.append({"column": col, "reason": f"tek-feature bin-safligi={purity:.3f} → hedef kilidi (leakage)"})
                elif purity >= pur_suspect:
                    suspect.append({"column": col, "reason": f"tek-feature bin-safligi={purity:.3f} (asiri ayirt edici)"})
            else:  # regression
                rho = s[mask].corr(yv[mask], method="spearman")
                if pd.isna(rho):
                    continue
                rho = abs(float(rho))
                if rho >= reg_drop:
                    drop.append({"column": col, "reason": f"tek-feature |Spearman|={rho:.4f} → hedef kilidi"})
                elif rho >= reg_suspect:
                    suspect.append({"column": col, "reason": f"tek-feature |Spearman|={rho:.3f} (asiri ayirt edici)"})
        except Exception:
            continue
    return drop, suspect


def _detect_leakage(
    df: pd.DataFrame,
    y: pd.Series,
    meta=None,  # ColumnMetadata, identifier tespiti için
    protected_high_card=None,  # SADECE kardinalite (KESİN 2) kuralından muaf kolonlar
) -> tuple[pd.DataFrame, dict]:
    """
    Leakage tespiti — iki seviyeli:
      KESİN  → otomatik drop (identifier, ~birebir korelasyon)
      ŞÜPHELİ → drop etme, uyarı listesine ekle

    ``protected_high_card``: time-series lag kolonlari gibi deterministik turetilmis,
    continuous oldugu icin dogal olarak yuksek-kardinaliteli ama ID OLMAYAN kolonlar.
    Bunlar YALNIZCA KESİN 2 (kardinalite) heuristiginden muaftir; identifier (KESİN 1)
    ve korelasyon/sizinti kontrolu onlara da AYNEN uygulanir.

    Dönüş:
      df              : temizlenmiş dataframe
      leakage_report  : {"dropped": [...], "suspected": [...]}
    """
    protected_high_card = protected_high_card or set()
    dropped = []      # otomatik atılanlar (kesin)
    suspected = []    # şüpheli, kullanıcıya raporlanacak
    n_rows = len(df)
    logger.info(f"\n[leakage] === BASLIYOR ===")
    logger.info(f"[leakage] df shape: {df.shape}, y dtype: {y.dtype}")

    # y'yi numeric'e çevir (korelasyon için), başarısızsa korelasyon atla
    try:
        y_num = pd.to_numeric(y, errors="coerce").astype(float)
        y_valid = y_num.notna().mean() > 0.8
    except Exception:
        y_valid = False
        y_num = None

    for col in list(df.columns):
        sem = meta.columns.get(col, {}).get("semantic_type", "") if meta is not None else ""

        # ── KESİN 1: Identifier semantic_type ──
        if sem == "identifier":
            df = df.drop(columns=[col], errors="ignore")
            dropped.append({"column": col, "reason": "identifier (semantic_type)"})
            logger.info(f"[leakage] KESIN drop: '{col}' (identifier)")
            continue

        # ── KESİN 2: Aşırı yüksek kardinalite (ID-benzeri) ──
        # KRİTİK: Sürekli ÖLÇÜMLER (PCA bileşenleri V1..V28, float metrikler, tutar) doğal
        # olarak yüksek-kardinalitelidir ama ID DEĞİLdir → onları ID sanıp atma. Gerçek
        # identifier'lar integer/string'dir. Aksi halde bütün ayırt edici sürekli feature'lar
        # silinir (creditcard V1-V28 vakası). Muaf: float dtype VEYA protected_high_card
        # (time-series lag). Korelasyon/sızıntı kontrolü onlara yine uygulanır.
        #
        # ÖNEMLİ: semantic_type 'metric'/'numeric' muafiyeti YALNIZCA ~%100 benzersiz
        # OLMAYAN kolonlar için geçerli. Intelligence engine bir tamsayı kimliği (örn.
        # customer_id) çoğu zaman "metric/numeric" etiketler; ama satır-başına benzersiz
        # bir integer/string kolon — etiketi ne olursa olsun — tanımlayıcıdır (gerçek bir
        # tamsayı ölçüm neredeyse hiç %100 benzersiz olmaz). Bu yüzden near-unique
        # kolonlarda semantic muafiyetini uygulama; sadece float ölçümleri koru.
        nunique = df[col].nunique()
        near_unique = n_rows > 0 and nunique / n_rows > 0.95
        is_continuous = (
            pd.api.types.is_float_dtype(df[col])
            or (sem in ("metric", "numeric") and not near_unique)
        )
        if col not in protected_high_card and not is_continuous and near_unique:
            df = df.drop(columns=[col], errors="ignore")
            dropped.append({"column": col, "reason": f"asiri kardinalite ({nunique}/{n_rows}, ID-benzeri)"})
            logger.info(f"[leakage] KESIN drop: '{col}' (kardinalite orani {nunique/n_rows:.2f}, ID-benzeri)")
            continue

    # ── Korelasyon tabanlı (yalnızca numeric + y numeric ise) ──
    if y_valid:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        logger.info(f"[leakage DEBUG] numeric cols: {num_cols}")
        logger.info(f"[leakage DEBUG] y_num NaN orani: {y_num.isna().mean():.2f}")
        for col in num_cols:
            if df[col].nunique() <= 1:
                continue
            try:
                col_vals = pd.to_numeric(df[col], errors="coerce").astype(float)
                corr = abs(col_vals.corr(y_num))
                logger.info(f"[leakage DEBUG] '{col}' corr = {corr}")
            except Exception as e:
                logger.error(f"[leakage DEBUG] '{col}' corr hatasi: {e}")
                continue
            if pd.isna(corr):
                continue

            if corr > 0.999:
                # KESİN: birebir kopya (örn. target'in kendisi/duplikası)
                df = df.drop(columns=[col], errors="ignore")
                dropped.append({"column": col, "reason": f"birebir korelasyon ({corr:.4f})"})
                logger.info(f"[leakage] KESIN drop: '{col}' (corr={corr:.4f})")
            elif corr > 0.85:
                # ŞÜPHELİ: yüksek ama kesin değil → drop etme, uyar
                suspected.append({"column": col, "reason": f"yuksek korelasyon ({corr:.3f})"})
                logger.warning(f"[leakage] SUPHELI: '{col}' (corr={corr:.3f}) — droplanmadi, uyari verildi")

    # ── Tek-feature hedef kilidi (NON-LINEER sizinti) ──
    # Korelasyonun kacirdigi 'esikten birebir ayiran' kolonlari (ALERT_ID, *_isna) yakalar.
    if y_valid:
        tl_drop, tl_suspect = _single_feature_target_lock(df, y)
        for d in tl_drop:
            if d["column"] in df.columns:
                df = df.drop(columns=[d["column"]], errors="ignore")
                dropped.append(d)
                logger.info(f"[leakage] KESIN drop: '{d['column']}' ({d['reason']})")
        already = {x["column"] for x in dropped} | {x["column"] for x in suspected}
        for s in tl_suspect:
            if s["column"] not in already:
                suspected.append(s)
                logger.warning(f"[leakage] SUPHELI: '{s['column']}' ({s['reason']}) — droplanmadi")

    return df, {"dropped": dropped, "suspected": suspected}


# ── Leakage-safe (train-fit) transform variantlari ───────────
# Asagidaki *_tt fonksiyonlari yukaridaki tekil-df siblinglerinin TRAIN/TEST
# ayrik versiyonudur: TUM istatistik (target ortalamasi, mean/std, varyans,
# korelasyon) YALNIZCA X_train'den ogrenilir ve X_test'e UYGULANIR. Boylece test
# dagilimi/target'i egitime SIZMAZ (grandmaster disiplini) — holdout skoru durust
# kalir. Kolon hizalamasi (one-hot sonrasi gorulmeyen/eksik kategori) train kolon
# setine gore yapilir. Unsupervised (clustering/anomaly) yol tekil-df siblinglerini
# kullanmaya devam eder: orada holdout yok, transform tum veriye fit edilir.
def _encode_categorical_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        return X_train, X_test
    # Kardinalite karari TRAIN'e gore (test'e bakmadan).
    low  = [c for c in cat_cols if X_train[c].nunique() <= 10]
    high = [c for c in cat_cols if X_train[c].nunique() >  10]
    if low:
        X_train = pd.get_dummies(X_train, columns=low, drop_first=True, dtype=int)
        X_test  = pd.get_dummies(X_test,  columns=low, drop_first=True, dtype=int)
        # Test'i train kolon setine hizala: train'de olmayan dummy (gorulmeyen
        # kategori) duser, test'te eksik dummy 0 ile doldurulur → ayni kolon seti.
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        logger.info(f"[feature_eng] One-hot ({len(low)}, train-fit): {low}")
    if high:
        try:
            enc = TargetEncoder(random_state=settings.RANDOM_SEED)
            X_train[high] = enc.fit_transform(X_train[high], y_train)
            X_test[high]  = enc.transform(X_test[high])
            logger.info(f"[feature_eng] Target encoding ({len(high)}, train-fit): {high}")
        except Exception as e:
            logger.error(f"[feature_eng] Target encoding hatasi → kolonlar drop ediliyor: {e}")
            X_train = X_train.drop(columns=high, errors="ignore")
            X_test  = X_test.drop(columns=high, errors="ignore")
    return X_train, X_test


def _scale_features_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, outlier_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = X_train.select_dtypes(include="number").columns.tolist()
    if not num:
        return X_train, X_test
    # inf temizligi: train median'i ile (test'e de TRAIN median uygulanir).
    inf_cols = [
        c for c in num
        if np.isinf(pd.to_numeric(X_train[c], errors="coerce").to_numpy(dtype="float64")).any()
        or np.isinf(pd.to_numeric(X_test[c],  errors="coerce").to_numpy(dtype="float64")).any()
    ]
    if inf_cols:
        X_train[inf_cols] = X_train[inf_cols].replace([np.inf, -np.inf], np.nan)
        X_test[inf_cols]  = X_test[inf_cols].replace([np.inf, -np.inf], np.nan)
        med = X_train[inf_cols].median()
        X_train[inf_cols] = X_train[inf_cols].fillna(med).fillna(0.0)
        X_test[inf_cols]  = X_test[inf_cols].fillna(med).fillna(0.0)
        logger.info(f"[feature_eng] inf temizlendi (train median, scaling oncesi): {inf_cols}")
    robust   = [c for c in outlier_cols if c in num]
    standard = [c for c in num if c not in robust]
    if robust:
        sc = RobustScaler().fit(X_train[robust])
        X_train[robust] = sc.transform(X_train[robust])
        X_test[robust]  = sc.transform(X_test[robust])
        logger.info(f"[feature_eng] RobustScaler (train-fit): {len(robust)} kolon")
    if standard:
        sc = StandardScaler().fit(X_train[standard])
        X_train[standard] = sc.transform(X_train[standard])
        X_test[standard]  = sc.transform(X_test[standard])
        logger.info(f"[feature_eng] StandardScaler (train-fit): {len(standard)} kolon")
    return X_train, X_test


def _select_features_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = X_train.select_dtypes(include="number").columns.tolist()
    if not num:
        return X_train, X_test
    # Dusuk varyans: TRAIN'de hesapla, ikisinden de dus.
    low_var = X_train[num].var()[lambda s: s < VARIANCE_THRESHOLD].index.tolist()
    if low_var:
        X_train = X_train.drop(columns=low_var)
        X_test  = X_test.drop(columns=low_var, errors="ignore")
        num = [c for c in num if c not in low_var]
        logger.info(f"[feature_eng] Low variance drop (train-fit): {low_var}")
    # Yuksek korelasyon: TRAIN korelasyon matrisine gore.
    if len(num) > 1:
        corr  = X_train[num].corr().abs()
        mask  = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.where(mask)  # type: ignore
        hc    = [c for c in upper.columns if any(upper[c] > CORRELATION_THRESHOLD)]
        if hc:
            X_train = X_train.drop(columns=hc)
            X_test  = X_test.drop(columns=hc, errors="ignore")
            logger.info(f"[feature_eng] High corr drop (train-fit): {hc}")
    return X_train, X_test


def _select_univariate_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, intent: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hedefle istatistiksel ilişkisi ANLAMSIZ (ANOVA/f_regression p-value > eşik)
    sayısal feature'ları eler — _select_features_tt feature-feature redundancy'yi
    atarken bu, target-irrelevant gürültü kolonlarını atar (notebook'taki
    SelectKBest/corrwith tekniğinin leakage-safe karşılığı).

    İstatistik YALNIZCA X_train'den öğrenilir, aynı kolonlar X_test'ten de düşürülür
    (P0 disiplini). Keyfi skor eşiği yerine p-value kullanılır. Over-prune koruması:
    feature seti küçükse hiç uygulanmaz ve seçim sonrası en az UNIVARIATE_MIN_KEEP
    feature garanti kalır. Drop edilen kolon adlarını döndürür."""
    from sklearn.feature_selection import f_classif, f_regression

    is_cls = intent.strip().lower() == "classification"
    score_fn = f_classif if is_cls else f_regression
    num = X_train.select_dtypes(include="number").columns.tolist()
    if len(num) < UNIVARIATE_MIN_COLS:
        return X_train, X_test, []
    y = np.asarray(y_train)
    if is_cls and len(np.unique(y)) < 2:  # tek sınıf → ANOVA tanımsız
        return X_train, X_test, []
    try:
        _f, pvals = score_fn(X_train[num].to_numpy(dtype="float64"), y)
    except Exception as e:
        logger.info(f"[feature_eng] Univariate seçim atlandı ({e})")
        return X_train, X_test, []
    pvals = np.nan_to_num(np.asarray(pvals, dtype="float64"), nan=1.0)  # sabit feature → anlamsız
    insignificant = [num[i] for i in range(len(num)) if pvals[i] > UNIVARIATE_PVALUE]
    if not insignificant:
        return X_train, X_test, []
    # En az UNIVARIATE_MIN_KEEP feature kalmalı: gerekirse en yüksek p-value'lulardan
    # başlayarak yalnızca izin verilen sayıda drop et.
    if len(num) - len(insignificant) < UNIVARIATE_MIN_KEEP:
        order = sorted(insignificant, key=lambda c: pvals[num.index(c)], reverse=True)
        insignificant = order[: max(0, len(num) - UNIVARIATE_MIN_KEEP)]
    if not insignificant:
        return X_train, X_test, []
    X_train = X_train.drop(columns=insignificant)
    X_test  = X_test.drop(columns=insignificant, errors="ignore")
    logger.info(f"[feature_eng] Univariate ({'ANOVA' if is_cls else 'f_regression'}) "
                f"irrelevant drop (p>{UNIVARIATE_PVALUE}, train-fit): {insignificant}")
    return X_train, X_test, insignificant


def _detect_leakage_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
    meta=None, protected_high_card=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """_detect_leakage'i YALNIZCA train uzerinde kosar (korelasyon/kardinalite
    train'den ogrenilir); tespit edilip dusurulen kolonlari test'ten de dusurur."""
    cols_before = set(X_train.columns)
    X_train, report = _detect_leakage(X_train, y_train, meta, protected_high_card=protected_high_card)
    dropped = cols_before - set(X_train.columns)
    if dropped:
        X_test = X_test.drop(columns=[c for c in dropped if c in X_test.columns], errors="ignore")
    return X_train, X_test, report


def _apply_domain_leakage_drops_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, domain_leakage_report: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """_apply_domain_leakage_drops'u train uzerinde uygular; dusurulen kolonlari
    test'ten de senkron dusurur (ayni kolon seti garantisi)."""
    cols_before = set(X_train.columns)
    X_train, domain_dropped = _apply_domain_leakage_drops(X_train, domain_leakage_report)
    dropped = cols_before - set(X_train.columns)
    if dropped:
        X_test = X_test.drop(columns=[c for c in dropped if c in X_test.columns], errors="ignore")
    return X_train, X_test, domain_dropped


# ── Grandmaster feature aileleri (leakage-safe) ──────────────
# Uc generik, target KULLANMAYAN feature ailesi. Hepsi deterministik + kapali (cap'li)
# → feature patlamasi yok. Missing-indicator row-wise (split oncesi); count-encoding ve
# groupby-deviation ISTATISTIK ogrenir → SADECE train'e fit, test'e uygulanir (P0
# disiplini). Hicbiri y'yi kullanmaz, dolayisiyla target sizintisi tasimaz.
_COUNT_ENCODE_MAX_CARD = 200    # bundan yuksek kardinalite (ID-benzeri) → count'lama
_COUNT_ENCODE_MIN_CARD = 2
_GROUPBY_MAX_CARD      = 50     # gruplama kolonu en fazla bu kart (yuksekse grup ort. anlamsiz)
_GROUPBY_MAX_NUM_COLS  = 3      # patlama korumasi: en fazla N sayisal × M kategorik
_GROUPBY_MAX_CAT_COLS  = 3
_GROUPBY_MIN_GROUP_SIZE = 30    # grup istatistigi ancak >= bu kadar satir varsa guvenilir; kucuk grup → global fallback
_GROUPBY_MAX_FEATURES  = 30     # sert backstop: dev+zdev+grpstd toplami bu kadari gecemez (patlama korumasi)
_MISSING_IND_MAX_COLS  = 30


def _leakage_exclude_cols(feature_plan: dict, meta, domain_leakage_report: dict) -> set:
    """Auto-feature ailelerinin TURETMEMESI gereken kolonlar: target + semantic_type
    identifier/timestamp/target + domain leakage suspect'leri. Bu kolonlardan _isna/
    count_/dev_ uretmek 'own-goal' leakage yaratir (orn. ALERT_TYPE_isna)."""
    excl: set = set()
    tgt = ((feature_plan or {}).get("target") or {}).get("column_name", "")
    if tgt:
        excl.add(tgt)
    cols_meta = getattr(meta, "columns", {}) or {}
    for col, info in cols_meta.items():
        if (info or {}).get("semantic_type", "") in ("identifier", "timestamp", "target"):
            excl.add(col)
    for c in (domain_leakage_report or {}).get("concerns", []):
        name = c.get("column_name") if isinstance(c, dict) else None
        if name:
            excl.add(name)
    return {c for c in excl if c}


def _add_missing_indicators(df: pd.DataFrame, max_cols: int = _MISSING_IND_MAX_COLS,
                            exclude: set | None = None) -> list[str]:
    """fillna ONCESI cagrilir: null tasiyan kolonlar icin '<col>_isna' (0/1) ekler →
    eksiklik sinyalini korur (impute bu bilgiyi siler). Row-wise: capraz-satir istatistik
    YOK → leakage YOK, split'ten once guvenli. Cok az/cok cok null tasiyan kolonlar dusuk
    varyansla _select_features'ta zaten elenir. df IN-PLACE; eklenen kolon adlarini dondurur.

    ``exclude``: target/identifier/leakage-suspect kolonlar — bunlardan '<col>_isna'
    TURETME (own-goal onleme): leaky bir kolon 'sadece pozitifte dolu' ise '<col>_isna'
    target'i birebir kopyalar; bu kolonu hic uretmeyiz."""
    exclude = exclude or set()
    null_cols = [c for c in df.columns if c not in exclude and df[c].isnull().any()]
    null_cols = sorted(null_cols, key=lambda c: df[c].isnull().mean(), reverse=True)[:max_cols]
    added: list[str] = []
    for c in null_cols:
        name = f"{c}_isna"
        if name in df.columns:
            continue
        df[name] = df[c].isnull().astype("int8")
        added.append(name)
    if added:
        logger.info(f"[feature_eng] Missing-indicator ({len(added)}): {added}")
    return added


def _count_encode_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, exclude: set | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Her uygun kategorik kolon icin 'count_<col>' = o kategorinin TRAIN frekansi.
    Frekanslar SADECE train'den ogrenilir; test train frekansiyla maplenir, gorulmeyen
    kategori → 0. Leakage-safe (target kullanmaz); ucuz ama guclu kategorik sinyal
    (nadir/sik kategori ayrimi). Encode ONCESI calismali (ham kategorikleri kullanir).
    ``exclude``: leakage-suspect/target kolonlardan 'count_' turetme (own-goal onleme)."""
    exclude = exclude or set()
    cat_cols = [c for c in X_train.select_dtypes(include="object").columns if c not in exclude]
    added: list[str] = []
    for c in cat_cols:
        card = X_train[c].nunique()
        if card < _COUNT_ENCODE_MIN_CARD or card > _COUNT_ENCODE_MAX_CARD:
            continue
        name = f"count_{c}"
        if name in X_train.columns:
            continue
        freq = X_train[c].value_counts()
        X_train[name] = X_train[c].map(freq).fillna(0).astype("int32")
        X_test[name]  = X_test[c].map(freq).fillna(0).astype("int32")
        added.append(name)
    if added:
        logger.info(f"[feature_eng] Count-encoding ({len(added)}, train-fit): {added}")
    return X_train, X_test, added


def _groupby_deviation_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, meta=None, exclude: set | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Grup-istatistik feature aileleri (her cat×num cifti icin, TRAIN-fit → test-map):
      * 'dev_<num>_by_<cat>'    = num − grup ortalamasi (satirin kendi grup-normundan sapmasi)
      * 'zdev_<num>_by_<cat>'   = (num − grup ort.) / grup std → OLCEK-BAGIMSIZ sapma
                                  (farkli olcekli kolonlari kiyaslanabilir yapar; sigma=0 → 0)
      * 'grpstd_<num>_by_<cat>' = grubun std'i (o kategorinin ic degiskenligi)
    Grandmaster tablo feature'larinin en guclulerinden (orn. bu musterinin harcamasi,
    ulkesi ortalamasindan kac std yukarida). TUM istatistikler SADECE train'den; gorulmeyen
    kategori → train GLOBAL istatistigi. Leakage-safe (target kullanmaz).

    MIN-GRUP KAPISI (``_GROUPBY_MIN_GROUP_SIZE``): grup istatistigi ancak grup >= N satir
    iceriyorsa guvenilir; kucuk gruplar (2-3 satir) gurultulu ortalama uretir → onlar global
    istatistige duser. Bir cat'te >=2 guvenilir grup yoksa o cat tamamen atlanir (feature
    dejenere/sabit olurdu). BACKSTOP (``_GROUPBY_MAX_FEATURES``): toplam uretimi cap'ler
    (deterministik sira: cat dis, num ic, aile sirasi dev→zdev→grpstd).

    Patlama korumasi: dusuk-kardinaliteli cat'ler × en yuksek normalize-varyansli sayisallar,
    cift sayisi cap'li. Encode ONCESI calismali (ham kategorikleri grup anahtari yapar).
    ``exclude``: leakage-suspect/target kolonlari ne grup anahtari ne sayisal girdi yap."""
    exclude = exclude or set()
    cat_cols = [c for c in X_train.select_dtypes(include="object").columns
                if c not in exclude and _GROUPBY_MAX_CARD >= X_train[c].nunique() >= 2]
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    # Turetilmis kolonlari grup-deviation GIRDISI yapma → gurultu/dongu onle.
    # zdev_/grpstd_ ayrica eklenir: startswith("dev_") 'zdev_'i YAKALAMAZ.
    num_cols = [c for c in num_cols
                if c not in exclude
                and not c.startswith(("count_", "dev_", "zdev_", "grpstd_", "is_zero_", "ratio_"))
                and not c.endswith("_isna")]
    if not cat_cols or not num_cols:
        return X_train, X_test, []

    # cat secimi: dusuk kardinalite once (grup basina daha cok satir → stabil ortalama)
    cat_cols = sorted(cat_cols, key=lambda c: X_train[c].nunique())[:_GROUPBY_MAX_CAT_COLS]

    # num secimi: olcek-bagimsiz yayilim (min-max normalize varyans) yuksek olanlar once
    def _norm_var(c: str) -> float:
        s = pd.to_numeric(X_train[c], errors="coerce")
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            return 0.0
        return float(((s - lo) / (hi - lo)).var())
    num_cols = sorted(num_cols, key=lambda c: (-_norm_var(c), c))[:_GROUPBY_MAX_NUM_COLS]

    added: list[str] = []

    def _emit(name: str, tr: "pd.Series", te: "pd.Series") -> None:
        if name in X_train.columns:
            return
        X_train[name] = tr.values
        X_test[name]  = te.values
        added.append(name)

    for cat in cat_cols:
        if len(added) >= _GROUPBY_MAX_FEATURES:
            break
        # Guvenilir grup kapisi: sadece >= N satirli gruplarin istatistigi kullanilir;
        # kalanlar (ve gorulmeyen test kategorileri) global istatistige duser.
        sizes = X_train[cat].value_counts()
        reliable = set(sizes[sizes >= _GROUPBY_MIN_GROUP_SIZE].index)
        if len(reliable) < 2:
            continue  # >=2 guvenilir kontrast grubu yok → feature dejenere olurdu, atla
        cat_tr, cat_te = X_train[cat], X_test[cat]
        for num in num_cols:
            if len(added) >= _GROUPBY_MAX_FEATURES:
                break
            xtr = pd.to_numeric(X_train[num], errors="coerce")
            xte = pd.to_numeric(X_test[num], errors="coerce")
            g = xtr.groupby(cat_tr)
            grp_mean = g.mean()
            grp_std  = g.std(ddof=0)   # ddof=0 → tek-degerli grup NaN degil 0 verir
            global_mean = float(xtr.mean())
            global_std  = float(xtr.std(ddof=0))
            if pd.isna(global_mean):
                global_mean = 0.0
            if pd.isna(global_std):
                global_std = 0.0
            # guvenilmez gruplari global'e sabitle (index bazli where)
            grp_mean = grp_mean.where(grp_mean.index.isin(reliable), global_mean)
            grp_std  = grp_std.where(grp_std.index.isin(reliable), global_std)

            mean_tr = cat_tr.map(grp_mean).fillna(global_mean)
            mean_te = cat_te.map(grp_mean).fillna(global_mean)
            std_tr  = cat_tr.map(grp_std).fillna(global_std)
            std_te  = cat_te.map(grp_std).fillna(global_std)

            # dev = x − grup ort.
            _emit(f"dev_{num}_by_{cat}",
                  (xtr - mean_tr).fillna(0.0), (xte - mean_te).fillna(0.0))
            # zdev = (x − grup ort.) / grup std ; sigma=0/NaN → 0 (guvenli)
            z_tr = ((xtr - mean_tr) / std_tr.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            z_te = ((xte - mean_te) / std_te.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            _emit(f"zdev_{num}_by_{cat}", z_tr, z_te)
            # grpstd = grubun std'i (grup ici degiskenlik)
            _emit(f"grpstd_{num}_by_{cat}", std_tr.fillna(0.0), std_te.fillna(0.0))

    if added:
        logger.info(f"[feature_eng] Groupby-stats ({len(added)}, train-fit, cap={_GROUPBY_MAX_FEATURES}): {added}")
    return X_train, X_test, added


# ── MAIN NODE ────────────────────────────────────────────────
_FEATURE_SPEC_EPS = 1e-9

# Bounds keep the deterministic ops numerically safe regardless of what the LLM picks.
_POWER_EXP_BOUNDS = (-3.0, 3.0)


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


# Safe, deterministic op palette the LLM may compose via FeatureSpec. Each fn
# takes ``(cols, params)`` — a list of pandas Series + a list of numeric constants —
# and returns one Series. Arity/params are validated against
# ``schemas.FEATURE_OP_SPEC`` (the single source of truth) before any fn runs.
_FEATURE_OPS = {
    "ratio":        lambda cols, p: cols[0] / (_prod_series(cols[1:]).abs() + _FEATURE_SPEC_EPS),
    "product":      lambda cols, p: _prod_series(cols),
    "difference":   lambda cols, p: cols[0] - cols[1],
    "zscore":       lambda cols, p: (cols[0] - cols[0].mean()) / (cols[0].std(ddof=0) + _FEATURE_SPEC_EPS),
    "log":          lambda cols, p: np.log1p(cols[0].clip(lower=0)),
    "power":        lambda cols, p: cols[0] ** float(p[0]),
    "weighted_sum": lambda cols, p: _weighted_sum_series(cols, p),
    "normalize":    lambda cols, p: (cols[0] - cols[0].min()) / ((cols[0].max() - cols[0].min()) + _FEATURE_SPEC_EPS),
}


def _sanitize_params(op: str, params: list) -> list:
    """Coerce/clamp an op's numeric params into a safe range. None if invalid."""
    try:
        out = [float(v) for v in (params or [])]
    except (TypeError, ValueError):
        return None
    if op == "power" and out:
        lo, hi = _POWER_EXP_BOUNDS
        out[0] = max(lo, min(hi, out[0]))
    return out


def _run_feature_specs(df: pd.DataFrame, specs: list, disallowed: set, *, source_label: str) -> list[str]:
    """Validate + deterministically apply a list of ``FeatureSpec``s in place.

    Shared executor for both the agent-driven path (``merchant_analysis``) and the
    generic plan-driven path (``feature_plan.derived_features``). Each spec is
    checked against the op palette (shape via ``FEATURE_OP_SPEC``), the actual
    schema (columns exist, name not already taken), and a leakage denylist before
    its op runs; anything invalid or unsafe is skipped, never executed. Returns the
    columns actually added.
    """
    from src.domains.schemas import validate_feature_spec_shape

    deny = {c for c in (disallowed or set()) if c}
    added: list[str] = []
    for spec in specs:
        try:
            if spec.op not in _FEATURE_OPS:
                continue
            params = _sanitize_params(spec.op, getattr(spec, "params", None))
            if params is None or not validate_feature_spec_shape(spec.op, len(spec.inputs), len(params)):
                logger.info(f"[feature_eng] FeatureSpec '{spec.name}' atlandi: gecersiz sekil ({spec.op})")
                continue
            if spec.name in df.columns or spec.name in added:
                continue
            missing = [c for c in spec.inputs if c not in df.columns]
            if missing:
                logger.info(f"[feature_eng] FeatureSpec '{spec.name}' atlandi: kolon yok {missing}")
                continue
            leaky = [c for c in spec.inputs if c in deny]
            if leaky:
                logger.info(f"[feature_eng] FeatureSpec '{spec.name}' atlandi: leakage girdisi {leaky}")
                continue
            cols = [pd.to_numeric(df[c], errors="coerce") for c in spec.inputs]
            series = _FEATURE_OPS[spec.op](cols, params)
            df[spec.name] = series.replace([np.inf, -np.inf], 0).fillna(0)
            added.append(spec.name)
            logger.info(f"[feature_eng] {source_label} feature'i uygulandi: "
                        f"{spec.name} = {spec.op}{tuple(spec.inputs)}"
                        + (f" params={params}" if params else ""))
        except Exception as e:
            logger.error(f"[feature_eng] FeatureSpec '{getattr(spec, 'name', '?')}' hatasi: {e}")
    return added


def _apply_feature_specs(df: pd.DataFrame, state: dict, disallowed: set) -> list[str]:
    """Apply agent-decided ``FeatureSpec``s (from ``merchant_analysis``) in place.

    This is the sector-specific "agent decides, builder executes" path: the expert
    agent chose these transforms over the REAL columns it investigated. Delegates to
    :func:`_run_feature_specs` for validation + execution. Returns columns added.
    """
    from src.domains.schemas import MerchantAnalysis

    analysis = MerchantAnalysis.from_dict(state.get("merchant_analysis"))
    if analysis is None:
        return []
    specs = [s for f in analysis.findings for s in (f.feature_specs or [])]
    if not specs:
        return []
    return _run_feature_specs(df, specs, disallowed, source_label="Ajan")


def _apply_plan_derived_features(df: pd.DataFrame, feature_plan: dict, disallowed: set) -> list[str]:
    """Apply the generic, sector-independent ``feature_plan.derived_features``.

    The feature planner LLM proposes dataset-meaningful derived columns (e.g.
    velocity = ratio(distance, time)) as structured FeatureSpecs — never raw code.
    This runs for ANY dataset, with or without a domain pack, through the same
    validated/leakage-guarded executor. Returns columns added.
    """
    from src.domains.schemas import FeatureSpec

    raw = feature_plan.get("derived_features") or []
    specs: list[FeatureSpec] = []
    for d in raw:
        try:
            specs.append(d if isinstance(d, FeatureSpec) else FeatureSpec(**d))
        except Exception as e:
            logger.info(f"[feature_eng] derived_feature atlandi (parse): {e}")
    if not specs:
        return []
    return _run_feature_specs(df, specs, disallowed, source_label="Plan")


def _apply_domain_recipes(df: pd.DataFrame, state: dict, meta: "ColumnMetadata") -> list[str]:
    """Run the active domain pack's deterministic FeatureRecipes on ``df`` in place.

    Returns the names of columns actually added. No-op (returns ``[]``) when no
    pack is active. Each recipe is individually guarded so one bad recipe can
    never block the others or the pipeline.
    """
    from src.domains import get_active_pack
    from src.domains.base import MerchantContext
    from src.domains.schemas import MerchantAnalysis

    pack = get_active_pack(state.get("domain_pack"))
    if pack.name == "none":
        return []

    ctx = MerchantContext.from_dict(state.get("merchant_context"))
    analysis = MerchantAnalysis.from_dict(state.get("merchant_analysis"))
    try:
        recipes = pack.feature_recipes(ctx, state.get("db_intelligence"), analysis) or []
    except Exception as e:
        logger.error(f"[feature_eng] feature_recipes() hatasi ({e}); atlaniyor.")
        return []

    added: list[str] = []
    for r in recipes:
        try:
            if not r.applies(df, meta):
                continue
            cols = r.build(df, meta) or {}
            for name, series in cols.items():
                if name in df.columns:
                    continue
                df[name] = series
                added.append(name)
        except Exception as e:
            logger.error(f"[feature_eng] Recipe '{getattr(r, 'name', '?')}' hatasi: {e}")
    return added


def _target_distribution(y_arr, intent: str) -> "tuple[dict | None, dict | None]":
    """Hedef dağılımını uygun temsile dönüştürür.

    Dönüş: ``(counts, histogram)`` — ikisinden biri her zaman ``None``, böylece
    frontend hangisini çizeceğini tek bakışta seçer:
      - ``counts``    : ``{deger: adet}`` value_counts — sınıflandırma / düşük
        kardinalite (mevcut donut davranışı AYNEN korunur).
      - ``histogram`` : ``{"bin_edges":[...], "counts":[...]}`` — regresyon /
        sürekli hedef; her benzersiz değeri ayrı "sınıf" yapan çorbayı önler.

    Regresyon kararı: ``intent == "regression"`` VEYA (sayısal hedef ve
    ``nunique > 20``). Anti-fabrication: tüm sayılar gerçek ``y_arr``'dan numpy
    ile hesaplanır; hiçbir değer uydurulmaz.
    """
    s = pd.Series(y_arr)
    is_numeric = pd.api.types.is_numeric_dtype(s)
    nunique = int(s.nunique(dropna=True))
    is_regression = (str(intent).strip().lower() == "regression") or (is_numeric and nunique > 20)

    if is_regression and is_numeric:
        data = s.dropna().to_numpy()
        if data.size > 0 and nunique > 1:
            n_bins = int(min(30, max(20, nunique)))  # ~20-30 bin
            counts_arr, edges = np.histogram(data, bins=n_bins)
            histogram = {
                "bin_edges": [float(e) for e in edges],    # n_bins + 1 kenar
                "counts":    [int(c) for c in counts_arr],  # n_bins adet
            }
            return None, histogram
        # tek-değerli / boş sürekli hedef: histogram anlamsız → value_counts'a düş

    # sınıflandırma / düşük kardinalite: mevcut value_counts davranışı
    # (native int/str'e cast → JSON güvenliği; frontend için anlamı değişmez)
    return {str(k): int(v) for k, v in s.value_counts().items()}, None


def _build_zero_missing_flags(
    df: pd.DataFrame, original_cols: set, intent: str, aggregated: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """DOKUNUS 2: is_zero (orijinal sayisal kolon) + was_missing (DOKUNUS 1 tasiyicisi)
    flag'lerini PLAIN INT olarak uret. Kapsam disi/aggregation/aday yok → tasiyicilari
    temizleyip cik. Yayginlik kapisi %1-%99 (dogum-aninda); global cap _MAX_DERIVED_FLAGS,
    asilirsa deterministik sirala (dengelilik min(p,1-p) desc, esitlikte ad asc). Yalnizca
    ORIJINAL kolonlar (turetilmis feature'in flag'i olmasin); isim cakisma guard'i.
    Tasiyicilar burada df'ten kaldirilir (gated-out olanlar encode/scale'e sizmaz)."""
    carriers = [c for c in df.columns if c.startswith(_DERIVED_FLAG_PREFIX)]

    # Kapsam disi (yalniz classif/reg/clustering) VEYA aggregation yolu → flag uretme
    if intent.strip().lower() not in _FLAG_INTENTS or aggregated:
        if carriers:
            df = df.drop(columns=carriers, errors="ignore")
        if aggregated:
            logger.info("[feature_eng] zero/missing flag: aggregation yolu → atlandi (satir-bazli anlamsiz)")
        return df, []

    candidates: list[tuple[str, np.ndarray, float]] = []

    # was_missing — tasiyicidan
    for carrier in carriers:
        name = f"was_missing_{carrier[len(_DERIVED_FLAG_PREFIX):]}"
        if name in df.columns:
            continue
        s = df[carrier].to_numpy().astype(int)
        p = float(s.mean()) if s.size else 0.0
        if _FLAG_MIN_PREVALENCE <= p <= _FLAG_MAX_PREVALENCE:
            candidates.append((name, s, min(p, 1.0 - p)))

    # is_zero — yalnizca ORIJINAL sayisal kolonlar (math/domain/lag/date turevleri haric)
    for col in df.select_dtypes(include="number").columns:
        if col.startswith(_DERIVED_FLAG_PREFIX) or col not in original_cols:
            continue
        name = f"is_zero_{col}"
        if name in df.columns:
            continue
        s = (pd.to_numeric(df[col], errors="coerce").to_numpy() == 0).astype(int)
        if not s.any() or s.all():        # sifir hic gorulmemis / hepsi sifir → bilgi yok
            continue
        p = float(s.mean())
        if _FLAG_MIN_PREVALENCE <= p <= _FLAG_MAX_PREVALENCE:
            candidates.append((name, s, min(p, 1.0 - p)))

    # Tasiyicilari her halukarda dusur (was_missing serileri yukarida numpy'a kopyalandi)
    if carriers:
        df = df.drop(columns=carriers, errors="ignore")

    if not candidates:
        return df, []

    candidates.sort(key=lambda t: (-t[2], t[0]))   # deterministik: dengelilik desc, ad asc
    if len(candidates) > _MAX_DERIVED_FLAGS:
        logger.info(f"[feature_eng] zero/missing flag: {len(candidates)} aday > cap "
                    f"{_MAX_DERIVED_FLAGS} → kirpildi")
        candidates = candidates[:_MAX_DERIVED_FLAGS]

    added: list[str] = []
    for name, s, _ in candidates:
        df[name] = s
        added.append(name)
    logger.info(f"[feature_eng] zero/missing flag uretildi ({len(added)}): {added}")
    return df, added


def _build_ratio_features(
    df: pd.DataFrame, original_cols: set, meta: "ColumnMetadata",
    intent: str, aggregated: bool, leaky_cols_high: set,
) -> tuple[pd.DataFrame, list[str]]:
    """Deterministik oran (a/b) motoru — modelin yapamadigi olcek-carprazi sinyal.
    Yalniz ORIJINAL sayisal kolonlar; binary/sabit (nunique<3), ID-benzeri (nunique/n_rows>0.95
    — DTYPE-FARKINDA: yalniz int/object'e uygulanir, float continuous MUAF), leaky kaynak ve
    cok-sifirli PAYDA haric. TEK YON (ad-sirali a<b → a/b) → resiprokal patlamasi
    yok. Payda kapisi + winsorize [q01,q99] → sifir-bolme/uc deger nötr. Cap _MAX_RATIO_FEATURES;
    asilirsa normalize-varyans (yayilim) ile sirala, esitlikte ad. Donen kolonlar _detect_leakage'da
    protected_high_card'a eklenir (continuous → kardinalite kuralindan MUAF; korelasyon/leakage/
    select TAM uygulanir — flag'den farki budur). ID on-filtresi, bu kardinalite muafiyetinin
    yanlis-etiketli ID'yi orana sizdirma acigini kapatir (LLM metadata hatasina dayaniklilik)."""
    if intent.strip().lower() not in _FLAG_INTENTS or aggregated:
        if aggregated:
            logger.info("[feature_eng] ratio feature: aggregation yolu → atlandi (orijinal kolonlar gruplandi)")
        return df, []

    # Aday: orijinal sayisal, leaky/flag/turetilmis degil
    n_rows = len(df)
    num_cols = [c for c in df.select_dtypes(include="number").columns
                if c in original_cols
                and not c.startswith(_DERIVED_FLAG_PREFIX)
                and not c.startswith(("is_zero_", "was_missing_"))
                and c not in leaky_cols_high]
    nunique = {c: int(pd.to_numeric(df[c], errors="coerce").nunique()) for c in num_cols}

    def _passes_id_guard(c: str) -> bool:
        # Kardinalite-tabanli ID-guard DTYPE-FARKINDA: yalniz integer ve object/string
        # kolonlara uygulanir (customer_id gibi int-unique ID'ler elenir). FLOAT (continuous)
        # kolonlar bu kuraldan MUAF — estimated_salary/balance gibi yuksek-kardinaliteli
        # surekli olcumler false-positive olarak elenmesin. Muafiyet SADECE bu guard'a ozeldir:
        # float oranlar yine winsorize/payda-kapisi/korelasyon/leakage/cap'e TAM tabidir.
        # Dinamik: isme degil dtype'a bakar (object/string num_cols'a zaten girmez → pratikte int kapisi).
        if pd.api.types.is_float_dtype(df[c]):
            return True
        return not (n_rows > 0 and nunique[c] / n_rows > _RATIO_MAX_CARD_RATIO)

    eligible = sorted(
        c for c in num_cols
        if nunique[c] >= _RATIO_MIN_UNIQUE   # binary/sabit haric (dtype-bagimsiz)
        and _passes_id_guard(c)              # ID-benzeri haric — yalniz int/object (float MUAF)
    )
    if len(eligible) < 2:
        return df, []

    zero_frac = {c: float((pd.to_numeric(df[c], errors="coerce") == 0).mean()) for c in eligible}

    candidates: list[tuple[str, np.ndarray, float]] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]            # ad-sirali: pay=a, payda=b (tek yon)
            if zero_frac[b] > _RATIO_DENOM_MAX_ZERO:    # cok-sifirli payda → ciftı atla
                continue
            name = f"ratio_{a}_{b}"
            if name in df.columns:                      # isim cakisma guard (math/domain)
                continue
            r = pd.to_numeric(df[a], errors="coerce") / pd.to_numeric(df[b], errors="coerce")
            r = r.replace([np.inf, -np.inf], np.nan)
            lo, hi = r.quantile(_RATIO_WINSOR_Q), r.quantile(1.0 - _RATIO_WINSOR_Q)
            r = r.clip(lo, hi).fillna(0.0)
            arr = r.to_numpy(dtype="float64")
            vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
            if not np.isfinite(vmin) or vmax == vmin:    # sabit oran → bilgi yok
                continue
            score = float(np.nanvar((arr - vmin) / (vmax - vmin)))   # olcek-bagimsiz yayilim
            candidates.append((name, arr, score))

    if not candidates:
        return df, []

    candidates.sort(key=lambda t: (-t[2], t[0]))   # deterministik: yayilim desc, ad asc
    if len(candidates) > _MAX_RATIO_FEATURES:
        logger.info(f"[feature_eng] ratio feature: {len(candidates)} aday > cap "
                    f"{_MAX_RATIO_FEATURES} → kirpildi")
        candidates = candidates[:_MAX_RATIO_FEATURES]

    added: list[str] = []
    for name, arr, _ in candidates:
        df[name] = arr
        added.append(name)
    logger.info(f"[feature_eng] ratio feature uretildi ({len(added)}): {added}")
    return df, added


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

    # Modality ekseni (domain_pack'ten bağımsız). Boş/"tabular" = mevcut davranış;
    # "timeseries" → holdout split kronolojik olur (geçmişle eğit, gelecekte test).
    modality         = (state.get("modality") or "").strip().lower()
    time_col         = (state.get("time_col") or "").strip()
    entity_col       = (state.get("entity_col") or "").strip()   # opsiyonel multi-entity (FAZ B2-a)

    if not feature_plan:
        return {"error":"Feature plan yok."}

    logger.info(f"\n{'='*60}")
    logger.info(f"[feature_eng] BASLIYOR — BigData Feature Engineering")
    logger.info(f"{'='*60}")
    logger.info(f"[feature_eng] Analiz: {feature_plan.get('analysis_summary','?')}")

    meta = ColumnMetadata(db_intelligence, profiling_report)

    try:
        df = _load_data(state)
    except Exception as e:
        import traceback
        logger.error(f"\n[feature_eng] HATA: {e}")
        logger.info(traceback.format_exc())
        return {"error": str(e)}

    if df.empty or len(df) < 5:
        return {"error": f"Veri seti yüklenemedi veya çok az veri var ({len(df)} satır)."}

    logger.info(f"[feature_eng] Veri: {len(df):,} satır × {len(df.columns)} kolon")

    # Multi-table join suffix'lerini temizle
    all_tables = [t["table_name"] for t in db_intelligence.get("tables", [])]
    df = _normalize_column_names(df, table_names=all_tables if all_tables else None)
    logger.info(f"[feature_eng] Normalizasyon sonrasi: {len(df.columns)} kolon")

    # Critic'ten gelen drop_features uygula
    drop_features = feature_plan.get("drop_features", [])
    if drop_features:
        existing_drops = [c for c in drop_features if c in df.columns]
        if existing_drops:
            df = df.drop(columns=existing_drops, errors="ignore")
            logger.info(f"[feature_eng] Critic drop_features: {existing_drops}")

    # original_cols: is_zero/ratio motorlari YALNIZCA orijinal kolonlarda calissin
    # (math/domain/date/lag/_isna turevleri haric). _add_missing_indicators _isna
    # kolonlari EKLEMEDEN ONCE yakalanmali → turetilmis kolonlar original_cols'a sizmaz.
    original_cols = set(df.columns)

    # Critic 'too-good/leakage' teşhisinden gelen DETERMINISTIK drop (retry).
    # LLM'in drop_features'ına bağlı kalmadan, importance-bazlı leakage_suspect'leri
    # doğrudan düşürür — schema'da drop_features alanı olmadığı için bu yol garanti.
    forced_drops = state.get("forced_leakage_drops") or []
    if forced_drops:
        existing_forced = [c for c in forced_drops if c in df.columns]
        if existing_forced:
            df = df.drop(columns=existing_forced, errors="ignore")
            logger.info(f"[feature_eng] Forced leakage drop (critic): {existing_forced}")

    # Auto-feature ailelerinin türetmemesi gereken leaky/target/identifier kolonlar.
    # ÖNEMLİ (own-goal önleme): leaky bir kolondan _isna/count_/dev_ türetmek target'ı
    # birebir kopyalayan bir feature yaratabilir (örn. ALERT_TYPE_isna).
    leakage_exclude = _leakage_exclude_cols(
        feature_plan, meta, state.get("domain_leakage_report") or {})
    leakage_exclude |= set(forced_drops)  # forced drop'lardan da auto-feature türetme
    if leakage_exclude:
        logger.info(f"[feature_eng] Auto-feature exclude (leakage-safe): {sorted(leakage_exclude)}")

    # Missing-value indicator'lari (fillna ONCESI — _clean_data dolduruyor): null
    # sinyalini koru. Row-wise → leakage yok, split'ten once guvenli. Eklenen kolonlar
    # auto_feature_cols'ta birikir; supervised yolda count/deviation ile genisler.
    auto_feature_cols: list[str] = _add_missing_indicators(df, exclude=leakage_exclude)

    # Temizlik
    df = _clean_data(df)

    # ── Time-series lag katmani (FAZ A) ──────────────────────────
    # Yalnizca modality==timeseries; tabular yol bu blogu hic gormez.
    # Konum: _clean_data sonrasi, _resolve_target ONCESI → target hala df'te
    # (gecmis target = mesru feature), degerler henuz scale/encode edilmemis,
    # zaman kolonu henuz drop edilmemis. NaN-lead drop fillna'dan (~asagida) ONCE.
    ts_lag_info = None
    if modality == "timeseries":
        resolved_time_col = _resolve_time_col(df, time_col)
        if resolved_time_col is None:
            logger.info(f"[feature_eng] WARN timeseries lag: time_col '{time_col}' df'te yok → lag atlandi")
        elif float(pd.to_datetime(df[resolved_time_col], errors="coerce").notna().mean()) < 0.5:
            logger.info(f"[feature_eng] WARN timeseries lag: '{resolved_time_col}' parse edilemedi → lag atlandi")
        else:
            # entity_col opsiyonel: _resolve_time_col generik ad-normalizasyonu yapar
            # (entity de _normalize_column_names suffix'inden etkilenebilir). Bos/bulunamaz
            # → GLOBAL yol (FAZ A/B1 birebir).
            resolved_entity_col = _resolve_time_col(df, entity_col) if entity_col else None
            if entity_col and resolved_entity_col is None:
                logger.info(f"[feature_eng] WARN timeseries: entity_col '{entity_col}' df'te yok → GLOBAL yol")
            df, ts_lag_info = _build_timeseries_features(
                df, feature_plan, resolved_time_col, meta,
                entity_col=resolved_entity_col or "",
            )

    # Hedef değişken
    df, y, target_name = _resolve_target(df, feature_plan, intent)

    # Aggregation — timeseries'te zaman eksenini korumak icin ATLANIR (kapatinca
    # domain recipe'leri de inert kalir: avg_/std_ kolonlari olusmaz → onlara dokunmuyoruz).
    aggregated = False
    if modality == "timeseries":
        logger.info("[feature_eng] timeseries: aggregation atlandi (zaman ekseni korunuyor)")
    else:
        entity_resolution = state.get("entity_resolution") or {}
        n_before_agg = len(df)
        df, y = _aggregate_if_needed(df, y, feature_plan, meta, entity_resolution=entity_resolution)
        aggregated = (len(df) != n_before_agg)   # satirlar gruplandiysa flag uretme (DOKUNUS 2 atlar)

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
        logger.info(f"[feature_eng] High-confidence leaky kolonlar (türev üretilmeyecek): {leaky_cols_high}")

    # Date feature extraction — leaky olan timestamp kolonları atlanır
    for col in meta.get_by_type("timestamp"):
        if col in df.columns:
            if col in leaky_cols_high:
                logger.info(f"[feature_eng] SKIP türev: '{col}' domain leakage (high) — year/month/dayofweek üretilmiyor")
                continue
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"]      = dt.dt.year
                df[f"{col}_month"]     = dt.dt.month
                df[f"{col}_dayofweek"] = dt.dt.dayofweek
            except Exception:
                pass

    # Timeseries: kronolojik split anahtarını drop'tan ÖNCE yakala — aşağıdaki blok
    # timestamp kolonlarını drop ediyor + duckdb USING SAMPLE satır sırasını bozduğu
    # için satır sırasına GÜVENİLMEZ; gerçek tarih değerine göre sıralanmalı. Anahtar
    # df.index'e hizalı tutulur (sonraki adımlar yalnızca kolon ekler/atar, satır ve
    # index'i korur), split anında pozisyonel olarak yeniden hizalanır.
    time_sort_key = None
    entity_split_key = None   # B2-b-1: entity-aware split icin — drop'tan ONCE yakala (identifier dusuyor)
    if modality == "timeseries":
        resolved_time_col = _resolve_time_col(df, time_col)
        if resolved_time_col is None:
            logger.info(f"[feature_eng] WARN timeseries: time_col '{time_col}' df'te bulunamadı → RASTGELE split fallback")
        else:
            parsed = pd.to_datetime(df[resolved_time_col], errors="coerce")
            valid_ratio = float(parsed.notna().mean())
            if valid_ratio < 0.5:
                logger.info(f"[feature_eng] WARN timeseries: '{resolved_time_col}' tarih olarak parse edilemedi "
                      f"(%{valid_ratio*100:.0f} geçerli) → RASTGELE split fallback")
            else:
                time_sort_key = parsed
                logger.info(f"[feature_eng] timeseries: kronolojik anahtar '{resolved_time_col}' yakalandı "
                      f"({parsed.notna().sum()}/{len(parsed)} geçerli tarih)")
                # entity anahtari: time_sort_key gibi df.index'e hizali, drop'tan ONCE yakalanir
                if entity_col:
                    resolved_entity_col = _resolve_time_col(df, entity_col)
                    if resolved_entity_col is not None:
                        entity_split_key = df[resolved_entity_col].copy()
                        logger.info(f"[feature_eng] timeseries: entity-aware split anahtari "
                              f"'{resolved_entity_col}' yakalandı ({entity_split_key.nunique()} entity)")
                    else:
                        logger.info(f"[feature_eng] WARN timeseries: entity_col '{entity_col}' split için "
                              f"bulunamadı → GLOBAL split")

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
        logger.info(f"[feature_eng] Drop (id/ts/target): {existing}")

    # Leakage denylist for every derived-feature path below: target + its aliases +
    # high-confidence leaky cols can never become an input.
    disallowed = {target_name, target_col} | set(related) | set(leaky_cols_high)

    # Generic (sector-independent) derived features: the feature planner LLM picks
    # an op from the safe palette over REAL columns (e.g. velocity = ratio(x, t)).
    # Validated + deterministically executed — replaces the old raw df.eval path,
    # which let the LLM write code with no leakage guard.
    derived_feature_cols = _apply_plan_derived_features(df, feature_plan, disallowed)
    if derived_feature_cols:
        logger.info(f"[feature_eng] Plan turevli feature'lar: {derived_feature_cols}")

    # Domain pack feature'ları (additive + guarded). Pack 'none' ise hiç çalışmaz.
    #   (a) Ajan-karar yolu: expert ajanının propose_feature ile seçtiği FeatureSpec'ler,
    #       gerçek şemaya + leakage denylist'e karşı doğrulanıp deterministik uygulanır.
    #   (b) Statik fallback: ajan bir şey üretmediyse, convention-bazlı recipe'ler devreye girer.
    domain_feature_cols: list[str] = []
    try:
        agent_cols = _apply_feature_specs(df, state, disallowed)
        domain_feature_cols.extend(agent_cols)
        if not agent_cols:
            domain_feature_cols.extend(_apply_domain_recipes(df, state, meta))
        if domain_feature_cols:
            logger.info(f"[feature_eng] Domain feature'lari: {domain_feature_cols}")
    except Exception as e:
        logger.error(f"[feature_eng] Domain feature hatasi ({e}); atlaniyor.")

    # ── DOKUNUS 2: deterministik sifir/oran motorlari (SPLIT ONCESI, row-wise) ──
    # is_zero = orijinal sayisal kolonlar (row-wise, deterministik → leakage-safe).
    # was_missing artik uretilmiyor: arkadasin _add_missing_indicators (_isna) ayni
    # sinyali own-goal korumasiyla uretiyor (carrier yok → _build_zero_missing_flags
    # yalniz is_zero dondurur). Ratio = orijinal sayisal cift (a/b, tek yon); leaky_cols_high
    # + ID-benzeri kaynak dislanir. Ikisi de split-oncesi feature-construction (arkadasin
    # _count_encode/_groupby_deviation'i da bu altitude'da) → split sonrasi _tt zinciri isler.
    df, derived_flag_cols = _build_zero_missing_flags(df, original_cols, intent, aggregated)
    df, ratio_feature_cols = _build_ratio_features(
        df, original_cols, meta, intent, aggregated, leaky_cols_high)

    # ── LEAKAGE-SAFE SIRA (GRANDMASTER DİSİPLİNİ) ────────────────
    # Encoder/scaler/feature-selection/sizinti tespiti İSTATİSTİK öğrenir (target
    # ortalaması, kolon mean/std, varyans, korelasyon). Bunlar SADECE train'e fit
    # edilmeli; aksi halde test dağılımı/target'ı eğitime sızar → holdout skoru
    # yapay şişer. Bu yüzden artık önce SPLIT, sonra fit-on-train → transform-test.
    # (Eski sıra hepsini tüm df'ye fit edip sonra böldüğü için sızıntılıydı.)
    # Oranlar continuous (yuksek-kardinaliteli ama ID DEGIL) → SADECE kardinalite
    # kuralindan muaf (ts lag'leriyle ayni mantik); _detect_leakage_tt'de korunur.
    ts_protected_cols: set = set(ratio_feature_cols)
    if ts_lag_info:
        ts_protected_cols |= (set(ts_lag_info.get("lag_columns", []))
                              | set(ts_lag_info.get("rolling_columns", [])))

    # 14. Train/test split
    split_imbalanced = 0
    univariate_dropped: list[str] = []   # denetimli univariate seçim (yalnızca supervised dalda dolar)
    if intent in ("clustering","anomaly_detection"):
        # Unsupervised: holdout yok → transform'lar TÜM veriye fit edilir (kümeleme
        # için doğru/istenen davranış; target sızıntısı kavramı zaten yok). Eski
        # tekil-df siblingleri burada aynen kullanılır.
        df = _encode_categorical(df, y)
        df = _scale_features(df, meta.get_outlier_cols())
        df = _select_features(df)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        df, leakage_report = _detect_leakage(df, y, meta, protected_high_card=ts_protected_cols)
        df, domain_dropped = _apply_domain_leakage_drops(df, domain_leakage_report)
        leakage_report["dropped"].extend(domain_dropped)
        X_train = X_test = df.copy()
        y_train = y_test = pd.Series([], dtype=int)
        split_strategy = "none"
        logger.info(f"[feature_eng] Unsupervised — split atlandi (transform'lar tüm veriye fit)")
    else:
        # ── (1) ÖNCE SPLIT — henüz encode/scale edilmemiş ham df üzerinden ──
        if modality == "timeseries" and time_sort_key is not None and entity_split_key is not None:
            # ENTITY-AWARE holdout: her entity'nin kendi son %TEST_SIZE'i test → hicbir entity
            # tamamen bir tarafa dusmez (her biri gecmisiyle train, gelecegiyle test). Pozisyonel
            # (df ve y konum bazli birlikte hizali) → kaydirmali zaman aralikli paneller dogru.
            train_pos, test_pos, split_imbalanced = _entity_aware_split_positions(
                time_sort_key, entity_split_key, df.index, TEST_SIZE)
            X_train, X_test = df.iloc[train_pos], df.iloc[test_pos]
            y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
            split_strategy = "entity_aware"
            if split_imbalanced:
                logger.info(f"[feature_eng] WARN entity-aware split: {split_imbalanced} entity train'i bos/yetersiz "
                      f"(cok kucuk entity) — modele zayif katki, sessiz gecilmiyor")
            logger.info(f"[feature_eng] ENTITY-AWARE split (timeseries): her entity son %{int(TEST_SIZE*100)} = test "
                  f"({entity_split_key.nunique()} entity, n_train={len(X_train)}, n_test={len(X_test)}), shuffle YOK")
        elif modality == "timeseries" and time_sort_key is not None:
            # Kronolojik holdout: geçmişle eğit, gelecekte test. shuffle/stratify YOK.
            # Pozisyonel sırala (df ve y konum bazlı hizalı) → index etiketlerinden bağımsız.
            order_pos = _chronological_order(time_sort_key, df.index)
            df_sorted = df.iloc[order_pos]
            y_sorted  = y.iloc[order_pos]
            n_test = max(1, int(round(len(df_sorted) * TEST_SIZE)))
            X_train, X_test = df_sorted.iloc[:-n_test], df_sorted.iloc[-n_test:]
            y_train, y_test = y_sorted.iloc[:-n_test], y_sorted.iloc[-n_test:]
            split_strategy = "global_chronological"
            logger.info(f"[feature_eng] KRONOLOJIK split (timeseries): son %{int(TEST_SIZE*100)} = test "
                  f"(n_train={len(X_train)}, n_test={len(X_test)}), shuffle/stratify YOK")
        else:
            stratify_arg = y if intent.strip().lower() == "classification" else None
            split_strategy = "random"
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    df, y, test_size=TEST_SIZE, random_state=settings.RANDOM_SEED, stratify=stratify_arg
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    df, y, test_size=TEST_SIZE, random_state=settings.RANDOM_SEED
                )

        # Split sonrası kopyala: aşağıdaki transform'lar kolon-atamaları yapar;
        # iloc/train_test_split dilimleri üzerinde SettingWithCopy uyarısını önler.
        X_train, X_test = X_train.copy(), X_test.copy()

        # ── (2) SONRA TRANSFORM — istatistik YALNIZCA X_train'den, X_test'e uygula ──
        # Grandmaster feature aileleri: count-encoding + groupby-deviation. ENCODE
        # ONCESI calisirlar (ham kategorikleri girdi/grup-anahtari olarak kullanirlar).
        # Ikisi de train-fit → leakage-safe; target kullanmazlar.
        _auto_exclude = disallowed | leakage_exclude
        X_train, X_test, _ce_cols = _count_encode_tt(X_train, X_test, exclude=_auto_exclude)
        X_train, X_test, _gd_cols = _groupby_deviation_tt(X_train, X_test, meta, exclude=_auto_exclude)
        auto_feature_cols = auto_feature_cols + _ce_cols + _gd_cols

        X_train, X_test = _encode_categorical_tt(X_train, X_test, y_train)
        X_train, X_test = _scale_features_tt(X_train, X_test, meta.get_outlier_cols())
        X_train, X_test = _select_features_tt(X_train, X_test)
        # Final NaN/numeric temizliği (sabit fillna(0) — deterministik, sızıntısız)
        X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_test  = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)
        # Denetimli univariate seçim: hedefle alakasız feature'ları ele (train-fit).
        # NaN/numeric temizliği SONRASI çalışır (ANOVA/f_regression NaN kabul etmez).
        if intent.strip().lower() in ("classification", "regression"):
            X_train, X_test, univariate_dropped = _select_univariate_tt(
                X_train, X_test, y_train, intent)
        # Leakage detection train'den öğrenir, test'ten de senkron düşürür
        X_train, X_test, leakage_report = _detect_leakage_tt(
            X_train, X_test, y_train, meta, protected_high_card=ts_protected_cols)
        X_train, X_test, domain_dropped = _apply_domain_leakage_drops_tt(
            X_train, X_test, domain_leakage_report)
        leakage_report["dropped"].extend(domain_dropped)

    y_train_arr = np.array(y_train)
    y_test_arr  = np.array(y_test)

    # Hedef dağılımı: sınıflandırmada value_counts (donut), regresyon/sürekli
    # hedefte histogram bin'leri. İkisinden biri None → frontend hangisini
    # çizeceğini bilir. Gerçek y_train/y_test'ten hesaplanır (anti-fabrication).
    dist_train_counts, hist_train = _target_distribution(y_train_arr, intent)
    dist_test_counts,  hist_test  = _target_distribution(y_test_arr, intent)

    logger.info(f"\n[feature_eng] TAMAMLANDI:")
    logger.info(f"  Hedef   : {target_name}")
    logger.info(f"  X_train : {X_train.shape}")
    logger.info(f"  X_test  : {X_test.shape}")
    if len(np.unique(y_train_arr)) <= 10:
        logger.info(f"  Train dist: {pd.Series(y_train_arr).value_counts().to_dict()}")  # type: ignore

    feature_report = {
        "target_name":              target_name,
        "analysis_summary":         feature_plan.get("analysis_summary",""),
        "final_features":           X_train.columns.tolist(),
        "n_features":               len(X_train.columns),
        "n_train":                  len(X_train),
        "n_test":                   len(X_test),
        "target_distribution_train": dist_train_counts,   # sınıflandırma → value_counts, regresyon → None
        "target_distribution_test":  dist_test_counts,
        "target_histogram_train":    hist_train,           # regresyon → {bin_edges, counts}, sınıflandırma → None
        "target_histogram_test":     hist_test,
        "intelligence_metadata":    meta.summary(),
        "data_source":              "parquet" if state.get("data_path") else "sql_results_preview",
        "total_rows_available":     None,  # execution_node profiling'den alınabilir
        "leakage_dropped":          leakage_report["dropped"],
        "leakage_suspected":        leakage_report["suspected"],
        "univariate_dropped":       univariate_dropped,
        "derived_features":         [c for c in derived_feature_cols if c in X_train.columns],
        "domain_features":          [c for c in domain_feature_cols if c in X_train.columns],
        "derived_flags":            [c for c in derived_flag_cols if c in X_train.columns],
        "ratio_features":           [c for c in ratio_feature_cols if c in X_train.columns],
        "auto_features":            [c for c in auto_feature_cols if c in X_train.columns],
        "timeseries_lag":           ts_lag_info,
        "split_strategy":           split_strategy,
        "split_imbalanced_entities": split_imbalanced,
    }

    # Ağır split dizilerini diske yaz; state'te yalnızca dosya yolu taşı
    # (checkpoint payload şişmesini önler). Yazma başarısız olursa inline'a düş.
    from src.utils.dataset_store import dump_split
    split_paths = dump_split(X_train, X_test, y_train, y_test, tag=f"p{state.get('project_id', '')}")
    out: dict = {"feature_report": feature_report, "error": None, **split_paths}
    if not split_paths.get("X_train_path"):   # disk yazımı kısmen/tamamen başarısız → inline fallback
        out["X_train"] = X_train.to_dict(orient="records")
        out["X_test"]  = X_test.to_dict(orient="records")
    if not split_paths.get("y_train_path"):
        out["y_train"] = y_train.tolist() if hasattr(y_train, "tolist") else list(y_train)
        out["y_test"]  = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
    return out