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

    # ── Training cap (dinamik, stratified) ───────────────────────────────────
    # total <= cap → tüm veri. total > cap → sınıf ORANINI KORUYAN stratified
    # örnekleme (classification) ya da temsili random (regression). Dengesizlik
    # DUZELTİLMEZ (downstream SMOTE/class_weight halleder) → test seti gerçekçi,
    # pozitif oranı korunur. Örnekleme SESSİZ DEĞİL: metadata df.attrs'a yazılır →
    # build_features → feature_report → rapor notu (TR/EN). Cap env-override'lı
    # (settings.MAX_TRAINING_ROWS) — kademeli açmak için.
    cap = settings.MAX_TRAINING_ROWS

    if total <= cap:
        df = conn.execute(f"SELECT * FROM {pq}").df()
        logger.info(f"[feature_eng] Tum veri yuklendi: {len(df):,} satır (cap={cap:,})")
        conn.close()
        df = df.convert_dtypes(convert_integer=False)
        df.attrs["_dq_sampling"] = {"sampled": False, "original_rows": total,
                                    "sampled_rows": total, "method": "none", "cap": cap}
        return df

    sampling_meta: dict = {"sampled": True, "original_rows": total, "cap": cap}

    # Stratified (sınıf oranını koru) — yalnız confirmed + classification-benzeri target
    if actual_target and confirmed:
        try:
            dist = conn.execute(
                f'SELECT "{actual_target}", COUNT(*) as cnt FROM {pq} '
                f'WHERE "{actual_target}" IS NOT NULL '
                f'GROUP BY "{actual_target}" ORDER BY cnt'
            ).fetchall()
            n_classes = len(dist)
            # 2..DQ_TARGET_MAX_CARD sınıf → classification-benzeri (stratify anlamlı).
            # Üstü → sürekli/regresyon target → random'a düş (stratify anlamsız).
            if 2 <= n_classes <= settings.DQ_TARGET_MAX_CARD:
                frames = []
                for cls_val, cls_cnt in dist:
                    # Proportional: her sınıf cap * (cls_cnt/total) → ORAN KORUNUR.
                    # (imbalance-aware "tüm azınlığı al" YOK — oranı bozuyordu.)
                    cls_n = max(1, round(cap * cls_cnt / total))
                    df_cls = conn.execute(
                        f'SELECT * FROM (SELECT * FROM {pq} WHERE "{actual_target}" = $1) '
                        f'USING SAMPLE {cls_n} ROWS', [cls_val]
                    ).df()
                    frames.append(df_cls)
                df = pd.concat(frames, ignore_index=True).sample(
                    frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

                before_min = min(c for _, c in dist) / max(total, 1)
                vc = df[actual_target].value_counts()
                after_min = (vc.min() / len(df)) if len(df) else 0.0
                sampling_meta.update({
                    "method": "stratified", "sampled_rows": len(df), "n_classes": n_classes,
                    "minority_pct_before": round(before_min * 100, 2),
                    "minority_pct_after": round(after_min * 100, 2),
                })
                logger.info(f"[feature_eng] Stratified cap: {total:,} → {len(df):,} satır "
                      f"(cap={cap:,}), azınlık %{before_min*100:.2f} → %{after_min*100:.2f} (oran korundu)")
                conn.close()
                df = df.convert_dtypes(convert_integer=False)
                df.attrs["_dq_sampling"] = sampling_meta
                return df
        except Exception as e:
            logger.error(f"[feature_eng] Stratified sampling hatasi: {e} → random fallback")

    # Random (regression / target yok / stratify uygulanamadı) — temsili örneklem
    df = conn.execute(f"SELECT * FROM {pq} USING SAMPLE {cap} ROWS").df()
    sampling_meta.update({"method": "random", "sampled_rows": len(df)})
    logger.info(f"[feature_eng] Random cap: {total:,} → {len(df):,} satır (cap={cap:,})")
    conn.close()
    df = df.convert_dtypes(convert_integer=False)
    df.attrs["_dq_sampling"] = sampling_meta
    return df
    
# Column suffix normalization
def _normalize_column_names(
    df: "pd.DataFrame",
    table_names: list[str] | None = None,
) -> "tuple[pd.DataFrame, list[dict]]":
    """
    Multi-table join sonrası oluşan suffix'leri temizler — ama SESSİZ VERİ KAYBI ASLA.

    Örnek: is_fraud_transactions → is_fraud   (aynı değerli join-duplicate)
            merchant_name_merchants → merchant_name

    Kural (SIRA-BAĞIMSIZ, deterministik — karar iterasyon sırasına değil hedef +
    değer-eşitliğine dayanır; aynı veri, farklı kolon sırası → aynı çıktı):
    - Suffix strip'i MEVCUT bir kolonla aynı ada düşerse:
        · değerler BİREBİR AYNI (df[col].equals) → gerçek join-duplicate, güvenle düş
        · değerler FARKLI → ÇARPIŞMA: silme, orijinal adı koru, collisions'a kaydet
      equals dtype-KATI: int64 vs float64 (aynı sayılar) → "aynı değil" → ikisi de
      tutulur; NaN aynı konumda eşit sayılır. İlke: ŞÜPHE VARSA KORU.
    - Strip sonucu boş hedef (o ada kolon yok):
        · explicit (tablo adları biliniyor) + TEK kaynak → güvenli rename
        · explicit + ÇOKLU kaynak aynı hedefe → çarpışma, hiçbirini rename etme
        · fallback (tablo adı yok) → SPEKÜLATİF RENAME YOK. '_number'/'_score' gerçek
          tablo son eki olmayabilir (products_number ≠ products tablosu); ismi mangle
          etme, orijinal korunur. Fallback yalnız değeri-doğrulanmış duplicate'i düşürür.

    Dönüş: (df, collisions). Çarpışmalar rapora taşınır (feature_report.
    normalization_collisions → Faz 2 deterministik risk/WARN listesi) — log yetmez,
    kullanıcının veri-kalitesi sinyalini görmesi gerek.
    """
    collisions: list[dict] = []
    if table_names:
        explicit = True
        suffixes = [f"_{t}" for t in table_names]
    else:
        explicit = False
        suffix_set: set[str] = set()
        for col in df.columns:
            parts = col.rsplit("_", 1)
            if len(parts) == 2 and len(parts[1]) >= 3:
                suffix_set.add(f"_{parts[1]}")
        suffixes = list(suffix_set)

    if not suffixes:
        return df, collisions

    existing = set(df.columns)
    suffixes_sorted = sorted(suffixes, key=len, reverse=True)

    # Her kolon → EN UZUN eşleşen suffix strip'i (kolon başına tek deterministik hedef)
    clean_of: dict[str, str] = {}
    for col in df.columns:
        for suffix in suffixes_sorted:
            if col.endswith(suffix) and len(col) > len(suffix):
                clean_of[col] = col[: -len(suffix)]
                break

    # Hedefe göre grupla → çoklu-çarpışma ve sıra-bağımsızlık burada çözülür
    from collections import defaultdict
    by_target: dict[str, list[str]] = defaultdict(list)
    for col, clean in clean_of.items():
        by_target[clean].append(col)

    drop_cols: list[str] = []
    rename_map: dict[str, str] = {}
    for clean in sorted(by_target):                 # deterministik hedef sırası
        srcs = sorted(by_target[clean])             # deterministik kaynak sırası
        if clean in existing:
            dups  = [c for c in srcs if df[c].equals(df[clean])]   # birebir → join-dup
            diffs = [c for c in srcs if c not in dups]             # farklı → çarpışma
            drop_cols.extend(dups)
            if diffs:
                collisions.append({
                    "normalized_name": clean,
                    "kept_columns": sorted([clean, *diffs]),
                    "dropped_duplicates": sorted(dups),
                    "issue": "normalization_name_collision",
                })
        elif explicit and len(srcs) == 1:
            rename_map[srcs[0]] = clean            # güvenli: hedef boş, tek kaynak
        elif explicit and len(srcs) > 1:
            collisions.append({                    # çoklu kaynak tek hedefe → rename etme
                "normalized_name": clean,
                "kept_columns": srcs,
                "dropped_duplicates": [],
                "issue": "normalization_name_collision",
            })
        # fallback + boş hedef → hiçbir şey yapma (orijinal adlar korunur)

    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"[feature_eng] Değeri birebir-aynı duplicate suffix kolonlar silindi: "
              f"{sorted(drop_cols)[:5]}{'...' if len(drop_cols) > 5 else ''}")
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"[feature_eng] {len(rename_map)} kolon suffix temizlendi "
              f"(örnek: {sorted(rename_map.items())[:3]})")
    if collisions:
        logger.warning(f"[feature_eng] Normalizasyon ÇARPIŞMASI ({len(collisions)}): farklı-değerli "
              f"kolonlar aynı normalize adına düşüyor → orijinal adlar KORUNDU (sessiz kayıp yok); "
              f"rapora taşınıyor: {[c['normalized_name'] for c in collisions]}")

    return df, collisions


# Cleaning
def _clean_data(df: pd.DataFrame, impute_missing: bool = True) -> pd.DataFrame:
    # T3a-1: ``impute_missing=False`` → impute BLOĞU atlanır; sahiplik orkestratörün pre-split
    # ``ImputeStep``'ine geçer (aynı konum, tüm-df fit → metrik-NÖTR). string-cast/dedup/currency
    # YERİNDE kalır. legacy/shadow modda ``True`` (mevcut davranış BİREBİR).
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

    if impute_missing:
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

    logger.info(f"[feature_eng] Temizlik sonrasi: {len(df):,} satır | null: {df.isnull().sum().sum()}"
                f"{' (impute orkestratörde)' if not impute_missing else ''}")
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

# Payda guvenligi: paydanin KUTUP KUTLESI = min(P(b>0), P(b<0)) — azinlik isaretin payi.
# Payda sifirin ICINDEN geciyorsa a/b'nin veri araliginda kutbu (pole) vardir: b=+e ve b=-e olan
# iki neredeyse-ozdes satir ZIT sonsuzluklara firlar. Bunu winsorize ONARAMAZ — kirpma kuyrugu
# keser ama SIRALAMAYI duzeltmez, oysa agac split'leri ve lineer katsayilar tam olarak siralamaya/
# olcege dayanir. Tek-tarafli kucuk payda (hep pozitif, sifira yaklasan) ise agir kuyruklu ama
# MONOTON'dur; winsorize onu gercekten onarir. Ayrim "payda kucuk mu" degil, "isaret degistiriyor mu".
#
# Esik winsorizer'in KENDI kuantili: kirpma q01/q99'da oldugu icin motor ancak o kadar kuyruk
# kirlenmesini sogurabilir. Azinlik isaret kutlesi bunun ALTINDA ise kutbun firlattigi satirlar
# zaten kirpilip gider (tek aykiri negatif deger feature'i oldurmez); USTUNDE ise kutup kirpmadan
# sag cikip feature'i zehirler. Yani kapi tam olarak "bunu winsorizer yutabilir mi" sorusudur.
#
# Kural SADECE matematiksel bozuklugu hedefler; target'a/importance'a BAKMAZ, "bu feature ise
# yaramaz" YARGISI ICERMEZ. Sifiri gecen kolon PAY olarak serbesttir — yalniz PAYDA olamaz.
# Olculdu (4 veri seti, 48 aday kolon): kredi/churn/e-ticaret paydalarinin tumu 0.000; ULB'nin
# V1-V28'i 0.380-0.498. Boslugun genisligi 0.38, her iki yonde sifir yanlis siniflama.
_RATIO_DENOM_MAX_SIGN_MASS = _RATIO_WINSOR_Q


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
    """**T5a — DELEGE (ikiz implementasyon tek noktaya indi).** Gövde artık orkestratörün
    ``OneHotStep/TargetEncodeStep (+ decide_encode_route)`` adımını çağırır; algoritma TEK yerde yaşar. Çağrı yüzeyi (imza + dönüş şekli)
    KORUNUR — çağıranlar ve testler değişmez. Doğrulama ZAMANSAL yapıldı: delegasyon öncesi
    legacy çıktısı ``data/_t5a_baseline.json``'a alındı, sonrası onunla karşılaştırıldı
    (modal kontrol arm'ı bu adımda öldüğü için tek geçerli ölçüt budur — bkz. handoff)."""
    from src.services.inference.orchestrator import decide_encode_route
    from src.services.inference.steps import OneHotStep, TargetEncodeStep
    # LEGACY SEMANTİĞİ: bu fonksiyonun ``exclude`` parametresi YOKTU → boş küme geçilir
    # (orkestratör yolunda ``encode_exclude`` ctx'ten gelir; burada gelmemeli).
    route = decide_encode_route(X_train, exclude=None)
    for anahtar, sinif in (("onehot", OneHotStep), ("target", TargetEncodeStep)):
        step = sinif(config={"columns": route[anahtar]})
        X_train, params = step.fit_transform(X_train, y_train)   # E: train cross-fit yolu
        X_test = step.transform(X_test, params)                  # serving yolu
    return X_train, X_test


def _scale_features_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, outlier_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """**T5a — DELEGE (ikiz implementasyon tek noktaya indi).** Gövde artık orkestratörün
    ``InfImputeStep + Robust/StandardScaleStep (+ decide_scaler_route)`` adımını çağırır; algoritma TEK yerde yaşar. Çağrı yüzeyi (imza + dönüş şekli)
    KORUNUR — çağıranlar ve testler değişmez. Doğrulama ZAMANSAL yapıldı: delegasyon öncesi
    legacy çıktısı ``data/_t5a_baseline.json``'a alındı, sonrası onunla karşılaştırıldı
    (modal kontrol arm'ı bu adımda öldüğü için tek geçerli ölçüt budur — bkz. handoff)."""
    from src.services.inference.orchestrator import decide_scaler_route
    from src.services.inference.steps import InfImputeStep, RobustScaleStep, StandardScaleStep
    inf_step = InfImputeStep()
    X_train, inf_params = inf_step.fit_transform(X_train, None)
    X_test = inf_step.transform(X_test, inf_params)
    route = decide_scaler_route(X_train, outlier_cols=outlier_cols)
    for anahtar, sinif in (("robust", RobustScaleStep), ("standard", StandardScaleStep)):
        step = sinif(config={"columns": route[anahtar]})
        X_train, params = step.fit_transform(X_train, None)
        X_test = step.transform(X_test, params)
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


# ── T5-D: HAM-SATIR SERVING CUTOVER ANAHTARI (geçici iskele) ──────────────────
# İkiz borç üç üreticide devrediliyor: D1 missing_indicator · D2 ratio_construct · D3 zero_flag.
# ÖLÇÜLDÜ: her devir, fully-built yolu O ADIM için BOZAR — eğitim kolonu artık adımda üretir
# (`delta.added` dolu) ama fully-built X_test'te kolon zaten vardır → adım atlar → "eksik ekleme".
# Yani ara durumlar üretimde koşulamaz. Bu yüzden devirler TEK anahtarla ATOMİK açılır:
#   False → bugünkü davranış (inline yazma + fully-built X_test) — üretim her adımda ÇALIŞIR
#   True  → tam ham-satır yolu (adımlar üretir + transform_pipeline ham satırdan başlar)
# D3 bitip 6-dataset G1/G2 ölçümü yeşil olunca anahtar KALKAR, ham yol tek yol olur.
SERVING_RAW_REPLAY = True    # T5-D3 sonrası CUTOVER: ham-satır serving tek yol


def _add_missing_indicators(df: pd.DataFrame, max_cols: int = _MISSING_IND_MAX_COLS,
                            exclude: set | None = None, write: bool = True) -> list[str]:
    """fillna ONCESI cagrilir: null tasiyan kolonlar icin '<col>_isna' (0/1) ekler →
    eksiklik sinyalini korur (impute bu bilgiyi siler). Row-wise: capraz-satir istatistik
    YOK → leakage YOK, split'ten once guvenli. Cok az/cok cok null tasiyan kolonlar dusuk
    varyansla _select_features'ta zaten elenir. df IN-PLACE; eklenen kolon adlarini dondurur.

    ``exclude``: target/identifier/leakage-suspect kolonlar — bunlardan '<col>_isna'
    TURETME (own-goal onleme): leaky bir kolon 'sadece pozitifte dolu' ise '<col>_isna'
    target'i birebir kopyalar; bu kolonu hic uretmeyiz.

    ``write=False`` (T5-D1, İKİZ BORÇ DEVRİ): SEÇİM yapılır, YAZMA yapılmaz — kolon adları
    döndürülür ama ``df`` değişmez. Sahiplik orkestratörün ``MissingIndicatorStep``'ine geçer
    (order 0). Gerekçe: inline üretim sürdükçe adım eğitimde "kolon zaten var" görüp ATLIYOR
    (``delta.added=[]``) ama ham serving replay'inde ÜRETİYOR → adımın kolon sözleşmesi eğitimle
    serving arasında AYRIŞIYOR. Delta-guard bunu yakaladı (§2.34). Seçim BURADA kalır (donmuş
    karar, ``missing_indicator_meta``); devredilen yalnız YAZMA."""
    exclude = exclude or set()
    null_cols = [c for c in df.columns if c not in exclude and df[c].isnull().any()]
    null_cols = sorted(null_cols, key=lambda c: df[c].isnull().mean(), reverse=True)[:max_cols]
    added: list[str] = []
    for c in null_cols:
        name = f"{c}_isna"
        if name in df.columns:
            continue
        if write:
            df[name] = df[c].isnull().astype("int8")
        added.append(name)
    if added:
        logger.info(f"[feature_eng] Missing-indicator ({len(added)}): {added}"
                    f"{'' if write else ' (SEÇİM; yazma orkestratörde — T5-D1)'}")
    return added


def _count_encode_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, exclude: set | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Her uygun kategorik kolon icin 'count_<col>' = o kategorinin TRAIN frekansi.
    Frekanslar SADECE train'den ogrenilir; test train frekansiyla maplenir, gorulmeyen
    kategori → 0. Leakage-safe (target kullanmaz); ucuz ama guclu kategorik sinyal
    (nadir/sik kategori ayrimi). Encode ONCESI calismali (ham kategorikleri kullanir).
    ``exclude``: leakage-suspect/target kolonlardan 'count_' turetme (own-goal onleme).

    **T5a — DELEGE (ikiz implementasyon tek noktaya indi).** Gövde artık orkestratörün
    ``CountEncodeStep`` adımını çağırır; algoritma TEK yerde yaşar. Çağrı yüzeyi (imza + dönüş şekli)
    KORUNUR — çağıranlar ve testler değişmez. Doğrulama ZAMANSAL yapıldı: delegasyon öncesi
    legacy çıktısı ``data/_t5a_baseline.json``'a alındı, sonrası onunla karşılaştırıldı
    (modal kontrol arm'ı bu adımda öldüğü için tek geçerli ölçüt budur — bkz. handoff).
    """
    from src.services.inference.steps import CountEncodeStep
    step = CountEncodeStep(config={"exclude": list(exclude or [])})
    X_train, params = step.fit_transform(X_train, None)
    return X_train, step.transform(X_test, params), list(params.keys())


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
    ``exclude``: leakage-suspect/target kolonlari ne grup anahtari ne sayisal girdi yap.

    **T5a — DELEGE (ikiz implementasyon tek noktaya indi).** Gövde artık orkestratörün
    ``GroupbyDeviationStep`` adımını çağırır; algoritma TEK yerde yaşar. Çağrı yüzeyi (imza + dönüş şekli)
    KORUNUR — çağıranlar ve testler değişmez. Doğrulama ZAMANSAL yapıldı: delegasyon öncesi
    legacy çıktısı ``data/_t5a_baseline.json``'a alındı, sonrası onunla karşılaştırıldı
    (modal kontrol arm'ı bu adımda öldüğü için tek geçerli ölçüt budur — bkz. handoff).
    """
    from src.services.inference.steps import GroupbyDeviationStep
    step = GroupbyDeviationStep(config={"exclude": list(exclude or [])})
    X_train, params = step.fit_transform(X_train, None)
    emitted = [name for rec in (params or []) for _, name in rec.get("emits", [])]
    return X_train, step.transform(X_test, params), emitted



# ── MAIN NODE ────────────────────────────────────────────────
# NOT (T5b): bu iki sabit T5a'da gövde-devri sırasında KAZAEN silinmişti (script "bir sonraki
# def'e kadar" temizlerken modül düzeyindeki bu atamaları da yuttu) → `_FEATURE_OPS` ve
# `_sanitize_params` NameError verdi ve `tests/test_feature_specs.py` 7 testle KIRILDI.
# Tam suite koşusu yakaladı; parity dosyası tek başına görmemişti (o yolu tetiklemiyor).
_FEATURE_SPEC_EPS = 1e-9
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


def _run_feature_specs(df: pd.DataFrame, specs: list, disallowed: set, *, source_label: str,
                       skip_ops: "frozenset | set | None" = None) -> list[str]:
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
    skip = set(skip_ops or ())
    added: list[str] = []
    for spec in specs:
        try:
            if spec.op not in _FEATURE_OPS:
                continue
            if spec.op in skip:
                # T4b — SAHİPLİK DEVRİ: bu op portlandı ve orkestratörün adımı üretiyor
                # (train-fit, #5 fiksi). Burada da çalışsaydı kolon ZATEN var olur, adımın
                # `name in X_train.columns` kapısı onu ATLAR ve fiks hiç devreye girmezdi
                # (ÇİFT-ÇALIŞMA sessizce fiksi iptal ederdi). `_clean_data(impute_missing=False)`
                # deseninin aynısı. Yalnız orchestrator modunda; legacy/shadow etkilenmez.
                logger.info(f"[feature_eng] FeatureSpec '{spec.name}' atlandi: "
                            f"'{spec.op}' orkestratorde (T4b sahiplik devri)")
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


def _apply_plan_derived_features(df: pd.DataFrame, feature_plan: dict, disallowed: set,
                                 skip_ops: "frozenset | set | None" = None) -> list[str]:
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
    return _run_feature_specs(df, specs, disallowed, source_label="Plan", skip_ops=skip_ops)


def _apply_domain_recipes(df: pd.DataFrame, state: dict, meta: "ColumnMetadata",
                          *, defer_to_orchestrator: bool = False) -> "tuple[list[str], list[dict], list[str]]":
    """Run the active domain pack's deterministic FeatureRecipes on ``df`` in place.

    Returns ``(added, frozen_specs, unported)``:
      * ``added``        — names of the columns this call is responsible for (written here,
        or — when deferred — declared for :class:`DomainFeaturesStep` to write).
      * ``frozen_specs`` — the ``FeatureRecipe.specs`` declarations, non-empty ONLY when the
        write was deferred to the orchestrator.
      * ``unported``     — bu koşuda orkestratöre DEVREDİLMEYEN kolonlar → Katman 1 kapsamı
        DIŞI (``unported_domain_features`` kapsam olgusu). Devir olduysa boştur.

    No-op when no pack is active. Each recipe is individually guarded so one bad recipe can
    never block the others or the pipeline.

    **SEÇİM/YAZMA AYRIMI (ikiz borç deseni, ratio + zero-flag ile aynı).** ``defer_to_orchestrator``
    yalnız TÜM uygulanabilir recipe'ler spec beyan ettiğinde fiilen devreye girer ("ya hepsi ya
    hiçbiri"): kısmi devir, kapsam reddi hâlinde yarı-yazılmış çerçeve bırakırdı. Devir olduğunda
    kolonları ``DomainFeaturesStep`` (order 7, pre-split) yazar — konum eskisiyle AYNI noktadır,
    çünkü ratio/zero-flag motorlarının seçimi bu kolonların varlığına bakar."""
    from src.domains import get_active_pack
    from src.domains.base import MerchantContext
    from src.domains.schemas import MerchantAnalysis

    pack = get_active_pack(state.get("domain_pack"))
    if pack.name == "none":
        return [], [], []

    ctx = MerchantContext.from_dict(state.get("merchant_context"))
    analysis = MerchantAnalysis.from_dict(state.get("merchant_analysis"))
    try:
        recipes = pack.feature_recipes(ctx, state.get("db_intelligence"), analysis) or []
    except Exception as e:
        logger.error(f"[feature_eng] feature_recipes() hatasi ({e}); atlaniyor.")
        return [], [], []

    # 1. GEÇİŞ — her recipe'nin çıktısı + (varsa) donmuş spec beyanı. Hiçbir şey YAZILMAZ:
    # devir kararı ancak TÜM recipe'ler görüldükten sonra verilebilir.
    plan: list[tuple[str, dict, "list | None"]] = []
    for r in recipes:
        ad = getattr(r, "name", "?")
        try:
            if not r.applies(df, meta):
                continue
            cols = {n: s for n, s in (r.build(df, meta) or {}).items() if n not in df.columns}
            if not cols:
                continue
            try:
                specs = r.specs(df, meta)
            except Exception as e:                    # spec beyanı patlarsa recipe PORTSUZ sayılır
                logger.error(f"[feature_eng] Recipe '{ad}' specs() hatasi ({e}); portsuz sayiliyor.")
                specs = None
            plan.append((ad, cols, specs))
        except Exception as e:
            logger.error(f"[feature_eng] Recipe '{ad}' hatasi: {e}")

    spec_yok = [n for _, cols, specs in plan if specs is None for n in cols]
    devret = bool(defer_to_orchestrator and plan and not spec_yok)

    added: list[str] = []
    frozen: list[dict] = []
    for ad, cols, specs in plan:
        if devret:
            frozen.extend(specs)
            added.extend(s["name"] for s in specs)
            continue
        for name, series in cols.items():
            if name in df.columns:
                continue
            df[name] = series
            added.append(name)
    # PORTSUZ = "bu koşuda orkestratöre DEVREDİLMEYEN her kolon" — spec beyan etmeyenler değil.
    # Fark kritik: devir kapalıyken (ör. SERVING_RAW_REPLAY=False) kolonlar inline yazılır ve
    # artifact'te onları üretecek adım YOKTUR; "spec'i vardı" demek kapsamı yanlış genişletirdi
    # ve serving'de kolon sessizce doğmazdı (fully-built X_test ile guard bunu göremez).
    unported = [] if devret else list(added)
    if devret:
        logger.info(f"[feature_eng] Domain recipe'leri orkestratore DEVREDILDI "
                    f"({len(frozen)} donmus spec): {[s['name'] for s in frozen]}")
    elif unported:
        logger.info(f"[feature_eng] Domain recipe kolonları inline yazıldı (portsuz sayılır"
                    + (f"; spec beyan etmeyen: {spec_yok}" if spec_yok else "") + f"): {unported}")
    return added, frozen, unported


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
    force_zero_cols: "set | None" = None, write: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """DOKUNUS 2: is_zero (orijinal sayisal kolon) + was_missing (DOKUNUS 1 tasiyicisi)
    flag'lerini PLAIN INT olarak uret. Kapsam disi/aggregation/aday yok → tasiyicilari
    temizleyip cik. Yayginlik kapisi %1-%99 (dogum-aninda); global cap _MAX_DERIVED_FLAGS,
    asilirsa deterministik sirala (dengelilik min(p,1-p) desc, esitlikte ad asc). Yalnizca
    ORIJINAL kolonlar (turetilmis feature'in flag'i olmasin); isim cakisma guard'i.
    Tasiyicilar burada df'ten kaldirilir (gated-out olanlar encode/scale'e sizmaz).

    force_zero_cols (INVARIANT): oran motorunun payda olarak kullandigi, SIFIR iceren
    kolonlar. Bu kolonlarin is_zero flag'i prevalence kapisindan VE global cap'ten MUAF
    uretilir — cunku oran sentinel'den aritilirken (bkz. _finalize_ratio_features_tt)
    sifir-payda bilgisi flag'de DURMALI, yoksa bilgi tamamen silinir. Dinamik: kolon
    adina bakmaz, oran motorunun bildirdigi paydalari alir.

    DIKKAT — invariant URETIMI garanti eder, HAYATTA KALMAYI degil: flag burada mutlaka
    olusur; ama downstream _select_univariate_tt/_detect_leakage_tt zayif-ongoruli flag'i
    dusurebilir (or. churn'de is_zero_tenure churn 0.23 vs 0.20 → selection eledi). Bu
    SESSIZ kayip degil, mesru bir selection karari (feature ise yaramiyorsa gider). Amac:
    bilginin sessizce oran sentinel'ine gomulup kaybolmasini onlemek, kalıcı tutmak degil."""
    force_zero_cols = force_zero_cols or set()
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

    # is_zero — yalnizca ORIJINAL sayisal kolonlar (math/domain/lag/date turevleri haric).
    # force_zero_cols'daki kolonlar (oran paydalari) prevalence kapisindan MUAF ve
    # cap-korumali → sifir-payda bilgisi flag'de her zaman durur (invariant).
    forced_added: list[tuple[str, np.ndarray]] = []
    for col in df.select_dtypes(include="number").columns:
        if col.startswith(_DERIVED_FLAG_PREFIX) or col not in original_cols:
            continue
        name = f"is_zero_{col}"
        if name in df.columns:
            continue
        num = pd.to_numeric(df[col], errors="coerce")
        s = (num.to_numpy() == 0).astype(int)
        if not s.any() or s.all():        # sifir hic gorulmemis / hepsi sifir → bilgi yok
            continue
        # #10 — BINARY KOLONDA is_zero URETME: 2-degerli kolonda is_zero = orijinalin
        # BIREBIR yeniden kodlamasi (is_zero = 1 - col/k, corr -1.0) → yeni bilgi yok,
        # garantili kopya (credit_card ↔ is_zero_credit_card). force_zero (Faz 0 invariant)
        # OLSA BILE uretme: invariant "sifir bilgisi MEVCUT olsun" der, "yeni kolon" degil;
        # binary'de bilgi zaten KORUNAN orijinal kolonda tam (oran motoru paydayi dusurmez)
        # → invariant saglanmis say, kopya uretme. 3+ degerli kolonda is_zero bir COARSENING'dir
        # (hangi sifir-disi deger oldugunu atar) → ayri sinyal tasiyabilir, KORUNUR.
        # (nunique==1 sabit kolon buraya hic ulasmaz: yukarida s.all()/not s.any() eler;
        #  ayrica low-variance drop onu zaten dusurur — sessiz kayip yok.)
        if num.dropna().nunique() <= 2:
            continue
        p = float(s.mean())
        if col in force_zero_cols:        # INVARIANT: prevalence kapisini atla, cap'ten muaf
            forced_added.append((name, s))
            continue
        if _FLAG_MIN_PREVALENCE <= p <= _FLAG_MAX_PREVALENCE:
            candidates.append((name, s, min(p, 1.0 - p)))

    # Tasiyicilari her halukarda dusur (was_missing serileri yukarida numpy'a kopyalandi)
    if carriers:
        df = df.drop(columns=carriers, errors="ignore")

    if not candidates and not forced_added:
        return df, []

    candidates.sort(key=lambda t: (-t[2], t[0]))   # deterministik: dengelilik desc, ad asc
    # Cap forced'lari SAYMAZ: normal adaylar kalan slot'a sigar, forced'lar her zaman eklenir.
    cap_normal = max(0, _MAX_DERIVED_FLAGS - len(forced_added))
    if len(candidates) > cap_normal:
        logger.info(f"[feature_eng] zero/missing flag: {len(candidates)} normal aday > kalan slot "
                    f"{cap_normal} (forced={len(forced_added)}) → kirpildi")
        candidates = candidates[:cap_normal]

    # T5-D3 (ikiz borç devri — EN DİKKATLİ olan): ``write=False`` → SEÇİM aynen yapılır,
    # YAZMA ``ZeroFlagStep``'e (order 6) geçer. **KURAL: seçim BURADA kalır.** ``force_zero_cols``
    # invariant'ı (oran motorunun sıfır-payda kolonları prevalence kapısından VE cap'ten MUAF,
    # §2.18 / Faz 0) yukarıdaki seçim mantığının içinde yaşıyor ve adımın içinde YENİDEN
    # KURULMAZ — kurulsaydı motorlar-arası kuplaj inference paketinde yeniden doğar (tam da
    # kaçındığımız drift). Devredilen yalnız ``df[name] = s`` satırı.
    added: list[str] = []
    for name, s in forced_added:          # invariant flag'ler once (cap-muaf)
        if write:
            df[name] = s
        added.append(name)
    for name, s, _ in candidates:
        if write:
            df[name] = s
        added.append(name)
    logger.info(f"[feature_eng] zero/missing flag uretildi ({len(added)}): {added}")
    if forced_added:
        logger.info(f"[feature_eng] zero/missing flag: invariant (sifir-payda) forced flag'ler: "
                    f"{[n for n, _ in forced_added]}")
    return df, added


def _build_ratio_features(
    df: pd.DataFrame, original_cols: set, meta: "ColumnMetadata",
    intent: str, aggregated: bool, leaky_cols_high: set, write: bool = True,
) -> tuple[pd.DataFrame, list[str], set, dict]:
    """Deterministik oran (a/b) motoru — modelin yapamadigi olcek-carprazi sinyal.
    Yalniz ORIJINAL sayisal kolonlar; binary/sabit (nunique<3), ID-benzeri (nunique/n_rows>0.95
    — DTYPE-FARKINDA: yalniz int/object'e uygulanir, float continuous MUAF), leaky kaynak, cok-sifirli
    PAYDA ve SIFIRI GECEN PAYDA (kutup — bkz. _RATIO_DENOM_MAX_SIGN_MASS) haric. Sifiri gecen kolon
    PAY olarak serbesttir. TEK YON (ad-sirali a<b → a/b) → resiprokal patlamasi
    yok. Payda kapilari + winsorize [q01,q99] → sifir-bolme/uc deger nötr. Cap _MAX_RATIO_FEATURES;
    asilirsa normalize-varyans (yayilim) ile sirala, esitlikte ad. Donen kolonlar _detect_leakage'da
    protected_high_card'a eklenir (continuous → kardinalite kuralindan MUAF; korelasyon/leakage/
    select TAM uygulanir — flag'den farki budur). ID on-filtresi, bu kardinalite muafiyetinin
    yanlis-etiketli ID'yi orana sizdirma acigini kapatir (LLM metadata hatasina dayaniklilik)."""
    if intent.strip().lower() not in _FLAG_INTENTS or aggregated:
        if aggregated:
            logger.info("[feature_eng] ratio feature: aggregation yolu → atlandi (orijinal kolonlar gruplandi)")
        return df, [], set(), {}

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
        return df, [], set(), {}

    zero_frac = {c: float((pd.to_numeric(df[c], errors="coerce") == 0).mean()) for c in eligible}
    # Kutup kutlesi: azinlik isaretin payi. >esik → payda sifirin icinden geciyor (bkz. sabit notu).
    sign_mass = {}
    for c in eligible:
        s = pd.to_numeric(df[c], errors="coerce")
        sign_mass[c] = float(min((s > 0).mean(), (s < 0).mean()))

    # HAM oran (a/b) — winsor sinirlari + impute median SPLIT SONRASI train'e fit edilir
    # (bkz. _finalize_ratio_features_tt). Burada yalniz row-wise bolme + inf/NaN normalize:
    # bolme deterministik/leakage-safe, ama kuantil/median cross-row ISTATISTIK → train-only.
    # NaN = payda-sifir (inf→NaN) ∪ kaynak-missing; ikisi de split sonrasi dagilim-ici impute
    # ile temizlenir (out-of-range 0.0 sentinel YOK). Siralama/cap RAW sonlu degerlerden.
    candidates: list[tuple[str, np.ndarray, float, str, str]] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a, b = eligible[i], eligible[j]            # ad-sirali: pay=a, payda=b (tek yon)
            if zero_frac[b] > _RATIO_DENOM_MAX_ZERO:    # cok-sifirli payda → ciftı atla
                continue
            if sign_mass[b] > _RATIO_DENOM_MAX_SIGN_MASS:   # payda sifiri geciyor → kutup, atla
                logger.debug(f"[feature_eng] ratio atlandi: '{a}'/'{b}' — payda sifiri geciyor "
                             f"(kutup kutlesi %{sign_mass[b]*100:.1f} > %{_RATIO_DENOM_MAX_SIGN_MASS*100:.0f})")
                continue
            name = f"ratio_{a}_{b}"
            if name in df.columns:                      # isim cakisma guard (math/domain)
                continue
            num = pd.to_numeric(df[a], errors="coerce")
            den = pd.to_numeric(df[b], errors="coerce")
            r = (num / den).replace([np.inf, -np.inf], np.nan)   # payda-sifir → inf → NaN
            finite = r[np.isfinite(r)]
            if finite.nunique() < 2:                     # sabit/degersiz oran → bilgi yok
                continue
            # Izlenebilirlik (Addition #3): NaN'in iki kaynagini ayri say — impute ikisini de
            # dagilim-ici doldurur, ama log hangi mekanizmanin kac satir urettigini gosterir.
            n_zero_denom = int((den == 0).sum())
            n_src_missing = int((num.isna() | den.isna()).sum())
            if n_zero_denom or n_src_missing:
                logger.debug(f"[feature_eng] ratio '{name}': tanimsiz satir — "
                             f"payda-sifir={n_zero_denom}, kaynak-missing={n_src_missing} "
                             f"(split sonrasi train-median ile impute)")
            fmin, fmax = float(finite.min()), float(finite.max())
            # olcek-bagimsiz yayilim (RAW sonlu deger — sentinel/impute yok, siralama icin)
            score = float((((finite - fmin) / (fmax - fmin)).var())) if fmax > fmin else 0.0
            candidates.append((name, r.to_numpy(dtype="float64"), score, a, b))

    if not candidates:
        return df, [], set(), {}

    candidates.sort(key=lambda t: (-t[2], t[0]))   # deterministik: yayilim desc, ad asc
    if len(candidates) > _MAX_RATIO_FEATURES:
        logger.info(f"[feature_eng] ratio feature: {len(candidates)} aday > cap "
                    f"{_MAX_RATIO_FEATURES} → kirpildi")
        candidates = candidates[:_MAX_RATIO_FEATURES]

    added: list[str] = []
    denom_zero_cols: set = set()   # sifir iceren + fiilen kullanilan payda → is_zero flag GARANTI
    ratio_meta: dict[str, tuple[str, str]] = {}   # {ratio_kolon: (pay, payda)} → etiket bundan kurulur
    for name, arr, _, num_name, denom in candidates:
        # T5-D2 (ikiz borç devri): ``write=False`` → SEÇİM + meta üretilir, KOLON YAZILMAZ.
        # Sahiplik ``RatioConstructStep``'e (order 5) geçer; adım aynı çekirdeği uygular
        # ((num/den) + inf→NaN). Seçim BURADA kalır — ``ratio_meta`` donmuş karar kaynağıdır
        # ve ``denom_zero_cols`` üzerinden zero-flag invariant'ını besler (§2.18 kuplajı).
        if write:
            df[name] = arr                   # HAM oran (NaN korunur; winsor/impute split sonrasi)
        added.append(name)
        ratio_meta[name] = (num_name, denom)   # string-parse belirsizligine karsi kesin kaynak
        if zero_frac.get(denom, 0.0) > 0.0:
            denom_zero_cols.add(denom)
    logger.info(f"[feature_eng] ratio feature uretildi ({len(added)}): {added}")
    if denom_zero_cols:
        logger.info(f"[feature_eng] ratio: sifir-payda kolonlari → is_zero flag garanti "
                    f"(invariant): {sorted(denom_zero_cols)}")
    return df, added, denom_zero_cols, ratio_meta


def _fit_ratio_stats(frame: pd.DataFrame, cols: list[str]) -> dict:
    """Oran kolonlari icin winsor sinirlari (q01,q99) + impute median'i YALNIZCA
    verilen cerceveden (supervised: X_train) ogren — sklearn transformer disiplini.
    Sonlu degerler uzerinden; train'de sonlu deger yoksa kolon None ile isaretlenir
    (transform drop eder). Median winsorlanmis dagilimdan → dagilim-ici, out-of-range
    0.0 sentinel URETMEZ (eski fillna(0.0) bug'inin koku)."""
    stats: dict = {}
    for c in cols:
        if c not in frame.columns:
            continue
        s = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = s[np.isfinite(s)]
        if finite.empty:
            stats[c] = None    # train'de hic sonlu deger yok → uygulanamaz (drop sinyali)
            continue
        lo = float(finite.quantile(_RATIO_WINSOR_Q))
        hi = float(finite.quantile(1.0 - _RATIO_WINSOR_Q))
        med = float(finite.clip(lo, hi).median())   # dagilim-ici impute degeri
        stats[c] = (lo, hi, med)
    return stats


def _apply_ratio_stats(frame: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """_fit_ratio_stats sonuclarini uygula: clip [lo,hi] + NaN→train-median. NaN iki
    kaynagi da (payda-sifir ∪ kaynak-missing) ayni dagilim-ici deger ile dolar → gizli
    sentinel yok. None istatistik → kolonu dusur (train'de sonlu deger yoktu)."""
    drop_cols = [c for c, st in stats.items() if st is None and c in frame.columns]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
        logger.info(f"[feature_eng] ratio finalize: train'de sonlu deger yok → dusuruldu: {drop_cols}")
    for c, st in stats.items():
        if st is None or c not in frame.columns:
            continue
        lo, hi, med = st
        s = pd.to_numeric(frame[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame[c] = s.clip(lo, hi).fillna(med)
    return frame


def _finalize_ratio_features_tt(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oran kolonlarini train'e FIT, train+test'e TRANSFORM et (leakage-safe). Winsor
    sinirlari ve impute median X_train'den; X_test yalniz transform gorur.

    **T5a — DELEGE (ikiz implementasyon tek noktaya indi).** Gövde artık orkestratörün
    ``RatioFinalizeStep`` adımını çağırır; algoritma TEK yerde yaşar. Çağrı yüzeyi (imza + dönüş şekli)
    KORUNUR — çağıranlar ve testler değişmez. Doğrulama ZAMANSAL yapıldı: delegasyon öncesi
    legacy çıktısı ``data/_t5a_baseline.json``'a alındı, sonrası onunla karşılaştırıldı
    (modal kontrol arm'ı bu adımda öldüğü için tek geçerli ölçüt budur — bkz. handoff).
    """
    from src.services.inference.steps import RatioFinalizeStep
    step = RatioFinalizeStep(config={"columns": list(cols)})
    X_train, params = step.fit_transform(X_train, None)
    return X_train, step.transform(X_test, params)


def _finalize_ratio_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Tek-cerceve varyant (unsupervised: holdout yok → tum df'ye fit+transform;
    kumeleme icin dogru davranis, target sizintisi kavrami zaten yok)."""
    if not cols:
        return df
    return _apply_ratio_stats(df, _fit_ratio_stats(df, cols))


def _admit_spec(s, deny: set) -> "dict | None":
    """Tek spec'i KABUL KURALLARINDAN geçir → normalize edilmiş kayıt, ya da ``None``.

    **TEK KAYNAK (§2.19'un 4. örneğinin gereği):** eski yolda (``_run_feature_specs``) her spec
    çalışmadan önce leakage denylist'inden ve şekil doğrulamasından geçiyordu; o kapılar op'un
    ÇEVRESİNDE yaşıyordu ve op emekli olunca kaybolacaktı. Kapılar burada, beyan yüzeyinde.
    İki farklı ÇIKTI ŞEKLİ (parametric çift vs rowwise altılı) bu tek kabul mantığını paylaşır.

    ``params`` BEYAN ANINDA clamp'lenir (``_sanitize_params``) ve clamp uygulandıysa ORİJİNAL
    değer de kaydedilir: kullanıcının planı ``power=99`` derken biz ``3.0`` hesaplıyorsak bu
    artifact'te GÖRÜNMELİ — sessizce başka bir şey hesaplamış olmayalım."""
    from src.domains.schemas import validate_feature_spec_shape

    op = s.get("op") if isinstance(s, dict) else getattr(s, "op", None)
    name = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
    inputs = s.get("inputs") if isinstance(s, dict) else getattr(s, "inputs", None)
    ham_params = s.get("params") if isinstance(s, dict) else getattr(s, "params", None)
    if not (op and name and inputs):
        return None
    params = _sanitize_params(op, ham_params)
    if params is None or not validate_feature_spec_shape(op, len(inputs), len(params)):
        logger.info(f"[pipeline_ctx] '{name}' beyan EDİLMEDİ: gecersiz sekil ({op})")
        return None
    leaky = [c for c in inputs if c in deny]
    if leaky:
        logger.info(f"[pipeline_ctx] '{name}' beyan EDİLMEDİ: leakage girdisi {leaky}")
        return None
    kayit = {"name": name, "op": op, "inputs": list(inputs), "params": list(params)}
    orijinal = [float(v) for v in (ham_params or [])] if ham_params else []
    if orijinal and orijinal != list(params):
        kayit["params_clamped"] = True                 # clamp SESSİZ DEĞİL
        kayit["params_original"] = orijinal
        logger.info(f"[pipeline_ctx] '{name}' params clamp'lendi: {orijinal} → {list(params)}")
    return kayit


def _plan_specs(feature_plan: dict, disallowed: "set | None") -> list:
    """Plan'daki TÜM derived_features spec'lerini kabul kurallarından geçir (sıra korunur)."""
    deny = {c for c in (disallowed or set()) if c}
    out = []
    for s in ((feature_plan or {}).get("derived_features") or []):
        k = _admit_spec(s, deny)
        if k:
            out.append(k)
    return out


def plan_op_specs(feature_plan: dict, op_name: str, disallowed: "set | None" = None) -> list:
    """PARAMETRIC ikili (zscore/normalize) için ``[[çıktı_adı, girdi_kolonu], ...]``.

    Şekil neden ayrı: bu iki op **parametric** (fitted μ/σ, min/max — #5 fiksi), diğer altısı
    **rowwise**. Ayrım fitted-state ekseninde; tek listede toplamak ya rowwise adımı parametric
    yapardı ya train-fit'i kaybederdi. Kova bütünlüğü testle çitlenir (her portlu op TAM OLARAK
    bir beyan listesinde)."""
    return [[k["name"], k["inputs"][0]] for k in _plan_specs(feature_plan, disallowed)
            if k["op"] == op_name]


def plan_derived_specs(feature_plan: dict, ops: "set | frozenset",
                       disallowed: "set | None" = None) -> list:
    """ROWWISE op'lar için tam şekil: ``[{name, op, inputs[], params[], params_clamped?}, ...]``."""
    return [k for k in _plan_specs(feature_plan, disallowed) if k["op"] in ops]


def plan_chained_specs(feature_plan: dict) -> list:
    """ZİNCİRLEME spec'ler: girdisi BAŞKA bir spec'in çıktısı olanlar (§2.19'un 6. örneği).

    Eski yolda ``_run_feature_specs`` tek ``df`` üzerinde PLAN SIRASINDA ilerliyor ve her spec'i
    yazdıktan sonra bir sonrakinin girdisini ``df.columns``'a karşı denetliyor (1657) → zincir
    ÇALIŞIYOR. Orkestratörde spec'ler op-kovalarına bölündüğü (parametric ↔ rowwise) ve sıra
    PIPELINE'dan geldiği için zincir SESSİZCE kırılırdı: girdi kolonu o an yoktur, spec atlanır,
    kolon hiç doğmaz. Garanti "tek df + plan sırası" YAPISINDAN geliyordu → açık beyana çevrildi.

    **ÖLÇÜLEMEDİ (üçüncü kez):** planner'ın zincir üretip üretmediği bilinmiyor — elimizdeki tüm
    gerçek plan verisi ``derived_features`` alanı eklenmeden ÖNCEYE ait (yerel app DB 2026-06-08→14,
    alan 2026-06-25). Şema zinciri artık AÇIKÇA yasaklıyor (birinci savunma); bu, son savunma."""
    derived = (feature_plan or {}).get("derived_features") or []
    adlar = {(s.get("name") if isinstance(s, dict) else getattr(s, "name", None)) for s in derived}
    adlar.discard(None)
    zincirli = []
    for s in derived:
        nm = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
        inp = s.get("inputs") if isinstance(s, dict) else getattr(s, "inputs", None)
        hits = [c for c in (inp or []) if c in adlar]
        if hits:
            zincirli.append({"name": nm, "chained_inputs": hits})
    return zincirli


def assemble_scope_facts(
    *, modality: str, aggregated: bool, timeseries_lag, unported_domain_features: list,
    column_renames: list, feature_plan: dict,
) -> dict:
    """``_SCOPE_RULES``'un baktığı olguların TEK kaynağı — ``check_pipeline_scope`` girdisi.

    NEDEN AYRI FONKSİYON (ÖN-UÇUŞ): kapsam kapısı ``_run_orchestrated_chain``'de, yani
    ratio/zero-flag SEÇİMİNDEN sonra çalışıyor. O noktada ``_orch_mode`` pre-split adımların
    sahipliğini ÇOKTAN devralmıştır (``write=False``) → orada reddetmek geri dönüşü olmayan bir
    yerde patlamak demekti: istisna node'a kadar çıkıyor ve **kullanıcı raporunu da götürüyordu**
    (ölçüldü: banking-churn koşusu, domain pack recipe'leri → `feature_engineering_node` FAIL).
    Oysa kapının anlamı "bu koşu DEPLOY edilemez", "bu koşu ÇALIŞAMAZ" değil.

    Bu olguların HEPSİ ratio/zero-flag'den ÖNCE bilinir (modality/aggregated erken, ts-lag
    ``_build_timeseries_features``'ta, domain feature'ları domain-pack bloğunda). Dolayısıyla
    kapsam ÖN-UÇUŞU orada yapılabilir ve ihlal hâlinde ``_orch_mode`` kapatılıp legacy zincire
    TEMİZ dönülür. Fonksiyonun ayrı olması, ön-uçuş ile asıl kapının AYNI değerleri görmesini
    yapısal olarak garantiler (kopyala-yapıştır sapması imkânsız).

    KİLİT: ``test_scope_facts_cover_all_rules`` — burada üretilen anahtar kümesi ``_SCOPE_RULES``'un
    tüm olgularını KAPSAMAK zorunda; kapsamazsa ön-uçuş bir ihlali göremez ve kapı yine geç patlar.
    """
    return {
        # dedup: _clean_data split ÖNCESİ tüm df'te yapar. Konumu orkestratörde YENİDEN
        # ÜRETİLEMEZ ("tüm df'i dedup'la sonra böl" ≠ "böl sonra train'i dedup'la") →
        # karar kuralı gereği ADIM değil ÖN-KOŞUL olarak beyan edilir.
        "deduplicated":   True,
        # T5-(2): currency dönüşümü `_clean_data`'da, portlanmadı → serving girdisi ZATEN
        # numerik olmalı. `build_features` bu dönüşümü kendi yaptığı için eğitim tarafında
        # olgu SAĞLANMIŞ sayılır; olgunun değeri "girdi normalize edilmiş mi" sorusudur.
        "currency_normalized": True,
        # §2.19'un 6. örneği — zincirleme türev (bir spec'in girdisi başka spec'in çıktısı).
        "chained_derived_features": plan_chained_specs(feature_plan),
        # NORMALİZASYON (fail-open kapatma): build_features'ta boş modality "tabular"
        # demektir (`if modality == "timeseries"` else tabular yol). Boş bırakılsaydı
        # kapsam kuralı `bool(v)` üzerinden onu SESSİZCE geçerdi — beyan var ama değer
        # yok = en kötü hâl. Burada niyet açıkça yazılır.
        "modality":        (modality or "tabular"),
        "aggregated":      bool(aggregated),
        "timeseries_lag":  timeseries_lag,
        # KAPSAM PORTU: reddedilen artık "domain feature ÜRETİLDİ" değil, "domain feature'ı
        # üretip DONMUŞ SPEC beyan ETMEYEN yol kullanıldı". Beyan edenleri DomainFeaturesStep
        # replay ediyor (bkz. FeatureRecipe.specs).
        "unported_domain_features": list(unported_domain_features or []),
        "column_renames":  list(column_renames or []),
    }


def assemble_pipeline_ctx(
    *, intent: str, modality: str, aggregated: bool, target_name: str,
    drop_columns: list, ratio_feature_meta: dict, missing_indicator_meta: dict,
    derived_flag_meta: dict, feature_plan: dict, encode_exclude: set,
    outlier_cols, timeseries_lag, domain_feature_cols: list, date_feature_cols: list,
    column_renames: list, disallowed: "set | None" = None, date_source_cols: "list | None" = None,
    domain_specs: "list | None" = None, unported_domain_features: "list | None" = None,
) -> dict:
    """Katman 1 BEYAN YÜZEYİ (T0) — orkestratörün ihtiyacı olan her şeyi TEK yerde beyan eder.

    Üç tür bilgi taşır:
      1. **Yapısal ön-koşullar** (``deduplicated``/``target_col``/``drop_columns``) —
         ``check_structural_preconditions`` bunları fiili çerçeve üzerinde denetler.
      2. **Donmuş seçimler** (ratio çiftleri, flag kolonları, zscore/normalize spec'leri,
         exclude/outlier listeleri) — pre-split sabit kararlar; orkestratör yeniden TÜRETMEZ,
         manifest'e ``frozen_choice`` olarak DONDURUR.
      3. **Kapsam olguları** (modality/aggregated/ts-lag/domain/date/rename/unported_ops) —
         ``check_pipeline_scope`` bunlara bakar.

    MEKANİZMANIN KİLİT ÖZELLİĞİ: burada BEYAN EDİLMEYEN her olgu, orkestratörde "bilinmiyor"
    sayılır ve artifact'i **``deployable=False``** yapar. Yani bu fonksiyona yeni bir olgu
    eklemeyi unutmak "sessizce destekleniyor" sonucunu DOĞURAMAZ — en kötü ihtimalle artifact
    deploy edilemez olur. Kapsam ancak bilinçli genişletilir (port → guard 0 → kural kaldır).

    SAF fonksiyon: ``build_features``'ın hesapladığı değerleri alır, hiçbir şey hesaplamaz →
    tek başına test edilebilir ve ``build_features``'tan bağımsız evrilir.
    """
    from src.services.inference.orchestrator import (PORTED_FEATURE_OPS,
                                                     PORTED_ROWWISE_OPS)

    derived = (feature_plan or {}).get("derived_features") or []

    def _specs(op_name: str) -> list:
        return plan_op_specs(feature_plan, op_name, disallowed)

    ops_used = {(s.get("op") if isinstance(s, dict) else getattr(s, "op", None)) for s in derived}
    unported_ops = sorted(o for o in ops_used if o and o not in PORTED_FEATURE_OPS)

    return {
        # ── 1. yapısal ön-koşullar ──────────────────────────────
        # `deduplicated` + `currency_normalized` + kapsam olgularının TAMAMI TEK kaynaktan
        # (`assemble_scope_facts`) gelir — ön-uçuş ile asıl kapı aynı değerleri görsün diye.
        **assemble_scope_facts(
            modality=modality, aggregated=aggregated, timeseries_lag=timeseries_lag,
            unported_domain_features=unported_domain_features, column_renames=column_renames,
            feature_plan=feature_plan),
        "target_col":     target_name,
        "drop_columns":   list(drop_columns),
        # ── 2. donmuş seçimler (pre-split sabit kararlar) ───────
        "ratio_pairs":    [[n, a, b] for n, (a, b) in (ratio_feature_meta or {}).items()],
        "missing_indicator_cols": list((missing_indicator_meta or {}).values()),
        "zero_flag_cols": list((derived_flag_meta or {}).values()),
        "zscore_specs":   _specs("zscore"),
        "normalize_specs": _specs("normalize"),
        # 6-OP PORTU: rowwise op'lar TAM ŞEKİLLE (op + tüm inputs + clamp'li params).
        "derived_specs":  plan_derived_specs(feature_plan, PORTED_ROWWISE_OPS, disallowed),
        # DOMAIN PACK PORTU: recipe'lerin dondurulmuş per-row spec'leri (DomainFeaturesStep
        # config kaynağı). Seçim fit-anında donduğu için serving'de YENİDEN TÜRETİLMEZ.
        "domain_specs":   list(domain_specs or []),
        "encode_exclude": sorted(encode_exclude or set()),
        "outlier_cols":   sorted(outlier_cols or []),
        "intent":         intent,
        # ── 3. kapsam olguları ─────────────────────────────────
        # (modality/aggregated/ts-lag/domain/rename + dedup/currency/chained → yukarıdaki
        #  `assemble_scope_facts` bloğu; aşağıdakiler kapsam KURALI olmayan beyanlar.)
        "date_features":   list(date_feature_cols or []),      # türev adları (raporlama; kapsam kuralı KALDIRILDI)
        "date_source_cols": list(date_source_cols or []),      # DateFeaturesStep (order 2) config kaynağı
        "unported_ops":    unported_ops,
    }


def _run_orchestrated_chain(
    builder, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train, pipeline_ctx: dict,
    meta, domain_leakage_report: dict, raw_test: "pd.DataFrame | None" = None,
) -> tuple:
    """POST-split zinciri (``PIPELINE[8:]``) TEK ``builder``da devret; pre-split orders 0-7
    (missing-indicator + impute + **drop_columns** (T4a) + **zscore + normalize** (T4b))
    ``build_features``'ta ZATEN fit edildi → tek order uzayı.

    ⚠️ **KAPI SIRASI (ÖLÇÜLDÜ, varsayılmadı):** ``check_scope`` BURADA çağrılır — yani split'ten
    SONRA, pre-split adımlardan SONRA. ``ArtifactBuilder`` kurulumunda DEĞİL (dataclass'ın
    ``__post_init__``'i yok; ``check_scope`` açık bir metottur). Sebep yapısal: ``pipeline_ctx``
    ancak ratio/zeroflag'den sonra kurulabiliyor. T4a ile ``drop_columns`` + yapısal ön-koşul
    denetimi pre-split'e kaydığı için **``check_scope``'un ÖNÜNE** geçtiler. Fail-closed HER İKİ
    sırada da korunur (ikisi de fırlatır, artifact kurulmaz); değişen, ikisini birden ihlal eden
    bir koşuda önce hangi teşhisin çıktığı. Kilit: ``test_control_gate_order_is_locked``.

    ``ImputeStep`` pre-split (SAHİPLİK ``_clean_data``'dan orkestratöre geçti, T3a-1) ve **T3b-fit'ten
    itibaren `fit_frame=df.iloc[_train_idx]`** → dolgu istatistiği YALNIZ train'den (#4 fiksi deploy).
    Artık bit-birebir DEĞİL: sapma yüzeyi = train-medyanı full-medyandan farklı olan kolonların
    TEST hücreleri (ölçüldü: yalnız housing/``total_bedrooms``, 207 hücre + ondan türeyen 3 oran).
    ``_FEATURE_OPS`` zscore/normalize hâlâ eski leaky yolda (T4). Beyan dışı sapma = porting bug'ı.

    KAPSAM/ÖN-KOŞUL REDDİ FIRLATIR (bilinçli): pre-split adımlar kapsam-kapısından ÖNCE koşar
    (pipeline_ctx ancak ratio/zeroflag sonrası hazır) ama fail-closed KORUNUR — ihlal burada
    ``check_scope``'ta firlar, artifact KURULMAZ. Desteklenen kapsamda (referans setler) fark yok.

    Dönüş: ``(X_train, X_test, mid_train, mid_test, ce_cols, gd_cols,
    univariate_dropped, leakage_report)`` — ``mid_*`` = ``ratio_finalize`` SONRASI, encode
    ÖNCESİ çerçeveler (fairness taşıyıcısı + Veri İlişkileri onları ister).
    """
    from src.services.inference.orchestrator import transform_pipeline, PIPELINE

    # ``pipeline_ctx`` YAYINLANAN beyan yüzeyidir (feature_report'a girer) ve JSON-temiz
    # kalmalıdır → ``meta`` gibi canlı objeler YALNIZ burada, çağrı yerinde eklenir.
    ctx_fit = {**pipeline_ctx, "meta": meta, "domain_leakage_report": domain_leakage_report}
    builder.check_scope(ctx_fit)                 # KAPSAM KAPISI (ctx hazır; pre-split adımlardan sonra)

    mid: dict = {}

    def _catch(key: str):
        def _obs(step_id, X):        # X = KOPYA (gözlemci yapısal olarak read-only)
            if step_id == "ratio_finalize":
                mid[key] = X
        return _obs

    # POST-split faz: PIPELINE[8:] (ratio_construct..select) X_train'de; builder order 7'den devam eder
    # (0-7 pre-split: missing_indicator, impute, **date_features**, drop_columns (T4a),
    # zscore+normalize (T4b), plan_derived, **domain_features** (domain pack portu)).
    # Post-split adımlar T2'deki ile BİREBİR (X_train fully-built).
    X_train_out = X_train.copy()
    for spec in PIPELINE[8:]:
        X_train_out = builder.fit_step(spec, X_train_out, y_train, ctx_fit, on_step=_catch("train"))
    artifact = builder.finalize(X_train_out.columns)
    # T5-(2): SERVING YOLU ARTIK ASIL YOL — `raw_test` verilmişse X_test HAM satırlardan
    # baştan üretilir (manifest replay). Böylece "serving parity" dolaylı değil DOĞRUDAN
    # ölçülür ve delta-guard incremental çerçevede fiilen çalışır (muafiyet gerekmez).
    # `raw_test=None` → eski davranış (fully-built X_test); geriye-uyum için korunur.
    X_test_out = transform_pipeline(
        X_test if raw_test is None else raw_test, artifact, on_step=_catch("test"),
        # ── ARA DURUM: girdi HAM DEĞİLSE iki denetim de anlamsız, AÇIKÇA kapatılır ──
        # `verify_input`: imza HAM sözleşmeyi tarif eder; fully-built çerçeveye uygulanamaz.
        # `verify_delta`: ÖLÇÜLDÜ (superstore) — pre-split adımların etkisi fully-built X_test'te
        #   ZATEN olmuş olduğu için adımlar orada hiçbir şey yapamaz (`drop_columns` eğitimde
        #   'Row ID' düşürür, fully-built çerçevede kolon zaten yoktur → `removed=[]`). Yani
        #   delta-guard YALNIZ ham yolla tutarlıdır. Cutover (SERVING_RAW_REPLAY) açılınca ikisi
        #   de kalıcı olarak AÇIK olacak. **Sessiz zayıflatma değil:** karar burada, gerekçesiyle.
        verify_input=(raw_test is not None), verify_delta=(raw_test is not None))

    # Yan-değerler: ÜRETİM SIRASIYLA params'tan (manifest'in ``columns_out``'u SIRALI
    # olduğu için oradan türetmek `auto_features` liste sırasını alfabetikleştirir → rapor sapar).
    ce_cols = list((artifact.param_store.get("count_encode") or {}).keys())
    gd_cols = [name for rec in (artifact.param_store.get("groupby_deviation") or [])
               for _, name in rec.get("emits", [])]
    sel = artifact.param_store.get("select_drop") or {}
    leakage_report = sel.get("leakage_report") or {"dropped": [], "suspected": []}
    leakage_report["dropped"].extend(sel.get("domain_dropped") or [])

    logger.info(f"[feature_eng] T3a-1 ORKESTRATOR: {len(artifact.manifest)} manifest kalemi "
                f"(pre-split 7 + post-split {len(PIPELINE) - 7}), "
                f"deployable={artifact.supported_scope.get('deployable')}")
    # WIRING: artifact artık YEREL DEĞİL — Faz A persistence için çağırana döndürülür
    # (build_features raw_train + save_artifact bağlamına sahip; burada disk I/O yapmıyoruz).
    return (X_train_out, X_test_out, mid.get("train", X_train), mid.get("test", X_test),
            ce_cols, gd_cols, sel.get("univariate_dropped") or [], leakage_report, artifact)


def _apply_date_features(df: pd.DataFrame, timestamp_cols, leaky_cols_high: set, *, write: bool
                         ) -> "tuple[list[str], list[str]]":
    """Timestamp kolonlarından ``_year``/``_month``/``_dayofweek`` türet (DURUMSUZ). §2.19 KAPSAM
    PORTU'nun legacy anchor'ı — ``DateFeaturesStep`` bunu devraldı (legacy_ref="_apply_date_features").

    ``write=True`` (legacy/shadow): kolonları ``df``'e YAZAR (eski inline davranış birebir).
    ``write=False`` (orchestrator+cutover): YALNIZ seçim döner; yazma order-2 adımında (ikiz borç deseni).
    Döner: (date_feature_cols=türev adları [raporlama], date_source_cols=genişletilecek ts kolonları [step config])."""
    date_feature_cols: list[str] = []
    date_source_cols: list[str] = []
    for col in timestamp_cols:
        if col not in df.columns:
            continue
        if col in leaky_cols_high:
            logger.info(f"[feature_eng] SKIP türev: '{col}' domain leakage (high) — year/month/dayofweek üretilmiyor")
            continue
        date_source_cols.append(col)
        date_feature_cols.extend([f"{col}_year", f"{col}_month", f"{col}_dayofweek"])
        if write:
            try:
                dt = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"]      = dt.dt.year
                df[f"{col}_month"]     = dt.dt.month
                df[f"{col}_dayofweek"] = dt.dt.dayofweek
            except Exception:
                pass
    return date_feature_cols, date_source_cols


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
        # Örnekleme metadatasını HEMEN yakala — df.attrs sonraki FE transformasyonlarında
        # düşebilir; feature_report'a taşınıp rapora yansıtılacak (sessiz değil).
        sampling_meta = dict(df.attrs.get("_dq_sampling", {}))
    except Exception as e:
        import traceback
        logger.error(f"\n[feature_eng] HATA: {e}")
        logger.info(traceback.format_exc())
        return {"error": str(e)}

    if df.empty or len(df) < 5:
        return {"error": f"Veri seti yüklenemedi veya çok az veri var ({len(df)} satır)."}

    logger.info(f"[feature_eng] Veri: {len(df):,} satır × {len(df.columns)} kolon")

    # Multi-table join suffix'lerini temizle. Çarpışmalar (farklı-değerli iki kolonun
    # aynı normalize adına düşmesi) rapora taşınır → sessiz veri kaybı asla (Faz 0.5).
    all_tables = [t["table_name"] for t in db_intelligence.get("tables", [])]
    _cols_before_norm = list(df.columns)          # Katman 1 beyani: yeniden-adlandirma oldu mu?
    df, normalization_collisions = _normalize_column_names(
        df, table_names=all_tables if all_tables else None)
    # Katman 1 KAPSAM OLGUSU: _normalize_column_names portlanmadi → ad degistiyse
    # orkestratorun ham ad → egitim adi eslemesi YOK; kapsam kapisi bunu reddeder.
    column_renames = [[o, n] for o, n in zip(_cols_before_norm, list(df.columns)) if o != n] \
        if len(_cols_before_norm) == len(df.columns) else [["<kolon-sayisi-degisti>", ""]]
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
    # Katman 1 T3a-1: pre-split adım SAHİPLİĞİ orkestratörde. ``orchestrator`` modunda
    # missing-indicator (order 0) + impute (order 1) TEK ``ArtifactBuilder``'a fit edilir; split'in
    # ÖTE yanında (post-split) AYNI builder devam eder → tek order uzayı (iki-fit_pipeline reddini
    # ihlal etmez, bkz. ArtifactBuilder docstring). legacy/shadow yolu BİREBİR DEĞİŞMEZ (kontrol arm'ı).
    _pipe_mode = settings.FEATURE_PIPELINE_MODE
    _orch_mode = (_pipe_mode == "orchestrator")
    _builder = None
    if _orch_mode:
        from src.services.inference.orchestrator import ArtifactBuilder, PIPELINE as _ORCH_PIPELINE
        _builder = ArtifactBuilder()

    # T5-D1: orchestrator modunda SEÇİM burada, YAZMA orkestratörde (ikiz borç devri).
    auto_feature_cols: list[str] = _add_missing_indicators(
        df, exclude=leakage_exclude,
        write=not (_orch_mode and SERVING_RAW_REPLAY))
    # Katman 1 (T0): FILTRELENMEMIS kayit — {uretilen: kaynak}. `auto_features` alani
    # X_train'e gore filtreli yayinlanir (hayatta-kalan listesi) → orkestratorun donmus
    # secim kaynagi olarak YETERSIZ; select_drop'un dusurdukleri kaybolur ve groupby'in
    # secim yaptigi ara cerceve eski yoldan sapar. `ratio_feature_meta` dersinin tekrari.
    # NOT: burada ad-cozumleme guvenli — suffix SABIT (`f"{c}_isna"`), ratio'nun
    # `ratio_a_b` belirsizligi (hangi alt cizgi boluyor) yok.
    missing_indicator_meta: dict[str, str] = {n: n[: -len("_isna")] for n in auto_feature_cols}
    if _orch_mode:
        # order 0 — MissingIndicatorStep: **T5-D1 İKİZ BORÇ KAPANDI** — `*_isna` kolonlarını artık
        # EĞİTİMDE DE bu adım üretiyor (inline yazma `write=False` ile kalktı). Sonuç: adımın kolon
        # sözleşmesi (`delta.added`) eğitimde ve ham serving replay'inde AYNI → delta-guard geçer.
        # Öncesinde adım eğitimde no-op'tu ve ayrışma SESSİZDİ (mutlak guard göremiyordu).
        df = _builder.fit_step(_ORCH_PIPELINE[0], df,
                               ctx={"missing_indicator_cols": list(missing_indicator_meta.values())})

    # Temizlik — orchestrator modunda impute BLOĞU atlanır (builder yapar; T3a-2'de resolve_target
    # + erken split ARADA → impute artık hedefi görmez, item(3) hedef null=0 → INERT).
    df = _clean_data(df, impute_missing=not _orch_mode)
    # ── T5-(2): SERVING GİRDİSİ ANLIK GÖRÜNTÜSÜ (ham-satır replay'in kaynağı) ──
    # Yakalama noktası BİLİNÇLİ olarak `_clean_data` SONRASI (ölçüldü, iki sebep):
    #  (1) DEDUP burada: credit'te 165 satır düşüyor → order-0 girdisinde yakalasaydık
    #      pozisyonel/etiket hizalaması split'le tutmazdı (ölçüm: data/_t5_rowalign.json).
    #  (2) CURRENCY burada: dönüşüm portlanmadı → daha erken yakalanan çerçevede para
    #      kolonları hâlâ string olurdu ve pipeline'da onları çeviren adım YOK.
    # Sözleşme böylece açık: **serving girdisi dedup'lanmış + currency-normalize edilmiş olmalı**
    # (`deduplicated` + `currency_normalized` kapsam olguları).
    # Inline üretilmiş `_isna` kolonları düşürülür — onları serving'de order 0 ÜRETİR (ikiz borç).
    _df_raw = None
    _resolved_early = False
    _train_idx = _test_idx = None
    if _orch_mode:
        # ── T3a-2: resolve_target impute-ÖNCESİNE (missing-indicator + currency'den SONRA) ──
        # Erken split indeksi stratify=y ister → resolve_target burada çözülür. ÖN-UÇUŞ (ölçüldü):
        # alias YOK (item 2) + hedef null=0 (item 3) → reorder bit-birebir INERT. ts-lag/aggregation
        # orch'ta kapsam-REDDİ (tabular) → hedefsiz koşmaları önemsiz (artifact zaten kurulmaz).
        df, y, target_name = _resolve_target(df, feature_plan, intent)
        _resolved_early = True
        # ── SERVING GİRDİSİ ANLIK GÖRÜNTÜSÜ (ham-satır replay'in kaynağı) ──
        # Konum BİLİNÇLİ: `_clean_data` SONRASI (dedup + currency portlanmadı → girdi zaten
        # normalize gelmeli) ve `_resolve_target` SONRASI (**serving sözleşmesi ETİKETSİZDİR**).
        # Inline üretilmiş `*_isna` düşürülür — serving'de onları order 0 ÜRETİR.
        # id kolonları serving İMZASINDAN çıkarılır — model feature'ı DEĞİL (DropColumnsStep order-3'te
        # zaten düşürür). Snapshot'ı "id-drop SONRASINA almak" = id'yi burada düş. **TIMESTAMP KORUNUR**
        # (date türevleri ham ts'yi ister → imzada required kalmalı). Target zaten resolve_target'ta çıktı.
        # Serving'de id gelmese bile delta-guard artık tolere eder (orchestrator eksik_silme: kolon _before'da
        # yoksa "silinmedi" ihlali sayılmaz — "çıktıda yok" zaten sağlanır).
        _sig_drop = [c for c in meta.get_by_type("identifier") if c in df.columns]
        _df_raw = df.drop(columns=[c for c in missing_indicator_meta if c in df.columns] + _sig_drop,
                          errors="ignore").copy()
        # datetime_cols: meta'nın timestamp tiplediği + _df_raw'da mevcut kolonlar → imzada kind="datetime"
        # (date portu sonrası ham ts kolonu required; string dtype de olsa authoritative işaretlenir).
        _dt_cols = [c for c in meta.get_by_type("timestamp") if c in _df_raw.columns]
        _builder.declare_raw_input(_df_raw, datetime_cols=_dt_cols)  # beyan; order-0'dan TÜRETME (ikiz borç)
        # ERKEN SPLIT İNDEKSİ — geç split'i (aşağıda) df.iloc[idx] ile değiştirir. Partition + SIRA
        # geç train_test_split(df,y,...) ile BİREBİR (kolon-bağımsız: C-ölçümü; G1 .equals() satır sırasını
        # kanıtlar). Rows bu noktadan split'e kadar DEĞİŞMEZ (agg scope-reddi) → pozisyonel indeks geçerli.
        _strat = y if intent.strip().lower() == "classification" else None
        try:
            _train_idx, _test_idx = train_test_split(
                np.arange(len(df)), test_size=TEST_SIZE, random_state=settings.RANDOM_SEED, stratify=_strat)
        except ValueError:
            _train_idx, _test_idx = train_test_split(
                np.arange(len(df)), test_size=TEST_SIZE, random_state=settings.RANDOM_SEED)
        # order 1 — ImputeStep. **T3b-fit: #4 FİKSİ BURADA DEPLOY OLUYOR.** Dolgu istatistiği artık
        # YALNIZ train indeksinden öğrenilir (`fit_frame`), uygulama tüm çerçevededir (`df`) — impute
        # KONUMU korunur, yani Bulgu A'nın dört pre-split tüketicisi (resolve_target/plan-op/ratio/
        # zeroflag) hâlâ null'suz çerçeve görür. Değişen TEK şey medyanın kaynağı (full → train).
        # T3b-gate ön koşuldu: `isnull().any()` kapısı dursaydı train'de null'suz kolon (housing
        # total_bedrooms, 207 null'un 207'si TEST'te) dolgu ÖĞRENMEZ ve terminal fillna(0)'a düşerdi.
        df = _builder.fit_step(_ORCH_PIPELINE[1], df, fit_frame=df.iloc[_train_idx])

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

    # Hedef değişken — T3a-2'de orch modunda ZATEN yukarıda (impute ÖNCESİ) çözüldü (erken split için).
    if not _resolved_early:
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

    # Date feature extraction — §2.19 KAPSAM PORTU: DateFeaturesStep'in legacy anchor'ı. T5-D1 deseni:
    # legacy/shadow (write=True) inline üretir; orchestrator+cutover (write=False) YALNIZ seçim döner,
    # YAZMA order-2 adımında. date_feature_cols=türev adları (raporlama), date_source_cols=step config.
    date_feature_cols, date_source_cols = _apply_date_features(
        df, meta.get_by_type("timestamp"), leaky_cols_high,
        write=not (_orch_mode and SERVING_RAW_REPLAY))
    if _orch_mode:
        # order 2 — DateFeaturesStep (§2.19 KAPSAM PORTU): YAZMA burada (inline write=False), impute
        # SONRASI / drop ÖNCESİ. Ham ts kolonu hâlâ df'te (drop order 3'te düşürür). Rowwise → fit_frame yok.
        df = _builder.fit_step(_ORCH_PIPELINE[2], df, ctx={"date_source_cols": date_source_cols})

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
    if _orch_mode:
        # T4a — SAHİPLİK DEVRİ: id/ts/target drop'u artık orkestratörün pre-split adımı yapıyor
        # (order 2). KONUM birebir korunur (impute'tan SONRA, plan-türevi/#5'ten ÖNCE) → çerçeve
        # aynı → bit-birebir beklenir; değişen yalnız SAHİPLİK. Yapısal ön-koşul denetimi
        # (`check_structural_preconditions`) adımın içinde tetiklenir → o da buraya kayar.
        # T4b'de ZScore/Normalize de pre-split'e taşınacağı için bu adımın önce taşınması ŞART:
        # ön-ek bloğu bitişik olmalı (order uzayı monoton; drop, derived'dan ÖNCE gelmeli).
        df = _builder.fit_step(_ORCH_PIPELINE[3], df,
                               ctx={"drop_columns": existing, "target_col": target_name})
    elif existing:
        df = df.drop(columns=existing, errors="ignore")
        logger.info(f"[feature_eng] Drop (id/ts/target): {existing}")

    # Leakage denylist for every derived-feature path below: target + its aliases +
    # high-confidence leaky cols can never become an input.
    disallowed = {target_name, target_col} | set(related) | set(leaky_cols_high)

    # Generic (sector-independent) derived features: the feature planner LLM picks
    # an op from the safe palette over REAL columns (e.g. velocity = ratio(x, t)).
    # Validated + deterministically executed — replaces the old raw df.eval path,
    # which let the LLM write code with no leakage guard.
    # 6-OP PORTU: orchestrator modunda plan'ın TÜM portlu op'ları (8/8) BURADA çalışmaz —
    # sahiplik orkestratörün pre-split adımlarında (order 3-4 parametric, 5 rowwise).
    # legacy/shadow arm'ında eski yol AYNEN çalışmaya devam eder (kontrol arm'ı bozulmaz).
    from src.services.inference.orchestrator import (PORTED_FEATURE_OPS as _PORTED,
                                                     PORTED_ROWWISE_OPS)
    derived_feature_cols = _apply_plan_derived_features(
        df, feature_plan, disallowed, skip_ops=(_PORTED if _orch_mode else None))
    if _orch_mode:
        # order 3-4-5 — ZScore/Normalize (parametric, **fit_frame = TRAIN indeksi**, #5 fiksi)
        # + PlanDerived (rowwise, fitted-state yok → fit_frame gereksiz).
        # havuzunda görünür, seçim sapmaz (Bulgu A dersi).
        # ctx henüz kurulmadı (assemble_pipeline_ctx ratio/zeroflag'den sonra) → spec'ler
        # AYNI TEK KAYNAKTAN (`_admit_spec` tabanlı yardımcılar) türetilir, kopyalanmaz.
        _rowwise = plan_derived_specs(feature_plan, PORTED_ROWWISE_OPS, disallowed)
        _pre_ctx = {"zscore_specs": plan_op_specs(feature_plan, "zscore", disallowed),
                    "normalize_specs": plan_op_specs(feature_plan, "normalize", disallowed),
                    "derived_specs": _rowwise}
        for _spec_idx in (4, 5):        # parametric ikili → train-fit (date portu order 2'yi kaydırdı)
            df = _builder.fit_step(_ORCH_PIPELINE[_spec_idx], df, ctx=_pre_ctx,
                                   fit_frame=df.iloc[_train_idx])
        df = _builder.fit_step(_ORCH_PIPELINE[6], df, ctx=_pre_ctx)   # rowwise altılı (fit yok)
        # Yan-değer: eski op'un döndürdüğü liste artık eksik → adımlardan TAMAMLA
        # (üretim sırasıyla; T2'nin `_ce_cols`/`_gd_cols` deseni).
        for _sid in ("zscore_feature", "normalize_feature"):
            derived_feature_cols = derived_feature_cols + list(
                (_builder.param_store.get(_sid) or {}).keys())
        derived_feature_cols = derived_feature_cols + [s["name"] for s in _rowwise]
    if derived_feature_cols:
        logger.info(f"[feature_eng] Plan turevli feature'lar: {derived_feature_cols}")

    # Domain pack feature'ları (additive + guarded). Pack 'none' ise hiç çalışmaz.
    #   (a) Ajan-karar yolu: expert ajanının propose_feature ile seçtiği FeatureSpec'ler,
    #       gerçek şemaya + leakage denylist'e karşı doğrulanıp deterministik uygulanır.
    #   (b) Statik fallback: ajan bir şey üretmediyse, convention-bazlı recipe'ler devreye girer.
    # KAPSAM PORTU: (b) yolunun spec BEYAN EDEN recipe'leri orkestratöre devredilir (yazma
    # order 7'de); (a) ajan yolu ve spec'siz recipe'ler PORTSUZ → `unported_domain_features`
    # kapsam olgusunu doldurur ve aşağıdaki ön-uçuş koşuyu legacy zincire alır.
    domain_feature_cols: list[str] = []
    domain_specs: list[dict] = []
    domain_unported: list[str] = []
    try:
        agent_cols = _apply_feature_specs(df, state, disallowed)
        domain_feature_cols.extend(agent_cols)
        domain_unported.extend(agent_cols)     # ajan FeatureSpec'leri henüz portlanmadı
        if not agent_cols:
            _rec_cols, domain_specs, _rec_unported = _apply_domain_recipes(
                df, state, meta, defer_to_orchestrator=(_orch_mode and SERVING_RAW_REPLAY))
            domain_feature_cols.extend(_rec_cols)
            domain_unported.extend(_rec_unported)
        if domain_feature_cols:
            logger.info(f"[feature_eng] Domain feature'lari: {domain_feature_cols}")
    except Exception as e:
        logger.error(f"[feature_eng] Domain feature hatasi ({e}); atlaniyor.")

    # ── KAPSAM ÖN-UÇUŞU — kapı burada, ratio/zero-flag SAHİPLİK ayrımından ÖNCE ──
    # Kapsam kapısı (`ArtifactBuilder.check_scope`) `_run_orchestrated_chain`'de DURUYOR ve
    # fail-closed KALIYOR (son savunma). Ama orada patlamak koşuyu komple öldürüyordu: istisna
    # feature_engineering_node'a kadar çıkar, kullanıcı raporunu da alamaz. Kapının anlamı ise
    # "bu koşu DEPLOY EDİLEMEZ" — "bu koşu ÇALIŞAMAZ" değil. Bu yüzden ihlal, olguların hepsinin
    # BİLİNDİĞİ ilk noktada (domain feature'ları yeni hesaplandı; ts-lag/modality/rename zaten
    # biliniyordu) SAPTANIR ve orkestratör modu KAPATILIR → legacy zincire temiz dönüş.
    # NEDEN TAM BURASI: bir sonraki satırdan itibaren `_orch_mode` sahiplik ayrımı yapıyor
    # (`write=not (_orch_mode and SERVING_RAW_REPLAY)`) — ratio/zero-flag kolonları ya inline
    # yazılır ya orkestratör adımına bırakılır. Karar bu çatallanmadan SONRA değişseydi çerçeve
    # yarı-yazılmış kalırdı (legacy `_finalize_ratio_features_tt` olmayan kolonu arardı).
    # Artifact SONUCU DEĞİŞMEZ: ihlalde eskiden de artifact kurulmuyordu, şimdi de kurulmuyor
    # (deployable=False + gerekçe raporda). Değişen tek şey: eğitim/rapor artık HAYATTA KALIYOR.
    _scope_block: list = []
    if _orch_mode:
        from src.services.inference.orchestrator import check_pipeline_scope as _check_scope
        _preflight = _check_scope(assemble_scope_facts(
            modality=modality, aggregated=aggregated, timeseries_lag=ts_lag_info,
            unported_domain_features=domain_unported, column_renames=column_renames,
            feature_plan=feature_plan))
        if _preflight["violations"]:
            _scope_block = _preflight["violations"]
            _orch_mode = False
            _ihlal = "; ".join(f"{v['fact']}={v['value']!r} → {v['reason']}" for v in _scope_block)
            logger.warning(
                f"[feature_eng] KAPSAM DIŞI ({len(_scope_block)} koşul) → Katman 1 artifact "
                f"KURULMAYACAK (deploy edilemez), koşu LEGACY zincirle SÜRÜYOR: {_ihlal}")

    # order 7 — DomainFeaturesStep: devredilen domain kolonlarının YAZILDIĞI yer. Konum
    # bilinçli: ratio/zero-flag SEÇİMİNDEN önce (onların "bu ad zaten var mı" kapısı bu
    # kolonlara bakar) → eski yolun gördüğü çerçevenin AYNISI.
    if domain_specs:
        if _orch_mode:
            df = _builder.fit_step(_ORCH_PIPELINE[7], df, ctx={"domain_specs": domain_specs})
        else:
            # KAPSAM BAŞKA bir olgudan reddedildi (ör. modality) → devir geçersiz; kolonları
            # AYNI adım sınıfıyla burada yaz (kod tek, replay ile bit-birebir; ikinci bir
            # inline uygulama yazmak sapma kaynağı olurdu).
            from src.services.inference.steps import DomainFeaturesStep as _DomainStep
            df = _DomainStep(config={"specs": domain_specs}).transform(df)

    # ── DOKUNUS 2: deterministik sifir/oran motorlari (SPLIT ONCESI, row-wise) ──
    # is_zero = orijinal sayisal kolonlar (row-wise, deterministik → leakage-safe).
    # was_missing artik uretilmiyor: arkadasin _add_missing_indicators (_isna) ayni
    # sinyali own-goal korumasiyla uretiyor (carrier yok → _build_zero_missing_flags
    # yalniz is_zero dondurur). Ratio = orijinal sayisal cift (a/b, tek yon); leaky_cols_high
    # + ID-benzeri kaynak dislanir. RATIO ONCE calisir → flag motoruna "sifir-payda"
    # setini bildirir; bu paydalarin is_zero flag'i garanti uretilir (invariant). Boylece
    # oran split sonrasi sentinel'den aritilirken (bkz. _finalize_ratio_features_tt)
    # sifir-payda bilgisi flag'de dururur. HAM oran degerleri (winsor/impute YOK) split
    # sonrasi train'e fit edilir; burada yalniz row-wise bolme + selection yapilir.
    # T5-D2: orchestrator + cutover açıkken SEÇİM burada, YAZMA orkestratörde (order 5).
    df, ratio_feature_cols, ratio_denom_zero_cols, ratio_feature_meta = _build_ratio_features(
        df, original_cols, meta, intent, aggregated, leaky_cols_high,
        write=not (_orch_mode and SERVING_RAW_REPLAY))
    # T5-D3: SEÇİM burada (force_zero invariant DAHİL), YAZMA orkestratörde (order 6).
    df, derived_flag_cols = _build_zero_missing_flags(
        df, original_cols, intent, aggregated, force_zero_cols=ratio_denom_zero_cols,
        write=not (_orch_mode and SERVING_RAW_REPLAY))
    # Katman 1 (T0): FILTRELENMEMIS kayit — {uretilen: kaynak}. `derived_flags` alani
    # X_train'e gore filtreli (hayatta-kalan) → frozen kaynagi olarak yetersiz (yukaridaki
    # missing_indicator_meta ile ayni gerekce). Prefix SABIT → ad-cozumleme belirsiz degil.
    derived_flag_meta: dict[str, str] = {
        n: n[len("is_zero_"):] for n in derived_flag_cols if n.startswith("is_zero_")}

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

    # Katman 1 BEYAN YÜZEYİ — BİR KEZ kurulur, iki tüketicisi var: gölge koşu (T1) ve
    # feature_report["pipeline_ctx"]. Tek kaynak → ikisi sapamaz.
    pipeline_ctx = assemble_pipeline_ctx(
        intent=intent, modality=modality, aggregated=aggregated, target_name=target_name,
        drop_columns=existing, ratio_feature_meta=ratio_feature_meta,
        missing_indicator_meta=missing_indicator_meta, derived_flag_meta=derived_flag_meta,
        feature_plan=feature_plan, encode_exclude=(disallowed | leakage_exclude),
        outlier_cols=meta.get_outlier_cols(), timeseries_lag=ts_lag_info,
        domain_feature_cols=domain_feature_cols, date_feature_cols=date_feature_cols,
        domain_specs=domain_specs, unported_domain_features=domain_unported,
        date_source_cols=date_source_cols, column_renames=column_renames,
        # T4 (3): eski op'un ÇEVRESİNDEKİ leakage denylist'i beyan yüzeyine taşınır — op
        # emekli olunca kaybolmasın (bkz. assemble_pipeline_ctx içindeki gerekçe).
        disallowed=disallowed)
    shadow_report: dict = {}             # T1 gölge koşu sonucu (yalnız shadow modunda dolar)

    # 14. Train/test split
    split_imbalanced = 0
    univariate_dropped: list[str] = []   # denetimli univariate seçim (yalnızca supervised dalda dolar)
    # Fairness segment taşıyıcısı — yalnızca supervised dalda dolar. Unsupervised'da
    # holdout/y_test yok, dolayısıyla grup başına performans karşılaştırması TANIMSIZ.
    # Sessizce boş bırakmak yerine SEBEBİ taşınır (rapor "uygulanmadı çünkü ..." der).
    _seg_df = None
    # SEBEP dil-nötr KOD taşır (rapor render anında R_REASON ile çevirir) — servis
    # katmanı ham Türkçe basmaz (kod-nötrlük ilkesi; bkz. report_i18n başlığı).
    segment_meta: dict = {
        "segment_columns": [], "carried_columns": [], "mode": "", "skipped": {}, "n_rows": 0,
        "reason": {"code": "seg_unsupervised"},
    }
    # Veri İlişkileri: supervised dalda gerçek train partisyonundan (train_only),
    # unsupervised'da holdout OLMADIĞI için tüm veriden (full_data_descriptive).
    # Etiket her iki durumda da hesabın fiilen geçtiği satırları söyler.
    relations_report: dict = {}
    if intent in ("clustering","anomaly_detection"):
        # İlişkiler HAM df'ten, encode/scale'den ÖNCE. Holdout olmadığı için hesap
        # tüm veriden geçer → "full_data_descriptive": rapor bunu tanımlayıcı olarak
        # sunar, tahmine dönük bir bulgu gibi DEĞİL.
        from src.services.relations import compute_relations
        relations_report = compute_relations(
            df, None, "", generated_from="full_data_descriptive",
        )
        # Unsupervised: holdout yok → transform'lar TÜM veriye fit edilir (kümeleme
        # için doğru/istenen davranış; target sızıntısı kavramı zaten yok). Eski
        # tekil-df siblingleri burada aynen kullanılır.
        # Oran finalize (winsor+impute) burada tüm df'ye fit+transform — sentinel yok.
        df = _finalize_ratio_features(df, ratio_feature_cols)
        ratio_feature_cols = [c for c in ratio_feature_cols if c in df.columns]
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
            split_strategy = "random"
            if _resolved_early:
                # T3a-2: ERKEN hesaplanan indeksle böl (yukarıda resolve_target sonrası). Aynı partition
                # + SIRA → geç train_test_split ile bit-birebir (pozisyonel; G1 .equals() satır sırasını kanıtlar).
                X_train, X_test = df.iloc[_train_idx], df.iloc[_test_idx]
                y_train, y_test = y.iloc[_train_idx], y.iloc[_test_idx]
            else:
                stratify_arg = y if intent.strip().lower() == "classification" else None
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

        # ── Oran finalize: winsor sınırları + impute median YALNIZCA X_train'den ──
        # (sklearn transformer disiplini). HAM oran kolonları (a/b, NaN korunmuş) burada
        # clip[q01,q99] + NaN→train-median görür; payda-sıfır ∪ kaynak-missing aynı
        # dağılım-içi değere iner → gizli 0.0 sentinel YOK. Encode/scale/relations'tan
        # ÖNCE: hepsi temiz sonlu değer görür, kuantiller test'ten sızmaz.
        # Katman 1 T1 — GÖLGE KOŞU kancası: post-split zincir ÖNCESİ X_train'i yakala.
        # Yalnız `shadow` modunda kopyalanır (legacy'de maliyet SIFIR). Bkz. inference/shadow.py.
        _mode = settings.FEATURE_PIPELINE_MODE
        _shadow_x_train = (X_train.copy() if _mode == "shadow" else None)

        # Katman 1 wiring — varsayılan (legacy/shadow): artifact YOK. orchestrator branch üzerine yazar.
        # KAPSAM GERİ DÜŞÜŞÜ: `_scope_block` doluysa mod orchestrator olsa da legacy zincir koştu →
        # gerekçe MOD DEĞİL İHLAL olarak yazılır (rapor "neden deploy edilemiyor"u göstersin).
        _artifact_id = None
        _artifact_block = {"artifact_id": None, "deployable": False, "phase": None,
                           "reason": ("unsupported_scope: " + "; ".join(
                               f"{v['fact']}={v['value']!r}" for v in _scope_block))
                           if _scope_block else "feature_pipeline_mode!=orchestrator"}
        if _scope_block:
            _artifact_block["scope_violations"] = _scope_block

        if _orch_mode:
            # ── YAPISAL DEVİR — pre-split (builder orders 0-1) + post-split (PIPELINE[2:]) ──
            # `ImputeStep` pre-split (T3a-1 sahiplik devri) + train-fit (T3b-fit, #4 deploy) → artık
            # metrik-NÖTR DEĞİL; sapma beyan edilen yüzeyle sınırlı. zscore/normalize hâlâ eski leaky
            # yolda (T4). `_builder` pre-split fazından (yukarıda) devralınır.
            # T5-(2): X_test artık SERVING YOLUNDAN üretilir — ham satırlar `transform_pipeline`'a
            # verilir. Hizalama **ETİKET-BAZLI** (`.loc`), pozisyonel DEĞİL: `_clean_data` dedup'ı
            # satır üyeliğini değiştirebiliyor (credit −165) ve `drop_duplicates` orijinal etiketleri
            # KORUYOR → etiketle seçim, satır üyeliği nerede değişirse değişsin doğru hizalar.
            # ── ARA DURUM (ÖLÇÜLDÜ, kayda geçti): ham-satır yolu İKİZ BORÇ tarafından
            # BLOKLU. `_add_missing_indicators`/`_build_zero_missing_flags`/`_build_ratio_features`
            # eğitimde HÂLÂ inline üretiyor → order 0/5/6 adımları eğitimde "kolon zaten var"
            # görüp ATLIYOR (delta.added=[]), ham replay'de ise ÜRETİYOR → delta-guard DOĞRU
            # şekilde sapma raporluyor. Guard'ı zayıflatmak yerine ham yol ERTELENDİ; ikiz borç
            # (§2.25 #2) kapanınca `_df_raw.loc[X_test.index]` verilecek. Ölçüm: fully-built
            # yolda delta'lar TUTUYOR → guard bu turda da tam kapsamlı çalışıyor (muafiyet YOK).
            _raw_test = (_df_raw.loc[X_test.index]
                         if (_df_raw is not None and SERVING_RAW_REPLAY) else None)
            (X_train, X_test, _mid_train, _mid_test, _ce_cols, _gd_cols,
             univariate_dropped, leakage_report, _artifact) = _run_orchestrated_chain(
                _builder, X_train, X_test, y_train, pipeline_ctx, meta, domain_leakage_report,
                raw_test=_raw_test)
            ratio_feature_cols = [c for c in ratio_feature_cols if c in _mid_train.columns]
            auto_feature_cols = auto_feature_cols + _ce_cols + _gd_cols
            # Zincir-ORTASI tüketiciler: gözlemcinin yakaladığı `ratio_finalize` sonrası
            # çerçevelerden beslenir (legacy'deki kanca noktasının BİREBİR aynısı). İkisi de
            # saf okuyucu ve RNG'yi seed'siz tüketmiyor (relations.py:267 `random_state=42`)
            # → çağrı sırasının kaymış olması bit-birebirliği bozmaz.
            from src.services.segment_carrier import build_raw_test_carrier
            _seg_df, segment_meta = build_raw_test_carrier(
                _mid_test, state.get("db_intelligence", {}) or {},
                exclude=disallowed | leakage_exclude | {target_name},
            )
            from src.services.relations import compute_relations
            relations_report = compute_relations(
                _mid_train, y_train, target_name, generated_from="train_only",
            )

            # ── KATMAN 1 FAZ A: artifact'i diske yaz (FAIL-OPEN) ──
            # İLKE: pipeline guard'ları fail-CLOSED (koşuyu durdurur); artifact PERSISTENCE fail-OPEN.
            # Serialize patlarsa kullanıcı raporunu KAYBETMEMELİ → logla, artifact_id=None, koşu SÜRER.
            # Faz A doldurur: manifest/param_store/raw_input_signature (finalize'da) + training_reference
            # .per_feature (raw TRAIN partisyonundan, imza.required kolonları). model_ref BOŞ (Faz B),
            # baseline_metrics None (Faz C). phase="features".
            try:
                from src.services.inference.references import build_training_reference
                from src.services.inference.serialize import save_artifact
                _req = (_artifact.raw_input_signature or {}).get("required") or []
                _raw_train = _df_raw.loc[X_train.index] if _df_raw is not None else None
                if _raw_train is not None:
                    # TİP-KARARI TEK KAYNAK: raw_input_signature (dtype değil) → ayrık int'ler
                    # (products_number/tenure/binary) categorical kurulur, dejenere quantile+sahte PSI biter.
                    _artifact.training_reference = build_training_reference(
                        _raw_train, columns=_req, baseline_metrics=None,
                        signature=_artifact.raw_input_signature)
                # İŞ ANAHTARI (Katman 2 gerçekleşen-perf v2 erken-şartı): entity kolonu bildirildiyse
                # skorlama onu predictions.business_key'e yazar. entity_col genelde model girdisi DEĞİL
                # (dropped id) → v1'de çoğunlukla NULL; ama kolon+yazım ŞİMDİ kurulur (geçmiş düzeltilemez).
                _artifact.business_key_column = entity_col or None
                _artifact.phase = "features"
                _dep = bool((_artifact.supported_scope or {}).get("deployable"))
                _artifact_id = save_artifact(_artifact)
                _artifact_block = {"artifact_id": _artifact_id, "deployable": _dep,
                                   "phase": "features", "reason": None}
                logger.info(f"[feature_eng] Faz A artifact yazıldı: id={_artifact_id} "
                            f"deployable={_dep} phase=features")
            except Exception as _e:
                logger.error(f"[feature_eng] Faz A artifact YAZILAMADI (fail-open, koşu sürüyor): {_e}")
                _artifact_id = None
                _artifact_block = {"artifact_id": None, "deployable": False, "phase": None,
                                   "reason": f"serialize_failed: {type(_e).__name__}: {str(_e)[:200]}"}
        else:
            X_train, X_test = _finalize_ratio_features_tt(X_train, X_test, ratio_feature_cols)
            _dropped_ratio = set(ratio_feature_cols) - set(X_train.columns)
            ratio_feature_cols = [c for c in ratio_feature_cols if c in X_train.columns]
            ts_protected_cols -= _dropped_ratio   # finalize'ta düşen oranı korumadan çıkar

            # ── Fairness segment taşıyıcısı — HAM hâlde, encode/scale'den ÖNCE ──
            # Buradan sonraki her transform kolonları encode/scale ediyor; ham
            # demografik değer bir daha erişilebilir olmayacak. Snapshot X_test ile
            # POZİSYONEL hizalı: aşağıdaki transform'lar satır eklemez/çıkarmaz.
            from src.services.segment_carrier import build_raw_test_carrier
            _seg_df, segment_meta = build_raw_test_carrier(
                X_test, state.get("db_intelligence", {}) or {},
                exclude=disallowed | leakage_exclude | {target_name},
            )

            # ── Veri İlişkileri — AYNI KANCA, aynı sebep ──
            # Hesap GERÇEK train partisyonundan geçer → "train_only" provenance'ı bir
            # iddia değil, yapısal olgu. Rapor anında split'i yeniden üretmek zaman
            # serisi/agrege planlarda sessizce farklı satırlar verirdi (bkz. relations.py).
            from src.services.relations import compute_relations
            relations_report = compute_relations(
                X_train, y_train, target_name, generated_from="train_only",
            )

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

        # Katman 1 T1 — GÖLGE KARŞILAŞTIRMASI. Üretim çıktısı YUKARIDA tamamlandı; buradan
        # sonrası salt gözlem. `run_shadow_comparison` ASLA fırlatmaz (kendi içinde yutar) →
        # gölge koşu üretim koşusunu bozamaz. Sonuç feature_report'a düşer (kanıt toplama).
        if _shadow_x_train is not None:
            from src.services.inference.shadow import run_shadow_comparison
            shadow_report = run_shadow_comparison(
                _shadow_x_train, y_train, pipeline_ctx, X_train)

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
        # Ratio etiket metadata'sı {kolon: (pay, payda)} — humanize_feature string-parse
        # belirsizliği yerine buradan kesin etiket kurar (Faz 1 #4). X_train'e GÖRE
        # FİLTRELENMEZ: bir lookup tablosudur, "hayatta kalan" listesi değil. relations
        # selection'dan ÖNCE hesaplandığı için orada düşmüş ratio'lar da geçer; hepsini
        # tutmak her bölümde AYNI etiketi garanti eder (aksi halde biri humanize biri ham).
        "ratio_feature_meta":       {k: list(v) for k, v in ratio_feature_meta.items()},
        # Katman 1 (T0) — FILTRELENMEMIS kayitlar (X_train'e gore FILTRELENMEZ; ustteki
        # ratio_feature_meta ile ayni sozlesme). Orkestratorun donmus secim kaynagi bunlar;
        # `auto_features`/`derived_flags` hayatta-kalan listeleridir, o amac icin KULLANILMAZ.
        "missing_indicator_meta":   dict(missing_indicator_meta),
        "derived_flag_meta":        dict(derived_flag_meta),
        # Katman 1 BEYAN YUZEYI — orkestratore verilecek ctx (beyan edilmeyen olgu →
        # artifact deployable=False; sessizce "destekleniyor" olamaz).
        "pipeline_ctx":             pipeline_ctx,
        # T1 golge kosu sonucu (yalniz FEATURE_PIPELINE_MODE=shadow'da dolar; legacy'de {}).
        "shadow_report":            shadow_report,
        "auto_features":            [c for c in auto_feature_cols if c in X_train.columns],
        "timeseries_lag":           ts_lag_info,
        "split_strategy":           split_strategy,
        "split_imbalanced_entities": split_imbalanced,
        # Training cap örnekleme bilgisi (sessiz değil → rapora yansır)
        "sampling":                 sampling_meta,
        # Normalizasyon çarpışmaları (farklı-değerli kolonlar aynı ada düşüyor) →
        # deterministik veri-kalitesi sinyali; Faz 2 risk/WARN listesi tüketir.
        "normalization_collisions": normalization_collisions,
    }

    # Ağır split dizilerini diske yaz; state'te yalnızca dosya yolu taşı
    # (checkpoint payload şişmesini önler). Yazma başarısız olursa inline'a düş.
    from src.utils.dataset_store import dump_split, dump_raw_test
    split_paths = dump_split(X_train, X_test, y_train, y_test, tag=f"p{state.get('project_id', '')}")
    out: dict = {"feature_report": feature_report, "error": None, **split_paths}

    # Ham test taşıyıcısını diske yaz (state'te yalnızca yol — checkpoint şişmesin).
    # Yazma başarısızsa taşıyıcı YOK sayılır ama sebep meta'da kalır → fairness ve
    # hata analizi "hesaplanamadı çünkü ..." diyebilir, sessiz 0 dönmez.
    raw_path = None
    if _seg_df is not None and len(_seg_df):
        raw_path = dump_raw_test(_seg_df, tag=f"p{state.get('project_id', '')}")
        if raw_path is None:
            segment_meta["reason"] = {"code": "seg_write_failed"}
    out["raw_test_path"] = raw_path or ""
    out["raw_test_meta"] = segment_meta
    out["relations_report"] = relations_report
    # Katman 1 wiring: artifact yolu (id) + durum bloğu (deployable/phase/reason) state'e.
    # Ağır Artifact diske gitti; state YALNIZ string id + küçük blok taşır (checkpoint'e sığar).
    out["inference_artifact_id"] = _artifact_id
    out["inference_artifact"] = _artifact_block
    if not split_paths.get("X_train_path"):   # disk yazımı kısmen/tamamen başarısız → inline fallback
        out["X_train"] = X_train.to_dict(orient="records")
        out["X_test"]  = X_test.to_dict(orient="records")
    if not split_paths.get("y_train_path"):
        out["y_train"] = y_train.tolist() if hasattr(y_train, "tolist") else list(y_train)
        out["y_test"]  = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
    return out