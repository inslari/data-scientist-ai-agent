"""
Analysis Engine Service
=======================
Derinlemesine veri analizi (EDA) motoru.

Mevcut DuckDBEngine.profile_table() çıktısını ALIR ve üstüne ekler:
  - correlation_matrix : sayısal sütunlar arası Pearson korelasyon (heatmap için)
  - histograms         : her sayısal sütun için bin dağılımı (histogram için)
  - target_relations   : hedef belirtilmişse, hedefle en ilişkili sütunlar
  - clustering         : PCA 2D + KMeans (scatter için)

Model EĞİTMEZ. Veriyi anlatır. Hiçbir şeyi silmez/değiştirmez.
Çıktı: frontend'in recharts ile çizeceği JSON.

Dataset-agnostik: hiçbir kolon adı hardcode edilmez.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════
#  Yardımcı: güvenli float (NaN/inf → None)
# ═══════════════════════════════════════════════════════════

def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _get_numeric_cols(profile: dict) -> list[str]:
    """profile dict'inden sayısal kolon listesini çıkarır."""
    summary = profile.get("summary", {})
    return summary.get("numeric_columns", [])


# ═══════════════════════════════════════════════════════════
#  1. KORELASYON MATRİSİ (heatmap için)
# ═══════════════════════════════════════════════════════════

def compute_correlation(df: pd.DataFrame, profile: dict, target_col: Optional[str] = None) -> Optional[dict]:
    """
    Sayısal sütunlar arası Pearson korelasyon matrisi +
    kategorik sütunlar dahil birleşik matris (Cramér's V).

    Returns:
        {
          "correlation_matrix": {          # geriye dönük uyumlu (sadece sayısal)
            "columns": [...], "matrix": [...], "top_pairs": [...]
          },
          "correlation_full": {            # sayısal + kategorik birleşik
            "columns": [...], "matrix": [...], "top_pairs": [...]
          }
        }
    None: 2'den az sütun varsa
    """
    numeric_cols = _get_numeric_cols(profile)
    cols = [c for c in numeric_cols if c in df.columns]

    # --- Pearson (mevcut davranış, geriye dönük uyumlu) ---
    pearson_result = None
    if len(cols) >= 2:
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        corr = sub.corr(method="pearson")

        matrix = []
        for a in cols:
            row = []
            for b in cols:
                v = corr.loc[a, b]
                row.append(_safe_float(v) if _safe_float(v) is not None else 0.0)
            matrix.append(row)

        pairs = []
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if i < j:
                    v = _safe_float(corr.loc[a, b])
                    if v is not None:
                        pairs.append({"a": a, "b": b, "corr": v})
        pairs.sort(key=lambda p: abs(p["corr"]), reverse=True)

        pearson_result = {
            "columns": cols,
            "matrix": matrix,
            "top_pairs": pairs[:10],
        }

    # --- Cramér's V (kategorik-kategorik) + birleşik matris ---
    summary = profile.get("summary", {})
    cat_cols = summary.get("categorical_columns", [])
    cat_cols = [c for c in cat_cols if c in df.columns]

    all_cols = cols + cat_cols
    if len(all_cols) < 2:
        if pearson_result:
            return {"correlation_matrix": pearson_result}
        return None

    # Birleşik matris hesapla
    full_size = len(all_cols)
    full_matrix = [[0.0] * full_size for _ in range(full_size)]

    # Sayısal-sayısal: Pearson'dan kopyala
    if len(cols) >= 2:
        sub_num = df[cols].apply(pd.to_numeric, errors="coerce")
        corr_num = sub_num.corr(method="pearson")
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                v = _safe_float(corr_num.loc[a, b])
                ai, bi = all_cols.index(a), all_cols.index(b)
                full_matrix[ai][bi] = v if v is not None else 0.0

    # Kategorik-kategorik: Cramér's V
    def _cramers_v(x: pd.Series, y: pd.Series) -> Optional[float]:
        """Cramér's V — chi-square tabanlı kategorik ilişki ölçüsü."""
        try:
            from scipy.stats import chi2_contingency
            ct = pd.crosstab(x, y)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                return None
            chi2, _, _, _ = chi2_contingency(ct)
            n = ct.sum().sum()
            min_dim = min(ct.shape[0], ct.shape[1]) - 1
            if min_dim == 0 or n == 0:
                return None
            return float(np.sqrt(chi2 / (n * min_dim)))
        except Exception:
            return None

    for i, a in enumerate(cat_cols):
        ai = all_cols.index(a)
        full_matrix[ai][ai] = 1.0  # diagonal
        for j, b in enumerate(cat_cols):
            if i >= j:
                continue
            bi = all_cols.index(b)
            v = _cramers_v(df[a].astype(str), df[b].astype(str))
            val = _safe_float(v) if v is not None else 0.0
            full_matrix[ai][bi] = val
            full_matrix[bi][ai] = val

    # Sayısal-kategorik: correlation ratio (eta) yaklaşımı
    for num_c in cols:
        ni = all_cols.index(num_c)
        num_series = pd.to_numeric(df[num_c], errors="coerce")
        for cat_c in cat_cols:
            ci = all_cols.index(cat_c)
            try:
                groups = num_series.groupby(df[cat_c].astype(str))
                grand_mean = num_series.mean()
                ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in groups)
                ss_total = ((num_series - grand_mean) ** 2).sum()
                eta = float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0
                val = _safe_float(eta) if not np.isnan(eta) else 0.0
            except Exception:
                val = 0.0
            full_matrix[ni][ci] = val
            full_matrix[ci][ni] = val

    # top_pairs (birleşik)
    full_pairs = []
    for i, a in enumerate(all_cols):
        for j, b in enumerate(all_cols):
            if i < j:
                v = full_matrix[i][j]
                if v and v != 0.0:
                    full_pairs.append({"a": a, "b": b, "corr": v})
    full_pairs.sort(key=lambda p: abs(p["corr"]), reverse=True)

    result = {}
    if pearson_result:
        result["correlation_matrix"] = pearson_result
    result["correlation_full"] = {
        "columns": all_cols,
        "matrix": full_matrix,
        "top_pairs": full_pairs[:15],
    }
    return result


# ═══════════════════════════════════════════════════════════
#  2. HİSTOGRAMLAR (bar/histogram için)
# ═══════════════════════════════════════════════════════════

def compute_histograms(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
    n_bins: int = 20,
) -> Optional[dict]:
    """
    Her sayısal sütun için histogram bin'leri.

    Returns:
        {
          "histograms": {
            "age": {
              "bin_edges": [20, 25, 30, ...],   # bin sınırları
              "counts":    [15, 42, 38, ...],   # her bin'deki sayı
            },
            ...
          }
        }
    """
    numeric_cols = _get_numeric_cols(profile)
    result = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 2:
            continue
        try:
            counts, edges = np.histogram(series, bins=n_bins)
            result[col] = {
                "bin_edges": [_safe_float(e) for e in edges],
                "counts": [int(c) for c in counts],
                "bin_indices": list(range(len(counts))),
            }
        except Exception:
            continue
    if not result:
        return None
    return {"histograms": result}


# ═══════════════════════════════════════════════════════════
#  3. HEDEF İLİŞKİSİ (hedef belirtilmişse)
# ═══════════════════════════════════════════════════════════

def compute_target_relations(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
) -> Optional[dict]:
    """
    Hedefle en ilişkili sayısal sütunlar.
    Hedef sayısalsa Pearson korelasyon, kategorikse her sütunun
    hedef gruplarındaki ortalama farkı.

    Returns:
        {
          "target_relations": {
            "target": "churn",
            "target_type": "categorical",
            "relations": [{"feature": "tenure", "score": 0.42}, ...]  # |score| azalan
          }
        }
    """
    if not target_col or target_col not in df.columns:
        return None

    numeric_cols = _get_numeric_cols(profile)
    feats = [c for c in numeric_cols if c in df.columns and c != target_col]
    if not feats:
        return None

    target = df[target_col]
    relations = []

    # Hedef sayısal mı kategorik mi?
    target_num = pd.to_numeric(target, errors="coerce")
    target_is_numeric = target_num.notna().sum() > 0.8 * len(target)

    for feat in feats:
        fseries = pd.to_numeric(df[feat], errors="coerce")
        try:
            if target_is_numeric:
                # Pearson korelasyon
                score = fseries.corr(target_num)
            else:
                # Kategorik hedef: hedef gruplarında feature ortalaması ne kadar ayrışıyor?
                # (basit ayrım gücü: gruplar arası std / genel std)
                grouped = fseries.groupby(target).mean()
                if len(grouped) >= 2 and fseries.std() > 0:
                    score = grouped.std() / fseries.std()
                else:
                    score = 0.0
            s = _safe_float(score)
            if s is not None:
                relations.append({"feature": feat, "score": s})
        except Exception:
            continue

    relations.sort(key=lambda r: abs(r["score"]), reverse=True)

    return {
        "target_relations": {
            "target": target_col,
            "target_type": "numeric" if target_is_numeric else "categorical",
            "relations": relations,
        }
    }


# ═══════════════════════════════════════════════════════════
#  4. KÜMELEME (PCA 2D + KMeans, scatter için)
# ═══════════════════════════════════════════════════════════

def compute_clustering(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
    max_k: int = 5,
    sample_size: int = 2000,
) -> Optional[dict]:
    """
    Sayısal sütunları PCA ile 2D'ye indirir, KMeans ile kümeler.
    Büyük veride sample alır (scatter okunaklı kalsın + hız).

    Returns:
        {
          "clustering": {
            "points": [{"x": .., "y": .., "cluster": 0}, ...],
            "n_clusters": 3,
            "explained_variance": 0.78,   # PCA 2 bileşenin açıkladığı oran
          }
        }
    None: 2'den az sayısal sütun veya yetersiz veri
    """
    numeric_cols = _get_numeric_cols(profile)
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return None

    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 10:
        return None

    # Büyük veride örnekle
    if len(sub) > sample_size:
        sub = sub.sample(n=sample_size, random_state=42)

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        X = StandardScaler().fit_transform(sub.values)

        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        explained = float(pca.explained_variance_ratio_.sum())

        # En iyi k (silhouette ile, 2..max_k)
        best_k, best_score, best_labels = 2, -1.0, None
        for k in range(2, min(max_k, len(sub) - 1) + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_k, best_score, best_labels = k, score, labels
            except Exception:
                continue

        if best_labels is None:
            return None

        points = [
            {"x": _safe_float(coords[i, 0]), "y": _safe_float(coords[i, 1]),
             "cluster": int(best_labels[i])}
            for i in range(len(coords))
        ]

        return {
            "clustering": {
                "points": points,
                "n_clusters": best_k,
                "silhouette": _safe_float(best_score),
                "explained_variance": round(explained, 4),
            }
        }
    except ImportError:
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
#  5. HEDEF GRUPLARI (box + violin, sınıflandırma için)
# ═══════════════════════════════════════════════════════════

def compute_target_groups(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
    sample_size: int = 5000,
) -> Optional[dict]:
    """
    Her sayısal feature'ın hedef sınıflarına göre dağılımı.
    Box istatistikleri (min, q25, median, q75, max, outliers) +
    violin yoğunluk eğrisi (~25 nokta).

    Sadece KATEGORİK hedef için çalışır. Sayısal hedefte None döner.

    Returns:
        {
          "target_groups": {
            "target_col": "Churn",
            "groups": ["0", "1"],
            "features": {
              "tenure": {
                "0": {"min": .., "q25": .., "median": .., "q75": .., "max": ..,
                      "outliers": [...], "density": [{"x": .., "y": ..}, ...]},
                "1": {...}
              }
            }
          }
        }
    """
    if not target_col or target_col not in df.columns:
        return None

    # Hedef tipi tespiti (compute_target_relations deseni)
    target = df[target_col]
    target_num = pd.to_numeric(target, errors="coerce")
    target_is_numeric = target_num.notna().sum() > 0.8 * len(target)

    if target_is_numeric:
        return None  # Regresyon hedefinde uygulanamaz

    numeric_cols = _get_numeric_cols(profile)
    feats = [c for c in numeric_cols if c in df.columns and c != target_col]
    if not feats:
        return None

    # Büyük veride örnekle (compute_clustering deseni, sınır 5000)
    sub = df[[target_col] + feats].copy()
    if len(sub) > sample_size:
        sub = sub.sample(n=sample_size, random_state=42)

    groups = sorted([str(g) for g in target.dropna().unique()])
    features_data: dict[str, dict] = {}

    for feat in feats:
        feat_series = pd.to_numeric(sub[feat], errors="coerce")
        feat_data: dict[str, dict] = {}

        for grp in groups:
            mask = sub[target_col].astype(str) == grp
            vals = feat_series[mask].dropna()

            if len(vals) < 2:
                continue

            q25 = float(vals.quantile(0.25))
            q75 = float(vals.quantile(0.75))
            iqr = q75 - q25
            lower_fence = q25 - 1.5 * iqr
            upper_fence = q75 + 1.5 * iqr

            outlier_vals = vals[(vals < lower_fence) | (vals > upper_fence)]
            # Outlier noktalarını sınırla (çok fazla olmasın)
            outliers = [_safe_float(v) for v in outlier_vals.head(50)]
            outliers = [o for o in outliers if o is not None]

            # Violin density: np.histogram density=True, ~25 bin
            try:
                n_density_bins = 25
                counts, edges = np.histogram(vals, bins=n_density_bins, density=True)
                density = []
                for i in range(len(counts)):
                    mid_x = (edges[i] + edges[i + 1]) / 2
                    density.append({
                        "x": _safe_float(mid_x),
                        "y": _safe_float(counts[i]),
                    })
            except Exception:
                density = []

            feat_data[grp] = {
                "min": _safe_float(vals.min()),
                "q25": _safe_float(q25),
                "median": _safe_float(vals.median()),
                "q75": _safe_float(q75),
                "max": _safe_float(vals.max()),
                "outliers": outliers,
                "density": density,
            }

        if feat_data:
            features_data[feat] = feat_data

    if not features_data:
        return None

    return {
        "target_groups": {
            "target_col": target_col,
            "groups": groups,
            "features": features_data,
        }
    }


# ═══════════════════════════════════════════════════════════
#  6. PAIRWISE SCATTER (sayısal çiftler arası scatter)
# ═══════════════════════════════════════════════════════════

def compute_pairwise(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
    max_features: int = 5,
    sample_size: int = 1500,
) -> Optional[dict]:
    """
    Sayısal feature çiftleri için scatter noktaları.
    Frontend küçük scatter ızgarası çizecek.

    - max_features: kalabalık olmasın diye ilk N sayısal feature
    - sample_size: her çift için örneklenmiş nokta sayısı
    - Hedef kategorikse noktaya sınıf etiketi eklenir

    Returns:
        {
          "pairwise": {
            "features": ["tenure", "MonthlyCharges", ...],
            "points": [
              {"x_feat": "tenure", "y_feat": "MonthlyCharges",
               "x": 12, "y": 70.3, "target": "1"},
              ...
            ]
          }
        }
    """
    numeric_cols = _get_numeric_cols(profile)
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return None

    # İlk max_features ile sınırla
    cols = cols[:max_features]

    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 10:
        return None

    # Büyük veride örnekle (compute_clustering deseni)
    if len(sub) > sample_size:
        sub = sub.sample(n=sample_size, random_state=42)

    # Hedef kategorik mi?
    has_target = False
    target_values = None
    if target_col and target_col in df.columns:
        target_series = df[target_col]
        target_num = pd.to_numeric(target_series, errors="coerce")
        target_is_numeric = target_num.notna().sum() > 0.8 * len(target_series)
        if not target_is_numeric:
            has_target = True
            target_values = target_series.loc[sub.index].astype(str)

    points = []
    for i, col_a in enumerate(cols):
        for j, col_b in enumerate(cols):
            if i >= j:
                continue  # her çifti bir kez, diagonal hariç
            for idx in sub.index:
                pt = {
                    "x_feat": col_a,
                    "y_feat": col_b,
                    "x": _safe_float(sub.loc[idx, col_a]),
                    "y": _safe_float(sub.loc[idx, col_b]),
                }
                if has_target and target_values is not None:
                    pt["target"] = target_values.loc[idx]
                points.append(pt)

    if not points:
        return None

    return {
        "pairwise": {
            "features": cols,
            "points": points,
        }
    }


# ═══════════════════════════════════════════════════════════
#  7. KDE (yoğunluk eğrisi, ~40 nokta)
# ═══════════════════════════════════════════════════════════

def compute_kde(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
    n_bins: int = 40,
) -> Optional[dict]:
    """
    Her sayısal feature için pürüzsüz yoğunluk eğrisi.
    np.histogram(density=True) ile ~40 nokta.

    Returns:
        {
          "kde": {
            "tenure": [{"x": .., "y": ..}, ...],
            ...
          }
        }
    """
    numeric_cols = _get_numeric_cols(profile)
    result = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 2:
            continue
        try:
            counts, edges = np.histogram(series, bins=n_bins, density=True)
            curve = []
            for i in range(len(counts)):
                mid_x = (edges[i] + edges[i + 1]) / 2
                curve.append({
                    "x": _safe_float(mid_x),
                    "y": _safe_float(counts[i]),
                })
            result[col] = curve
        except Exception:
            continue

    if not result:
        return None

    return {"kde": result}


# ═══════════════════════════════════════════════════════════
#  8. EKSİK VERİ DESENİ (missing pattern ısı-matrisi)
# ═══════════════════════════════════════════════════════════

def compute_missing_pattern(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
    max_rows: int = 200,
) -> Optional[dict]:
    """
    Eksik veri eş-oluşum deseni. Satırlardan örnekle,
    her sütun için eksik=1 / dolu=0 matrisi üret.
    Frontend bunu ısı-matrisi gibi çizecek.

    Sadece en az bir eksiği olan sütunları dahil eder.
    Hiç eksik yoksa None döner.

    Returns:
        {
          "missing_pattern": {
            "columns": ["TotalCharges", "MonthlyCharges", ...],
            "rows": [[0, 1, 0, ...], [1, 0, 0, ...], ...]
          }
        }
    """
    # Sadece en az 1 eksik değeri olan sütunları bul
    missing_cols = [c for c in df.columns if df[c].isna().sum() > 0]
    if not missing_cols:
        return None

    sub = df[missing_cols]

    # Sadece en az bir eksik değeri olan satırları al, sonra örnekle
    has_any_missing = sub.isna().any(axis=1)
    missing_rows = sub[has_any_missing]

    if len(missing_rows) == 0:
        return None

    if len(missing_rows) > max_rows:
        missing_rows = missing_rows.sample(n=max_rows, random_state=42)

    # 1=eksik, 0=dolu matrisi
    matrix = missing_rows.isna().astype(int).values.tolist()

    return {
        "missing_pattern": {
            "columns": missing_cols,
            "rows": matrix,
        }
    }


# ═══════════════════════════════════════════════════════════
#  9. ECDF (kümülatif dağılım, ~50 nokta)
# ═══════════════════════════════════════════════════════════

def compute_ecdf(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
    n_points: int = 50,
) -> Optional[dict]:
    """
    Her sayısal feature için kümülatif dağılım fonksiyonu.
    Sıralı eşik değerleri + o değerin altındaki oran (~50 nokta).

    Returns:
        {
          "ecdf": {
            "tenure": [{"x": .., "p": ..}, ...],
            ...
          }
        }
    """
    numeric_cols = _get_numeric_cols(profile)
    result = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna().sort_values()
        if len(series) < 2:
            continue

        try:
            # n_points eşit aralıklı quantile noktası
            n = len(series)
            if n <= n_points:
                indices = list(range(n))
            else:
                indices = [int(i * (n - 1) / (n_points - 1)) for i in range(n_points)]

            curve = []
            for idx in indices:
                x_val = series.iloc[idx]
                p_val = (idx + 1) / n  # kümülatif oran
                curve.append({
                    "x": _safe_float(x_val),
                    "p": _safe_float(p_val),
                })
            result[col] = curve
        except Exception:
            continue

    if not result:
        return None

    return {"ecdf": result}


# ═══════════════════════════════════════════════════════════
#  ANALİZ MODÜL KAYIT DEFTERİ (Registry)
# ═══════════════════════════════════════════════════════════

# E-ticaret sektör modülleri (analysis_type="ecommerce" ile eklenir).
from src.services.analysis_ecommerce import compute_rfm, compute_basket_affinity

# Sağlık sektör modülleri (analysis_type="healthcare" ile eklenir).
from src.services.analysis_healthcare import (
    compute_survival_summary,
    compute_readmission_rate,
)

# Perakende sektör modülleri (analysis_type="retail" ile eklenir).
from src.services.analysis_retail import (
    compute_demand_trend,
    compute_price_elasticity,
)

# İK sektör modülleri (analysis_type="hr" ile eklenir).
from src.services.analysis_hr import (
    compute_attrition_overview,
    compute_compensation_gaps,
)

# Pazarlama sektör modülleri (analysis_type="marketing" ile eklenir).
from src.services.analysis_marketing import (
    compute_channel_performance,
    compute_campaign_roi,
)

from src.core.logging_config import get_logger
logger = get_logger(__name__)

# Analiz modülleri. "general" her veride çalışır; sektör grupları "general"in
# ÜSTÜNE eklenir (bkz. run_deep_analysis). Yeni sektör eklemek için: compute_*
# fonksiyonlarını yaz + buraya yeni bir grup anahtarı ekle, pack.analysis_type
# ile eşleştir.
ANALYSIS_MODULES: dict[str, list] = {
    "general": [
        compute_correlation,
        compute_histograms,
        compute_target_relations,
        compute_clustering,
        compute_target_groups,
        compute_pairwise,
        compute_kde,
        compute_missing_pattern,
        compute_ecdf,
    ],
    "fintech": [],
    "ecommerce": [
        compute_rfm,
        compute_basket_affinity,
    ],
    "healthcare": [
        compute_survival_summary,
        compute_readmission_rate,
    ],
    "retail": [
        compute_demand_trend,
        compute_price_elasticity,
    ],
    "hr": [
        compute_attrition_overview,
        compute_compensation_gaps,
    ],
    "marketing": [
        compute_channel_performance,
        compute_campaign_roi,
    ],
}


# ═══════════════════════════════════════════════════════════
#  ANA FONKSİYON
# ═══════════════════════════════════════════════════════════

def run_deep_analysis(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
    analysis_type: str = "general",
) -> dict:
    """
    profile_table çıktısını ALIR, üstüne analiz katmanları ekler.

    Her compute_* fonksiyonu aynı imzayı kullanır:
        compute_x(df, profile, target_col) -> dict | None

    Her fonksiyon kendi key'ini içeren bir dict döndürür
    (örn. {"correlation_matrix": {...}}).
    None dönerse o analiz atlanır.

    Args:
        df: ham veri (parquet'ten okunmuş)
        profile: DuckDBEngine.profile_table() çıktısı
        target_col: hedef sütun (opsiyonel)
        analysis_type: hangi sektör modül grubu eklensin ("general", "fintech",
            "ecommerce", …). "general" her zaman çalışır; sektör grubu onun
            ÜSTÜNE eklenir (domain pack'in analysis_type'ından gelir).

    Returns:
        profile + correlation_matrix + histograms + target_relations + clustering
        + (varsa) sektöre özel anahtarlar (örn. rfm, basket_affinity)
    """
    # Mevcut profile'ı koru, üstüne ekle
    analysis = dict(profile)

    # Önce "general" (her veride), sonra varsa sektör grubu — tekrar çalıştırma.
    groups = ["general"]
    if analysis_type and analysis_type not in ("general", "none"):
        groups.append(analysis_type)

    for group in groups:
        for fn in ANALYSIS_MODULES.get(group, []):
            try:
                result = fn(df, profile, target_col)
                if result and isinstance(result, dict):
                    analysis.update(result)
            except Exception as e:
                logger.error(f"[analysis_engine] {fn.__name__} hatası: {e}")
                continue

    return analysis


# ═══════════════════════════════════════════════════════════
#  PLAN HINTS — analist → feature_plan_node kanalı
# ═══════════════════════════════════════════════════════════
# Bu bölüm run_deep_analysis'ten AYRIDIR ve ANALYSIS_MODULES'e KAYITLI DEĞİLDİR.
# Buradaki çıktı LLM feature planına akan TEK analist kanalıdır, dolayısıyla
# leakage sınırı burada çizilir:
#
#   extract_plan_hints() HEDEFİ HİÇ GÖRMEZ — imzasında target parametresi YOKTUR.
#   Hedef-farkındalıklı bir istatistik (target korelasyonu, grup ayrışması, feature
#   önem sırası) üretmesi bu yüzden disipline değil, TASARIMA bağlı olarak imkânsızdır.
#   Hedef kolonun kendisi çağıran tarafından exclude_columns ile dışarıda bırakılır,
#   böylece redundant_pairs bir (feature, target) çifti dahi üretemez.
#
# Payload YALNIZCA kolon adı taşır — tek bir sayı bile taşımaz. Sebep: "şu kolon
# target ile 0.83 korelasyonlu" bilgisi karar düzeyine indirgense bile ("target ile
# en ilişkili 5 kolon") test etiketleriyle feature seçimi yapmak demektir; ROC'u
# şişiren budur, sayının kendisi değil. Hedef-agnostik yapısal bulgular (sabit kolon,
# çarpıklık, kardinalite, X-X redundancy) y'yi hiç kullanmadıkları için bu riski
# taşımaz ve karar düzeyinde akabilir.
#
# DS ajanı bu ipuçlarını yalnızca DROP/TRANSFORM kararında kullanır; her sayı aşağıda
# X_train'den yeniden hesaplanır (_select_features_tt, _select_univariate_tt).
# Değişmezler tests/test_analysis_hints_leakage.py ile sabitlenmiştir.

PLAN_HINT_NEAR_CONSTANT_SHARE   = 0.99   # baskın değer payı → "tek değerli sayılır"
PLAN_HINT_HIGH_CARD_UNIQUE_RATE = 0.90   # unique/satır → one-hot anlamsız
PLAN_HINT_HIGH_CARD_MIN_UNIQUE  = 50     # mutlak eşik (profile'daki "high" ile aynı)
PLAN_HINT_HIGH_MISSING_RATE     = 0.40   # null oranı
PLAN_HINT_SKEW                  = 1.00   # |skew(col)| → log adayı (df varken; gerçek çarpıklık)
PLAN_HINT_SKEW_RATIO            = 0.20   # |mean - median| / std → yalnızca df YOKKEN vekil ölçü
PLAN_HINT_OUTLIER_RATE          = 0.05   # IQR dışı satır oranı
PLAN_HINT_REDUNDANT_CORR        = 0.95   # |corr(a,b)| → çiftten biri gereksiz
PLAN_HINT_MAX_ROWS              = 50_000 # X-X korelasyonu için örneklem tavanı
PLAN_HINT_MAX_ITEMS             = 25     # prompt şişmesin (liste başına)

# Bu eşikler TEK KAYNAK: data_executor._generate_recommendations (rapor yolu) da
# buradan okur. Aynı olguya iki farklı eşik (null %30 vs %40, outlier %1 vs %5)
# uygulanması → aynı kolon bir yerde "sorunlu" diğerinde "temiz" görünürdü.

_DATE_SUFFIXES = ("_date", "_at", "_time")
_DATE_EXACT    = {"created_at", "updated_at", "timestamp", "birth_date", "order_date"}


def is_date_like(col: str, dtype: str = "") -> bool:
    """Tarih/zaman kolonu mu — dtype veya isim sezgiseli. Hedef-kör, satır görmez."""
    d = (dtype or "").upper()
    c = (col or "").lower()
    return ("DATE" in d or "TIME" in d
            or c in _DATE_EXACT
            or any(c.endswith(s) for s in _DATE_SUFFIXES))


def extract_plan_hints(
    df: Optional[pd.DataFrame],
    profile: dict,
    exclude_columns: Optional[set[str]] = None,
) -> dict:
    """
    feature_plan_node'a akacak HEDEF-KÖR, karar düzeyinde yapısal ipuçlarını çıkarır.

    Bilinçli olarak target parametresi ALMAZ (bkz. yukarıdaki bölüm notu).

    Args:
        df: ham veri. VARSA tüm istatistikler BUNDAN hesaplanır — kilitli-holdout
            ölçüm modunda buraya yalnızca train satırları gelir, dolayısıyla ipuçları
            gerçekten train-only olur. None/boş ise istatistikler profile'dan alınır
            (tüm veri) ve redundant_pairs boş kalır.
        profile: DuckDBEngine.profile_table() çıktısı. Kolon TİPLERİ (sayısal/kategorik)
            her hâlükârda buradan okunur — tip yapısaldır, satır bilgisi taşımaz.
        exclude_columns: hiç bakılmayacak kolonlar — çağıran hedef kolonu (ve hedef
            adaylarını) buraya koyar. Bu kolonlar hiçbir listede ve hiçbir çiftte
            görünmez.

    Returns:
        Yalnızca kolon adı taşıyan (float/int İÇERMEYEN) dict:
        {"generated_from", "target_aware": False, "constant_columns", "near_constant",
         "high_cardinality", "high_missing", "skewed", "outlier_heavy", "redundant_pairs"}
    """
    excl = {c for c in (exclude_columns or set()) if c}
    summary = profile.get("summary", {}) or {}
    col_profiles: dict[str, dict] = profile.get("columns", {}) or {}
    numeric_cols = [c for c in summary.get("numeric_columns", []) if c not in excl]
    cat_cols     = [c for c in summary.get("categorical_columns", []) if c not in excl]

    # Tarih/zaman kolonları: dtype'ı DATE/TIMESTAMP olanlar + isim sezgiseliyle
    # yakalananlar (ör. VARCHAR tutulan 'transaction_time'). Plan yolunda gerçekten
    # eksik olan tek sinyal buydu — LLM'in derived_features'ta karşılığı var
    # (yıl/ay/gün/saat parçaları). Hedef-kör, sıfır sayı.
    date_like = sorted({
        c for c in col_profiles
        if c not in excl and is_date_like(c, (col_profiles.get(c) or {}).get("dtype", ""))
    })

    sample = _hint_sample(df)
    stats  = _column_stats(sample, profile, numeric_cols + cat_cols)

    constant: list[str]      = []
    near_constant: list[str] = []
    high_card: list[str]     = []
    high_missing: list[str]  = []
    skewed: list[str]        = []
    outlier_heavy: list[str] = []

    for col in numeric_cols + cat_cols:
        st = stats.get(col)
        if col in excl or not st:
            continue

        # Sabit / tek değerli — hiç sinyal yok
        if st["unique_count"] <= 1:
            constant.append(col)
            continue

        # Neredeyse sabit: tek bir değer satırların ≥%99'unu kaplıyor
        if st["top_share"] >= PLAN_HINT_NEAR_CONSTANT_SHARE:
            near_constant.append(col)

        if st["null_rate"] >= PLAN_HINT_HIGH_MISSING_RATE:
            high_missing.append(col)

        # Tarih kolonları kardinalite listesine GİRMEZ: metin tutulan bir timestamp
        # ('transaction_time') hem "yüksek kardinaliteli → at" hem "tarih → parçala"
        # diye görünürdü; LLM'e çelişkili iki emir gitmiş olurdu. Doğru olan parçalamak.
        if col in cat_cols and col not in date_like:
            if (st["unique_rate"] >= PLAN_HINT_HIGH_CARD_UNIQUE_RATE
                    or st["unique_count"] > PLAN_HINT_HIGH_CARD_MIN_UNIQUE):
                high_card.append(col)

        if col in numeric_cols:
            # Çarpıklık yalnızca SÜREKLİ kolonlar için anlamlı. Dengesiz bir 0/1 bayrağının
            # skew'i doğal olarak >1 çıkar ama log(bayrak) saçmadır — binary/ikili kolonları
            # ele, yoksa LLM'e anlamsız türetme önerisi gitmiş olur.
            if (st["skew"] is not None and abs(st["skew"]) >= PLAN_HINT_SKEW
                    and st["unique_count"] > 2):
                skewed.append(col)
            if st["outlier_rate"] >= PLAN_HINT_OUTLIER_RATE:
                outlier_heavy.append(col)

    redundant_pairs = _redundant_pairs(sample, numeric_cols)

    def _cap(items: list) -> list:
        return sorted(items)[:PLAN_HINT_MAX_ITEMS] if items and isinstance(items[0], str) else items[:PLAN_HINT_MAX_ITEMS]

    return {
        # Denetlenebilirlik: ipuçları hangi satırlardan üretildi. "full_data" =
        # tüm veri (hedef-agnostik olduğu için ROC'u şişirmemesi beklenir),
        # "train_only" = ölçüm modunda yeniden üretilmiş train partisyonu.
        "generated_from":   "full_data",
        "target_aware":     False,     # değişmez — testle sabitlenmiştir
        "constant_columns": _cap(constant),
        "near_constant":    _cap(near_constant),
        "high_cardinality": _cap(high_card),
        "high_missing":     _cap(high_missing),
        "skewed":           _cap(skewed),
        "outlier_heavy":    _cap(outlier_heavy),
        "redundant_pairs":  _cap(redundant_pairs),
        "date_like":        _cap(date_like),
    }


def _hint_sample(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Deterministik örneklem — ipuçları eşik kararı olduğu için tam tarama gereksiz."""
    if df is None or df.empty:
        return None
    if len(df) > PLAN_HINT_MAX_ROWS:
        return df.sample(n=PLAN_HINT_MAX_ROWS, random_state=42)
    return df


def _column_stats(
    sample: Optional[pd.DataFrame], profile: dict, columns: list[str]
) -> dict[str, dict]:
    """
    Eşik kararlarını besleyen kolon istatistikleri. Bu sayılar YALNIZCA İÇERİDE kalır;
    dışarı çıkan payload sadece kolon adı taşır.

    sample verilmişse istatistikler ONDAN hesaplanır. Ölçüm modunda (kilitli holdout)
    sample = train partisyonu olduğu için ipuçları gerçekten train-only olur —
    profiling_report tüm veriden geldiği hâlde test satırları hesaba girmez.
    sample yoksa profile'a düşülür (tüm veri).
    """
    col_profiles: dict[str, dict] = profile.get("columns", {}) or {}
    total_rows = (profile.get("shape", {}) or {}).get("rows", 0) or 0
    stats: dict[str, dict] = {}

    for col in columns:
        cp = col_profiles.get(col) or {}
        if cp.get("error"):
            continue

        if sample is not None and col in sample.columns:
            s = sample[col]
            n = max(len(s), 1)
            uniq = int(s.nunique(dropna=True))
            counts = s.value_counts(dropna=True)
            num = pd.to_numeric(s, errors="coerce")
            stats[col] = {
                "unique_count": uniq,
                "unique_rate":  uniq / n,
                "null_rate":    float(s.isna().mean()),
                "top_share":    float(counts.iloc[0] / n) if len(counts) else 0.0,
                "skew":         _safe_float(num.skew()) if num.notna().any() else None,
                "outlier_rate": _outlier_rate(num),
            }
            continue

        # Fallback: profiling_report (tüm veri). |ortalama − medyan| / std yalnızca bir
        # ÇARPIKLIK VEKİLİ — gerçek skew() kadar iyi değil (çift-tepeli kolonu çarpık
        # sanabilir), o yüzden sadece df yokken kullanılır.
        top_values = cp.get("top_values") or {}
        uniq = int(cp.get("unique_count") or 0)
        mean, median, std = cp.get("mean"), cp.get("median"), cp.get("std")
        proxy_skew = None
        if (uniq > 2 and all(isinstance(v, (int, float)) for v in (mean, median, std))
                and std and std > 0):
            ratio = abs(float(mean) - float(median)) / float(std)
            # Vekili gerçek eşikle kıyaslanabilir ölçeğe taşı (Pearson: skew ≈ 3·oran).
            proxy_skew = PLAN_HINT_SKEW if ratio >= PLAN_HINT_SKEW_RATIO else 0.0
        stats[col] = {
            "unique_count": uniq,
            "unique_rate":  float(cp.get("unique_rate") or 0.0),
            "null_rate":    float(cp.get("null_rate") or 0.0),
            "top_share":    (max(top_values.values()) / total_rows) if (top_values and total_rows) else 0.0,
            "skew":         proxy_skew,
            "outlier_rate": float(cp.get("has_outliers") or 0.0),
        }

    return stats


def _outlier_rate(num: pd.Series) -> float:
    """IQR dışı satır oranı (profile'daki has_outliers ile aynı tanım)."""
    try:
        q1, q3 = float(num.quantile(0.25)), float(num.quantile(0.75))
    except Exception:  # pragma: no cover - defensive
        return 0.0
    iqr = q3 - q1
    if iqr <= 0:
        return 0.0
    return float(((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).mean())


def _redundant_pairs(sample: Optional[pd.DataFrame], numeric_cols: list[str]) -> list[list[str]]:
    """
    |corr(a, b)| ≥ eşik olan sayısal kolon ÇİFTLERİ (X-X; hedef zaten dışlanmış).

    Katsayı döndürmez — yalnızca çift. "İkisinden biri gereksiz" kararı için yeterli;
    hangisinin tutulacağını feature_builder X_train üzerinden yeniden hesaplar.
    """
    if sample is None:
        return []
    cols = [c for c in numeric_cols if c in sample.columns]
    if len(cols) < 2:
        return []

    try:
        corr = sample[cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[analysis_engine] redundant_pairs hesaplanamadı: {e}")
        return []

    pairs: list[list[str]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = _safe_float(corr.loc[a, b])
            if v is not None and abs(v) >= PLAN_HINT_REDUNDANT_CORR:
                pairs.append([a, b])
    return pairs


def format_plan_hints(hints: dict) -> str:
    """
    Hint dict'ini feature_plan prompt'una gömülecek metne çevirir.

    Metin de sayı taşımaz — LLM'e yalnızca kolon adları ve ne yapabileceği gider.
    Boş hint → boş string (prompt'a hiçbir şey eklenmez).
    """
    if not hints:
        return ""

    sections = [
        ("constant_columns", "Constant / single-valued columns (no signal — drop them)"),
        ("near_constant",    "Near-constant columns (one value dominates — drop them)"),
        ("high_cardinality", "High-cardinality categorical columns (do NOT one-hot; drop or group them)"),
        ("high_missing",     "Columns with heavy missingness (prefer dropping, or use them only as an indicator)"),
        ("skewed",           "Skewed numeric columns (good candidates for a log(...) derived feature)"),
        ("outlier_heavy",    "Outlier-heavy numeric columns (avoid raw products/ratios built on them)"),
        ("date_like",        "Date/time columns (never feed them raw — derive parts such as year/month/day/hour, or a recency difference)"),
    ]

    lines: list[str] = []
    for key, label in sections:
        cols = hints.get(key) or []
        if cols:
            lines.append(f"- {label}: {', '.join(cols)}")

    pairs = hints.get("redundant_pairs") or []
    if pairs:
        pair_txt = "; ".join(f"({a} ≈ {b})" for a, b in pairs)
        lines.append(f"- Redundant column pairs (near-duplicate information — keep at most one of each pair): {pair_txt}")

    if not lines:
        return ""

    return (
        "\n\n📊 ANALYST HINTS (deterministic structural findings):\n"
        + "\n".join(lines)
        + "\n\nThese hints were computed WITHOUT looking at the target column — they contain NO "
          "information about predictive power. Use them ONLY to drop columns or to choose "
          "transformations (e.g. log for skewed columns). NEVER use them to rank, score or "
          "\"keep\" features by how useful they look; feature selection is decided later on the "
          "training split alone."
    )