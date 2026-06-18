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

# Analiz modülleri. "general" her veride çalışır.
# "fintech" ileride doldurulacak (WoE/IV, target-rate vb.) — şimdilik boş yuva.
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
        analysis_type: registry'den hangi modül grubunu çalıştır ("general", "fintech")

    Returns:
        profile + correlation_matrix + histograms + target_relations + clustering
    """
    # Mevcut profile'ı koru, üstüne ekle
    analysis = dict(profile)

    for fn in ANALYSIS_MODULES.get(analysis_type, []):
        try:
            result = fn(df, profile, target_col)
            if result and isinstance(result, dict):
                analysis.update(result)
        except Exception as e:
            print(f"[analysis_engine] {fn.__name__} hatası: {e}")
            continue

    return analysis