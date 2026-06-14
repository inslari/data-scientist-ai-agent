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
Çıktı: frontend'in Plotly ile çizeceği JSON.

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


# ═══════════════════════════════════════════════════════════
#  1. KORELASYON MATRİSİ (Plotly heatmap için)
# ═══════════════════════════════════════════════════════════

def compute_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> Optional[dict]:
    """
    Sayısal sütunlar arası Pearson korelasyon matrisi.

    Returns (Plotly heatmap formatı):
        {
          "columns": ["age", "tenure", ...],   # x ve y ekseni
          "matrix":  [[1.0, 0.3, ...], ...],   # z değerleri
          "top_pairs": [{"a","b","corr"}, ...] # en güçlü ilişkiler (özet için)
        }
    None: 2'den az sayısal sütun varsa
    """
    cols = [c for c in numeric_cols if c in df.columns]
    if len(cols) < 2:
        return None

    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="pearson")

    # Matrisi listeye çevir (NaN → 0, heatmap render edilebilsin)
    matrix = []
    for a in cols:
        row = []
        for b in cols:
            v = corr.loc[a, b]
            row.append(_safe_float(v) if _safe_float(v) is not None else 0.0)
        matrix.append(row)

    # En güçlü ilişkiler (mutlak değer, kendisiyle olan 1.0'ları hariç tut)
    pairs = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i < j:  # her çifti bir kez
                v = _safe_float(corr.loc[a, b])
                if v is not None:
                    pairs.append({"a": a, "b": b, "corr": v})
    pairs.sort(key=lambda p: abs(p["corr"]), reverse=True)

    return {
        "columns": cols,
        "matrix": matrix,
        "top_pairs": pairs[:10],  # en güçlü 10 ilişki
    }


# ═══════════════════════════════════════════════════════════
#  2. HİSTOGRAMLAR (Plotly bar/histogram için)
# ═══════════════════════════════════════════════════════════

def compute_histograms(
    df: pd.DataFrame,
    numeric_cols: list[str],
    n_bins: int = 20,
) -> dict:
    """
    Her sayısal sütun için histogram bin'leri.

    Returns:
        {
          "age": {
            "bin_edges": [20, 25, 30, ...],   # bin sınırları
            "counts":    [15, 42, 38, ...],   # her bin'deki sayı
          },
          ...
        }
    """
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
            }
        except Exception:
            continue
    return result


# ═══════════════════════════════════════════════════════════
#  3. HEDEF İLİŞKİSİ (hedef belirtilmişse)
# ═══════════════════════════════════════════════════════════

def compute_target_relations(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
) -> Optional[dict]:
    """
    Hedefle en ilişkili sayısal sütunlar.
    Hedef sayısalsa Pearson korelasyon, kategorikse her sütunun
    hedef gruplarındaki ortalama farkı.

    Returns:
        {
          "target": "churn",
          "relations": [{"feature": "tenure", "score": 0.42}, ...]  # |score| azalan
        }
    """
    if target_col not in df.columns:
        return None

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
        "target": target_col,
        "target_type": "numeric" if target_is_numeric else "categorical",
        "relations": relations,
    }


# ═══════════════════════════════════════════════════════════
#  4. KÜMELEME (PCA 2D + KMeans, Plotly scatter için)
# ═══════════════════════════════════════════════════════════

def compute_clustering(
    df: pd.DataFrame,
    numeric_cols: list[str],
    max_k: int = 5,
    sample_size: int = 2000,
) -> Optional[dict]:
    """
    Sayısal sütunları PCA ile 2D'ye indirir, KMeans ile kümeler.
    Büyük veride sample alır (scatter okunaklı kalsın + hız).

    Returns:
        {
          "points": [{"x": .., "y": .., "cluster": 0}, ...],
          "n_clusters": 3,
          "explained_variance": 0.78,   # PCA 2 bileşenin açıkladığı oran
        }
    None: 2'den az sayısal sütun veya yetersiz veri
    """
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
            "points": points,
            "n_clusters": best_k,
            "silhouette": _safe_float(best_score),
            "explained_variance": round(explained, 4),
        }
    except ImportError:
        return None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
#  ANA FONKSİYON
# ═══════════════════════════════════════════════════════════

def run_deep_analysis(
    df: pd.DataFrame,
    profile: dict,
    target_col: Optional[str] = None,
) -> dict:
    """
    profile_table çıktısını ALIR, üstüne analiz katmanları ekler.

    Args:
        df: ham veri (parquet'ten okunmuş)
        profile: DuckDBEngine.profile_table() çıktısı
        target_col: hedef sütun (opsiyonel)

    Returns:
        profile + correlation_matrix + histograms + target_relations + clustering
    """
    summary = profile.get("summary", {})
    numeric_cols = summary.get("numeric_columns", [])

    # Mevcut profile'ı koru, üstüne ekle
    analysis = dict(profile)

    analysis["correlation_matrix"] = compute_correlation(df, numeric_cols)
    analysis["histograms"] = compute_histograms(df, numeric_cols)
    analysis["clustering"] = compute_clustering(df, numeric_cols)

    if target_col:
        analysis["target_relations"] = compute_target_relations(df, target_col, numeric_cols)

    return analysis