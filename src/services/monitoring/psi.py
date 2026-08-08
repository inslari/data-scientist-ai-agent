"""Katman 2 Parça 3 — PSI (input drift) çekirdeği. SAF fonksiyonlar (DB yok → tek başına testli).

Referans = `training_reference.per_feature` (raw-space). Prod = pencerenin örneklem `input_row`'ları.

KOVA KENARLARI (sürekli): quantile SEVİYELERİNDEN türetilir, sabit `[.01,.24,.25,.25,.24,.01]`
DEĞİL. Benzersiz kenar değerleri + sağ-sürekli CDF (`F_upper`) → referans oranı = kenardaki seviye
farkı. Binary (0/1) böylece özel-kılıfsız çözülür: quantile `[0,0,0,1,1]` → kenar `[-inf,0,1,+inf]`
= 3 kova. Ölçüm 2: churn'ün adj-collision kolonları DEJENERE DEĞİL, ölçülür.

DEJENERASYON = YALNIZ REFERANS özelliği: distinct interior quantile < 2 (gerçek sabit kolon) →
psi=null, degenerate=true, reason. **Prod tarafında tek-değer DRİFT'tir** (referans sağlamsa PSI
hesaplanır ve YÜKSEK çıkar — tüm kütle bir kovada); sessizce degenerate SAYILMAZ.

ZERO-KOVA: Laplace (α, `settings.MONITORING_PSI_LAPLACE_ALPHA`) HEM ref HEM prod pseudo-count'una.
"""
from __future__ import annotations

import math
from bisect import bisect_left

# per_feature quantile anahtarı → seviye (references._REF_QUANTILES ile hizalı)
_Q_LEVELS = (("0.01", 0.01), ("0.25", 0.25), ("0.5", 0.50), ("0.75", 0.75), ("0.99", 0.99))


# ── kova inşası (referans) ───────────────────────────────────────────────────

def _continuous_bins(stat: dict):
    """Döner: ((edges, ref_props), None) ya da (None, reason) dejenere ise.

    edges = [-inf, *benzersiz_interior, +inf]; ref_props = ardışık seviye farkları (hepsi > 0)."""
    q = stat.get("quantiles") or {}
    pts = []                                  # (value, level) — mevcut interior quantile'lar
    for key, lvl in _Q_LEVELS:
        v = q.get(key)
        if v is not None:
            pts.append((float(v), lvl))
    interior = sorted({v for v, _ in pts})
    if len(interior) < 2:
        return None, ("single_value" if len(interior) == 1 else "insufficient_distinct_edges")
    # F_upper(value) = o değere sahip quantile noktalarının EN BÜYÜK seviyesi (sağ-sürekli CDF).
    fupper: dict = {}
    for v, lvl in pts:
        fupper[v] = max(fupper.get(v, 0.0), lvl)
    edges = [-math.inf] + interior + [math.inf]

    def _F(e):
        if e == -math.inf:
            return 0.0
        if e == math.inf:
            return 1.0
        return fupper[e]

    ref_props = [_F(edges[i + 1]) - _F(edges[i]) for i in range(len(edges) - 1)]
    return (edges, ref_props), None


def _categorical_bins(stat: dict):
    """Döner: ((cats, ref_cat_props, tail), None) ya da (None, reason).

    "other" kovası DAİMA eklenir (prod unseen'i yakalar; ref tarafı tail_mass, 0 olabilir → Laplace).
    Dejenere yalnız hiç bilinen kategori yoksa (karşılaştıracak referans yok)."""
    vcn = stat.get("value_counts_normalized") or {}
    cats = list(vcn.keys())
    if len(cats) < 1:
        return None, "insufficient_distinct_edges"
    tail = float(stat.get("tail_mass") or 0.0)
    return (cats, [float(vcn[c]) for c in cats], tail), None


# ── PSI hesabı ───────────────────────────────────────────────────────────────

def _laplace_psi(ref_props, prod_counts, n_train: int, alpha: float) -> float:
    """PSI = Σ (p_prod − p_ref)·ln(p_prod/p_ref); iki taraf da Laplace pseudo-count (α, k kova)."""
    k = len(ref_props)
    n_train = max(int(n_train), 1)
    n_prod = sum(prod_counts)
    ref_s = [((p * n_train) + alpha) / (n_train + alpha * k) for p in ref_props]
    prod_s = [(c + alpha) / (n_prod + alpha * k) for c in prod_counts]
    return sum((p - r) * math.log(p / r) for r, p in zip(ref_s, prod_s))


def _continuous_psi(bins, values, n_train, alpha):
    """Döner: (psi, n_bins, distribution). distribution = histogram overlay (Faz 0 item 2, detay endpoint):
    kenarlar + referans oranları + prod sayımları. ±inf kenar → `null` (JSON-güvenli; psycopg2 inf reddeder)."""
    edges, ref_props = bins
    k = len(ref_props)
    counts = [0] * k
    for x in values:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if math.isnan(xf):
            continue
        j = min(max(bisect_left(edges, xf) - 1, 0), k - 1)   # (edges[j], edges[j+1]] yarı-açık
        counts[j] += 1
    dist = {"kind": "continuous",
            "edges": [None if math.isinf(e) else round(float(e), 6) for e in edges],
            "ref_props": [round(float(p), 6) for p in ref_props],
            "prod_counts": counts, "prod_total": sum(counts)}
    return _laplace_psi(ref_props, counts, n_train, alpha), k, dist


def _categorical_psi(bins, values, n_train, alpha):
    """Döner: (psi, n_bins, distribution). "other" kovası dahil kategoriler + ref oranları + prod sayımları."""
    cats, ref_cat_props, tail = bins
    idx = {c: i for i, c in enumerate(cats)}
    ref_props = ref_cat_props + [tail]                       # son kova = "other"
    k = len(ref_props)
    counts = [0] * k
    for v in values:
        if v is None:
            continue
        counts[idx.get(str(v), k - 1)] += 1                  # bilinmeyen → "other"
    dist = {"kind": "categorical",
            "categories": [str(c) for c in cats] + ["__other__"],
            "ref_props": [round(float(p), 6) for p in ref_props],
            "prod_counts": counts, "prod_total": sum(counts)}
    return _laplace_psi(ref_props, counts, n_train, alpha), k, dist


# ── pencere özeti ─────────────────────────────────────────────────────────────

def compute_psi_summary(per_feature: dict, prod_rows: list, n_train: int, alpha: float,
                        raw_importance: "dict | None" = None) -> dict:
    """per_feature (referans) + prod_rows (input_row dict listesi) → pencere PSI özeti.

    `raw_importance` (Parça 3b, raw_col→model payı): verilirse psi_overall importance-AĞIRLIKLI
    (ölçülen küme üzerinde renormalize) + her kolona importance_weight. YOKSA düz ortalama, weight None
    (fallback — importance çıkarılamamış / regresyon). psi_max HER ZAMAN ağırlıksız (kritik kolon
    ortalamada erimesin). degenerate_importance_mass = ölçülemeyen kolonların model ağırlığı (sayı değil)."""
    has_imp = bool(raw_importance)
    per_feat_out: dict = {}
    psi_scalars: dict = {}                        # {col: psi} — Faz 0 item 3 (list sparkline, hafif projeksiyon)
    psi_distribution: dict = {}                   # {col: dist} — Faz 0 item 2 (detay overlay, ayrı kolon/deferred)
    measured: list = []                          # (col, psi) — dejenere OLMAYAN
    degen = 0
    for col, stat in (per_feature or {}).items():
        kind = stat.get("kind")
        bins, reason = (_continuous_bins(stat) if kind == "continuous" else _categorical_bins(stat))
        iw = (float(raw_importance.get(col, 0.0)) if has_imp else None)
        if bins is None:
            per_feat_out[col] = {"psi": None, "n_bins": 0, "degenerate": True,
                                 "reason": reason, "importance_weight": iw}
            degen += 1
            continue
        values = [r[col] for r in prod_rows if isinstance(r, dict) and col in r]
        psi, n_bins, dist = (_continuous_psi(bins, values, n_train, alpha) if kind == "continuous"
                             else _categorical_psi(bins, values, n_train, alpha))
        psi_r = round(float(psi), 6)
        per_feat_out[col] = {"psi": psi_r, "n_bins": n_bins, "degenerate": False,
                             "reason": None, "importance_weight": iw}
        psi_scalars[col] = psi_r
        psi_distribution[col] = dist
        measured.append((col, float(psi)))

    # psi_overall: importance-ağırlıklı (ölçülen küme üzerinde RENORMALİZE — dejenereler düştüğü için);
    # importance yoksa ya da ölçülen ağırlık toplamı 0 ise DÜZ ortalama (fallback).
    if not measured:
        psi_overall = None
    elif has_imp and sum(raw_importance.get(c, 0.0) for c, _ in measured) > 0:
        wsum = sum(raw_importance.get(c, 0.0) for c, _ in measured)
        psi_overall = sum(raw_importance.get(c, 0.0) * p for c, p in measured) / wsum
    else:
        psi_overall = sum(p for _, p in measured) / len(measured)

    psi_max = max((p for _, p in measured), default=None)
    degen_mass = (round(sum(raw_importance.get(col, 0.0)
                            for col, o in per_feat_out.items() if o["degenerate"]), 6)
                  if has_imp else None)
    return {
        "psi_overall": (round(psi_overall, 6) if psi_overall is not None else None),
        "psi_max": (round(psi_max, 6) if psi_max is not None else None),
        "psi_degenerate_count": degen,
        "degenerate_importance_mass": degen_mass,
        "psi_per_feature": per_feat_out,
        "psi_scalars": psi_scalars,               # {col: psi} — list sparkline (hafif)
        "psi_distribution": psi_distribution,     # {col: dist} — detay overlay (ayrı kolon, deferred)
    }
