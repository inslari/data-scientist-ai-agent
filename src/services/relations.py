from __future__ import annotations

"""
Veri İlişkileri (Relations)
---------------------------
Sütunlar arası ve sütun-hedef ilişkilerinin RAPOR ÖZETİ.

NEDEN SPLIT ANINDA HESAPLANIYOR
===============================
Bu istatistikler ``feature_builder``'ın split'inden HEMEN SONRA, encode/scale'den
ÖNCE, gerçek ``X_train`` partisyonundan hesaplanır (segment taşıyıcısıyla aynı
kanca). Provenance ``train_only`` etiketi bu yüzden bir İDDİA değil, YAPISAL bir
olgudur: hesap fiilen train satırlarından geçer.

Alternatif — split'i rapor anında ``train_test_split`` ile yeniden üretmek —
BİLİNÇLİ olarak reddedildi. Gerçek split ``entity_aware`` veya
``global_chronological`` olabiliyor (bkz. feature_builder), üstelik FE sırasında
satır düşüyor (dropna, sampling cap). Yeniden üretim bu durumlarda SESSİZCE farklı
satırlar verir ve "train_only" etiketi yalan söylerdi. ``feature_plan_node``'daki
kilitli-holdout yeniden üretimi ölçüm modu için yeterli, rapor kanıtı için değil.

NE HESAPLANIR
=============
  X-y (hedefle ilişki) : hedef-farkındalıklı → train-only olması ŞART
  X-X (sütunlar arası) : hedefi hiç görmez → zaten güvenli, aynı partisyondan
                         geldiği için bedavaya train_only olur

``analysis_engine``'in test edilmiş ``compute_correlation`` / ``compute_target_relations``
fonksiyonları yeniden kullanılır; onlara X_train'den türetilmiş minimal bir profil
verilir (profiling_report TÜM veriden gelir ve türetilmiş feature'ları içermez —
kullanılamaz).

Ağır çıktılar (pairwise noktaları, kde eğrileri, histogram binleri) BİLEREK
üretilmez: onlar frontend grafiği içindir, raporun kanıt katmanı için değil ve
state'i şişirirler.
"""

import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger

logger = get_logger(__name__)

# Korelasyon hesabı için satır tavanı (extract_plan_hints ile aynı disiplin:
# eşik/özet kararı için tam tarama gereksiz, deterministik örneklem yeter).
RELATIONS_MAX_ROWS = 50_000

# Bir ilişkinin kanıt listesine girmesi için gereken minimum |skor|. Altındaki
# her şey gürültü sayılır ve 50 kolonluk veri setinde kanıtı boğar.
RELATIONS_MIN_ABS_SCORE = 0.01

# X-X çiftlerinde "neredeyse aynı bilgi" eşiği (analysis_engine ile aynı sabit).
RELATIONS_REDUNDANT_CORR = 0.95


# ── Türetilmiş kolon tanıma (dataset-agnostik) ───────────────────────────────
# Motorun ürettiği kolonların KENDİ kaynak kolonuyla ilişkisi tanımı gereği
# yüksektir (age ↔ age_log = 0.98; credit_card ↔ is_zero_credit_card = -1) —
# bilgi değil GÜRÜLTÜdür ve ana ilişki listesini boğar. Kolon ADLARINDAN kaynağı
# çıkarırız; hiçbir dataset-spesifik isim hard-code edilmez (feature_naming.py
# ile aynı prefix/suffix sözlüğü). İki tür ayrımı:
#   _derived_sources  → TÜM türevlerin kaynak kümesi (X-X definitional eleme için)
#   _alias_base       → yalnız AYNA türevler (is_zero/was_missing/isna) — bunlar
#                       ham kolonun işaret-çevrik kopyasıdır; hedefle ilişkide
#                       (active_member/is_zero_active_member) biri yeterli.
_RE_IS_ZERO = re.compile(r"^is_zero_(.+)$")
_RE_WAS_MISSING = re.compile(r"^was_missing_(.+)$")
_RE_COUNT = re.compile(r"^count_(.+)$")
_RE_ISNA = re.compile(r"^(.+)_isna$")
_RE_RATIO = re.compile(r"^ratio_(.+)_([^_]+)$")
_RE_GRPSTD = re.compile(r"^(?:dev|zdev|grpstd)_(.+)_by_(.+)$")
_RE_MATH_SUF = re.compile(r"^(.+)_(?:log1p|log|squared|sqrt|zscore)$")
_RE_MATH_PRE = re.compile(r"^(?:log1p|log|sqrt)_(.+)$")


def _derived_sources(name: str) -> set[str]:
    """Bir kolonun TÜRETİLDİĞİ kaynak kolon(lar). Ham kolonsa boş küme."""
    for rx in (_RE_IS_ZERO, _RE_WAS_MISSING, _RE_COUNT, _RE_MATH_PRE):
        m = rx.match(name)
        if m:
            return {m.group(1)}
    for rx in (_RE_ISNA, _RE_MATH_SUF):
        m = rx.match(name)
        if m:
            return {m.group(1)}
    for rx in (_RE_RATIO, _RE_GRPSTD):
        m = rx.match(name)
        if m:
            return {m.group(1), m.group(2)}
    return set()


def _alias_base(name: str) -> Optional[str]:
    """Kolon ham bir kolonun İŞARET-ÇEVRİK kopyasıysa o kaynağı döndürür, değilse None.

    Yalnız ayna göstergeler: is_zero_{X}, was_missing_{X}, {X}_isna. Bunların hedefle
    ilişkisi ham kolonunkinin aynısıdır (ters işaretli) → ikisi de listede olursa
    biri yeterli. Math türevleri (log/sqrt) AYNA DEĞİLDİR (ilişkiyi değiştirir) →
    onlar ayrı bilgi sayılır, burada None döner.
    """
    for rx in (_RE_IS_ZERO, _RE_WAS_MISSING):
        m = rx.match(name)
        if m:
            return m.group(1)
    m = _RE_ISNA.match(name)
    if m:
        return m.group(1)
    return None


# Türev ekleri — token-sınırlı kaynak aramada affiks soymak için.
_DERIV_PREFIXES = ("is_zero_", "was_missing_", "count_", "log1p_", "log_", "sqrt_",
                   "ratio_", "grpstd_", "zdev_", "dev_")
_DERIV_SUFFIXES = ("_log1p", "_log", "_squared", "_sqrt", "_zscore", "_isna")


def _has_source_token(derived: str, base: str) -> bool:
    """`derived`'ın türev-ekini soyduktan sonra kalan gövdede `base` bir TOKEN dizisi mi?

    Alt çizgili kolon adları (credit_score) ratio/grpstd'de kesin ayrıştırılamaz;
    'ratio_age_credit_score' → gövde 'age_credit_score', hem 'age' hem 'credit_score'
    token-sınırlı olarak burada yakalanır (regex son alt çizgiden bölünce kaçırıyordu).
    '_by_' ayracı grpstd'de token sınırına indirgenir."""
    rem = None
    for p in _DERIV_PREFIXES:
        if derived.startswith(p):
            rem = derived[len(p):]
            break
    if rem is None:
        for s in _DERIV_SUFFIXES:
            if derived.endswith(s):
                rem = derived[:-len(s)]
                break
    if not rem:
        return False
    rem = rem.replace("_by_", "_").strip("_")
    return (rem == base or rem.startswith(base + "_")
            or rem.endswith("_" + base) or ("_" + base + "_") in rem)


def _is_definitional_pair(a: str, b: str) -> bool:
    """İki kolonun ilişkisi TANIM gereği mi (biri diğerinden türemiş / ortak kaynak)?"""
    sa, sb = _derived_sources(a), _derived_sources(b)
    if b in sa or a in sb or (sa and sb and (sa & sb)):
        return True
    return _has_source_token(a, b) or _has_source_token(b, a)


def _dedup_target_mirrors(rels: list[dict]) -> list[dict]:
    """Hedef ilişkilerinde AYNA çiftleri teke indirir (X ve is_zero_X).

    Kimlik anahtarı: ayna türevse kaynak kolon, değilse kolonun kendisi. Aynı
    anahtarda: ham kolon (anahtarla aynı ad) varsa o kalır; yoksa |skor| en büyük
    olan. Sıra korunur (girişteki güçlü→zayıf sıralaması bozulmaz)."""
    best: dict[str, dict] = {}
    order: list[str] = []
    for r in rels:
        feat = r["feature"]
        key = _alias_base(feat) or feat
        if key not in best:
            best[key] = r
            order.append(key)
            continue
        cur = best[key]
        # Ham kolonu (adı anahtara eşit) tercih et; değilse büyük |skor|.
        cur_is_raw = cur["feature"] == key
        new_is_raw = feat == key
        if new_is_raw and not cur_is_raw:
            best[key] = r
        elif new_is_raw == cur_is_raw and abs(r.get("score") or 0) > abs(cur.get("score") or 0):
            best[key] = r
    return [best[k] for k in order]


def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def fisher_z_interval(r: float, n: int, z: float = 1.96) -> Optional[dict]:
    """Pearson korelasyonu için %95 güven aralığı (Fisher z dönüşümü).

    Neden Fisher z: r'nin örneklem dağılımı çarpıktır ve |r|→1'de sıkışır; düz
    normal yaklaşım aralığı [-1,1] dışına taşırır. arctanh uzayında dağılım
    yaklaşık normaldir, aralık geri dönüştürülünce doğal olarak sınırlar içinde kalır.

    n <= 3 → aralık tanımsız (None). Sıfır genişlikte aralık ÜRETİLMEZ.
    """
    if r is None or n is None or n <= 3:
        return None
    r = max(min(float(r), 0.999999), -0.999999)   # arctanh(±1) = ±inf
    zr = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(zr - z * se), np.tanh(zr + z * se)
    return {"lower": round(float(lo), 4), "upper": round(float(hi), 4), "method": "fisher-z"}


def _profile_from_frame(df: pd.DataFrame) -> dict:
    """X_train'den analysis_engine'in beklediği minimal profil.

    profiling_report KULLANILAMAZ: o TÜM ham veriden gelir, split sonrası
    türetilmiş feature'ları (oran/bayrak kolonları) içermez ve kolon listesi
    X_train ile uyuşmaz.
    """
    return {
        "summary": {
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
        },
        "shape": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "columns": {},
    }


def compute_relations(
    X_train_raw: pd.DataFrame,
    y_train,
    target_name: str,
    *,
    generated_from: str,
) -> dict:
    """Split anındaki HAM train partisyonundan ilişki özetini çıkarır.

    Args:
        X_train_raw: encode/scale ÖNCESİ train partisyonu (hedef kolon İÇİNDE DEĞİL).
        y_train: train hedef değerleri (X_train_raw ile hizalı).
        target_name: hedefin ham adı.
        generated_from: "train_only" (supervised, gerçek split) veya
            "full_data_descriptive" (unsupervised — holdout yok).

    Returns:
        {"generated_from", "n_rows", "target", "target_type",
         "target_relations": [{"feature","score","ci"}],   # |skor| azalan, TAMAMI
         "top_pairs": [{"a","b","corr"}],
         "redundant_pairs": [{"a","b","corr"}],
         "reason": ""}                                     # boşsa hesap yapıldı
    """
    from src.services.analysis_engine import compute_correlation, compute_target_relations

    out: dict = {
        "generated_from": generated_from,
        "n_rows": 0,
        "target": target_name or "",
        "target_type": "",
        "target_relations": [],
        "top_pairs": [],
        "redundant_pairs": [],
        "reason": "",
    }

    if X_train_raw is None or X_train_raw.empty:
        out["reason"] = {"code": "rel_train_empty"}
        return out

    df = X_train_raw.copy()
    # Deterministik örneklem (RANDOM_SEED yerine sabit 42 — analysis_engine deseni).
    if len(df) > RELATIONS_MAX_ROWS:
        df = df.sample(n=RELATIONS_MAX_ROWS, random_state=42)
    n_rows = int(len(df))
    out["n_rows"] = n_rows

    # ── X-X: sütunlar arası (hedefi HİÇ görmez) ──
    try:
        corr_res = compute_correlation(df, _profile_from_frame(df), target_col=None) or {}
        pearson = corr_res.get("correlation_matrix") or {}
        pairs = pearson.get("top_pairs") or []
        kept = [
            {"a": p["a"], "b": p["b"], "corr": p["corr"]}
            for p in pairs if abs(p.get("corr") or 0) >= RELATIONS_MIN_ABS_SCORE
        ]
        # Türetilmiş kolonun kendi kaynağıyla ilişkisi ANA listeden elenir (gürültü);
        # near-duplicate bölümünde kalır — orası tam olarak "aynı bilgiyi taşıyan"
        # çiftler içindir (kullanıcının açık şartı).
        definitional = [p for p in kept if _is_definitional_pair(p["a"], p["b"])]
        informative = [p for p in kept if not _is_definitional_pair(p["a"], p["b"])]
        out["top_pairs"] = informative
        out["redundant_pairs"] = definitional + [
            p for p in informative if abs(p["corr"]) >= RELATIONS_REDUNDANT_CORR
        ]
    except Exception as e:
        logger.warning(f"[relations] X-X korelasyonu hesaplanamadı: {e}")

    # ── X-y: hedefle ilişki (train-only olması ŞART olan kısım) ──
    if target_name and y_train is not None and len(y_train) == len(X_train_raw):
        try:
            y_ser = pd.Series(np.asarray(y_train).ravel(), index=X_train_raw.index)
            joined = df.join(y_ser.rename(target_name), how="left")
            tr_res = compute_target_relations(
                joined, _profile_from_frame(joined), target_col=target_name
            ) or {}
            tr = tr_res.get("target_relations") or {}
            out["target_type"] = tr.get("target_type", "")
            rels = []
            for r in tr.get("relations", []):
                score = r.get("score")
                if score is None or abs(score) < RELATIONS_MIN_ABS_SCORE:
                    continue
                # Güven aralığı YALNIZCA gerçek Pearson korelasyonunda anlamlı.
                # Kategorik hedefte skor bir "ayrışma oranı"dır (gruplar arası std /
                # genel std), korelasyon değildir → Fisher z UYGULANMAZ; uygulanırsa
                # olmayan bir kesinlik iddia edilmiş olurdu.
                ci = (fisher_z_interval(score, n_rows)
                      if out["target_type"] == "numeric" else None)
                rels.append({"feature": r["feature"], "score": score, "ci": ci})
            # Ayna çiftleri (X ve is_zero_X hedefe karşı) teke indir — biri yeterli.
            out["target_relations"] = _dedup_target_mirrors(rels)
        except Exception as e:
            logger.warning(f"[relations] X-y ilişkileri hesaplanamadı: {e}")
            out["reason"] = {"code": "rel_target_failed", "params": {"e": str(e)}}
    elif not target_name:
        out["reason"] = {"code": "rel_no_target"}

    logger.info(
        f"[relations] {generated_from}: {n_rows:,} satır | "
        f"hedefle ilişki={len(out['target_relations'])} | "
        f"sütun çifti={len(out['top_pairs'])} (gereksiz={len(out['redundant_pairs'])})"
    )
    return out
