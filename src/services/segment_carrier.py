from __future__ import annotations

"""
Segment Carrier
---------------
Fairness'ın HAM grup değerlerine erişebilmesi için split anında alınan snapshot.

NEDEN VAR
=========
Fairness bugüne kadar ``X_test`` üzerinde çalışıyordu. Orası encode + scale
edilmiş bir matris; ham demografik değer orada YOKTUR. Üstelik
``dataset_store.dump_split`` parquet'i ``index=False`` yazdığı için test satırını
ham veriye geri join etmenin hiçbir yolu yok. Sonuç: fairness ham gruba hiç
ulaşamıyordu ve bunu telafi etmek için ``X_test`` üzerinde üç ayrı varsayım
yapıyordu — üçü de sessizce kırılıyordu:

  1. ``mask = X_test[col] > 0.5`` → one-hot 0/1 varsayar. Ama dummy'ler
     ``dtype=int`` olduğu için ``_scale_features_tt``'te StandardScaler'dan
     geçiyor; standartlaştırılmış bir dummy'de 0.5 eşiği anlamsız. Grup oranı
     ~%80'i geçtiğinde maske TAMAMEN boşalıyor → grup "< min_size" kuralına
     takılıp sessizce atlanıyordu. Yani en kalabalık gruplar hiç denetlenmiyordu.
  2. ``drop_first=True`` (``_encode_categorical_tt``) bir kategoriyi referans
     olarak düşürüyor → o grubun kolonu hiç oluşmuyor, fairness onu hiç görmüyor.
  3. Kardinalitesi 10'u aşan kolon TargetEncoder'a gidiyor → binary olmaktan
     çıkıyor, ``nunique()==2`` kontrolüne takılıp atlanıyordu.

Taşıyıcı üçünü birden çözer: ham değere bakılır, hiçbir kategori düşmez, çok
sınıflı gruplar binary'ye indirgenmeden analiz edilir.

HİZALAMA
========
Snapshot split'ten HEMEN SONRA, encode/scale'den ÖNCE alınır ve ``X_test`` ile
POZİSYONEL hizalıdır. Split sonrası transform'lar (count-encode, groupby-deviation,
encode, scale, feature-selection, leakage-drop) yalnızca KOLON ekler/düşürür;
satır sırasını veya sayısını değiştirmez. Bu değişmez testle sabitlenmiştir.

KAPSAM (ayar, kod değil)
========================
``settings.FAIRNESS_SEGMENT_MODE``:
  "narrow" (varsayılan) → demografik sezgisel + db_intelligence semantic_type.
      Bugünkü fairness kapsamı, küçük state.
  "wide" → tüm düşük kardinaliteli kategorikler. "Her segmentte performans"
      analizini açar. Geçiş kod değişikliği DEĞİL, ayar.
"""

from typing import Optional

import pandas as pd

from src.core.config import settings
from src.core.feature_naming import is_derived_feature
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# Demografik sezgisel — evaluator._analyze_bias'taki listeyle AYNI KAYNAK olsun
# diye buraya taşındı (iki yerde iki liste → bir kolon bir yerde denetlenip
# diğerinde atlanırdı).
DEMOGRAPHIC_KEYWORDS = (
    "age", "sex", "gender", "race", "ethnicity", "marital", "country", "nativity",
)


def _is_demographic(col: str, db_intelligence: dict) -> bool:
    """Kolon demografik mi — semantic_type veya isim sezgiseli."""
    for c in (db_intelligence or {}).get("columns", []) or []:
        if c.get("column_name") == col and c.get("semantic_type") == "demographic":
            return True
    low = (col or "").lower()
    return any(k in low for k in DEMOGRAPHIC_KEYWORDS)


def select_segment_columns(
    df: pd.DataFrame,
    db_intelligence: dict,
    *,
    exclude: Optional[set[str]] = None,
) -> tuple[list[str], dict[str, str]]:
    """Taşınacak segment kolonlarını seçer.

    Returns:
        (seçilen kolonlar, {atlanan kolon: görünür sebep})

    Atlananlar da DÖNER — sessiz eleme yok. Fairness bölümü "şu kolon segment
    olarak taşınmadı çünkü ..." diyebilsin diye.
    """
    excl = {c for c in (exclude or set()) if c}
    mode = (settings.FAIRNESS_SEGMENT_MODE or "narrow").strip().lower()
    max_card = settings.FAIRNESS_SEGMENT_MAX_CARDINALITY

    selected: list[str] = []
    skipped: dict[str, str] = {}

    # Aday havuzu: kategorik + düşük kardinaliteli sayısal (0/1 bayrak, kod alanı).
    for col in df.columns:
        if col in excl:
            continue

        # (1) MOTOR TÜREVİ kolonlar segment ADAYI DEĞİLDİR. age_log, ratio_age_*,
        #     is_zero_* demografik grup değil, türetilmiş sayısaldır. "Ölçülemedi"
        #     listesine bile GİRMEZ — orası gerçek demografik adaylara ayrılmıştır
        #     (kullanıcının açık şartı). Sessiz ama DOĞRU: türev bir grup tanımı değildir.
        if is_derived_feature(col):
            continue

        is_demo = _is_demographic(col, db_intelligence)
        if mode == "narrow" and not is_demo:
            continue  # dar modda demografik olmayanlar aday bile değil (sebep gerekmez)

        try:
            nuniq = int(df[col].nunique(dropna=True))
        except Exception as e:  # pragma: no cover - defensive
            skipped[col] = {"code": "seg_skip_card_unreadable", "params": {"e": str(e)}}
            continue

        if nuniq < 2:
            skipped[col] = {"code": "seg_skip_single_value"}
            continue
        if nuniq > max_card:
            # (2) YÜKSEK KARDİNALİTELİ SÜREKLİ SAYISAL bir grup değildir (age, ratio,
            #     ID). "Ölçülemedi" listesine GİRMEZ — orası gerçek demografik
            #     adaylara ayrılıdır. Ama yüksek kardinaliteli KATEGORİK (çok ülkeli
            #     bir kolon) gerçek bir adaydır → sebebiyle görünür kalır (binleme işi ayrı).
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            skipped[col] = {"code": "seg_skip_high_card",
                            "params": {"n": nuniq, "max": max_card}}
            continue

        selected.append(col)

    return selected, skipped


def select_concentration_columns(
    df: pd.DataFrame,
    *,
    exclude: Optional[set[str]] = None,
) -> tuple[list[str], dict[str, str]]:
    """D3 aksiyon-yoğunlaşması için aday kolonlar — DEMOGRAFİ-AGNOSTİK.

    ``select_segment_columns``'ın kardinalite [2, max_card] + motor-türevi eleme
    kurallarını YENİDEN KULLANIR ama ``_is_demographic`` filtresini KULLANMAZ.
    Sebep: D3'ün sorusu adalet değil — "modelin aksiyon aldığı en riskli %10 tek
    bir gruba mı yığılmış". Bunun için grup demografik olmak zorunda değil;
    loan_grade bir kredi segmenti ama yığılma teşhisi için tamamen geçerli.

    İKİ KAVRAM AYRI KALIR: fairness'ın ``select_segment_columns``'ı (demografik)
    HİÇ değişmez; bu ayrı bir seçici, ayrı bir çıktı. Bir kolonun ikisinde birden
    çıkması (churn'de country/gender) sorun değil — farklı sorular, ayrı raporlar.
    """
    excl = {c for c in (exclude or set()) if c}
    max_card = settings.FAIRNESS_SEGMENT_MAX_CARDINALITY
    selected: list[str] = []
    skipped: dict[str, str] = {}

    for col in df.columns:
        if col in excl:
            continue
        if is_derived_feature(col):
            continue   # motor türevi (ratio_*, is_zero_*, *_log) bir grup tanımı değildir
        try:
            nuniq = int(df[col].nunique(dropna=True))
        except Exception as e:  # pragma: no cover - defensive
            skipped[col] = {"code": "seg_skip_card_unreadable", "params": {"e": str(e)}}
            continue
        if nuniq < 2:
            skipped[col] = {"code": "seg_skip_single_value"}
            continue
        if nuniq > max_card:
            # Yüksek kardinaliteli SÜREKLİ sayısal (age, income, ID) grup değildir → sessiz.
            # Yüksek kardinaliteli KATEGORİK gerçek aday ama binleme ayrı iş → sebebiyle görünür.
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            skipped[col] = {"code": "seg_skip_high_card", "params": {"n": nuniq, "max": max_card}}
            continue
        selected.append(col)

    return selected, skipped


def build_raw_test_carrier(
    X_test_raw: pd.DataFrame,
    db_intelligence: dict,
    *,
    exclude: Optional[set[str]] = None,
) -> tuple[Optional[pd.DataFrame], dict]:
    """Split sonrası HAM X_test'in TAM snapshot'ı + fairness'ın analiz kapsamı.

    NEDEN TÜM KOLONLAR TAŞINIYOR (yalnızca demografikler değil)
    ===========================================================
    İki tüketici var ve ihtiyaçları farklı:
      - fairness      → HAM KATEGORİK değerler (sex, race, marital.status)
      - hata analizi  → HAM SAYISAL değerler ("hatalı vakalarda 42 saat, doğrularda 17")

    Hata analizi ``X_test`` üzerinde çalışıyordu; orası StandardScaler'dan geçmiş,
    dolayısıyla "42 saat" bilgisi YOK EDİLMİŞ durumda — kullanıcıya "0.3682" diye
    bir sayı gösteriliyordu. Ham değeri geri getirmenin tek yolu split anında
    snapshot almak (scaler saklanmıyor, tersine çevrilemez).

    Taşıma maliyeti DİSK'tir, state değil: parquet'e yazılır, state'te yalnızca yol
    taşınır (X_test'in kendisi zaten aynı şekilde yazılıyor). ``FAIRNESS_SEGMENT_MODE``
    artık NEYİN TAŞINDIĞINI değil, fairness'ın NEYİ ANALİZ ETTİĞİNİ belirler —
    dar varsayılan korunur, ama "her segmentte performans" için ham veri hazır durur.

    Returns:
        (tam ham snapshot | None, meta)
        meta: {"segment_columns", "carried_columns", "mode", "skipped", "n_rows", "reason"}
    """
    mode = (settings.FAIRNESS_SEGMENT_MODE or "narrow").strip().lower()
    seg_cols, skipped = select_segment_columns(X_test_raw, db_intelligence, exclude=exclude)

    meta: dict = {
        "segment_columns": seg_cols,     # fairness'ın analiz edeceği kolonlar
        "carried_columns": [],           # parquet'e fiilen yazılanlar
        "mode": mode,
        "skipped": skipped,
        "n_rows": int(len(X_test_raw)),
        "reason": "",
    }

    if X_test_raw is None or X_test_raw.empty:
        meta["reason"] = {"code": "seg_empty_partition"}
        return None, meta

    snapshot = X_test_raw.copy()
    meta["carried_columns"] = snapshot.columns.tolist()

    if not seg_cols:
        # Ham veri yine taşınır (hata analizi kullanacak); yalnızca FAIRNESS kapsamı boş.
        meta["reason"] = ({"code": "seg_no_demo_narrow"} if mode == "narrow"
                          else {"code": "seg_no_demo_broad"})
        logger.info(f"[raw_test_carrier] Fairness segment kolonu yok ({mode}): {meta['reason']}")
    else:
        logger.info(
            f"[raw_test_carrier] {len(seg_cols)} segment kolonu analiz kapsamında ({mode}): {seg_cols}"
            + (f" | fairness dışı: {list(skipped)}" if skipped else "")
        )
    logger.info(f"[raw_test_carrier] {len(meta['carried_columns'])} ham kolon taşındı "
                f"({meta['n_rows']:,} satır)")
    return snapshot, meta
