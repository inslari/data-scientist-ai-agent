from __future__ import annotations
from typing import Any

from src.core.llm import get_llm
from src.core.config import settings
from src.core.prompt_loader import load_prompt
from src.schemas.leakage_schemas import LeakageReport
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
import pandas as pd

from src.core.logging_config import get_logger
logger = get_logger(__name__)



def _wilson_lower_bound(p: float, n: int, z: float = 1.96) -> float:
    """Bir dilimdeki sınıf oranının %95 güven alt sınırı.

    Ham oran küçük dilimlerde yalan söyler: %2 pozitifli bir veride 600 satırlık bir
    dilimde hiç pozitif görmemek sürpriz değildir — bu "sonuç kesin" demek değil,
    "ölçecek kadar veri yok" demektir. Ham p=1.0 ise kilit ilan etmek, olmayan bir
    kesinliği uydurmaktır. Alt sınır, saflık iddiasını örneklemin taşıyabileceği
    kadarıyla sınırlar: aynı dilim 6000 satırda sıfır pozitif içeriyorsa bu gerçekten
    güçlü kanıttır ve alt sınır 1'e yaklaşır.
    """
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * float(np.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)))
    return max(0.0, (centre - margin) / denom)


def detect_target_lock(X: pd.DataFrame, y: np.ndarray,
                       feature_importance: dict,
                       intent: str = "classification",
                       importance_threshold: float = 0.05,
                       gain_threshold: float = 0.95,
                       min_bin_fraction: float = 0.05,
                       variance_collapse_threshold: float = 0.05,
                       n_bins: int = 10) -> list[dict]:
    """
    Yüksek-importance sayısal feature'ları dilimlere böler.
    Classification: Bir dilimde target tek sınıfa kilitleniyorsa (saflık kazancı > eşik)
    Regression: Bir dilimde target varyansı global varyansa göre çöküyorsa (oran < eşik)
    ve o dilim yeterince büyükse (> min_bin_fraction), sızıntı şüphesi raporlar.

    Classification'da ham saflık (dilimdeki çoğunluk sınıfının oranı) KULLANILAMAZ:
    dengesiz veride (fraud %2, churn %20) her dilim zaten trivially saf görünür ve
    hedefle hiç ilişkisi olmayan bir kolon bile sızıntı diye işaretlenir. Bunun
    yerine saflık, sınıfın global oranına (prior) göre normalize edilir:

        gain_c = (p_c - π_c) / (1 - π_c)

    p_c dilimdeki, π_c ise veri genelindeki sınıf oranı. 0 = dilim prior'dan
    farksız, 1 = tam kilit. Prior her çağrıda y'den okunur, yani eşik veri setine
    kendini uyarlar: dengesiz veride yanlış alarm sönerken dengeli veride klasik
    kilit yakalanmaya devam eder. Kazanç tüm sınıflar üzerinden maksimum alınır —
    kilit her iki yönde de (ve multiclass'ta) olabilir.

    Güçlü sinyal ile kilit farklı şeylerdir: fraud oranı %2 iken %50 fraud'lu bir
    dilim çok güçlü bir sinyaldir (gain≈0.49) ama kilit değildir; sızıntı, sonucun
    neredeyse KESİNLEŞMESİdir.

    Eşik 0.95: ölçülen veri setlerinde meşru-ama-baskın feature'lar 0.27-0.69 bandında
    kalırken gerçek kilit 0.97+'ye çıkıyor; 0.95 bu boşluğun ortasına düşüyor. Kazanç
    hem prior'a hem örneklem büyüklüğüne göre düzeltildiği için eşik veri setinden
    bağımsız olarak aynı şeyi ifade eder — dataset'e özel ayar gerekmez.

    Confidence ayrımı (lokal sıçrama analizi):
      - high: şüpheli dilim komşusundan ANİ kopuyor → yapay leakage şüphesi
      - low:  şüpheli dilim komşusuna KADEMELİ bağlanıyor → meşru güçlü sinyal
    """
    concerns = []
    y = np.asarray(y)
    n = len(y)

    is_class = intent.strip().lower() == "classification"
    global_std = float(np.std(y)) if not is_class else 0.0

    priors: dict = {}
    if is_class:
        priors = pd.Series(y).value_counts(normalize=True).to_dict()
        # Tek sınıflı hedefte "kilit" tanımsız — ölçecek bir belirsizlik yok.
        if len(priors) < 2:
            return []

    for feat, imp in feature_importance.items():
        if imp < importance_threshold:
            continue
        if feat not in X.columns:
            continue
        col = X[feat]
        if col.nunique() < n_bins:
            continue

        try:
            bins = pd.qcut(col, q=n_bins, duplicates='drop')
        except Exception:
            continue

        df = pd.DataFrame({'bin': bins, 'y': y})
        
        flagged_bin = None
        flagged_purity = 0.0
        flagged_gain = 0.0
        flagged_baseline = 0.0
        flagged_class = None
        flagged_size = 0

        if is_class:
            bin_stats = df.groupby('bin', observed=True)['y'].agg(['count'])
            gains = []

            for bin_val, row in bin_stats.iterrows():
                grp = df[df['bin'] == bin_val]['y']
                shares = grp.value_counts(normalize=True)

                # Her sınıf için prior'a göre normalize edilmiş saflık kazancı.
                # Ham oran değil, güven alt sınırı kullanılır — küçük dilimde tesadüfen
                # boş çıkan bir sınıf "kesinlik" sayılmasın.
                # Dilimde hiç görünmeyen sınıf p=0 → kazanç negatif → zaten elenir.
                best_gain, best_cls, best_p = -np.inf, None, 0.0
                cnt = int(row['count'])
                for cls, prior in priors.items():
                    if prior >= 1.0:
                        continue
                    p = float(shares.get(cls, 0.0))
                    p_lb = _wilson_lower_bound(p, cnt)
                    g = (p_lb - float(prior)) / (1.0 - float(prior))
                    if g > best_gain:
                        best_gain, best_cls, best_p = g, cls, p
                if best_cls is None:
                    continue

                gains.append(best_gain)

                if (flagged_bin is None
                        and row['count'] >= min_bin_fraction * n
                        and best_gain >= gain_threshold):
                    flagged_bin = bin_val
                    flagged_gain = float(best_gain)
                    flagged_purity = float(best_p)
                    flagged_baseline = float(priors[best_cls])
                    flagged_class = best_cls
                    flagged_size = int(row['count'])

            jump_metrics = np.asarray(gains, dtype=float)
        else:
            if global_std == 0:
                continue
            
            bin_stats = df.groupby('bin', observed=True)['y'].agg(['std', 'count'])
            bin_stats['std'] = bin_stats['std'].fillna(0.0)
            rates = (bin_stats['std'] / global_std).values
            
            for idx, (bin_val, row) in enumerate(bin_stats.iterrows()):
                if row['count'] < min_bin_fraction * n:
                    continue
                oran = rates[idx]
                if oran <= variance_collapse_threshold:
                    flagged_bin = bin_val
                    flagged_purity = float(oran) # We keep the key name as purity for compatibility
                    flagged_size = int(row['count'])
                    break
            jump_metrics = rates

        if flagged_bin is None:
            continue

        bin_list = list(bin_stats.index)
        flagged_idx = bin_list.index(flagged_bin)

        avg_jump = float(np.abs(np.diff(jump_metrics)).mean()) if len(jump_metrics) >= 2 else 0.0

        if flagged_idx > 0:
            local_jump = abs(jump_metrics[flagged_idx] - jump_metrics[flagged_idx - 1])
        elif flagged_idx < len(jump_metrics) - 1:
            local_jump = abs(jump_metrics[flagged_idx] - jump_metrics[flagged_idx + 1])
        else:
            local_jump = 0.0

        local_jump_ratio = float(local_jump / (avg_jump + 1e-9))
        confidence = "high" if local_jump_ratio >= 2.0 else "low"

        if is_class:
            # Ham oranın yanına genel oranı da yazıyoruz: "%99 aynı sonuç" ifadesi
            # tek başına yanıltıcı — dengesiz veride zaten her aralık öyle görünür.
            ctx = (f"bu aralıkta sonuçların %{flagged_purity*100:.1f}'i '{flagged_class}' "
                   f"(veri genelinde %{flagged_baseline*100:.1f})")
            if confidence == "high":
                reason = (
                    f"'{feat}' {ctx} — sonucu neredeyse kesinleştiriyor ve komşu aralıklardan "
                    f"ani şekilde kopuyor. Model gerçek davranıştan değil, bu kopuştan "
                    f"öğreniyor olabilir → olası veri sızıntısı"
                )
            else:
                reason = (
                    f"'{feat}' {ctx} — sonucu neredeyse kesinleştiriyor ama bu kademeli ve "
                    f"tutarlı bir ilişki → büyük olasılıkla gerçek bir sinyal, sızıntı değil"
                )
        else:
            if confidence == "high":
                reason = (
                    f"'{feat}' bu aralıkta hedefin varyansını global varyansın %{flagged_purity*100:.1f}'sine "
                    f"kadar düşürüyor (neredeyse sabitliyor) ve komşu aralıklardan ani şekilde kopuyor. "
                    f"Model gerçek davranıştan değil, bu çöküşten öğreniyor olabilir → olası veri sızıntısı"
                )
            else:
                reason = (
                    f"'{feat}' bu aralıkta hedefin varyansını global varyansın %{flagged_purity*100:.1f}'sine düşürüyor "
                    f"ama bu kademeli ve tutarlı bir ilişki → büyük olasılıkla gerçek bir sinyal, sızıntı değil"
                )

        concern = {
            "feature": feat,
            "importance": round(imp, 4),
            "bin": str(flagged_bin),
            # regresyonda "purity" varyans-çöküşü oranıdır (geriye dönük uyumluluk)
            "purity": round(flagged_purity, 4),
            "bin_size": flagged_size,
            "confidence": confidence,
            "local_jump_ratio": round(local_jump_ratio, 2),
            "reason": reason,
        }
        if is_class:
            concern["purity_gain"] = round(flagged_gain, 4)
            concern["baseline"] = round(flagged_baseline, 4)
            concern["locked_class"] = str(flagged_class)
        concerns.append(concern)

    return concerns

def _build_human_message(state: dict) -> str:
    """LLM'e gönderilecek human mesajını oluşturur."""
    target_def = state.get("target_definition") or {}
    db_intel   = state.get("db_intelligence") or {}
    feature_plan = state.get("feature_plan") or {}

    target_col   = target_def.get("column_name", "?")
    target_desc  = target_def.get("description", "")
    target_table = target_def.get("table_name", "")

    # db_intelligence'tan kolon bilgilerini al
    columns_info = db_intel.get("columns", [])
    for c in columns_info:
        if c.get("table_name") == target_table:
            logger.info(f"[DEBUG] {c.get('column_name')}: semantic_type={c.get('semantic_type')}")
    # Sadece target'in tablosundaki kolonları al (büyük olasılıkla feature'lar buradan)
    relevant_cols = [
        c for c in columns_info
        if c.get("table_name", "") == target_table
        and c.get("column_name", "") != target_col
        and c.get("semantic_type") != "target"
    ]

    feature_lines = []
    for c in relevant_cols:
        name = c.get("column_name", "")
        desc = c.get("description", "")
        dtype = c.get("data_type", "")
        sem = c.get("semantic_type", "")
        feature_lines.append(f"- {name} ({dtype}, {sem}): {desc}")

    feature_block = "\n".join(feature_lines) if feature_lines else "(feature listesi boş)"

    return f"""## Target
- Kolon: {target_col}
- Tablo: {target_table}
- Açıklama: {target_desc}

## Feature Adayları
{feature_block}

## Görev
Bu target'ı tahmin etmek için yukarıdaki feature'lar kullanılacak. Aralarında domain-level leakage var mı? Varsa hangileri, neden, ne kadar emin?
"""


def detect_leakage(state: dict) -> dict:
    """
    Domain-based leakage detection using LLM.

    Çıktı:
        domain_leakage_report: {
            "concerns": [...],
            "overall_assessment": "...",
        }
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"[leakage_detector] BASLIYOR — Domain-based leakage detection")
    logger.info(f"{'='*60}")

    target_def = state.get("target_definition") or {}
    if not target_def.get("column_name"):
        logger.warning(f"[leakage_detector] Target yok, atlaniyor")
        return {"domain_leakage_report": {"concerns": [], "overall_assessment": "Target belirlenmemis, kontrol atlandi"}}

    try:
        system_prompt = load_prompt("leakage_detection_system.md")
        human_msg = _build_human_message(state)

        llm = get_llm(
            model_name=settings.DEFAULT_LLM_MODEL,
            temperature=0.1,  # düşük: tutarlı kararlar
        ).with_structured_output(LeakageReport)

        result: LeakageReport = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_msg),
        ])

        report_dict = result.model_dump()
        n_concerns = len(report_dict.get("concerns", []))

        logger.info(f"[leakage_detector] {n_concerns} suphe tespit edildi")
        for c in report_dict.get("concerns", []):
            logger.info(f"  - {c['column_name']} ({c['relationship_type']}, {c['confidence']}): {c['reasoning'][:80]}")
        logger.info(f"[leakage_detector] Genel: {report_dict.get('overall_assessment', '')}")

        return {"domain_leakage_report": report_dict}

    except Exception as e:
        import traceback
        logger.error(f"[leakage_detector] HATA: {e}")
        logger.info(traceback.format_exc())
        return {
            "domain_leakage_report": {
                "concerns": [],
                "overall_assessment": f"Leakage detection hata verdi: {str(e)}"
            }
        }