from __future__ import annotations
from typing import Any

from src.core.llm import get_llm
from src.core.config import settings
from src.core.prompt_loader import load_prompt
from src.schemas.leakage_schemas import LeakageReport
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
import pandas as pd



def detect_target_lock(X: pd.DataFrame, y: np.ndarray,
                       feature_importance: dict,
                       intent: str = "classification",
                       importance_threshold: float = 0.05,
                       purity_threshold: float = 0.97,
                       min_bin_fraction: float = 0.05,
                       variance_collapse_threshold: float = 0.05,
                       n_bins: int = 10) -> list[dict]:
    """
    Yüksek-importance sayısal feature'ları dilimlere böler.
    Classification: Bir dilimde target tek sınıfa kilitleniyorsa (purity > eşik)
    Regression: Bir dilimde target varyansı global varyansa göre çöküyorsa (oran < eşik)
    ve o dilim yeterince büyükse (> min_bin_fraction), sızıntı şüphesi raporlar.

    Confidence ayrımı (lokal sıçrama analizi):
      - high: şüpheli dilim komşusundan ANİ kopuyor → yapay leakage şüphesi
      - low:  şüpheli dilim komşusuna KADEMELİ bağlanıyor → meşru güçlü sinyal
    """
    concerns = []
    y = np.asarray(y)
    n = len(y)
    
    is_class = intent.strip().lower() == "classification"
    global_std = float(np.std(y)) if not is_class else 0.0

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
        flagged_size = 0

        if is_class:
            bin_stats = df.groupby('bin', observed=True)['y'].agg(['mean', 'count'])
            rates = bin_stats['mean'].values
            
            for bin_val, row in bin_stats.iterrows():
                if row['count'] < min_bin_fraction * n:
                    continue
                grp = df[df['bin'] == bin_val]['y']
                purity = grp.value_counts(normalize=True).max()
                if purity >= purity_threshold:
                    flagged_bin = bin_val
                    flagged_purity = float(purity)
                    flagged_size = int(row['count'])
                    break
            jump_metrics = rates
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
            if confidence == "high":
                reason = (
                    f"'{feat}' bu aralıkta sonucu %{flagged_purity*100:.0f} oranında "
                    f"tek bir değere kilitliyor ve komşu aralıklardan ani şekilde kopuyor. "
                    f"Model gerçek davranıştan değil, bu kopuştan öğreniyor olabilir → olası veri sızıntısı"
                )
            else:
                reason = (
                    f"'{feat}' bu aralıkta sonucu %{flagged_purity*100:.0f} oranında belirliyor "
                    f"ama bu kademeli ve tutarlı bir ilişki → büyük olasılıkla gerçek bir sinyal, sızıntı değil"
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

        concerns.append({
            "feature": feat,
            "importance": round(imp, 4),
            "bin": str(flagged_bin),
            "purity": round(flagged_purity, 4),
            "bin_size": flagged_size,
            "confidence": confidence,
            "local_jump_ratio": round(local_jump_ratio, 2),
            "reason": reason,
        })

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
            print(f"[DEBUG] {c.get('column_name')}: semantic_type={c.get('semantic_type')}")
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
    print(f"\n{'='*60}")
    print(f"[leakage_detector] BASLIYOR — Domain-based leakage detection")
    print(f"{'='*60}")

    target_def = state.get("target_definition") or {}
    if not target_def.get("column_name"):
        print(f"[leakage_detector] Target yok, atlaniyor")
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

        print(f"[leakage_detector] {n_concerns} suphe tespit edildi")
        for c in report_dict.get("concerns", []):
            print(f"  - {c['column_name']} ({c['relationship_type']}, {c['confidence']}): {c['reasoning'][:80]}")
        print(f"[leakage_detector] Genel: {report_dict.get('overall_assessment', '')}")

        return {"domain_leakage_report": report_dict}

    except Exception as e:
        import traceback
        print(f"[leakage_detector] HATA: {e}")
        print(traceback.format_exc())
        return {
            "domain_leakage_report": {
                "concerns": [],
                "overall_assessment": f"Leakage detection hata verdi: {str(e)}"
            }
        }