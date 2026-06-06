from __future__ import annotations
from typing import Any

from src.core.llm import get_llm
from src.core.config import settings
from src.core.prompt_loader import load_prompt
from src.schemas.leakage_schemas import LeakageReport
from langchain_core.messages import SystemMessage, HumanMessage


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