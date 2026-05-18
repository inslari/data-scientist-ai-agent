from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from state import AgentState


class AnalysisStep(BaseModel):
    model_config = {"extra": "forbid"}

    step: str = Field(
        description="Araç adı: sql_query | profiling | feature_engineering | ml_training | stats_analysis"
    )
    description: str = Field(description="Bu adımın ne yaptığını açıklayan tek cümle (Türkçe olarak yaz)")
    table_names: list[str] = Field(default_factory=list, description="Bu adımda kullanılacak tablolar")
    target_column: Optional[str] = Field(default=None, description="ML görevleri için hedef kolon")
    extra_params: Optional[str] = Field(default=None, description="JSON string olarak ek parametreler")
    order: int = Field(description="1'den başlayan yürütme sırası")


class AnalysisPlan(BaseModel):
    model_config = {"extra": "forbid"}

    steps: list[AnalysisStep]
    reasoning: str = Field(description="Bu planın neden seçildiğini açıklayan Türkçe metin")



SYSTEM_PROMPT = """Sen özerk bir veri bilimi ajanının Analiz Planlayıcısısın.
 
Görevin: Kullanıcının sorusuna, tespit edilen intent'e ve veritabanı zekâ modeline dayanarak adım adım bir yürütme planı üretmek.
 
## Kullanılabilir adımlar (araçlar):
- sql_query           : Veritabanından SQL ile veri çekme
- profiling           : Çekilen verinin istatistiksel profilini çıkarma
- feature_engineering : Dönüştürme, kodlama, ölçekleme, özellik seçimi
- ml_training         : Makine öğrenmesi modeli eğitme (sınıflandırma/regresyon/kümeleme)
- stats_analysis      : ML olmadan istatistiksel testler veya EDA
 
## Kurallar:
- Her zaman sql_query ile başla
- sql_query'yi her zaman profiling takip etmeli
- intent=eda veya report ise: sql_query → profiling → stats_analysis
- intent=classification veya regression ise: sql_query → profiling → feature_engineering → ml_training
- intent=clustering ise: sql_query → profiling → feature_engineering → ml_training
- Planı minimal tut — sadece gerekli adımları ekle
- Kolon adlarını ASLA uydurma — yalnızca db_intelligence'daki kolonları kullan
"""


def planner_node(state: AgentState) -> AgentState:
    """
    Node 3: Takes the database intelligence and user query as input and produces an analysis plan.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(AnalysisPlan)

    db_intelligence = state.get("db_intelligence", {})
    intent = state.get("intent", "eda")
    user_query = state.get("user_query", "")
    relevant_tables = state.get("relevant_tables", [])

    tables_summary = _summarize_db_intelligence(db_intelligence, relevant_tables)

    essages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
Kullanıcı sorusu: {user_query}
Tespit edilen intent: {intent}
İlgili tablolar: {relevant_tables}

Veritabanı zekâ özeti:
{tables_summary}

Minimal ve yürütülebilir bir analiz planı üret.
"""
        ),
    ]

    result: AnalysisPlan = structured_llm.invoke(messages)

    plan = sorted(
        [step.model_dump() for step in result.steps],
        key=lambda s: s["order"],
    )

    return {
        **state,
        "analysis_plan": plan,
        "messages": [m.model_dump() for m in messages],
        "current_step": "planner_node",
    }


def _summarize_db_intelligence(db_intelligence: dict, relevant_tables: list[str]) -> str:
    """
    Convert db intelligence to a summary string to be used as a context for the planner.
    """
    if not db_intelligence:
        return "No database intelligence available."

    lines = []

    # Sadece relevant tablolar
    for table in db_intelligence.get("tables", []):
        if table["table_name"] not in relevant_tables:
            continue
        lines.append(
            f"Table: {table['table_name']} | Entity: {table['inferred_entity']} "
            f"| Quality: {table['data_quality_score']:.2f} "
            f"| Analytics potential: {table['analytics_potential_score']:.2f}"
        )

    # Kolonlar (sadece importance > 0.5)
    lines.append("\nImportant columns:")
    for col in db_intelligence.get("columns", []):
        if col["table_name"] not in relevant_tables:
            continue
        if col.get("importance_score", 0) >= 0.5:
            lines.append(
                f"  {col['table_name']}.{col['column_name']} "
                f"[{col['semantic_type']}] importance={col['importance_score']:.2f}"
            )

    # Capabilities
    caps = db_intelligence.get("capabilities", [])
    if caps:
        lines.append(f"\nDetected capabilities: {', '.join(caps)}")

    # Relationships
    lines.append("\nRelationships:")
    for rel in db_intelligence.get("relationships", []):
        lines.append(
            f"  {rel['source_table']}.{rel['source_column']} → "
            f"{rel['target_table']}.{rel['target_column']} ({rel['relationship_type']})"
        )

    return "\n".join(lines)