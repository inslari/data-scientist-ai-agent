from ..state import AgentState
from ..core.llm import get_llm
from ..core.config import settings
from ..core.prompt_loader import create_chat_prompt
from ..schemas.planner_schemas import AnalysisPlan


def planner_node(state: AgentState) -> AgentState:
    """
    Node 3: Takes the database intelligence and user query as input and produces an analysis plan.
    """
    print("\n========================================\n[NODE START] planner_node başlıyor...\n========================================")
    llm = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(AnalysisPlan)

    db_intelligence = state.get("db_intelligence", {})
    intent = state.get("intent", "eda")
    user_query = state.get("user_query", "")
    relevant_tables = state.get("relevant_tables", [])

    tables_summary = _summarize_db_intelligence(db_intelligence, relevant_tables)

    PLANNER_PROMPT = create_chat_prompt("planner_system.md", "planner_human.md")
    chain = PLANNER_PROMPT | structured_llm

    result: AnalysisPlan = chain.invoke({  # type: ignore[assignment]
        "user_query": user_query,
        "intent": intent,
        "relevant_tables": relevant_tables,
        "tables_summary": tables_summary
    })

    plan = sorted(
        [step.model_dump() for step in result.steps],
        key=lambda s: s["order"],
    )

    return {
        **state,
        "analysis_plan": plan,
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