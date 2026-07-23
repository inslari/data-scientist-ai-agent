import json
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from src.graphs.talktodata.utils import _get_llm, _build_column_profiles, _StructuredJsonParser
from src.domains import get_active_pack
from src.domains.base import MerchantContext

from ..state import VizState, _VizPlanResult, _VizSpec

from src.core.logging_config import get_logger
logger = get_logger(__name__)

VIZ_PLAN_SYSTEM = """\
You are a senior data-visualization expert. Given a database schema and a user's request
(or a pre-selected suggestion), you design ONE meaningful chart and the SQL query that
feeds it.

Return a `spec` with:
- chart_type: see the chart-type guidance below
- x: the category / axis / label column (or the numeric column for `hist`/`box`)
- y: the numeric value column (leave null for hist/box where only one column is needed)
- group: optional column to split series by (grouped_bar, stacked100)
- agg: none | sum | mean | count  (aggregation of y grouped by x; use `none` for raw rows)
- title: a short, human title for the chart
- sql: a SINGLE SELECT that returns EXACTLY the columns the chart needs
- subtitle, x_label, y_label, value_format, currency, sort, top_n, data_labels, target,
  compare_to: presentation fields, see below

Presentation fields (they decide whether a non-technical reader understands the chart):
- x_label / y_label: what a business reader would call it — "Bölge", "Toplam gelir" —
  NEVER the raw column name like `avg_balance` or `geography`.
- value_format: number | percent | currency | compact. Use `currency` for money (and set
  `currency` to the symbol), `percent` for rates/shares, `compact` for large counts.
- sort: desc | asc | none. Rank comparisons desc so the biggest bar is read first;
  keep `none` for time series and any naturally ordered axis.
- top_n: cap the categories when there would be more than ~8; the rest merge into "Other".
- data_labels: true when there are few enough marks that the value fits on each one.
- target: a reference value to draw as a line (trend) or compare against (kpi).
- compare_to: a column holding the previous-period value, for the KPI delta badge.

Chart-type guidance — pick by WHAT THE REQUEST ASKS, and default to the business form.
The reader is usually non-technical, so a statistical chart must be EARNED by the request:

Business forms (the default — use one of these unless the request is explicitly statistical):
- A single headline number (total, rate, average) → kpi (add compare_to or target when a
  previous period / goal exists)
- Category comparison → bar; long category names or more than ~6 categories → hbar
- Time / ordered trend → line; a trend judged against a goal → trend (set `target`)
- Share of a whole, ≤ 6 slices → donut (preferred over pie)
- Composition compared ACROSS categories → stacked100 (x, group, y)
- Sequential stages that shrink (conversion, pipeline) → funnel
- What drove a change, as +/- contributions → waterfall
- Category split by a second category → grouped_bar

Statistical forms — ONLY when the request is genuinely about the shape of the data, i.e. it
asks about distribution, spread, outliers, correlation or density (words like "dağılım",
"distribution", "korelasyon", "ilişki", "aykırı", "outlier", "yayılım", "histogram"):
- Relationship between two numerics → scatter
- Distribution of one numeric column → hist (x = that column, y = null, agg = none)
- Spread / outliers of a numeric column, optionally by category → box
- Two categoricals × a numeric intensity → heatmap (x, y categorical, value numeric)
If the request is a plain comparison ("gelir bölgelere göre"), a statistical chart is the
WRONG answer — use the business form.

Presentation is NOT optional, whichever form you pick:
- title reads like a plain sentence a person would say, not `avg_balance by geography`.
- ALWAYS fill x_label and y_label with human-readable wording.
- ALWAYS choose a value_format that matches the unit; set sort=desc for rankings.
- Keep categorical charts to at most 8 categories (use top_n); set data_labels=true when
  there are ≤ 12 marks.

SQL rules (SQLite/standard SQL):
- SELECT only. Use GROUP BY for aggregations.
- For scatter/hist/box return raw rows and append LIMIT 5000.
- For kpi return ONE row (and a second column when you set compare_to).
- Use column names EXACTLY as in the schema (case-sensitive); wrap names with spaces/dots in double quotes.
- Prefer columns that TRULY match the concept (right type and unit); do not pick a proxy column.

If the request is too vague to choose columns confidently, set clarification_needed = true
and provide a short clarification_question; still return a best-effort `spec`.

--- Raw Schema ---
{schema}

--- Column Semantic Profiles ---
{column_profiles}

{domain_hint}

--- User request ---
{user_request}

{fix_hint}
"""

VIZ_PLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", VIZ_PLAN_SYSTEM),
    ("human", "Design the visualization specification. Return structured output."),
])


def _describe_request(state: VizState) -> str:
    """Serbest NL isteği, seçilen kartı veya iyileştirme isteğini LLM metnine çevirir."""
    # Refine: mevcut grafiği bağlam ver, istenen DEĞİŞİKLİĞİ uygulat
    refine = state.get("refine_context")
    if refine:
        prev_spec = {k: v for k, v in (refine.get("spec") or {}).items() if v not in (None, "", [])}
        return (
            "The user is MODIFYING an existing chart, not asking for a new one.\n\n"
            f"Current chart spec:\n{json.dumps(prev_spec, ensure_ascii=False, indent=2)}\n\n"
            f"Current SQL:\n{refine.get('sql') or '(none)'}\n\n"
            f"It originally answered: {refine.get('original_request') or '(unknown)'}\n\n"
            f"Requested change: {state.get('user_query') or ''}\n\n"
            "Apply ONLY this change and keep everything else from the current spec — same "
            "columns, title, labels and formatting unless the change requires otherwise. "
            "Rewrite the SQL only if the change actually needs different data."
        )

    suggestion = state.get("selected_suggestion")
    if suggestion:
        parts = [f"The user selected a suggested visualization:"]
        for key in ("title", "description", "chart_type", "x", "y", "agg"):
            val = suggestion.get(key)
            if val:
                parts.append(f"  - {key}: {val}")
        parts.append("Build exactly this chart. Fill in / correct any missing fields using the schema.")
        return "\n".join(parts)
    return state.get("user_query") or "Produce a useful overview visualization of this dataset."


async def viz_plan_node(state: VizState, config: RunnableConfig) -> VizState:
    """
    Şema + kullanıcı isteği (veya seçilen öneri) → _VizSpec (matplotlib tipi, kolonlar,
    aggregation, başlık, çizim SQL'i) üretir. Belirsizse clarification_needed set eder.
    """
    logger.info("\n[VIZ] viz_plan_node başlıyor...")

    llm             = _get_llm(temperature=0)
    db_intelligence = state.get("db_intelligence") or {}
    raw_schema      = state.get("raw_schema") or "Schema not available."
    column_profiles = _build_column_profiles(db_intelligence)

    # Domain pack query ipuçları (varsa)
    domain_hint = ""
    try:
        pack = get_active_pack(state.get("domain_pack"))
        mc = MerchantContext.from_dict(state.get("merchant_context"))
        if pack.name != "none" and mc is not None:
            domain_hint = pack.query_hints(mc, db_intelligence) or ""
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[VIZ] Domain pack ipucu başarısız ({e}); atlanıyor.")

    # Retry: önceki SQL hatası / validator düzeltme ipucu
    fix_parts = []
    if state.get("sql_error"):
        fix_parts.append(
            f"[URGENT] The previous SQL failed:\nQuery: {state.get('sql_query_raw')}\n"
            f"Error: {state['sql_error']}\nRewrite the sql to fix this."
        )
    if state.get("validation_fix_hint"):
        fix_parts.append(
            f"[VALIDATOR FEEDBACK] The previous result was rejected "
            f"(issue: {state.get('validation_issue')}).\nSuggested fix: {state['validation_fix_hint']}"
        )
    fix_hint = "\n\n".join(fix_parts)

    invoke_kwargs = {
        "schema":          raw_schema,
        "column_profiles": column_profiles,
        "user_request":    _describe_request(state),
        "domain_hint":     domain_hint,
        "fix_hint":        fix_hint,
    }

    plan: _VizPlanResult
    try:
        # Strategy 1: tool-call structured output
        plan = (VIZ_PLAN_PROMPT | llm.with_structured_output(_VizPlanResult)).invoke(invoke_kwargs)  # type: ignore[assignment]
    except Exception as e1:
        logger.error(f"[VIZ] plan structured output failed ({type(e1).__name__}), trying text parser")
        try:
            # Strategy 2: text-based JSON parser
            raw_text = llm.invoke(VIZ_PLAN_PROMPT.format_messages(**invoke_kwargs)).content
            plan = _StructuredJsonParser(pydantic_object=_VizPlanResult).parse(raw_text)  # type: ignore[assignment]
        except Exception as e2:
            logger.error(f"[VIZ] plan text parser failed ({e2}); clarification'a düşülüyor")
            return {
                **state,
                "clarification_needed":   True,
                "clarification_question": "I couldn't determine which columns to chart. Which metric and dimension would you like to see?",
                "viz_spec":               None,
                "current_step":           "viz_plan_node",
            }  # type: ignore[return-value]

    spec: _VizSpec = plan.spec
    logger.info(
        f"[VIZ] plan: type={spec.chart_type} x={spec.x} y={spec.y} agg={spec.agg} "
        f"clarify={plan.clarification_needed}"
    )

    return {
        **state,
        "viz_spec":               spec.model_dump(),
        "sql_query_raw":          spec.sql,      # sql_node bunu çalıştıracak
        "clarification_needed":   plan.clarification_needed,
        "clarification_question": plan.clarification_question,
        "current_step":           "viz_plan_node",
    }  # type: ignore[return-value]
