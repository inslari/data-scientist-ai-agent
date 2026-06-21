"""The experts of the HR / People Analytics Intelligence Engine.

Each expert carries the domain knowledge injected at the pipeline's four touch
points (intent biasing, query hints, feature hints, interpretation lens). Keyword
sets (Turkish + English) power the deterministic routing fallback in the generic
router.
"""

from __future__ import annotations

from src.domains.base import Expert

# ── Routing keywords (TR + EN), per expert key ──────────────────────────────
EXPERT_KEYWORDS: dict[str, list[str]] = {
    "attrition": [
        "attrition", "ayrılma", "ayrilma", "işten ayrılma", "isten ayrilma",
        "turnover", "devir", "çalışan kaybı", "calisan kaybi", "churn",
        "resignation", "istifa", "quit", "terminate", "ayrılan", "leaver",
        "flight risk", "kaçış riski", "kacis riski",
    ],
    "performance": [
        "performance", "performans", "kpi", "rating", "değerlendirme",
        "degerlendirme", "appraisal", "productivity", "verimlilik",
        "goal attainment", "hedef gerçekleşme", "high performer",
        "yüksek performans", "yuksek performans", "review score", "başarı",
        "basari",
    ],
    "recruitment": [
        "recruitment", "işe alım", "ise alim", "hiring", "funnel", "huni",
        "candidate", "aday", "applicant", "başvuru", "basvuru", "time-to-hire",
        "sourcing", "interview", "mülakat", "mulakat", "offer acceptance",
        "teklif kabul", "pipeline",
    ],
    "compensation": [
        "compensation", "ücret", "ucret", "maaş", "maas", "salary", "pay gap",
        "ücret farkı", "ucret farki", "bonus", "prim", "equity", "benefits",
        "yan haklar", "raise", "zam", "pay band", "market rate", "piyasa ücreti",
    ],
    "engagement": [
        "engagement", "bağlılık", "baglilik", "satisfaction", "memnuniyet",
        "enps", "survey", "anket", "morale", "moral", "well-being", "refah",
        "sentiment", "absenteeism", "devamsızlık", "devamsizlik", "burnout",
        "tükenmişlik", "tukenmislik",
    ],
}


ATTRITION_EXPERT = Expert(
    key="attrition",
    name="Attrition & Turnover Expert",
    summary=(
        "Specialist in employee attrition: turnover and resignation prediction, "
        "flight-risk drivers and retention."
    ),
    signals=[
        "attrition / turnover flag", "tenure", "voluntary vs. involuntary exit",
        "flight-risk drivers", "engagement & comp signals", "manager / team effects",
    ],
    intent_bias=["classification", "report", "deep_analysis"],
    query_hint=(
        "Focus on who leaves and why: per-employee attrition flag, tenure, and rates by "
        "department, role and manager. Compare leavers vs. stayers on tenure, pay and "
        "engagement. Define attrition from a clear exit event."
    ),
    feature_hint=(
        "Engineer attrition features at the employee grain WITHOUT leakage: tenure, role, "
        "pay relative to band, recent rating, engagement and commute/overtime signals. "
        "Measure features before the exit window — never use post-exit data as a feature."
    ),
    interpret_lens=(
        "Read results as a people-analytics lead: quantify attrition risk and its cost, "
        "name the strongest flight-risk drivers, and recommend targeted retention actions "
        "per segment."
    ),
)

PERFORMANCE_EXPERT = Expert(
    key="performance",
    name="Performance Analysis Expert",
    summary=(
        "Specialist in performance analytics: ratings and KPIs, productivity drivers, "
        "and high- vs. low-performer differentiation."
    ),
    signals=[
        "performance ratings / review scores", "KPI / goal attainment",
        "productivity metrics", "high vs. low performers", "rating distribution",
        "manager calibration",
    ],
    intent_bias=["classification", "regression", "eda", "report"],
    query_hint=(
        "Focus on performance outcomes: distribution of ratings and KPIs, and how they "
        "vary by team, tenure and role. Compare high vs. low performers on measurable "
        "drivers; watch for rating inflation or manager bias."
    ),
    feature_hint=(
        "Engineer performance features at the employee grain: tenure, role, training, "
        "goal-completion, peer/manager signals and prior ratings. Avoid leaking the "
        "current rating period's outcome into its own predictors."
    ),
    interpret_lens=(
        "Read results as a talent-management lead: describe the rating distribution, name "
        "what differentiates high performers, and recommend concrete development and "
        "calibration actions."
    ),
)

RECRUITMENT_EXPERT = Expert(
    key="recruitment",
    name="Recruitment Funnel Expert",
    summary=(
        "Specialist in the hiring funnel: applicant-to-hire conversion, time-to-hire, "
        "sourcing channel quality and offer acceptance."
    ),
    signals=[
        "applicant → interview → offer → hire funnel", "stage conversion rates",
        "time-to-hire", "sourcing channel quality", "offer acceptance rate",
        "drop-off by stage",
    ],
    intent_bias=["classification", "report", "eda", "regression"],
    query_hint=(
        "Focus on funnel metrics: counts and conversion at each hiring stage, time-to-hire "
        "by role, and quality by sourcing channel. Compute stage_n / stage_(n-1) and "
        "offer-acceptance rates."
    ),
    feature_hint=(
        "Engineer recruitment features at the candidate/req grain: stage reached, source "
        "channel, time-in-stage, role and seniority, and screen scores. For offer-accept "
        "prediction use only pre-decision signals."
    ),
    interpret_lens=(
        "Read results as a talent-acquisition lead: pinpoint the biggest funnel drop-offs, "
        "name the best and worst sourcing channels, and recommend concrete pipeline and "
        "offer improvements."
    ),
)

COMPENSATION_EXPERT = Expert(
    key="compensation",
    name="Compensation & Pay-Equity Expert",
    summary=(
        "Specialist in compensation: pay distribution and equity, pay gaps, bonus/benefit "
        "structure and market positioning."
    ),
    signals=[
        "salary / total comp distribution", "pay gaps by group", "pay band position",
        "bonus & benefits", "market-rate comparison", "compa-ratio",
    ],
    intent_bias=["regression", "eda", "deep_analysis", "report"],
    query_hint=(
        "Focus on pay distribution and equity: compare salary by department, level, "
        "gender and tenure, and position against pay bands. Surface gaps after controlling "
        "for role and experience."
    ),
    feature_hint=(
        "Engineer compensation features at the employee grain: base, total comp, compa-"
        "ratio, pay-band position, tenure and role. For equity analysis keep protected "
        "attributes separate and control for legitimate drivers."
    ),
    interpret_lens=(
        "Read results as a total-rewards analyst: report pay distribution and any gaps in "
        "plain terms, distinguish explained from unexplained differences, and recommend "
        "concrete pay actions."
    ),
)

ENGAGEMENT_EXPERT = Expert(
    key="engagement",
    name="Engagement & Sentiment Expert",
    summary=(
        "Specialist in engagement: satisfaction and eNPS, survey sentiment, morale and "
        "burnout/absenteeism signals."
    ),
    signals=[
        "engagement / satisfaction scores", "eNPS", "survey sentiment", "absenteeism",
        "burnout / morale signals", "trend over time",
    ],
    intent_bias=["eda", "classification", "report", "deep_analysis"],
    query_hint=(
        "Focus on engagement signals: distribution and trend of satisfaction/eNPS scores, "
        "differences by team and tenure, and links to absenteeism. Aggregate survey "
        "responses to team and time grain."
    ),
    feature_hint=(
        "Engineer engagement features at the employee/team grain: survey scores, score "
        "trend, absenteeism rate, tenure and manager. Treat sparse survey responses "
        "carefully; avoid leaking later outcomes (e.g. exit) into engagement features."
    ),
    interpret_lens=(
        "Read results as an employee-experience lead: describe engagement levels and "
        "trends, name the teams and drivers most at risk, and recommend concrete "
        "morale and well-being actions."
    ),
)


ALL_EXPERTS = [
    ATTRITION_EXPERT,
    PERFORMANCE_EXPERT,
    RECRUITMENT_EXPERT,
    COMPENSATION_EXPERT,
    ENGAGEMENT_EXPERT,
]
