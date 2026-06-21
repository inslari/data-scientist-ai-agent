"""The experts of the Healthcare Intelligence Engine.

Each expert carries the domain knowledge injected at the pipeline's four touch
points (intent biasing, query hints, feature hints, interpretation lens). Keyword
sets (Turkish + English) power the deterministic routing fallback in the generic
router.
"""

from __future__ import annotations

from src.domains.base import Expert

# ── Routing keywords (TR + EN), per expert key ──────────────────────────────
EXPERT_KEYWORDS: dict[str, list[str]] = {
    "survival": [
        "survival", "sağkalım", "sagkalim", "kaplan-meier", "kaplan meier",
        "hazard", "time-to-event", "olaya kadar", "censoring", "sansür",
        "sansur", "life table", "yaşam tablosu", "median survival",
        "ortanca sağkalım", "cox", "log-rank",
    ],
    "risk": [
        "risk score", "risk skoru", "risk stratification", "risk sınıflama",
        "risk siniflama", "comorbidity", "komorbidite", "severity", "şiddet",
        "siddet", "scoring model", "skor modeli", "mortality risk", "ölüm riski",
        "olum riski", "acuity", "early warning", "erken uyarı",
    ],
    "coding": [
        "icd", "icd-10", "icd-9", "diagnosis code", "tanı kodu", "tani kodu",
        "cpt", "procedure code", "işlem kodu", "islem kodu", "coding",
        "kodlama", "billing code", "drg", "mapping", "eşleme", "esleme",
        "snomed", "diagnosis grouping",
    ],
    "readmission": [
        "readmission", "yeniden yatış", "yeniden yatis", "tekrar yatış",
        "tekrar yatis", "30-day", "30 günlük", "30 gunluk", "bounce-back",
        "return visit", "geri dönüş yatışı", "avoidable admission",
        "length of stay", "yatış süresi", "yatis suresi", "discharge",
        "taburcu",
    ],
    "cohort": [
        "cohort", "kohort", "outcome", "sonuç", "sonuc", "patient group",
        "hasta grubu", "treatment effect", "tedavi etkisi", "prognosis",
        "prognoz", "epidemiolog", "baseline characteristics",
        "başlangıç özellikleri", "incidence", "insidans", "prevalence",
        "prevalans",
    ],
}


SURVIVAL_EXPERT = Expert(
    key="survival",
    name="Survival Analysis Expert",
    summary=(
        "Specialist in time-to-event analysis: Kaplan-Meier curves, hazard rates, "
        "censoring and median survival across patient groups."
    ),
    signals=[
        "time-to-event / duration", "event vs. censored indicator",
        "survival probability over time", "hazard / risk over time",
        "median survival", "group survival differences (log-rank)",
    ],
    intent_bias=["deep_analysis", "regression", "report"],
    query_hint=(
        "Focus on duration and event columns: per-subject follow-up time and a binary "
        "event/censoring flag. Compute event rates, median time-to-event, and survival "
        "by cohort. Never treat censored subjects as event-free completers."
    ),
    feature_hint=(
        "Engineer survival features WITHOUT leaking the outcome: baseline covariates at "
        "time-zero, follow-up duration, and a clean event indicator. Keep time and event "
        "as separate fields; do not derive predictors from post-event observations."
    ),
    interpret_lens=(
        "Read results as a clinical biostatistician: report median survival and event "
        "rates per group, name which cohorts diverge, and frame differences in absolute "
        "and relative survival terms with appropriate caution about censoring."
    ),
)

RISK_EXPERT = Expert(
    key="risk",
    name="Clinical Risk Scoring Expert",
    summary=(
        "Specialist in patient risk stratification: severity/comorbidity scoring, "
        "mortality and deterioration risk, and early-warning thresholds."
    ),
    signals=[
        "risk / severity scores", "comorbidity burden", "vital-sign abnormality",
        "mortality / deterioration risk", "risk tiers (low/med/high)",
        "calibration of predicted risk",
    ],
    intent_bias=["classification", "regression", "report"],
    query_hint=(
        "Focus on per-patient risk drivers: comorbidities, severity indicators and "
        "outcome flags. Aggregate event rates by risk tier and check that higher tiers "
        "carry higher observed risk (monotonicity)."
    ),
    feature_hint=(
        "Engineer risk features at the patient grain: comorbidity counts, severity "
        "indices, abnormal-vital flags and prior-utilization. Bin continuous risk into "
        "interpretable tiers; never include the outcome or post-outcome data as a feature."
    ),
    interpret_lens=(
        "Read results as a clinical risk lead: report discrimination and calibration in "
        "plain terms, name the strongest risk drivers, and recommend actionable risk "
        "thresholds for escalation."
    ),
)

CODING_EXPERT = Expert(
    key="coding",
    name="Medical Coding & ICD Expert",
    summary=(
        "Specialist in diagnosis/procedure coding: ICD/CPT/DRG mapping, code grouping, "
        "and turning raw codes into analyzable categories."
    ),
    signals=[
        "ICD / CPT / DRG codes", "code → chapter/category mapping",
        "diagnosis grouping", "coding completeness", "comorbidity flags from codes",
        "billing / case-mix groups",
    ],
    intent_bias=["eda", "report", "sql_query", "classification"],
    query_hint=(
        "Focus on code columns: parse and group ICD/CPT codes into clinical chapters or "
        "comorbidity flags, count the most frequent codes, and join codes to patient "
        "encounters. Treat codes as categorical, not numeric."
    ),
    feature_hint=(
        "Engineer coding features: one-hot or grouped indicators for major diagnosis "
        "chapters, comorbidity flags derived from code sets, and code counts per "
        "encounter. Prefer grouped chapters over thousands of raw code dummies."
    ),
    interpret_lens=(
        "Read results as a clinical coding analyst: name the dominant diagnosis groups, "
        "flag coding gaps or anomalies, and explain how the code mix shapes the case-mix "
        "and downstream models."
    ),
)

READMISSION_EXPERT = Expert(
    key="readmission",
    name="Readmission & LOS Expert",
    summary=(
        "Specialist in readmission and length-of-stay: 30-day readmission prediction, "
        "avoidable returns, discharge risk and LOS drivers."
    ),
    signals=[
        "30-day readmission flag", "length of stay (LOS)", "discharge disposition",
        "prior admissions", "post-discharge follow-up", "avoidable returns",
    ],
    intent_bias=["classification", "regression", "report"],
    query_hint=(
        "Focus on encounter-level readmission and LOS: define a readmission label from a "
        "fixed window (e.g. 30 days), compute LOS, and rate readmission by service line "
        "and discharge disposition. Anchor the window to the index discharge date."
    ),
    feature_hint=(
        "Engineer readmission features WITHOUT leakage: index-stay LOS, prior admission "
        "count, discharge disposition, comorbidity burden and index diagnosis group. "
        "Build the label from the post-discharge window only — never from the index stay's outcome."
    ),
    interpret_lens=(
        "Read results as a care-transitions analyst: quantify readmission risk and LOS "
        "drivers in operational terms, name avoidable-return patterns, and recommend "
        "discharge and follow-up interventions."
    ),
)

COHORT_EXPERT = Expert(
    key="cohort",
    name="Cohort & Outcomes Expert",
    summary=(
        "Specialist in cohort and outcomes analysis: defining patient groups, baseline "
        "characteristics, treatment effects and outcome comparisons."
    ),
    signals=[
        "cohort definition / inclusion criteria", "baseline characteristics",
        "outcome rates per cohort", "treatment vs. control", "incidence / prevalence",
        "prognostic factors",
    ],
    intent_bias=["deep_analysis", "eda", "report", "classification"],
    query_hint=(
        "Focus on cohort construction and outcomes: group patients by inclusion criteria, "
        "summarize baseline characteristics, and compare outcome rates across cohorts. "
        "Keep cohort definitions explicit and reproducible."
    ),
    feature_hint=(
        "Engineer cohort features at the patient grain: baseline demographics and "
        "clinical characteristics measured at cohort entry, plus exposure/treatment "
        "indicators. Measure covariates at baseline only to avoid outcome leakage."
    ),
    interpret_lens=(
        "Read results as a clinical epidemiologist: describe each cohort's size and "
        "baseline profile, compare outcomes in absolute and relative terms, and flag "
        "confounders before implying any causal effect."
    ),
)


ALL_EXPERTS = [
    SURVIVAL_EXPERT,
    RISK_EXPERT,
    CODING_EXPERT,
    READMISSION_EXPERT,
    COHORT_EXPERT,
]
