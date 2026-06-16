"""The five experts of the Merchant Intelligence Engine.

Each expert carries the domain knowledge that gets injected at the pipeline's
four touch points: intent biasing, query hints, feature hints and an
interpretation lens. Keyword sets (Turkish + English) power the deterministic
routing fallback in :mod:`router`.
"""

from __future__ import annotations

from src.domains.base import Expert

# ── Routing keywords (TR + EN), per expert key ──────────────────────────────
# Used by the deterministic fallback router and to bias the LLM router.
EXPERT_KEYWORDS: dict[str, list[str]] = {
    "payment": [
        "payment", "ödeme", "authorization", "otorizasyon", "approval", "onay",
        "decline", "red", "reddedilen", "settlement", "mutabakat", "retry",
        "auth rate", "onay oranı", "card", "kart", "pos", "issuer", "acquirer",
        "method", "yöntem", "mcc",
    ],
    "fraud": [
        "fraud", "sahte", "sahtekarlık", "dolandırıcılık", "anomaly", "anomali",
        "outlier", "aykırı", "suspicious", "şüpheli", "chargeback", "ters ibraz",
        "risk", "velocity", "hız", "fraudulent", "dolandırıcı", "kara liste",
        "blacklist", "fake",
    ],
    "growth": [
        "growth", "büyüme", "trend", "acquisition", "kazanım", "yeni müşteri",
        "new customer", "onboarding", "aktivasyon", "activation", "forecast",
        "tahmin", "projection", "projeksiyon", "expansion", "volume trend",
        "hacim trendi", "cohort", "kohort", "adoption",
    ],
    "customer": [
        "customer", "müşteri", "segment", "segmentasyon", "segmentation",
        "churn", "kayıp", "terk", "retention", "elde tutma", "clv", "ltv",
        "lifetime value", "yaşam boyu değer", "rfm", "loyalty", "sadakat",
        "behavior", "davranış", "persona",
    ],
    "revenue": [
        "revenue", "gelir", "ciro", "turnover", "sales", "satış", "income",
        "kazanç", "profit", "kar", "kâr", "margin", "marj", "interchange",
        "fee", "komisyon", "ücret", "average ticket", "ortalama sepet",
        "basket", "sepet", "monetization", "gelir kırılımı",
    ],
}


PAYMENT_EXPERT = Expert(
    key="payment",
    name="Payment Expert",
    summary=(
        "Specialist in payment operations: authorization/approval rates, decline "
        "reasons, retries, settlement, and payment-method/MCC mix."
    ),
    signals=[
        "authorization/approval rate", "decline & response codes", "retry & settlement",
        "payment method mix", "merchant category codes (MCC)", "card type (credit/debit)",
    ],
    intent_bias=["eda", "report", "sql_query", "regression"],
    query_hint=(
        "Focus on payment-flow metrics: approval vs. decline counts and rates, "
        "decline-reason breakdowns, retry behavior, settlement timing, and mix by "
        "payment method / card type / MCC. Join transactions to card and MCC tables "
        "when available; aggregate rates as approved/total."
    ),
    feature_hint=(
        "Engineer payment-operations features: per-merchant/per-card approval rate, "
        "decline-reason ratios, retry counts, time-to-settlement, payment-method share, "
        "and MCC category one-hots. Prefer ratios over raw counts to normalize across volume."
    ),
    interpret_lens=(
        "Read results as a payments operations analyst: explain approval/decline dynamics, "
        "flag method- or MCC-specific friction, and recommend concrete operational levers "
        "(retry strategy, routing, issuer-specific handling)."
    ),
)

FRAUD_EXPERT = Expert(
    key="fraud",
    name="Fraud Expert",
    summary=(
        "Specialist in fraud & risk: anomalous transactions, velocity, geo/device "
        "mismatch, chargebacks, and supervised fraud detection."
    ),
    signals=[
        "fraud labels", "transaction velocity (count/amount over time windows)",
        "geo/device mismatch", "chargebacks", "high-risk MCC", "amount outliers",
    ],
    intent_bias=["anomaly_detection", "classification"],
    query_hint=(
        "Focus on risk signals: transaction velocity over time windows (count and amount "
        "per card/account per hour/day), unusually large amounts, repeated declines, "
        "high-risk MCC concentration, and chargeback rates. When a fraud label table exists "
        "(e.g. train_fraud_labels), join it as the outcome."
    ),
    feature_hint=(
        "Engineer fraud-detection features WITHOUT leakage: rolling transaction velocity "
        "(count/sum over trailing windows), amount z-score vs. the card's own history, "
        "time-since-last-transaction, geo/device change flags, and high-risk MCC indicators. "
        "Use train_fraud_labels as the target. Never derive features from the label itself."
    ),
    interpret_lens=(
        "Read results as a fraud analyst: quantify detection performance in business terms "
        "(caught fraud vs. false positives / customer friction), name the strongest risk "
        "drivers, and propose monitoring thresholds and review rules."
    ),
)

GROWTH_EXPERT = Expert(
    key="growth",
    name="Growth Expert",
    summary=(
        "Specialist in growth: volume/customer trends, acquisition & activation, "
        "cohorts, and forecasting."
    ),
    signals=[
        "transaction volume over time", "new vs. returning customers", "activation",
        "cohort retention", "seasonality", "forecastable time series",
    ],
    intent_bias=["regression", "report", "eda", "sql_query"],
    query_hint=(
        "Focus on growth trends: transaction/customer volume by period (day/week/month), "
        "new vs. returning split, activation funnels, and cohort tables. Use date/time "
        "columns with GROUP BY time bucket and ORDER BY time."
    ),
    feature_hint=(
        "Engineer growth features: period-over-period volume change, customer tenure, "
        "activation recency, rolling growth rates, and seasonality (day-of-week, month) "
        "encodings. For forecasting, build lag and moving-average features."
    ),
    interpret_lens=(
        "Read results as a growth analyst: describe the trajectory and inflection points, "
        "separate acquisition from expansion effects, and recommend where growth is "
        "accelerating or stalling."
    ),
)

CUSTOMER_EXPERT = Expert(
    key="customer",
    name="Customer Expert",
    summary=(
        "Specialist in customer analytics: segmentation (RFM), churn, retention, "
        "and lifetime value."
    ),
    signals=[
        "recency / frequency / monetary (RFM)", "churn", "retention", "CLV/LTV",
        "demographics", "behavioral segments",
    ],
    intent_bias=["clustering", "classification", "regression"],
    query_hint=(
        "Focus on customer-level aggregates: per-customer recency (days since last "
        "transaction), frequency (transaction count), and monetary (total/avg amount). "
        "Join users/cards to transactions and GROUP BY customer."
    ),
    feature_hint=(
        "Engineer customer features at the customer grain: RFM (recency, frequency, "
        "monetary), tenure, category diversity, average basket, and trend in activity. "
        "For churn, define the label from an inactivity window; for segmentation, scale "
        "RFM features for clustering."
    ),
    interpret_lens=(
        "Read results as a customer analytics lead: describe the segments or churn drivers "
        "in plain business language, size each group, and recommend targeted actions "
        "(retention, upsell, win-back)."
    ),
)

REVENUE_EXPERT = Expert(
    key="revenue",
    name="Revenue Expert",
    summary=(
        "Specialist in revenue: net volume, interchange/fees, average ticket, and "
        "revenue breakdowns/forecasts."
    ),
    signals=[
        "net transaction volume", "interchange / fees", "average ticket size",
        "revenue by segment/MCC/method", "margin",
    ],
    intent_bias=["regression", "report", "eda", "sql_query"],
    query_hint=(
        "Focus on monetary outcomes: total and average transaction amount, revenue by "
        "MCC / method / segment / period, and fee or interchange where present. Use SUM/AVG "
        "with GROUP BY the relevant dimension and ORDER BY revenue DESC."
    ),
    feature_hint=(
        "Engineer revenue features: total/avg/median ticket per merchant or customer, "
        "revenue concentration (share of top categories), fee ratios, and period revenue "
        "trends. For forecasting build lagged revenue and moving averages."
    ),
    interpret_lens=(
        "Read results as a revenue/finance analyst: attribute revenue to its main drivers "
        "(volume vs. ticket vs. mix), highlight concentration and risk, and recommend where "
        "to defend or grow revenue."
    ),
)


ALL_EXPERTS = [
    PAYMENT_EXPERT,
    FRAUD_EXPERT,
    GROWTH_EXPERT,
    CUSTOMER_EXPERT,
    REVENUE_EXPERT,
]
