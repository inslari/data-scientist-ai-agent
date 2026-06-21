"""The experts of the E-Commerce Intelligence Engine.

Each expert carries the domain knowledge injected at the pipeline's four touch
points (intent biasing, query hints, feature hints, interpretation lens). Keyword
sets (Turkish + English) power the deterministic routing fallback in the generic
router.
"""

from __future__ import annotations

from src.domains.base import Expert

# ── Routing keywords (TR + EN), per expert key ──────────────────────────────
EXPERT_KEYWORDS: dict[str, list[str]] = {
    "basket": [
        "basket", "sepet", "market basket", "association", "birliktelik",
        "cross-sell", "çapraz satış", "co-occurrence", "birlikte alınan",
        "bundle", "paket", "apriori", "lift", "support", "confidence",
        "frequently bought", "birlikte sıkça",
    ],
    "customer": [
        "rfm", "segment", "segmentasyon", "segmentation", "recency", "frequency",
        "monetary", "müşteri", "customer", "clv", "ltv", "lifetime value",
        "yaşam boyu değer", "loyalty", "sadakat", "vip", "persona", "cohort",
        "kohort",
    ],
    "churn": [
        "churn", "kayıp", "terk", "retention", "elde tutma", "repeat",
        "tekrar", "geri dönüş", "win-back", "geri kazanım", "inactive",
        "pasif", "lapsed", "dormant", "uykuda", "reactivation", "abandon",
        "repurchase", "tekrar satın",
    ],
    "recommendation": [
        "recommend", "tavsiye", "öneri", "recommendation", "öneri sistemi",
        "collaborative", "işbirlikçi", "content-based", "içerik tabanlı",
        "affinity", "yakınlık", "similar products", "benzer ürün", "upsell",
        "üst satış", "personalization", "kişiselleştirme", "next best",
    ],
    "conversion": [
        "conversion", "dönüşüm", "funnel", "huni", "cart abandonment",
        "sepet terk", "checkout", "ödeme adımı", "add to cart", "sepete ekle",
        "bounce", "hemen çıkma", "session", "oturum", "click-through", "ctr",
        "landing", "açılış sayfası", "product page", "ürün sayfası",
    ],
}


BASKET_EXPERT = Expert(
    key="basket",
    name="Basket Analysis Expert",
    summary=(
        "Specialist in market-basket analysis: which products are bought together, "
        "association rules (support/confidence/lift), bundling and cross-sell."
    ),
    signals=[
        "order/basket line items", "product co-occurrence", "association rules (lift)",
        "average basket size & value", "category affinity", "bundle opportunities",
    ],
    intent_bias=["eda", "report", "sql_query", "clustering"],
    query_hint=(
        "Focus on order-line composition: group line items by order/basket id, count "
        "product and category co-occurrence, and compute basket size and value. Join "
        "order_items to products/categories; rank co-purchased pairs by frequency and lift."
    ),
    feature_hint=(
        "Engineer basket features at the order or customer grain: basket size (distinct "
        "items), basket value, category diversity, share of top category, and pairwise "
        "co-purchase indicators for high-lift pairs. Prefer ratios/shares over raw counts."
    ),
    interpret_lens=(
        "Read results as a merchandising analyst: name the strongest product/category "
        "affinities, quantify cross-sell lift, and recommend concrete bundles, placements "
        "and 'frequently bought together' rules."
    ),
)

CUSTOMER_EXPERT = Expert(
    key="customer",
    name="Customer & RFM Expert",
    summary=(
        "Specialist in customer analytics: RFM segmentation, lifetime value, loyalty "
        "tiers and cohort behavior."
    ),
    signals=[
        "recency / frequency / monetary (RFM)", "customer lifetime value (CLV)",
        "order count & tenure", "average order value", "loyalty/VIP tiers", "cohorts",
    ],
    intent_bias=["clustering", "regression", "classification"],
    query_hint=(
        "Focus on customer-level aggregates: per-customer recency (days since last order), "
        "frequency (order count), and monetary (total/avg spend). Join customers to orders "
        "and GROUP BY customer; bucket into RFM scores."
    ),
    feature_hint=(
        "Engineer customer features at the customer grain: RFM (recency, frequency, "
        "monetary), tenure, average order value, order cadence, category diversity, and "
        "trend in spend. Scale RFM features for clustering; for CLV build spend/frequency "
        "trends. Never derive features from a future-period label."
    ),
    interpret_lens=(
        "Read results as a CRM lead: describe the segments in plain business language, size "
        "each group and its value, and recommend targeted actions (VIP nurture, upsell, "
        "reactivation) per segment."
    ),
)

CHURN_EXPERT = Expert(
    key="churn",
    name="Retention & Churn Expert",
    summary=(
        "Specialist in retention: repeat-purchase behavior, churn/lapsed prediction, "
        "reactivation and win-back."
    ),
    signals=[
        "inter-purchase time", "repeat vs. one-time buyers", "inactivity window",
        "retention/repeat rate", "tenure", "declining engagement",
    ],
    intent_bias=["classification", "regression", "report"],
    query_hint=(
        "Focus on repeat behavior: per-customer time between orders, days since last order, "
        "repeat-purchase rate, and the share of one-time vs. returning buyers. Define a "
        "churn label from an inactivity window (e.g. no order in N days)."
    ),
    feature_hint=(
        "Engineer churn features WITHOUT leakage: recency, average inter-purchase gap, "
        "order frequency trend, tenure, last-order value, and engagement decay. Build the "
        "label from an inactivity window measured in a holdout period — never from the "
        "same orders used as features."
    ),
    interpret_lens=(
        "Read results as a retention analyst: quantify churn risk in business terms (lost "
        "revenue), name the strongest early-warning drivers, and propose win-back triggers "
        "and retention thresholds."
    ),
)

RECOMMENDATION_EXPERT = Expert(
    key="recommendation",
    name="Product Recommendation Expert",
    summary=(
        "Specialist in recommendation: product affinity, collaborative/content signals, "
        "personalization and next-best-offer."
    ),
    signals=[
        "user-item interactions", "product affinity / similarity", "view→purchase signals",
        "category/brand preference", "popularity & recency of items", "upsell paths",
    ],
    intent_bias=["clustering", "classification", "eda"],
    query_hint=(
        "Focus on user-item interactions: per-customer purchased products/categories, "
        "item popularity, and product-to-product co-purchase. Build a user×item view and "
        "rank candidate items by affinity and recency."
    ),
    feature_hint=(
        "Engineer recommendation features: customer category/brand preference shares, item "
        "popularity and recency, customer-item affinity scores, and price-tier preference. "
        "Avoid leaking the held-out target purchase into the feature window."
    ),
    interpret_lens=(
        "Read results as a personalization lead: explain which affinities drive good "
        "recommendations, where popularity vs. personalization wins, and recommend concrete "
        "next-best-offer and upsell rules."
    ),
)

CONVERSION_EXPERT = Expert(
    key="conversion",
    name="Conversion & Funnel Expert",
    summary=(
        "Specialist in conversion-rate optimization: funnel drop-off, cart abandonment, "
        "session→order conversion and checkout friction."
    ),
    signals=[
        "session→cart→checkout→order funnel", "cart abandonment rate", "conversion rate",
        "bounce rate", "add-to-cart rate", "checkout step drop-off",
    ],
    intent_bias=["classification", "report", "eda", "regression"],
    query_hint=(
        "Focus on funnel metrics: counts and rates at each stage (sessions, add-to-cart, "
        "checkout, completed orders), abandonment between stages, and conversion by traffic "
        "source / device / product page. Aggregate rates as stage_n / stage_(n-1)."
    ),
    feature_hint=(
        "Engineer conversion features: per-session add-to-cart and checkout flags, funnel "
        "stage reached, time-on-step, device/source one-hots, and cart value at abandonment. "
        "For purchase prediction, use only pre-conversion signals (no post-order leakage)."
    ),
    interpret_lens=(
        "Read results as a CRO analyst: pinpoint the biggest funnel drop-offs, quantify "
        "abandonment cost, and recommend concrete checkout/UX levers and re-engagement triggers."
    ),
)


ALL_EXPERTS = [
    BASKET_EXPERT,
    CUSTOMER_EXPERT,
    CHURN_EXPERT,
    RECOMMENDATION_EXPERT,
    CONVERSION_EXPERT,
]
