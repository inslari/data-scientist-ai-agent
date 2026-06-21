"""The experts of the Marketing Intelligence Engine.

Each expert carries the domain knowledge injected at the pipeline's four touch
points (intent biasing, query hints, feature hints, interpretation lens). Keyword
sets (Turkish + English) power the deterministic routing fallback in the generic
router.
"""

from __future__ import annotations

from src.domains.base import Expert

# ── Routing keywords (TR + EN), per expert key ──────────────────────────────
EXPERT_KEYWORDS: dict[str, list[str]] = {
    "campaign": [
        "campaign", "kampanya", "roi", "return on investment", "yatırım getirisi",
        "yatirim getirisi", "roas", "promotion", "promosyon", "ad spend",
        "reklam harcaması", "reklam harcamasi", "uplift", "a/b test", "a-b test",
        "conversion rate", "dönüşüm oranı", "donusum orani",
    ],
    "attribution": [
        "attribution", "atıf", "atif", "ilişkilendirme", "iliskilendirme",
        "multi-touch", "çok temaslı", "cok temasli", "first-touch", "last-touch",
        "touchpoint", "temas noktası", "temas noktasi", "marketing mix",
        "pazarlama karması", "pazarlama karmasi", "path to conversion",
        "dönüşüm yolu", "donusum yolu", "mmm",
    ],
    "ltv": [
        "ltv", "lifetime value", "yaşam boyu değer", "yasam boyu deger", "cac",
        "customer acquisition cost", "müşteri edinme maliyeti",
        "musteri edinme maliyeti", "payback", "geri ödeme", "geri odeme",
        "ltv/cac", "ltv cac", "retention value", "cohort value", "kohort değeri",
    ],
    "segmentation": [
        "segmentation", "segmentasyon", "segment", "persona", "cluster",
        "kümeleme", "kumeleme", "audience", "hedef kitle", "targeting",
        "hedefleme", "lookalike", "benzer kitle", "demographic", "demografik",
        "rfm",
    ],
    "channel": [
        "channel mix", "kanal karması", "kanal karmasi", "channel", "kanal",
        "omnichannel", "çok kanallı", "cok kanalli", "media mix", "medya karması",
        "medya karmasi", "paid search", "ücretli arama", "ucretli arama",
        "organic", "organik", "social", "sosyal medya", "email", "eposta",
        "traffic source", "trafik kaynağı", "trafik kaynagi",
    ],
}


CAMPAIGN_EXPERT = Expert(
    key="campaign",
    name="Campaign ROI Expert",
    summary=(
        "Specialist in campaign performance: ROI/ROAS, promo lift, ad-spend efficiency "
        "and conversion-rate measurement."
    ),
    signals=[
        "campaign spend & revenue", "ROI / ROAS", "conversion rate", "promo lift / uplift",
        "cost per acquisition", "A/B test results",
    ],
    intent_bias=["report", "regression", "eda", "classification"],
    query_hint=(
        "Focus on campaign economics: per-campaign spend, revenue, conversions and ROI/"
        "ROAS. Compute uplift vs. control and cost-per-acquisition; group by campaign and "
        "time window."
    ),
    feature_hint=(
        "Engineer campaign features at the campaign (or campaign×segment) grain: spend, "
        "impressions, CTR, conversion rate, CPA and creative/offer flags. For uplift, "
        "keep treatment/control clean and avoid post-campaign leakage."
    ),
    interpret_lens=(
        "Read results as a performance-marketing lead: rank campaigns by ROI/ROAS, name "
        "what drove winners and losers, and recommend concrete budget reallocation and "
        "offer changes."
    ),
)

ATTRIBUTION_EXPERT = Expert(
    key="attribution",
    name="Attribution Expert",
    summary=(
        "Specialist in attribution: multi-touch credit assignment, path-to-conversion, "
        "and marketing-mix contribution."
    ),
    signals=[
        "touchpoint sequences", "first/last/multi-touch credit", "path to conversion",
        "channel contribution", "assisted conversions", "marketing-mix effects",
    ],
    intent_bias=["report", "deep_analysis", "regression", "eda"],
    query_hint=(
        "Focus on conversion paths: order touchpoints per user, count assisting vs. "
        "converting channels, and distribute credit under first/last/multi-touch rules. "
        "Aggregate channel contribution to conversions and revenue."
    ),
    feature_hint=(
        "Engineer attribution features at the journey grain: touchpoint counts per "
        "channel, position-in-path, time-to-conversion and path length. Respect "
        "chronological order so later touches never leak into earlier-step features."
    ),
    interpret_lens=(
        "Read results as an attribution analyst: compare credit across models, name which "
        "channels are under- or over-valued by last-click, and recommend concrete budget "
        "shifts grounded in assisted contribution."
    ),
)

LTV_EXPERT = Expert(
    key="ltv",
    name="LTV / CAC Expert",
    summary=(
        "Specialist in unit economics: customer lifetime value, acquisition cost, LTV/CAC "
        "ratio and payback period."
    ),
    signals=[
        "customer lifetime value (LTV)", "customer acquisition cost (CAC)", "LTV/CAC ratio",
        "payback period", "retention / cohort value", "margin per customer",
    ],
    intent_bias=["regression", "deep_analysis", "report", "clustering"],
    query_hint=(
        "Focus on unit economics: per-customer revenue/margin over time vs. acquisition "
        "cost by channel/cohort. Compute LTV, CAC, the LTV/CAC ratio and payback period; "
        "group by acquisition cohort."
    ),
    feature_hint=(
        "Engineer LTV features at the customer grain: early-window spend/frequency, "
        "acquisition channel, margin and retention signals. Predict LTV from early "
        "behavior only — never from the full realized lifetime you are trying to estimate."
    ),
    interpret_lens=(
        "Read results as a growth/finance analyst: report LTV, CAC and payback by channel "
        "and cohort, name where acquisition is profitable vs. underwater, and recommend "
        "concrete spend and retention moves."
    ),
)

SEGMENTATION_EXPERT = Expert(
    key="segmentation",
    name="Customer Segmentation Expert",
    summary=(
        "Specialist in audience segmentation: clustering and personas, RFM and "
        "demographic targeting, and lookalike audiences."
    ),
    signals=[
        "behavioral / demographic segments", "RFM", "cluster personas",
        "audience size & value", "targeting / lookalike", "segment migration",
    ],
    intent_bias=["clustering", "eda", "classification", "report"],
    query_hint=(
        "Focus on customer-level aggregates to segment: behavioral (RFM, engagement) and "
        "demographic attributes. Build a customer×feature view and cluster into "
        "interpretable personas; size and value each segment."
    ),
    feature_hint=(
        "Engineer segmentation features at the customer grain: RFM, engagement, category "
        "preference and demographics. Scale features for clustering; keep them stable and "
        "leak-free so segments generalize."
    ),
    interpret_lens=(
        "Read results as an audience-strategy lead: describe each segment in plain "
        "language, size and value it, and recommend concrete targeting and messaging per "
        "segment."
    ),
)

CHANNEL_EXPERT = Expert(
    key="channel",
    name="Channel Mix Expert",
    summary=(
        "Specialist in channel mix: paid/organic/social/email performance, omnichannel "
        "balance and media-mix allocation."
    ),
    signals=[
        "per-channel spend & return", "paid vs. organic mix", "traffic source quality",
        "channel saturation", "media-mix allocation", "cross-channel overlap",
    ],
    intent_bias=["report", "regression", "eda", "deep_analysis"],
    query_hint=(
        "Focus on per-channel performance: spend, traffic, conversions and revenue by "
        "channel/source, and the resulting mix. Compute efficiency per channel and "
        "compare paid vs. organic contribution."
    ),
    feature_hint=(
        "Engineer channel features at the channel (or channel×time) grain: spend share, "
        "traffic share, conversion rate, efficiency and saturation indicators. Use shares "
        "and rates so channels of different sizes compare fairly."
    ),
    interpret_lens=(
        "Read results as a media-planning lead: name the most and least efficient "
        "channels, flag saturation, and recommend a concrete reallocation of the channel "
        "mix."
    ),
)


ALL_EXPERTS = [
    CAMPAIGN_EXPERT,
    ATTRIBUTION_EXPERT,
    LTV_EXPERT,
    SEGMENTATION_EXPERT,
    CHANNEL_EXPERT,
]
