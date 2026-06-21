"""The experts of the Retail Intelligence Engine.

Each expert carries the domain knowledge injected at the pipeline's four touch
points (intent biasing, query hints, feature hints, interpretation lens). Keyword
sets (Turkish + English) power the deterministic routing fallback in the generic
router.
"""

from __future__ import annotations

from src.domains.base import Expert

# ── Routing keywords (TR + EN), per expert key ──────────────────────────────
EXPERT_KEYWORDS: dict[str, list[str]] = {
    "inventory": [
        "inventory", "envanter", "stock", "stok", "reorder", "yeniden sipariş",
        "yeniden siparis", "safety stock", "güvenlik stoğu", "guvenlik stogu",
        "out-of-stock", "stoksuz", "stockout", "overstock", "fazla stok",
        "replenishment", "ikmal", "holding cost", "stok maliyeti",
        "abc analysis", "abc analizi",
    ],
    "pricing": [
        "price elasticity", "fiyat esnekliği", "fiyat esnekligi", "pricing",
        "fiyatlandırma", "fiyatlandirma", "markdown", "indirim oranı", "margin",
        "marj", "discount", "price optimization", "fiyat optimizasyonu",
        "willingness to pay", "price sensitivity", "fiyat hassasiyeti",
    ],
    "demand": [
        "demand forecast", "talep tahmini", "forecasting", "tahminleme",
        "seasonality", "mevsimsellik", "sales forecast", "satış tahmini",
        "satis tahmini", "demand planning", "talep planlama", "lead time",
        "tedarik süresi", "trend",
    ],
    "assortment": [
        "assortment", "ürün çeşitliliği", "urun cesitliligi", "çeşitlilik",
        "cesitlilik", "product mix", "ürün karması", "urun karmasi", "planogram",
        "shelf space", "raf alanı", "raf alani", "category management",
        "kategori yönetimi", "sku rationalization", "range", "variety",
    ],
    "store": [
        "store performance", "mağaza performansı", "magaza performansi",
        "footfall", "ayak trafiği", "ayak trafigi", "sales per square",
        "metrekare", "store comparison", "mağaza karşılaştırma",
        "magaza karsilastirma", "location", "lokasyon", "regional sales",
        "bölgesel satış", "bolgesel satis", "outlet",
    ],
}


INVENTORY_EXPERT = Expert(
    key="inventory",
    name="Inventory Optimization Expert",
    summary=(
        "Specialist in stock and inventory: reorder points, safety stock, stockouts "
        "and overstock, replenishment and ABC analysis."
    ),
    signals=[
        "on-hand stock / inventory levels", "stockout & overstock frequency",
        "reorder point / safety stock", "lead time", "inventory turnover",
        "holding cost", "ABC class",
    ],
    intent_bias=["regression", "eda", "report", "anomaly_detection"],
    query_hint=(
        "Focus on stock dynamics: per-SKU on-hand levels, demand vs. cover, stockout and "
        "overstock days, and turnover. Join inventory to sales to compute days-of-cover "
        "and reorder gaps."
    ),
    feature_hint=(
        "Engineer inventory features at the SKU (and SKU×store) grain: average demand, "
        "demand variability, lead time, current cover, turnover and ABC class. Use only "
        "data available before the replenishment decision — no future demand leakage."
    ),
    interpret_lens=(
        "Read results as a supply-chain planner: name the SKUs most at risk of stockout "
        "or overstock, quantify excess holding cost, and recommend concrete reorder "
        "points and safety-stock levels."
    ),
)

PRICING_EXPERT = Expert(
    key="pricing",
    name="Pricing & Elasticity Expert",
    summary=(
        "Specialist in pricing: price elasticity of demand, markdown and margin "
        "optimization, and price-sensitivity segmentation."
    ),
    signals=[
        "price vs. quantity relationship", "price elasticity", "margin & markdown",
        "discount depth", "promo lift", "price tiers / sensitivity",
    ],
    intent_bias=["regression", "deep_analysis", "eda", "report"],
    query_hint=(
        "Focus on price–quantity relationships: aggregate units sold against price points "
        "and discounts, controlling for seasonality. Estimate elasticity from log price "
        "vs. log quantity at the product or category grain."
    ),
    feature_hint=(
        "Engineer pricing features: unit price, relative price vs. category, discount "
        "depth, margin, and log-price for elasticity. Avoid mixing promo and base periods "
        "without a promo flag; never leak realized post-price demand into base features."
    ),
    interpret_lens=(
        "Read results as a pricing analyst: report elasticity by product/category, name "
        "where demand is price-sensitive vs. inelastic, and recommend concrete price and "
        "markdown moves with their margin impact."
    ),
)

DEMAND_EXPERT = Expert(
    key="demand",
    name="Demand Forecasting Expert",
    summary=(
        "Specialist in demand forecasting: sales trend and seasonality, demand planning "
        "and lead-time-aware forecasts."
    ),
    signals=[
        "sales / units over time", "trend & seasonality", "weekday/holiday effects",
        "demand variability", "forecast horizon", "lead time",
    ],
    intent_bias=["regression", "deep_analysis", "report", "eda"],
    query_hint=(
        "Focus on time-series demand: aggregate units sold by day/week and by SKU or "
        "store, expose trend and seasonality, and align to the forecast horizon. Group by "
        "calendar fields to surface weekday and holiday effects."
    ),
    feature_hint=(
        "Engineer demand features from past windows only: lagged sales, rolling means, "
        "trend, weekday/month/holiday flags, and recent variability. Never use same- or "
        "future-period sales as a feature for the period being forecast."
    ),
    interpret_lens=(
        "Read results as a demand planner: describe the trend and seasonal pattern, "
        "quantify forecast confidence, and recommend stocking and planning actions per "
        "SKU or store."
    ),
)

ASSORTMENT_EXPERT = Expert(
    key="assortment",
    name="Assortment & Category Expert",
    summary=(
        "Specialist in assortment planning: product mix and variety, planogram and shelf "
        "space, category management and SKU rationalization."
    ),
    signals=[
        "product / category mix", "SKU breadth & depth", "shelf space allocation",
        "cannibalization", "long-tail vs. core SKUs", "category contribution",
    ],
    intent_bias=["eda", "clustering", "report", "deep_analysis"],
    query_hint=(
        "Focus on the assortment: count SKUs and sales share per category, identify "
        "long-tail vs. core items, and compare variety across stores. Rank SKUs by "
        "contribution to spot rationalization candidates."
    ),
    feature_hint=(
        "Engineer assortment features at the category/store grain: SKU breadth, sales "
        "concentration (top-N share), category contribution, and overlap/cannibalization "
        "indicators. Prefer shares over raw counts."
    ),
    interpret_lens=(
        "Read results as a category manager: name over- and under-assorted categories, "
        "flag low-contribution SKUs to cut, and recommend concrete mix and shelf-space "
        "changes."
    ),
)

STORE_EXPERT = Expert(
    key="store",
    name="Store Performance Expert",
    summary=(
        "Specialist in store performance: footfall and conversion, sales per square "
        "metre, store comparison and regional/location effects."
    ),
    signals=[
        "sales per store / per sqm", "footfall & conversion", "basket size per store",
        "regional differences", "store ranking", "location attributes",
    ],
    intent_bias=["eda", "clustering", "report", "regression"],
    query_hint=(
        "Focus on store-level performance: aggregate sales, transactions and basket size "
        "per store, normalize by size or footfall, and compare across regions. Rank and "
        "cluster stores by performance."
    ),
    feature_hint=(
        "Engineer store features at the store grain: sales per sqm, conversion, average "
        "basket, footfall, region one-hots and location attributes. Normalize for store "
        "size so big and small stores compare fairly."
    ),
    interpret_lens=(
        "Read results as a retail operations lead: name top and bottom stores and why, "
        "quantify the gap to the benchmark, and recommend concrete location-level actions."
    ),
)


ALL_EXPERTS = [
    INVENTORY_EXPERT,
    PRICING_EXPERT,
    DEMAND_EXPERT,
    ASSORTMENT_EXPERT,
    STORE_EXPERT,
]
