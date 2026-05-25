from __future__ import annotations

"""
Feature Plan Node
-----------------
Kullanıcının sorgusunu, intent'i ve db_intelligence metadata'sını okuyarak
feature engineering için dinamik bir plan üretir.

LLM burada sadece "ne yapılacak" kararını verir:
- Hedef değişken ne?
- Nasıl tanımlanıyor?
- Hangi seviyede aggregation?
- Hangi feature grupları üretilmeli?

"Nasıl yapılacak" feature_engineering_node'un kendi kodunda.
Hiçbir pandas kodu LLM tarafından üretilmez.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from src.core.llm import get_llm
from pydantic import BaseModel, Field

from src.state import AgentState

load_dotenv(override=True)


class TargetDefinition(BaseModel):
    model_config = {"extra": "forbid"}

    column_name: str = Field(
        description="Hedef değişkenin adı. Mevcut değilse türetilecek isim (örn: 'churn', 'revenue', 'segment')"
    )
    is_derived: bool = Field(
        description="True ise hedef değişken veriden türetilecek, False ise zaten mevcut"
    )
    derivation_type: Literal["time_based", "threshold_based", "rank_based", "none"] = Field(
        description=(
            "time_based: tarih kolonundan türet (churn gibi). "
            "threshold_based: sayısal eşikten türet. "
            "rank_based: sıralamaya göre türet. "
            "none: zaten mevcut kolon"
        )
    )
    date_column: str = Field(
        default="",
        description="time_based için kullanılacak tarih kolonu adı"
    )
    threshold_days: int = Field(
        default=0,
        description="time_based için gün eşiği. 0 ise dinamik hesaplanır (verinin %80'lik noktası)"
    )
    threshold_value: float = Field(
        default=0.0,
        description="threshold_based için sayısal eşik değeri"
    )
    positive_class_description: str = Field(
        description="Pozitif sınıfın (1) ne anlama geldiği. Örn: 'Son 180 günde sipariş vermemiş müşteri'"
    )


class AggregationConfig(BaseModel):
    model_config = {"extra": "forbid"}

    required: bool = Field(
        description="True ise veri aggregation gerektirir (müşteri bazında gibi)"
    )
    group_by_column: str = Field(
        default="",
        description="Aggregation için groupby kolonu. Örn: 'customer_id'"
    )
    level: Literal["customer", "product", "order", "time", "none"] = Field(
        description="Aggregation seviyesi"
    )


class FeaturePlan(BaseModel):
    model_config = {"extra": "forbid"}

    analysis_summary: str = Field(
        description="Kullanıcının ne yapmak istediğinin tek cümlelik Türkçe özeti"
    )
    target: TargetDefinition
    aggregation: AggregationConfig
    feature_groups: list[Literal["rfm", "temporal_trend", "behavioral", "demographic", "product", "none"]] = Field(
        description=(
            "Üretilecek feature grupları. "
            "rfm: Recency/Frequency/Monetary. "
            "temporal_trend: Zamansal trend (dönem bazlı sipariş/harcama). "
            "behavioral: Davranışsal (kategori çeşitliliği, basket size). "
            "demographic: Demografik (yaş, hesap yaşı). "
            "product: Ürün bazlı (kategori, rating, stok). "
            "none: Aggregation gerekmez, ham veri kullanılır"
        )
    )
    use_data_split_for_label: bool = Field(
        description=(
            "True ise verinin son %20'si label için, ilk %80'i feature için kullanılır. "
            "time_based hedeflerde genellikle True olmalı."
        )
    )
    math_interactions: list[str] = Field(
        default_factory=list,
        description=(
            "LLM tarafından önerilen mantıksal matematiksel etkileşimler. "
            "Örn: ['revenue / max(1, order_count)', 'price * quantity']. "
            "Sadece sayısal kolonlar için kullanılmalıdır."
        )
    )
    reasoning: str = Field(
        description="Bu planın neden seçildiğini açıklayan Türkçe metin"
    )


SYSTEM_PROMPT = """Sen bir veri bilimi uzmanısın. Görevin, kullanıcının analiz isteğini ve veritabanı yapısını inceleyerek feature engineering için minimal ve doğru bir plan üretmektir.

## Kurallar:
- Yalnızca veritabanında GERÇEKTEN OLAN kolonları kullan
- Kolon isimlerini db_intelligence'daki isimlerle birebir eşleştir
- Hedef değişkeni kullanıcının sorgusundan çıkar
- Aggregation seviyesini analiz tipine göre belirle (churn → müşteri bazında, satış tahmini → ürün/zaman bazında)
- time_based hedeflerde use_data_split_for_label = True olmalı
- Gereksiz feature grubu ekleme — sadece analize katkı sağlayanları seç
- Gerekliyse `math_interactions` alanında oran (A/B) veya çarpım (A*B) gibi mantıksal feature'lar öner.
- Türkçe cevap ver
"""


def feature_plan_node(state: AgentState) -> AgentState:
    """
    Node 4.5: Kullanıcı sorgusundan feature engineering planı üretir.

    LLM sadece JSON plan üretir — pandas kodu yazmaz.
    Plan feature_engineering_node tarafından execute edilir.

    State girdileri:
        user_query     : Kullanıcının ham sorusu
        intent         : Tespit edilen intent
        db_intelligence: intelligence_engine metadata'sı

    State çıktıları:
        feature_plan   : FeaturePlan dict olarak
    """
    user_query = state.get("user_query", "")
    intent = state.get("intent", "classification")
    db_intelligence = state.get("db_intelligence", {})

    if not db_intelligence:
        return {
            **state,
            "error": "db_intelligence bulunamadı.",
            "current_step": "feature_plan_node",
        }

    tables_summary = _build_db_summary(db_intelligence)

    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o")
    llm = get_llm(model_name=model_name, temperature=0)
    structured_llm = llm.with_structured_output(FeaturePlan)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Kullanıcı sorusu: {user_query}
Tespit edilen intent: {intent}

Veritabanı yapısı:
{tables_summary}

Bu analiz için feature engineering planını üret.
"""),
    ]

    plan: FeaturePlan = structured_llm.invoke(messages)

    print(f"\n[feature_plan] Analiz özeti  : {plan.analysis_summary}")
    print(f"[feature_plan] Hedef değişken: {plan.target.column_name} "
          f"({'türetilecek' if plan.target.is_derived else 'mevcut'})")
    print(f"[feature_plan] Tanım         : {plan.target.positive_class_description}")
    print(f"[feature_plan] Aggregation   : {plan.aggregation.level} "
          f"(group_by: {plan.aggregation.group_by_column})")
    print(f"[feature_plan] Feature grupları: {plan.feature_groups}")
    if plan.math_interactions:
        print(f"[feature_plan] Math etkileşimler: {plan.math_interactions}")
    print(f"[feature_plan] Data split    : {plan.use_data_split_for_label}")
    print(f"[feature_plan] Neden         : {plan.reasoning}")

    return {
        **state,
        "feature_plan": plan.model_dump(),
        "current_step": "feature_plan_node",
    }


def _build_db_summary(db_intelligence: dict) -> str:
    """db_intelligence'ı LLM için okunabilir formata dönüştürür."""
    lines = []

    for table in db_intelligence.get("tables", []):
        lines.append(
            f"Tablo: {table['table_name']} | "
            f"Varlık: {table['inferred_entity']} | "
            f"Potansiyel: {table['analytics_potential_score']:.2f}"
        )

    lines.append("\nKolonlar (semantic_type | importance):")
    for col in db_intelligence.get("columns", []):
        lines.append(
            f"  {col['table_name']}.{col['column_name']} → "
            f"{col['semantic_type']} | importance: {col['importance_score']:.2f}"
        )

    rels = db_intelligence.get("relationships", [])
    if rels:
        lines.append("\nİlişkiler:")
        for rel in rels:
            lines.append(
                f"  {rel['source_table']}.{rel['source_column']} → "
                f"{rel['target_table']}.{rel['target_column']}"
            )

    caps = db_intelligence.get("capabilities", [])
    if caps:
        lines.append(f"\nKapasiteler: {', '.join(caps)}")

    return "\n".join(lines)