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
- Hedef değişken seçerken KOLON ADI'na değil KOLON AÇIKLAMASI'na bak
- Aggregation seviyesini analiz tipine göre belirle (churn → müşteri bazında, satış tahmini → ürün/zaman bazında)
- time_based hedeflerde use_data_split_for_label = True olmalı
- Gereksiz feature grubu ekleme — sadece analize katkı sağlayanları seç
- Gerekliyse `math_interactions` alanında oran (A/B) veya çarpım (A*B) gibi mantıksal feature'lar öner.
- Her satırın bir olay/işlem/örnek olduğu görevlerde aggregation.required = False, aggregation.level = 'none' olmalı.
- Türkçe cevap ver

## Hedef kolon seçimi:
Veritabanı özetindeki "HEDEF KOLON ADAYLARI" bölümüne bak.
Bu adaylar intelligence engine tarafından ML tahmin hedefi olarak etiketlenmiş kolonlardır.
Kullanıcının sorusuna en uygun olan adayı seç. Açıklamaları dikkatlice oku.
Eğer hiç aday yoksa, kullanıcı sorusuna en uygun binary/categorical kolonu seç.

## RETRY durumunda:
- Önceki planda neden başarısız olundu? Critic feedback'i dikkatlice oku.
- Eğer AUC ~0.5 ise: model veriyi öğrenememiş. Daha ayırt edici feature grupları seç, math_interactions ekle.
- Eğer Overfitting varsa: daha az feature grubu kullan.
- Her retry'da öncekinden FARKLI bir strateji dene.
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
    user_query      = state.get("user_query", "")
    intent          = state.get("intent", "classification")
    db_intelligence = state.get("db_intelligence", {})
    critic_feedback = state.get("critic_feedback", [])
    retry_count     = state.get("retry_count", 0)

    target_def_initial = state.get("target_definition", {})
    if target_def_initial.get("column_name"):
        print(f"[feature_plan] target_discovery_node önerisi: '{target_def_initial['column_name']}' "
              f"(confidence={target_def_initial.get('confidence', 0):.2f})")
    
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

    
    # retry_context'ten önce ekle
    target_hint = ""
    if target_def_initial.get("column_name"):
        target_hint = f"""

📍 ÖN-TESPİT EDİLEN HEDEF:
target_discovery_node şu kolonu önerdi: '{target_def_initial['column_name']}'
Confidence: {target_def_initial.get('confidence', 0):.2f}
Bu öneriyi dikkate al ama kendi guardrail kontrolünü yap."""

    # Retry bağlamı: critic feedback'i prompt'a ekle
    retry_context = ""
    if retry_count > 0 and critic_feedback:
        feedback_str = "\n".join(f"  - {fb}" for fb in critic_feedback)
        retry_context = f"""

⚠️ RETRY #{retry_count} — ÖNCEKİ PLAN BAŞARISIZ OLDU:
{feedback_str}

Yukarıdaki sorunları çözmek için FARKLI bir feature engineering stratejisi üret.
Önceki planla aynı yapıyı tekrarlama."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
Kullanıcı sorusu: {user_query}
Tespit edilen intent: {intent}

Veritabanı yapısı:
{tables_summary}{target_hint}{retry_context}

Bu analiz için feature engineering planını üret.
"""),
    ]

    plan: FeaturePlan = structured_llm.invoke(messages)

    # ── Post-processing: Dinamik target validation ────────────────────────────
    # LLM'in hedef kolon seçimini hem target_discovery_node önerisi
    # hem de db_intelligence semantic_type etiketlerine göre validate eder.
    plan_dict = plan.model_dump()
    target_def_initial = state.get("target_definition", {}) or {}
    _validate_target_selection(plan_dict, user_query, db_intelligence, target_def_initial)
    plan = FeaturePlan(**plan_dict)
    # ──────────────────────────────────────────────────────────────────────────

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
    """
    db_intelligence'ı LLM için okunabilir formata dönüştürür.
    Target adaylarını öne çıkarır ve açıklamaları gösterir.
    """
    lines = []

    # Tablolar
    for table in db_intelligence.get("tables", []):
        lines.append(
            f"Tablo: {table['table_name']} | "
            f"Varlık: {table['inferred_entity']} | "
            f"Potansiyel: {table['analytics_potential_score']:.2f}"
        )

    all_cols = db_intelligence.get("columns", [])

    # ── 1. Hedef kolon adaylarını öne çıkar ──
    target_cols = [c for c in all_cols if c.get("semantic_type") == "target"]
    if target_cols:
        lines.append("\n⚠️  HEDEF KOLON ADAYLARI (semantic_type=target):")
        for col in target_cols:
            desc = col.get("description", "Açıklama yok")
            samples = col.get("sample_values", "")
            lines.append(
                f"  *** {col['table_name']}.{col['column_name']} ***\n"
                f"      Açıklama: {desc}\n"
                f"      Örnek değerler: {samples}"
            )
    else:
        lines.append("\nℹ️  Hedef kolon adayı bulunamadı. LLM kullanıcı sorgusuna göre seçecek.")

    # ── 2. Diğer kolonlar ──
    lines.append("\nDiğer kolonlar (semantic_type | importance | açıklama):")
    for col in all_cols:
        if col.get("semantic_type") == "target":
            continue  # zaten yukarıda gösterildi
        desc = col.get("description", "")
        desc_str = f" | {desc[:80]}" if desc else ""
        lines.append(
            f"  {col['table_name']}.{col['column_name']} → "
            f"{col['semantic_type']} | importance: {col['importance_score']:.2f}{desc_str}"
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


# Dinamik Target Validation 
def _validate_target_selection(
    plan_dict: dict,
    user_query: str,
    db_intelligence: dict,
    target_definition: dict | None = None,
) -> None:
    """
    LLM'in seçtiği hedef kolonu hem target_discovery_node önerisi hem de
    db_intelligence semantic_type etiketlerine göre dinamik olarak validate eder.

    Algoritma:
    1. Önce target_discovery_node'un önerisi var mı kontrol et (yüksek confidence ise kabul et)
    2. db_intelligence'daki semantic_type='target' kolonları al (IE'nin ground truth'u)
    3. Eğer LLM'in seçimi zaten target-etiketli ise OK
    4. Eğer değilse: target-etiketli kolonlar arasından kullanıcı sorgusuna
       en yakın olanı word-overlap + description scoring ile seç
    5. Hiç target-etiketli kolon yoksa: LLM kararına saygı duy (IE kendisi hatalı)

    Hardcoded isim KULLANMAZ. Her dataset için çalışır.
    plan_dict in-place değiştirilir.
    """
    current_target = plan_dict.get("target", {}).get("column_name", "")
    all_cols       = db_intelligence.get("columns", [])

    # check target_discovery_node 
    if target_definition:
        discovery_target = target_definition.get("column_name", "")
        confidence       = target_definition.get("confidence", 0.0)

        if discovery_target and confidence >= 0.7:
            # high confidence 
            if current_target != discovery_target:
                print(f"[feature_plan] ⚠️  LLM='{current_target}' ama target_discovery_node "
                      f"='{discovery_target}' (confidence={confidence:.2f}). Override.")
                plan_dict["target"]["column_name"]    = discovery_target
                plan_dict["target"]["is_derived"]     = False
                plan_dict["target"]["derivation_type"] = "none"
                current_target = discovery_target
            else:
                print(f"[feature_plan] ✓  LLM ve target_discovery_node uyumlu: '{current_target}' "
                      f"(confidence={confidence:.2f})")

    target_candidates = [
        col for col in all_cols
        if col.get("semantic_type") == "target"
    ]

    if not target_candidates:
        print(f"[feature_plan] ℹ️  IE'de target-etiketli kolon yok. Mevcut seçim korunuyor: '{current_target}'")
        return

    target_col_names = {col["column_name"] for col in target_candidates}
    if current_target in target_col_names:
        print(f"[feature_plan] ✓  Target geçerli: '{current_target}' (semantic_type=target)")
        return

    print(
        f"[feature_plan] ⚠️  '{current_target}' kolonu semantic_type=target değil. "
        f"Düzeltiliyor..."
    )

    best = _score_target_by_query_overlap(target_candidates, user_query)
    print(f"[feature_plan] ✓  Target override: '{current_target}' → '{best['column_name']}' "
          f"(skor: {best['_score']:.3f})")

    plan_dict["target"]["column_name"]    = best["column_name"]
    plan_dict["target"]["is_derived"]     = False
    plan_dict["target"]["derivation_type"] = "none"


def _score_target_by_query_overlap(candidates: list[dict], user_query: str) -> dict:
    """
    Aday target kolonları kullanıcı sorgusuna göre skorlar.

    Skor = kullanıcı sorgusu ile (kolon_adı + açıklama + örnek_değerler) arasındaki
           word-overlap. Tamamen dinamik — hardcoded kelime yok.

    Args:
        candidates: db_intelligence'dan gelen, semantic_type='target' olan kolon listesi
        user_query: kullanıcının ham sorusu

    Returns:
        En yüksek skorlu kolon dict'i (+ '_score' anahtarı eklenerek)
    """
    # Normalize the query
    query_words = set(
        w.strip(".,!?;:").lower()
        for w in user_query.split()
        if len(w) > 2
    )

    scored = []
    for col in candidates:
        # Combine all column metadata
        col_text = " ".join([
            col.get("column_name", ""),
            col.get("description", ""),
            col.get("sample_values", ""),
            col.get("table_name", ""),
        ]).lower()

        col_words = set(
            w.strip(".,!?;:").lower()
            for w in col_text.split()
            if len(w) > 2
        )

        overlap  = len(query_words & col_words)
        # Bonus: more words in description give more signals
        desc_len = len(col.get("description", "")) / 200.0
        # Bonus: importance_score (IE's confidence)
        importance = col.get("importance_score", 0.5)

        score = overlap + desc_len + importance
        scored.append({**col, "_score": score})

    # Return the highest scoring candidate
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored[0]