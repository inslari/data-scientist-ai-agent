from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # --- Input ---
    user_query: str                  # Kullanıcının ham sorusu

    # --- Intent Node çıktısı ---
    intent: str                      # "eda", "classification", "regression", "clustering", "report"
    intent_confidence: float         # 0-1 arası güven skoru

    # --- DB Context Node çıktısı ---
    raw_schema: str                  # MCP get_schema() çıktısı (ham metin)
    db_intelligence: dict            # DatabaseIntelligenceModel.model_dump()
    relevant_tables: list[str]       # Planner'a gönderilecek filtrelenmiş tablolar

    # --- Planner çıktısı ---
    analysis_plan: list[dict]        # [{"step": "sql", "description": "...", "config": {...}}, ...]

    # --- Feature Plan çıktısı ---
    feature_plan: dict               # FeaturePlan — LLM'in ürettiği feature engineering planı

    # --- Execution Layer çıktısı ---
    sql_results: list[dict]          # Çekilen ham veri (records formatında)
    profiling_report: dict           # Kolon istatistikleri, null oranları, dtype'lar

    # --- Feature Engineering çıktısı ---
    feature_report: dict             # Yapılan dönüşümlerin özeti
    X_train: list[dict]              # Eğitim verisi
    X_test: list[dict]               # Test verisi
    y_train: list                    # Eğitim label'ları
    y_test: list                     # Test label'ları

    # --- AutoML çıktısı ---
    model_name: str                  # Seçilen model adı
    model_params: dict               # Hiperparametreler
    model_artifact: Any              # Eğitilmiş model objesi (pickle edilebilir)

    # --- Evaluation çıktısı ---
    metrics: dict                    # {"accuracy": 0.91, "f1": 0.88, ...}
    insights: list[str]              # LLM'in ürettiği yorumlar
    evaluation_report: dict          # Kapsamlı evaluation raporu (evaluation_node)
    optimal_threshold: float         # Youden's J ile hesaplanan optimal karar eşiği

    # --- Report çıktısı ---
    report: str                      # Markdown formatında final rapor

    # --- Graph kontrolü ---
    current_step: str                # Hangi node'dayız (debug için)
    error: Optional[str]             # Herhangi bir node'da hata olduysa buraya yaz
    messages: list                   # LLM conversation history (Planner için)