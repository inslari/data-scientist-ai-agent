from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    user_query: str                  # Raw user query

    # Domain (Sector) Pack — pluggable Merchant Intelligence layer
    domain_pack: str                 # Active pack name ("none" disables the layer)
    merchant_context: dict           # MerchantContext.to_dict() — selected expert(s) & framing
    domain_doc: str                  # Per-data-group dataset description (markdown context)
    merchant_analysis: dict          # MerchantAnalysis.to_dict() — agentic expert findings

    # Intent Node output
    intent: str                      # "eda", "classification", "regression", "clustering", "report", "sql_query"
    intent_confidence: float         # 0-1 confidence score

    # DB Context Node output
    raw_schema: str                  # Raw MCP schema output
    db_intelligence: dict            # DatabaseIntelligenceModel dict
    relevant_tables: list[str]       # Filtered tables for planner

    # Manuel hedef seçimi (UI'dan gelir) — doluysa target_discovery_node bypass edilir
    target_col: str                  # Kullanıcının UI'dan el ile seçtiği hedef sütun

    # Target Discovery Node output
    target_definition: dict          # Target definition for modeling

    # Planner output
    analysis_plan: list[dict]        # Plan steps

    # Feature Plan output
    feature_plan: dict               # Feature engineering plan from LLM

    # Execution Node output
    data_path: str
    sql_results: list[dict]          # Raw data records
    profiling_report: dict           # Column statistics, null rates, dtypes

    # Feature Engineering output
    feature_report: dict             # Summary of applied transformations
    X_train: list[dict]              # Training features
    X_test: list[dict]               # Test features
    y_train: list                    # Training labels
    y_test: list                     # Test labels

    # AutoML output
    model_name: str                  # Selected model name
    model_params: dict               # Hyperparameters
    model_artifact: Any              # Picklable trained model
    top_models: list[str]            # Top 2 Best Models
    
    # Evaluation output
    metrics: dict                    # Model evaluation metrics
    insights: list[str]              # LLM generated insights
    evaluation_report: dict          # Comprehensive evaluation report
    optimal_threshold: float         # Youden's J optimal decision threshold

    # Report Node output
    report: str                      # Final markdown report

    # SQL Query Node output
    sql_query_raw: str               # SQL Query 
    sql_query_results: list[dict]    # Raw query results (list of rows)
    sql_query_columns: list[str]     # Column names
    sql_query_chart: dict            # Chart data: {type, labels, datasets, title}
    sql_query_summary: str           # Generated short description
    sql_query_warning: str           # Warning message if approximate columns were used (empty = exact match)

    # Critic & Self-Correction
    retry_count: int                 # Number of retry attempts made
    max_retries: int                 # Maximum allowed retries
    # Append-only across retry loops — nodes emit only the new items (deltas);
    # the reducer concatenates them so feedback/history accumulate automatically.
    critic_feedback: Annotated[list[str], operator.add]   # Feedback from the critic node
    next_retry_node: str             # Node to route to after critic
    
    # Leakage Detection
    domain_leakage_report: dict

    # Entity Resolution
    entity_resolution: dict          # EntityResolution raporu (action, signals, stats)

    # Quality Gate
    quality_gate: dict

    # Retry History — append-only delta per turn (see critic_feedback note)
    retry_history: Annotated[list[dict], operator.add]
    # Audit Trail / Tracking
    experiment_logs: list[dict]      # Logs of completed ML experiments

    # Graph Control
    current_step: str                # Current active node
    error: Optional[str]             # Errors
    error_category: str              # classify_error() kategorisi: data/db/llm/timeout/memory/unknown
    messages: list                   # LLM conversation history
    excluded_models: list[str]       # Models excluded due to 
    
    # Analysis Report
    analysis_report: dict            # Deep analysis report