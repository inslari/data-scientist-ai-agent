from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    user_query: str                  # Raw user query

    # Intent Node output
    intent: str                      # "eda", "classification", "regression", "clustering", "report", "sql_query"
    intent_confidence: float         # 0-1 confidence score

    # DB Context Node output
    raw_schema: str                  # Raw MCP schema output
    db_intelligence: dict            # DatabaseIntelligenceModel dict
    relevant_tables: list[str]       # Filtered tables for planner

    # Planner output
    analysis_plan: list[dict]        # Plan steps

    # Feature Plan output
    feature_plan: dict               # Feature engineering plan from LLM

    # Execution Node output
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

    # Graph Control
    current_step: str                # Current active node
    error: Optional[str]             # Errors
    messages: list                   # LLM conversation history