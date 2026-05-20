from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    user_query: str                  # Raw user query

    # Intent Node output
    intent: str                      # "eda", "classification", "regression", "clustering", "report"
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

    # Graph Control
    current_step: str                # Current active node
    error: Optional[str]             # Execution errors
    messages: list                   # LLM conversation history