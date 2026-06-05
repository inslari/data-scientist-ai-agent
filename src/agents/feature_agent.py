import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import duckdb
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv(override=True)

from deepagents import create_deep_agent
from src.core.config import settings


class FinalFeaturesInput(BaseModel):
    math_interactions: List[str] = Field(description="A list of Pandas math string expressions that passed the tests and improve the model. e.g. ['amount / (tenure_months + 1)']")
    drop_features: List[str] = Field(description="A list of existing features that are redundant, useless, or highly correlated (leakage) and should be dropped.")
    explanation: str = Field(description="Brief explanation of why these features were chosen.")


def get_feature_agent(data_path: str, target_name: str, sample_limit: int = 10000):
    
    def test_feature_expression(expression: str) -> dict:
        """
        Evaluates a Pandas math expression on a sample of the data and returns its correlation with the target.
        Use this tool to test feature ideas before committing them.
        Example expression: 'amount / (tenure_months + 1)'
        """
        if not data_path or not target_name:
            return {"error": "Data path or target name is missing. Cannot evaluate."}
            
        try:
            conn = duckdb.connect(":memory:")
            # We sample data to make tests extremely fast
            query = f"SELECT * FROM read_parquet('{data_path}') USING SAMPLE {sample_limit}"
            df = conn.execute(query).df()
            conn.close()
            
            if target_name not in df.columns:
                return {"error": f"Target column '{target_name}' not found in the dataset."}
            
            # Apply the expression
            safe_expr = expression
            for col in df.columns:
                if "." in col and col in safe_expr:
                    safe_expr = safe_expr.replace(col, f"`{col}`")
                    
            series = df.eval(safe_expr)
            
            if series.isnull().all():
                return {"error": "Expression evaluated to entirely null values."}
                
            # Convert target to numeric for correlation
            y = pd.to_numeric(df[target_name], errors='coerce')
            x = pd.to_numeric(series, errors='coerce')
            
            valid_idx = ~(x.isnull() | y.isnull())
            if not valid_idx.any():
                 return {"error": "No valid numeric rows to calculate correlation."}
                 
            corr = np.corrcoef(x[valid_idx], y[valid_idx])[0, 1]
            variance = np.var(x[valid_idx])
            
            return {
                "status": "success",
                "expression": expression,
                "correlation_with_target": float(corr) if not np.isnan(corr) else 0.0,
                "variance": float(variance) if not np.isnan(variance) else 0.0
            }
        except Exception as e:
            return {"error": f"Failed to evaluate expression: {e}"}

    def commit_best_features(math_interactions: List[str], drop_features: List[str], explanation: str) -> dict:
        """
        Saves the best tested math interactions and lists any redundant features to drop.
        Call this tool ONCE at the end to finish your task.
        """
        return {
            "status": "success",
            "message": "Features successfully committed. Agent execution complete."
        }
        
    system_prompt = f"""You are an elite Feature Engineering Agent.
Your goal is to brainstorm and test mathematical feature combinations that improve the predictive power for the target variable: '{target_name}'.

Workflow:
1. Examine the column metadata provided by the user.
2. Formulate hypotheses for new features using pandas `eval` compatible math expressions. (e.g. `revenue / (age + 1)`, `is_active * total_spent`).
3. TEST your expressions using the `test_feature_expression` tool.
4. Analyze the correlation and variance returned by the tool. If the correlation is very low (e.g., < 0.01) or variance is 0, DO NOT keep the feature. If an expression throws an error, fix it or discard it.
5. Identify any useless or redundant existing features that should be dropped.
6. Once you have found the best features, call `commit_best_features` to finalize.

CRITICAL RULES:
- ONLY commit features that you have successfully tested via `test_feature_expression` and which yielded valid correlations.
- Do NOT hallucinate column names. Use exactly what is provided in the metadata.
"""

    provider = getattr(settings, "DEFAULT_LLM_PROVIDER", "openai").lower()
    return create_deep_agent(
        model=f"{provider}:{settings.INTELLIGENCE_ENGINE_MODEL}",
        tools=[test_feature_expression, commit_best_features],
        system_prompt=system_prompt
    )
