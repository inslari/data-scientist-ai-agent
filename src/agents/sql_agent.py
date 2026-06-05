import json
from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv(override=True)

from deepagents import create_deep_agent
from src.tools.db_tools import get_schema, run_sql_query
from src.core.config import settings

# System prompt for the SQL Agent
SQL_AGENT_SYSTEM_PROMPT = """You are an elite SQL Database Agent.
Your goal is to write a highly accurate SQL query based on the user's request and the database schema.

Here is your workflow:
1. Examine the user's request, the provided schema, and the column profiles.
2. If you are unsure about the data or table names, use the `run_sql_query` tool to test queries (e.g. `SELECT * FROM table LIMIT 5` or `SELECT COUNT(*) ...`).
3. Analyze the results or errors from `run_sql_query`. If you get an error (e.g., column not found), CORRECT your query and test again.
4. Once you have a working and accurate SQL query that fulfills the user's request, you MUST call the `submit_final_sql` tool to save your work and finish.

CRITICAL RULES:
- If the required columns/data do NOT exist, you must still call `submit_final_sql` but provide a query like 'SELECT "NO_VALID_COLUMN"' and include a `data_warning`.
- Always wrap table and column names in double quotes if they contain uppercase letters or spaces (e.g., "User Data", "CustomerID").
- Ensure your SQL is compatible with DuckDB or the target DB dialect if specified.
"""

class FinalSqlSubmission(BaseModel):
    sql_query: str = Field(description="The final, tested, and correct SQL query.")
    db_name: str = Field(description="The name of the database being queried.")
    data_warning: str = Field(description="Any warnings if the data was not perfect or columns were missing.")
    is_approximation: bool = Field(description="True if the query is an approximation due to missing exact data.")
    approximation_warning: str = Field(description="Explanation of the approximation if applicable.")

def submit_final_sql(sql_query: str, db_name: str, data_warning: str = "", is_approximation: bool = False, approximation_warning: str = "") -> dict:
    """
    Submits the final validated SQL query. You MUST call this tool to finish your job.
    """
    return {
        "status": "success",
        "message": "Final SQL submitted successfully. Agent execution complete."
    }

provider = getattr(settings, "DEFAULT_LLM_PROVIDER", "openai").lower()

sql_agent = create_deep_agent(
    model=f"{provider}:{settings.INTELLIGENCE_ENGINE_MODEL}",
    tools=[get_schema, run_sql_query, submit_final_sql],
    system_prompt=SQL_AGENT_SYSTEM_PROMPT
)
