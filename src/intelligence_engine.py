import json
import os
from typing import List, Optional, Any
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from src.core.llm import get_llm
from src.core.config import settings
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv(override=True)


# Pydantic Models for Structured Output     

class TableUnderstanding(BaseModel):
    db_name: str = Field(description="The name of the database this table belongs to (from Database: 'X' tags in the schema)", default="default")
    table_name: str
    inferred_entity: str
    description: str
    data_quality_score: float = Field(description="Score between 0-1 based on missing data likelihood")
    analytics_potential_score: float = Field(description="Score between 0-1 based on relationships, metrics, time columns")

class ColumnProfile(BaseModel):
    db_name: str = Field(description="The name of the database this column's table belongs to", default="default")
    table_name: str
    column_name: str
    data_type: str
    semantic_type: str = Field(description="E.g., identifier, metric, timestamp, category, target")
    nullability: str
    uniqueness: str = Field(description="high/medium/low")
    importance_score: float = Field(description="Score between 0-1 for analytics relevance")
    description: str = Field(
        default="",
        description="One sentence describing what this column actually contains (units, range, meaning)"
    )
    sample_values: str = Field(
        default="",
        description="Comma-separated list of representative sample values from this column (3-5 values)"
    )

class Relationship(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str = Field(description="one-to-one, one-to-many, many-to-many")

class SchemaGraph(BaseModel):
    nodes: List[str] = Field(description="List of tables and/or columns as nodes")
    edges: List[str] = Field(description="List of relationships/joins between nodes")

class DatabaseIntelligenceModel(BaseModel):
    tables: List[TableUnderstanding]
    columns: List[ColumnProfile]
    relationships: List[Relationship]
    capabilities: List[str] = Field(description="List of inferred analytics capabilities (e.g., time_series_analysis, cohort_analysis)")
    schema_graph: SchemaGraph
    global_insights: List[str] = Field(description="Short observations about dataset structure")

class DatabaseIntelligenceEngine:
    def __init__(self, llm=None):
        """
        Initialize the intelligence engine.
        Defaults to ChatOpenAI with gpt-4o if no LLM is provided.
        """
        if llm is None:
            self.llm = get_llm(model_name=settings.INTELLIGENCE_ENGINE_MODEL, temperature=0)
        else:
            self.llm = llm

    def analyze_schema(self, schema_text: str, sample_data: Optional[str] = None, foreign_keys: Optional[str] = None) -> DatabaseIntelligenceModel:
        """
        Analyzes a database schema and returns a structured intelligence representation.
        """
        
        system_prompt = """You are a Database Exploration and Intelligence Engine.
        
# ROLE
You are a Database Exploration and Intelligence Engine responsible for transforming relational databases into structured, AI-ready knowledge representations.

# GOAL
Your primary goal is to deeply understand the database structure and produce a high-quality metadata model that can later be used by analytics, feature engineering, and machine learning agents.

# BACKSTORY
You are part of an autonomous data science system. You are NOT responsible for business reporting or decision-making. Instead, you act as the foundation layer that interprets raw database structures into meaningful analytical representations.

Other downstream agents depend on your output to perform:
- analysis planning
- feature engineering
- machine learning
- reporting

Therefore your output must be precise, structured, and conservative.

---

# INPUT PROVIDED
- Database Schema: {schema_text}
- Sample Data: {sample_data}
- Foreign Keys: {foreign_keys}

---

# EXECUTION INSTRUCTIONS (CREW STYLE BEHAVIOR)

You must behave like a professional data engineer + data architect.

Before producing the final output, internally:
1. Understand each table as a real-world entity
2. Infer relationships and dependencies
3. Identify analytical potential of each column
4. Determine what kinds of ML/analytics tasks are possible

You MUST think step-by-step but do NOT reveal your reasoning.

If uncertain:
- prefer "unknown"
- do NOT hallucinate business meaning

---

# YOUR TASK
Build a structured "Database Intelligence Model" with the following considerations:

## 1. TABLE UNDERSTANDING
Identify real-world entity, estimate business meaning, and carefully extract the `db_name` from the "Database: 'X'" tags in the schema text.

## 2. COLUMN PROFILING
Infer:
- data_type
- semantic_type (identifier, metric, timestamp, category, target)
- nullability
- uniqueness
- importance_score
- description: ONE sentence explaining what the column truly contains (be specific about units and value types)
  Good examples: "Weekly hours worked per week (integer, range 1-99)"
                 "Capital gains from investments in USD (can be 0 for most records)"
                 "Binary income bracket: '>50K' or '<=50K' strings, NOT an actual salary number"
- sample_values: 3-5 representative actual values from sample data (e.g. "Male, Female" or "0, 5178, 99999")

## 3. RELATIONSHIP DETECTION
Infer:
- primary/foreign key relations
- implicit joins

## 4. DATA CAPABILITY INFERENCE
Infer possible analytics capabilities:
- time_series_analysis
- cohort_analysis
- segmentation
- forecasting
- classification_ready
- regression_ready
- funnel_analysis

## 5. ANALYTICS READINESS SCORE
For each table:
- data_quality_score (0–1)
- analytics_potential_score (0–1)

## 6. SCHEMA GRAPH BUILDING
Build a graph representation:
- nodes = tables / columns
- edges = relationships

---

# RULES (STRICT)
- Do NOT perform business reporting or trend analysis
- Do NOT assume real-world distributions unless explicitly provided
- Focus ONLY on structure + analytical potential
- Be conservative and avoid hallucinations
- Prefer deterministic reasoning over assumptions

## CRITICAL: semantic_type Decision Framework

For each column, assign semantic_type using this strict decision tree:

### Step 1 — Is it an identifier or key?
  → PK, FK, UUID, name, email, phone, address: assign `identifier`
  → Rule: identifiers uniquely address an entity row, not measure anything

### Step 2 — Is it a date, time, or datetime?
  → Assign `timestamp`

### Step 3 — Is it numeric and measurable?
  → Amounts, counts, scores, rates, durations, ages: assign `metric`

### Step 4 — Is it a label of an ML outcome? Apply the PREDICTABILITY TEST:
  Ask yourself: "Could a machine learning model be built where OTHER features, observed BEFORE this column's value was determined, are used to PREDICT this column?"
  AND: "Does this column represent a discrete BUSINESS EVENT or OUTCOME that happened (or didn't happen) at a specific point in time?"
  
  Examples of `target`:
  - A label saying a transaction was fraudulent (the event: fraud occurred)
  - A label saying a customer stopped buying (the event: churn occurred)
  - A label saying a loan was not repaid (the event: default occurred)
  - A label saying a patient was diagnosed with a disease
  
  NOT `target` — these are `category` or `metric`:
  - Current status/state flags that describe the PRESENT state of an entity ("is this account currently active?", "is this merchant on a blacklist?")
  - Boolean flags that are simultaneously true/false with all other data, not a post-hoc outcome
  - Risk scores (they are input metrics, not outcome labels)
  - Operational flags set by the system (is_foreign, is_weekend, etc.)
  
  Decision rule: `target` ONLY if the column encodes a concrete OUTCOME/EVENT that a business would want to predict IN ADVANCE.
  When in doubt: assign `category`.

### Step 5 — Everything else (string labels, low-cardinality categoricals, boolean flags):
  → Assign `category`

---

# OUTPUT FORMAT
{format_instructions}
"""
        structured_llm = self.llm.with_structured_output(DatabaseIntelligenceModel)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Please analyze the provided schema and return the structured JSON.")
        ])
        
        chain = prompt | structured_llm
        
        response = chain.invoke({
            "schema_text": schema_text or "No schema provided.",
            "sample_data": sample_data or "No sample data provided.",
            "foreign_keys": foreign_keys or "No explicit foreign keys provided.",
            "format_instructions": "Return a structured JSON object according to the schema."
        })
        
        return response
