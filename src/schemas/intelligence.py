from typing import List
from pydantic import BaseModel, Field

# --- Pydantic Models for Structured Output ---

class TableUnderstanding(BaseModel):
    table_name: str
    inferred_entity: str
    description: str
    data_quality_score: float = Field(description="Score between 0-1 based on missing data likelihood")
    analytics_potential_score: float = Field(description="Score between 0-1 based on relationships, metrics, time columns")

class ColumnProfile(BaseModel):
    table_name: str
    column_name: str
    data_type: str
    semantic_type: str = Field(description="E.g., identifier, metric, timestamp, category, target")
    nullability: str
    uniqueness: str = Field(description="high/medium/low")
    importance_score: float = Field(description="Score between 0-1 for analytics relevance")

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
