from pydantic import BaseModel, Field

class AnalysisStep(BaseModel):
    model_config = {"extra": "forbid"}
    step: str = Field(description="Adım başlığı")
    order: int = Field(description="Sıra no")
    sql_query: str = Field(description="Varsa SQL sorgusu", default="")

class AnalysisPlan(BaseModel):
    model_config = {"extra": "forbid"}
    steps: list[AnalysisStep]
