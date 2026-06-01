from pydantic import BaseModel, Field

class SQLGenerationResult(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = Field(description="SQL sorgusu")
    explanation: str = Field(description="Sorgu açıklaması")

class ChartConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str = Field(description="Grafik türü")
    x: str = Field(description="X ekseni")
    y: str = Field(description="Y ekseni")
    title: str = Field(description="Grafik başlığı")
