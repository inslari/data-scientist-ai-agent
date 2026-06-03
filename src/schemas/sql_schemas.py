from pydantic import BaseModel, Field
from typing import Optional

class SQLGenerationResult(BaseModel):
    model_config = {"extra": "forbid"}
    sql: str = Field(description="SQL sorgusu")
    db_name: str = Field(description="Sorgunun çalıştırılacağı veritabanı adı")
    data_warning: Optional[str] = Field(default="", description="Uygun kolon bulunamadığında verilecek uyarı mesajı")
    is_approximation: bool = Field(default=False, description="Sorgu tam eşleşme değilse true, tam eşleşmeyse false")
    approximation_warning: Optional[str] = Field(default="", description="Yaklaşık bir eşleşme yapıldıysa verilecek uyarı mesajı")

class ChartConfig(BaseModel):
    model_config = {"extra": "forbid"}
    type: str = Field(description="Grafik türü")
    x: str = Field(description="X ekseni")
    y: str = Field(description="Y ekseni")
    title: str = Field(description="Grafik başlığı")
