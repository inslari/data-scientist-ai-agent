from pydantic import BaseModel, Field
from typing import Literal


class LeakageConcern(BaseModel):
    """Tek bir leakage şüphesi."""
    model_config = {"extra": "forbid"}

    column_name: str = Field(
        description="Şüpheli feature kolonunun adı"
    )
    relationship_type: Literal[
        "derived",      # Feature target'tan matematiksel olarak türetilmiş (örn: Profit = Sales - Cost)
        "temporal",     # Feature gelecek bilgisi içeriyor olabilir (örn: son işlem tarihi)
        "identifier",   # Feature aslında bir kimlik tanımlayıcı (örn: hash, ID)
        "duplicate",    # Feature target'in farklı bir gösterimi (örn: income vs income_bracket)
        "other"         # Diğer tipte ilişki
    ] = Field(
        description="Şüphelenilen ilişki tipi"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Şüphe derecesi. high = neredeyse kesin leakage, low = ihtimal var"
    )
    reasoning: str = Field(
        description="Neden bu feature'ın leakage olabileceğini açıklayan kısa metin"
    )


class LeakageReport(BaseModel):
    """LLM'in tüm leakage analizinin sonucu."""
    model_config = {"extra": "forbid"}

    concerns: list[LeakageConcern] = Field(
        default_factory=list,
        description="Tespit edilen leakage şüphelerinin listesi. Yoksa boş liste döndür."
    )
    overall_assessment: str = Field(
        description="Genel değerlendirme: hiç leakage yoksa 'Belirgin leakage tespit edilmedi', "
                    "varsa kısa özet"
    )