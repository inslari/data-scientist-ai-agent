from pydantic import BaseModel, Field
from typing import Literal

class IntentResult(BaseModel):
    model_config = {"extra": "ignore"}
    intent: Literal["eda", "report", "classification", "regression"] = Field(
        description="Tespit edilen intent"
    )
    confidence: float = Field(
        description="Confidence score"
    )
