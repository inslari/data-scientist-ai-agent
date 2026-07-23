from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ── Providers ─────────────────────────────────────────────────────────────────
# A provider row carries exactly ONE model: the admin enters the model name on the
# same form as the credentials, so there is no "provider with no model" state and no
# nested model management. Two models behind the same key = two provider rows.
class AiProviderCreate(BaseModel):
    label: str
    provider_type: str  # openai | anthropic | google | ollama
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class AiProviderUpdate(BaseModel):
    label: Optional[str] = None
    model_name: Optional[str] = None
    # An omitted/empty api_key PRESERVES the stored key; set clear_key=True to wipe it.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    clear_key: bool = False


class AiProviderResponse(BaseModel):
    id: int
    label: str
    provider_type: str
    base_url: Optional[str] = None
    has_key: bool = False
    api_key_masked: Optional[str] = None  # e.g. "sk-…wxyz"; never the plaintext key
    created_at: datetime
    updated_at: Optional[datetime] = None
    # The provider's single model. ``model_id`` is what role assignments point at.
    model_id: Optional[int] = None
    model_name: Optional[str] = None

    class Config:
        from_attributes = True


# ── Role assignments ──────────────────────────────────────────────────────────
class AiAssignmentUpdate(BaseModel):
    # None clears the assignment for the role (→ env fallback).
    model_id: Optional[int] = None


class AiAssignmentResponse(BaseModel):
    role: str
    model_id: Optional[int] = None
    provider_id: Optional[int] = None
    provider_type: Optional[str] = None
    provider_label: Optional[str] = None
    model_name: Optional[str] = None
