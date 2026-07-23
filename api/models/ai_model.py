from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from api.core.database import Base


class AiModelModel(Base):
    """
    A model enabled under a provider connection (e.g. "gpt-4o-mini" under an OpenAI
    provider row). Model names come from the curated catalog (api/core/ai_catalog.py)
    or free-text for Ollama. A model can be assigned to a role in ai_role_assignments.
    """
    __tablename__ = "ai_models"
    __table_args__ = (UniqueConstraint("provider_id", "model_name", name="uq_ai_model_provider_name"),)

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(Integer, ForeignKey("ai_providers.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    label = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    provider = relationship("AiProviderModel", back_populates="models")
