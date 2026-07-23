from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from api.core.database import Base


class AiProviderModel(Base):
    """
    A single provider *connection* configured by an admin (credentials + endpoint).

    Multiple rows may share the same ``provider_type`` (e.g. two OpenAI keys), so
    ``provider_type`` is intentionally NOT unique — the ``label`` disambiguates.
    ``api_key_encrypted`` holds a Fernet-encrypted key (see api/core/secrets_crypto.py);
    it is nullable because Ollama needs only a ``base_url``, not a key.
    """
    __tablename__ = "ai_providers"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String(255), nullable=False)
    provider_type = Column(String(50), nullable=False, index=True)  # openai | anthropic | google | ollama
    api_key_encrypted = Column(String, nullable=True)
    base_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    models = relationship(
        "AiModelModel",
        back_populates="provider",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
