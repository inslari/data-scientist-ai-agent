from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from api.core.database import Base

# Roles the DB config can drive. Each maps to a get_llm(role=...) call-site.
# Embeddings are intentionally excluded (dead env config, no consumer in src/api).
AI_ROLES = ("main", "intelligence", "council")


class AiRoleAssignmentModel(Base):
    """
    Which model each system role uses. One row per role (``role`` unique). A missing
    row — or one whose ``model_id`` is NULL — means "fall back to the env defaults for
    that role" (legacy behaviour). Deleting the assigned model SET NULLs this row.
    """
    __tablename__ = "ai_role_assignments"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(50), nullable=False, unique=True, index=True)  # main | intelligence | council
    model_id = Column(Integer, ForeignKey("ai_models.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    model = relationship("AiModelModel")
