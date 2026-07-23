from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from api.core.database import Base


class PendingPasswordResetModel(Base):
    """
    A mail-free password-reset request awaiting admin approval (closed platform).

    The user submits their desired new password; it is hashed **immediately**
    (``hashed_new_password``) so no plaintext is ever stored and the admin can never
    see it — approval just copies this hash onto the user. One active request per
    user (``user_id`` unique); a new request replaces the old one. The row is deleted
    once the admin approves or rejects it.
    """
    __tablename__ = "pending_password_resets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False)  # denormalised for admin display
    hashed_new_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("UserModel")
