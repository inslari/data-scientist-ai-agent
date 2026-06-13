from sqlalchemy import Column, Integer, String, DateTime
from api.core.database import Base

class TokenBlacklistModel(Base):
    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(512), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
