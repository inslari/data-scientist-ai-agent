from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class AdminUserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    is_admin: bool
    created_at: datetime
    has_pending_reset: bool = False

    class Config:
        from_attributes = True


class PendingResetResponse(BaseModel):
    """Never includes the hashed/plaintext password — only who requested and when."""
    id: int
    user_id: int
    email: str
    created_at: datetime

    class Config:
        from_attributes = True
