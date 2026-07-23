from pydantic import BaseModel, EmailStr, field_validator
import re
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr

    @field_validator('email')
    @classmethod
    def lower_email(cls, v: str) -> str:
        return v.lower()

class UserCreate(UserBase):
    password: str
    full_name: Optional[str] = None

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one number')
        return v

class UserResponse(UserBase):
    id: int
    full_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    is_admin: bool = False
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

    @field_validator('email')
    @classmethod
    def lower_email(cls, v: str) -> str:
        return v.lower()

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one number')
        return v

class ResendVerificationRequest(BaseModel):
    email: EmailStr

    @field_validator('email')
    @classmethod
    def lower_email(cls, v: str) -> str:
        return v.lower()

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one number')
        return v

class RefreshTokenRequest(BaseModel):
    refresh_token: str

# Mail-free password reset: user submits a new password that is hashed immediately and
# held until an admin approves it. Same password strength rules as registration.
class RequestPasswordResetRequest(UserBase):
    new_password: str

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one number')
        return v

# Break-glass reset for admin accounts using the setup-time ADMIN_RESET_CODE.
class AdminResetWithCodeRequest(UserBase):
    new_password: str
    code: str

    @field_validator('new_password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if not re.search(r"[A-Z]", v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r"\d", v):
            raise ValueError('Password must contain at least one number')
        return v

class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None

class DeleteAccountRequest(BaseModel):
    password: str
