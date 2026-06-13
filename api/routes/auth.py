import secrets
import jwt
from jwt.exceptions import InvalidTokenError
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from api.core.database import get_db
from api.core.security import get_password_hash, verify_password, create_access_token, create_refresh_token, ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM
from api.models.user import UserModel
from api.models.token_blacklist import TokenBlacklistModel
from api.schemas.auth import UserCreate, UserResponse, Token, ForgotPasswordRequest, ResetPasswordRequest, ResendVerificationRequest, RefreshTokenRequest, ChangePasswordRequest, UpdateProfileRequest, DeleteAccountRequest
from api.core.deps import get_current_user, oauth2_scheme
from api.core.disposable_emails import is_disposable_email
from api.core.limiter import limiter
from api.core.email import send_verification_email, send_reset_password_email

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
def register(request: Request, user_in: UserCreate, db: Session = Depends(get_db)):
    if is_disposable_email(user_in.email):
        raise HTTPException(
            status_code=400,
            detail="Disposable or temporary email addresses are not allowed.",
        )

    # Check if user exists
    user = db.query(UserModel).filter(UserModel.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    
    # Generate verification token
    verification_token = secrets.token_urlsafe(32)
    
    # Create new user
    user = UserModel(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
        is_verified=False,
        verification_token=verification_token,
        verification_token_expires=datetime.now(timezone.utc) + timedelta(hours=24)
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
        
        # Send verification email asynchronously via background tasks or synchronously here
        # For simplicity we'll do it synchronously here
        send_verification_email(user.email, verification_token)
        
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Database integrity error")
    
    return user

@router.post("/resend-verification")
@limiter.limit("3/minute")
def resend_verification(request: Request, data: ResendVerificationRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.email == data.email).first()
    if user and not user.is_verified:
        verification_token = secrets.token_urlsafe(32)
        user.verification_token = verification_token
        user.verification_token_expires = datetime.now(timezone.utc) + timedelta(hours=24)
        db.commit()
        send_verification_email(user.email, verification_token)
    return {"message": "If the email is registered and unverified, a new verification link has been sent."}

@router.get("/verify-email")
def verify_email(token: str, db: Session = Depends(get_db)):    
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")
        
    user = db.query(UserModel).filter(UserModel.verification_token == token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")
        
    if user.verification_token_expires:
        expires = user.verification_token_expires
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires:
            raise HTTPException(status_code=400, detail="Verification token has expired. Please request a new one.")
        
    user.is_verified = True
    user.verification_token = None
    db.commit()
    
    return {"message": "Email verified successfully"}

@router.post("/login", response_model=Token)
@limiter.limit("10/minute")
def login(request: Request, db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    # OAuth2PasswordRequestForm expects username and password, we use email as username
    user = db.query(UserModel).filter(UserModel.email == form_data.username.lower()).first()
    
    if user and user.locked_until:
        locked_until = user.locked_until
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) < locked_until:
            raise HTTPException(status_code=403, detail="Account is temporarily locked due to too many failed login attempts. Please try again later.")
            
    if not user or not verify_password(form_data.password, user.hashed_password):
        if user:
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
            db.commit()
            
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if user.failed_login_attempts > 0 or user.locked_until:
        user.failed_login_attempts = 0
        user.locked_until = None
        db.commit()
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    if not user.is_verified:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Email not verified. Please check your inbox.")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "v": user.token_version}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": str(user.id), "v": user.token_version})
    return {"access_token": access_token, "token_type": "bearer", "refresh_token": refresh_token}

@router.post("/forgot-password")
@limiter.limit("5/minute")
def forgot_password(request: Request, data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.email == data.email).first()
    if user:
        reset_token = secrets.token_urlsafe(32)
        user.reset_password_token = reset_token
        user.reset_password_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        db.commit()
        send_reset_password_email(user.email, reset_token)
    
    # Always return success to prevent email enumeration
    return {"message": "If that email is in our system, we have sent a password reset link."}

@router.post("/reset-password")
@limiter.limit("5/minute")
def reset_password(request: Request, data: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.reset_password_token == data.token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
        
    # Check if token is expired (assuming reset_password_expires is a naive or aware datetime, let's compare properly)
    if user.reset_password_expires:
        # SQLite datetime might not have tzinfo, make it UTC if it doesn't
        expires = user.reset_password_expires
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
            
        if datetime.now(timezone.utc) > expires:
            raise HTTPException(status_code=400, detail="Reset token has expired.")
            
    user.hashed_password = get_password_hash(data.new_password)
    user.token_version += 1
    user.reset_password_token = None
    user.reset_password_expires = None
    db.commit()
    
    return {"message": "Password has been reset successfully."}

@router.post("/refresh", response_model=Token)
def refresh_token(data: RefreshTokenRequest, db: Session = Depends(get_db)):
    is_blacklisted = db.query(TokenBlacklistModel).filter(TokenBlacklistModel.token == data.refresh_token).first()
    if is_blacklisted:
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")
        
    try:
        payload = jwt.decode(data.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        token_type = payload.get("type")
        if user_id is None or token_type != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
        
    user = db.query(UserModel).filter(UserModel.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
        
    token_version = payload.get("v")
    if token_version is not None and user.token_version != token_version:
        raise HTTPException(status_code=401, detail="Invalid token version. Session has been revoked.")

    # Blacklist old refresh token
    expires_at = datetime.fromtimestamp(float(payload.get("exp", 0.0)), tz=timezone.utc)
    blacklisted_refresh = TokenBlacklistModel(token=data.refresh_token, expires_at=expires_at)
    db.add(blacklisted_refresh)
    db.commit()
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        data={"sub": str(user.id), "v": user.token_version}, expires_delta=access_token_expires
    )
    new_refresh_token = create_refresh_token(data={"sub": str(user.id), "v": user.token_version})
    return {"access_token": new_access_token, "token_type": "bearer", "refresh_token": new_refresh_token}

@router.post("/logout")
def logout(
    access_token: str = Depends(oauth2_scheme), 
    refresh_data: RefreshTokenRequest | None = None,
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        expires_at = datetime.fromtimestamp(float(payload.get("exp", 0.0)), tz=timezone.utc)
        
        blacklisted_access = TokenBlacklistModel(token=access_token, expires_at=expires_at)
        db.add(blacklisted_access)
        
        if refresh_data and refresh_data.refresh_token:
            try:
                refresh_payload = jwt.decode(refresh_data.refresh_token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
                r_expires_at = datetime.fromtimestamp(float(refresh_payload.get("exp", 0.0)), tz=timezone.utc)
                blacklisted_refresh = TokenBlacklistModel(token=refresh_data.refresh_token, expires_at=r_expires_at)
                db.add(blacklisted_refresh)
            except Exception:
                pass
                
        db.commit()
    except Exception:
        pass
        
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: UserModel = Depends(get_current_user)):
    return current_user

@router.post("/change-password")
@limiter.limit("5/minute")
def change_password(
    request: Request,
    data: ChangePasswordRequest, 
    current_user: UserModel = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    if not verify_password(data.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")
        
    current_user.hashed_password = get_password_hash(data.new_password)
    current_user.token_version += 1
    db.commit()
    
    return {"message": "Password changed successfully. You will be logged out of all devices."}

@router.put("/me", response_model=UserResponse)
def update_profile(
    data: UpdateProfileRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if data.full_name is not None:
        current_user.full_name = data.full_name
    db.commit()
    db.refresh(current_user)
    return current_user

@router.delete("/me")
@limiter.limit("5/minute")
def delete_account(
    request: Request,
    data: DeleteAccountRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not verify_password(data.password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect password")
    
    db.delete(current_user)
    db.commit()
    return {"message": "Account deleted successfully."}
