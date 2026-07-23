import hmac
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
from api.models.password_reset_request import PendingPasswordResetModel
from api.schemas.auth import UserCreate, UserResponse, Token, RefreshTokenRequest, ChangePasswordRequest, UpdateProfileRequest, DeleteAccountRequest, RequestPasswordResetRequest, AdminResetWithCodeRequest
from api.core.deps import get_current_user, oauth2_scheme
from api.core.disposable_emails import is_disposable_email
from api.core.limiter import limiter
from src.core.config import settings

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
    
    # Closed platform (no email): the very first registrant bootstraps as admin AND is
    # auto-verified (no one exists yet to approve them). Everyone after registers
    # unverified and must be approved by an admin in the panel before they can log in.
    is_first_user = db.query(UserModel).count() == 0

    # Create new user
    user = UserModel(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
        is_verified=is_first_user,
        is_admin=is_first_user,
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Database integrity error")

    return user

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
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Your account is pending admin approval.")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id), "v": user.token_version}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": str(user.id), "v": user.token_version})
    return {"access_token": access_token, "token_type": "bearer", "refresh_token": refresh_token}

@router.post("/request-password-reset")
@limiter.limit("5/minute")
def request_password_reset(request: Request, data: RequestPasswordResetRequest, db: Session = Depends(get_db)):
    """
    Mail-free password reset (closed platform). The user submits their desired new
    password; it is hashed immediately and held in ``pending_password_resets`` until
    an admin approves it — the admin never sees the plaintext. One pending request per
    user (a new one replaces the old). Always returns a generic success to avoid email
    enumeration.
    """
    user = db.query(UserModel).filter(UserModel.email == data.email).first()
    if user:
        pending = (
            db.query(PendingPasswordResetModel)
            .filter(PendingPasswordResetModel.user_id == user.id)
            .first()
        )
        if pending is None:
            pending = PendingPasswordResetModel(user_id=user.id, email=user.email)
            db.add(pending)
        pending.email = user.email
        pending.hashed_new_password = get_password_hash(data.new_password)
        pending.created_at = datetime.now(timezone.utc)
        db.commit()

    return {"message": "If that email is registered, your request was submitted and awaits admin approval."}


@router.post("/admin-reset-with-code")
@limiter.limit("5/minute")
def admin_reset_with_code(request: Request, data: AdminResetWithCodeRequest, db: Session = Depends(get_db)):
    """
    Break-glass reset for **admin accounts only**, using the setup-time
    ``ADMIN_RESET_CODE``. Lets a locked-out admin recover without another admin to
    approve them. Disabled (always fails) when the env code is unset. Generic error on
    any failure to avoid leaking which condition failed.
    """
    generic_error = HTTPException(status_code=400, detail="Invalid reset code or account.")

    configured = settings.ADMIN_RESET_CODE
    # Constant-time compare; bail if break-glass isn't configured.
    if not configured or not hmac.compare_digest(data.code, configured):
        raise generic_error

    user = db.query(UserModel).filter(UserModel.email == data.email).first()
    if not user or not user.is_admin:
        raise generic_error

    user.hashed_password = get_password_hash(data.new_password)
    user.token_version += 1  # invalidate existing sessions
    # Drop any pending self-service request now that the password is set directly.
    db.query(PendingPasswordResetModel).filter(PendingPasswordResetModel.user_id == user.id).delete()
    db.commit()

    return {"message": "Admin password has been reset successfully."}

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
