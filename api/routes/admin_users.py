"""
Admin-only user management for the closed platform: approve (verify) new accounts and
approve/reject mail-free password-reset requests. All routes require an admin.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.deps import get_current_admin
from api.models.user import UserModel
from api.models.password_reset_request import PendingPasswordResetModel
from api.schemas.admin_users import AdminUserResponse, PendingResetResponse

router = APIRouter(prefix="/admin/users", tags=["Admin Users"])


@router.get("", response_model=list[AdminUserResponse])
def list_users(
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    users = db.query(UserModel).order_by(UserModel.id).all()
    pending_ids = {r.user_id for r in db.query(PendingPasswordResetModel.user_id).all()}
    return [
        AdminUserResponse(
            id=u.id, email=u.email, full_name=u.full_name,
            is_active=u.is_active, is_verified=u.is_verified, is_admin=u.is_admin,
            created_at=u.created_at, has_pending_reset=u.id in pending_ids,
        )
        for u in users
    ]


@router.post("/{user_id}/verify", response_model=AdminUserResponse)
def verify_user(
    user_id: int,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Approve a pending account so it can log in (replaces email verification)."""
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.is_verified = True
    db.commit()
    db.refresh(user)
    has_pending = db.query(PendingPasswordResetModel).filter(
        PendingPasswordResetModel.user_id == user.id
    ).first() is not None
    return AdminUserResponse(
        id=user.id, email=user.email, full_name=user.full_name,
        is_active=user.is_active, is_verified=user.is_verified, is_admin=user.is_admin,
        created_at=user.created_at, has_pending_reset=has_pending,
    )


# ── Password-reset approval ───────────────────────────────────────────────────
reset_router = APIRouter(prefix="/admin/password-resets", tags=["Admin Users"])


@reset_router.get("", response_model=list[PendingResetResponse])
def list_pending_resets(
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    rows = db.query(PendingPasswordResetModel).order_by(PendingPasswordResetModel.created_at).all()
    return [PendingResetResponse.model_validate(r) for r in rows]


@reset_router.post("/{reset_id}/approve")
def approve_reset(
    reset_id: int,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """
    Apply the user's hashed new password (never seen by the admin) and invalidate their
    existing sessions. The admin must verify the requester's identity out-of-band first.
    """
    req = db.query(PendingPasswordResetModel).filter(PendingPasswordResetModel.id == reset_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Reset request not found.")
    user = db.query(UserModel).filter(UserModel.id == req.user_id).first()
    if not user:
        db.delete(req)
        db.commit()
        raise HTTPException(status_code=404, detail="User no longer exists.")

    user.hashed_password = req.hashed_new_password
    user.token_version += 1  # log the user out everywhere; the new password now applies
    db.delete(req)
    db.commit()
    return {"message": "Password reset approved."}


@reset_router.post("/{reset_id}/reject")
def reject_reset(
    reset_id: int,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    req = db.query(PendingPasswordResetModel).filter(PendingPasswordResetModel.id == reset_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Reset request not found.")
    db.delete(req)
    db.commit()
    return {"message": "Password reset request rejected."}
