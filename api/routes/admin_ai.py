"""
Admin-only management of DB-stored LLM providers, their models, and per-role
assignments. All routes require an admin (get_current_admin). API keys are stored
Fernet-encrypted and only ever returned masked.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.core.database import get_db
from api.core.deps import get_current_admin
from api.core import ai_settings
from api.core.secrets_crypto import encrypt, mask
from api.core.ai_catalog import get_catalog, SUPPORTED_PROVIDER_TYPES, REQUIRES_API_KEY
from api.models.user import UserModel
from api.models.ai_provider import AiProviderModel
from api.models.ai_model import AiModelModel
from api.models.ai_role_assignment import AiRoleAssignmentModel, AI_ROLES
from api.schemas.ai_provider import (
    AiProviderCreate, AiProviderUpdate, AiProviderResponse,
    AiAssignmentUpdate, AiAssignmentResponse,
)

router = APIRouter(prefix="/admin/ai", tags=["Admin AI Providers"])


def _serialize_provider(p: AiProviderModel) -> AiProviderResponse:
    # One provider row = one model (created together, see AiProviderCreate). The
    # ai_models table keeps its own rows because role assignments FK to a model id;
    # ``[0]`` is safe for anything this API creates. Rows left over from the earlier
    # multi-model shape would surface only their first model — nothing is deleted.
    model = p.models[0] if p.models else None
    return AiProviderResponse(
        id=p.id,
        label=p.label,
        provider_type=p.provider_type,
        base_url=p.base_url,
        has_key=bool(p.api_key_encrypted),
        api_key_masked=mask(p.api_key_encrypted),
        created_at=p.created_at,
        updated_at=p.updated_at,
        model_id=model.id if model else None,
        model_name=model.model_name if model else None,
    )


# ── Catalog ───────────────────────────────────────────────────────────────────
@router.get("/catalog")
def read_catalog(current_admin: UserModel = Depends(get_current_admin)):
    """Curated provider→models map + UI hints for the dropdowns."""
    return get_catalog()


# ── Providers ─────────────────────────────────────────────────────────────────
@router.get("/providers", response_model=list[AiProviderResponse])
def list_providers(
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    providers = db.query(AiProviderModel).order_by(AiProviderModel.id).all()
    return [_serialize_provider(p) for p in providers]


@router.post("/providers", response_model=AiProviderResponse, status_code=201)
def create_provider(
    data: AiProviderCreate,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    ptype = data.provider_type.lower().strip()
    if ptype not in SUPPORTED_PROVIDER_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported provider type: {data.provider_type}")
    if not data.label.strip():
        raise HTTPException(status_code=400, detail="Label is required.")
    model_name = data.model_name.strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required.")

    provider = AiProviderModel(
        label=data.label.strip(),
        provider_type=ptype,
        api_key_encrypted=encrypt(data.api_key) if data.api_key else None,
        base_url=(data.base_url.strip() or None) if data.base_url else None,
    )
    # Provider and its model are written in ONE transaction: a provider without a
    # model can't be assigned to a role, so a partial commit would leave dead rows.
    provider.models.append(AiModelModel(model_name=model_name))
    db.add(provider)
    db.commit()
    db.refresh(provider)
    ai_settings.invalidate()
    return _serialize_provider(provider)


@router.put("/providers/{provider_id}", response_model=AiProviderResponse)
def update_provider(
    provider_id: int,
    data: AiProviderUpdate,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    provider = db.query(AiProviderModel).filter(AiProviderModel.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found.")

    if data.label is not None and data.label.strip():
        provider.label = data.label.strip()
    if data.base_url is not None:
        provider.base_url = data.base_url.strip() or None

    # Renaming edits the existing model row in place rather than replacing it, so the
    # model id — and therefore any role assignment pointing at it — survives the edit.
    if data.model_name is not None and data.model_name.strip():
        new_name = data.model_name.strip()
        if provider.models:
            provider.models[0].model_name = new_name
        else:
            provider.models.append(AiModelModel(model_name=new_name))

    # Key handling: clear wins; else a non-empty key replaces; empty/None preserves.
    if data.clear_key:
        provider.api_key_encrypted = None
    elif data.api_key:
        provider.api_key_encrypted = encrypt(data.api_key)

    db.commit()
    db.refresh(provider)
    ai_settings.invalidate()
    return _serialize_provider(provider)


@router.delete("/providers/{provider_id}", status_code=204)
def delete_provider(
    provider_id: int,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    provider = db.query(AiProviderModel).filter(AiProviderModel.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found.")
    db.delete(provider)  # cascades to models; assignments SET NULL via FK
    db.commit()
    ai_settings.invalidate()
    return None


# NOTE: there are no standalone model endpoints. A model is created with its provider
# and renamed through PUT /providers/{id}; deleting the provider deletes it (cascade).


# ── Role assignments ──────────────────────────────────────────────────────────
def _assignment_detail(db: Session, role: str, assignment: AiRoleAssignmentModel | None) -> AiAssignmentResponse:
    if not assignment or assignment.model_id is None:
        return AiAssignmentResponse(role=role)
    model = db.query(AiModelModel).filter(AiModelModel.id == assignment.model_id).first()
    if not model:
        return AiAssignmentResponse(role=role)
    provider = db.query(AiProviderModel).filter(AiProviderModel.id == model.provider_id).first()
    return AiAssignmentResponse(
        role=role,
        model_id=model.id,
        provider_id=provider.id if provider else None,
        provider_type=provider.provider_type if provider else None,
        provider_label=provider.label if provider else None,
        model_name=model.model_name,
    )


@router.get("/assignments", response_model=list[AiAssignmentResponse])
def list_assignments(
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    rows = {a.role: a for a in db.query(AiRoleAssignmentModel).all()}
    return [_assignment_detail(db, role, rows.get(role)) for role in AI_ROLES]


@router.put("/assignments/{role}", response_model=AiAssignmentResponse)
def set_assignment(
    role: str,
    data: AiAssignmentUpdate,
    current_admin: UserModel = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    if role not in AI_ROLES:
        raise HTTPException(status_code=400, detail=f"Unknown role: {role}. Valid roles: {', '.join(AI_ROLES)}")

    if data.model_id is not None:
        model = db.query(AiModelModel).filter(AiModelModel.id == data.model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found.")
        provider = db.query(AiProviderModel).filter(AiProviderModel.id == model.provider_id).first()
        # Non-local providers can't be used without a key.
        if provider and provider.provider_type in REQUIRES_API_KEY and not provider.api_key_encrypted:
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{provider.label}' has no API key configured; add one before assigning it.",
            )

    assignment = db.query(AiRoleAssignmentModel).filter(AiRoleAssignmentModel.role == role).first()
    if not assignment:
        assignment = AiRoleAssignmentModel(role=role)
        db.add(assignment)
    assignment.model_id = data.model_id
    db.commit()
    db.refresh(assignment)
    ai_settings.invalidate()
    return _assignment_detail(db, role, assignment)
