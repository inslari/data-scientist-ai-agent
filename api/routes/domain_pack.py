from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List

from src.domains import list_packs
from api.core.deps import get_current_user
from api.models.user import UserModel

router = APIRouter(prefix="/domain-packs", tags=["Domain Packs"])


class DomainPackInfo(BaseModel):
    name: str
    label: str


@router.get("", response_model=List[DomainPackInfo])
def get_domain_packs(current_user: UserModel = Depends(get_current_user)):
    """
    Returns the selectable Merchant Intelligence / sector packs for UI dropdowns.
    Always includes "none" (layer disabled). Newly registered sectors appear
    automatically.
    """
    return list_packs()
