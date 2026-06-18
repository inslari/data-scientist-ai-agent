import time as _time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from api.core.database import get_db
from api.core.deps import get_current_user
from api.models.user import UserModel
from api.models.data_group import DataGroupModel
from api.models.data_group_query import DataGroupQueryModel

router = APIRouter(prefix="/data-groups", tags=["Data Group Queries"])


class DataGroupQueryResponse(BaseModel):
    id: int
    data_group_id: int
    title: str
    description: Optional[str]
    sql: str
    expert: Optional[str]
    pack_name: Optional[str]
    created_at: int

    model_config = {"from_attributes": True}


def _assert_group_owner(group_id: int, db: Session, current_user: UserModel) -> DataGroupModel:
    dg = db.query(DataGroupModel).filter(
        DataGroupModel.id == group_id,
        DataGroupModel.user_id == current_user.id,
    ).first()
    if not dg:
        raise HTTPException(status_code=404, detail="Data group not found")
    return dg


@router.get("/{group_id}/queries", response_model=List[DataGroupQueryResponse])
def list_queries(
    group_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    _assert_group_owner(group_id, db, current_user)
    return (
        db.query(DataGroupQueryModel)
        .filter(DataGroupQueryModel.data_group_id == group_id)
        .order_by(DataGroupQueryModel.created_at.desc())
        .all()
    )


@router.delete("/{group_id}/queries/{query_id}")
def delete_query(
    group_id: int,
    query_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    _assert_group_owner(group_id, db, current_user)
    q = db.query(DataGroupQueryModel).filter(
        DataGroupQueryModel.id == query_id,
        DataGroupQueryModel.data_group_id == group_id,
    ).first()
    if not q:
        raise HTTPException(status_code=404, detail="Query not found")
    db.delete(q)
    db.commit()
    return {"status": "success", "message": "Query deleted"}
