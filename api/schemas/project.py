from pydantic import BaseModel, ConfigDict
from typing import Optional
from api.schemas.data_group import DataGroupResponse

class ProjectBase(BaseModel):
    title: str
    description: Optional[str] = None
    query: Optional[str] = None
    data_group_id: Optional[int] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    query: Optional[str] = None
    data_group_id: Optional[int] = None

class ProjectResponse(ProjectBase):
    id: int
    created_at: int
    updated_at: int
    data_group: Optional[DataGroupResponse] = None
    
    model_config = ConfigDict(from_attributes=True)
