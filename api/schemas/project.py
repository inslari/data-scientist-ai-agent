from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any, Dict
from api.schemas.data_group import DataGroupResponse

class ProjectStepBase(BaseModel):
    node_name: str
    step_data: Optional[Dict[str, Any]] = None

class ProjectStepResponse(ProjectStepBase):
    id: int
    project_id: int
    created_at: int
    
    model_config = ConfigDict(from_attributes=True)

class ProjectBase(BaseModel):
    title: str
    description: Optional[str] = None
    query: Optional[str] = None
    data_group_id: Optional[int] = None
    intent: Optional[str] = None
    target_col: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    query: Optional[str] = None
    data_group_id: Optional[int] = None
    intent: Optional[str] = None
    target_col: Optional[str] = None

class ProjectResponse(ProjectBase):
    id: int
    user_id: Optional[int] = None
    status: str
    report: Optional[str] = None
    error: Optional[str] = None
    created_at: int
    updated_at: int
    data_group: Optional[DataGroupResponse] = None
    
    model_config = ConfigDict(from_attributes=True)

class ProjectWithStepsResponse(ProjectResponse):
    steps: List[ProjectStepResponse] = []
