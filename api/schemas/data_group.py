from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from api.schemas.mcp import MCPConfigResponse
from api.schemas.csv import CSVFileResponse

class DataGroupBase(BaseModel):
    name: str
    source_type: str

class DataGroupResponse(DataGroupBase):
    id: int
    created_at: int
    
    mcp_config: Optional[MCPConfigResponse] = None
    csv_files: Optional[List[CSVFileResponse]] = None
    
    model_config = ConfigDict(from_attributes=True)
