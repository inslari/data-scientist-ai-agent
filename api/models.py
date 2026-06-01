from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

class MCPConfigBase(BaseModel):
    name: str = Field(..., description="Unique name for the MCP connection")
    type: str = Field(..., description="Type of connection: 'stdio' or 'sse'")
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None

class MCPConfigCreate(MCPConfigBase):
    pass

class MCPConfigUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Unique name for the MCP connection")
    type: Optional[str] = Field(None, description="Type of connection: 'stdio' or 'sse'")
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    env: Optional[Dict[str, str]] = None

class MCPConfigResponse(MCPConfigBase):
    id: int
    
    model_config = ConfigDict(from_attributes=True)

class ConnectionTestRequest(BaseModel):
    config_id: int

class ToolInfo(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any]

class ConnectionTestResponse(BaseModel):
    status: str
    message: str
    tools: Optional[List[ToolInfo]] = None

class MCPToolRequest(BaseModel):
    config_id: int = Field(..., description="ID of the MCP config to use for the tool call")

class MCPSchemaResponse(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None

class MCPColumnStatsResponse(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None
