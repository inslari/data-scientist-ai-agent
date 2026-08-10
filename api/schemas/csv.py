from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any

class CSVFileResponse(BaseModel):
    id: int
    user_id: Optional[int] = None
    filename: str
    file_path: str
    file_size: int
    row_count: int
    columns_schema: Optional[List[Dict[str, str]]] = None
    created_at: int
    data_group_id: int
    
    model_config = ConfigDict(from_attributes=True)

class CSVUploadWarning(BaseModel):
    """Dosya yüklendi ama yanlış okunmuş olabilir — arayüz i18n koduyla gösterir."""
    filename: str
    code: str
    params: Dict[str, Any] = {}


class CSVUploadResponse(BaseModel):
    status: str
    message: str
    data: Optional[List[CSVFileResponse]] = None
    errors: Optional[List[Dict[str, str]]] = None
    warnings: Optional[List[CSVUploadWarning]] = None

class CSVSchemaResponse(BaseModel):
    status: str
    data: Optional[List[Dict[str, str]]] = None
    message: Optional[str] = None

class CSVSampleDataRequest(BaseModel):
    limit: int = Field(10, description="Number of rows to fetch")

class CSVSampleDataResponse(BaseModel):
    status: str
    data: Optional[Any] = None
    message: Optional[str] = None

class CSVColumnStatsResponse(BaseModel):
    status: str
    data: Optional[Any] = None
    message: Optional[str] = None
