from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from api.schemas.csv import CSVFileResponse


class ConverterOptionSchema(BaseModel):
    name: str
    type: str
    default: Any = None
    label_key: str
    help_key: Optional[str] = None
    choices: Optional[List[Dict[str, Any]]] = None
    source: Optional[str] = None
    default_source: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None


class ConverterInfo(BaseModel):
    id: str
    label_key: str
    description_key: Optional[str] = None
    source_extensions: List[str]
    target_extension: str
    produces_multiple: bool
    max_file_mb: int
    live_preview: bool = False
    can_save_to_assets: bool
    options: List[ConverterOptionSchema]


class ConvertersResponse(BaseModel):
    status: str
    data: List[ConverterInfo]


class ConvertInspectResponse(BaseModel):
    status: str
    filename: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ConvertedFileInfo(BaseModel):
    filename: str
    size: int
    rows: Optional[int] = None
    columns: Optional[int] = None
    source_filename: Optional[str] = None
    sheet: Optional[str] = None


class ConvertToAssetsResponse(BaseModel):
    status: str
    message: str
    data_group_id: Optional[int] = None
    data: Optional[List[CSVFileResponse]] = None
    converted: Optional[List[ConvertedFileInfo]] = None
    errors: Optional[List[Dict[str, str]]] = None
