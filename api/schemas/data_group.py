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

    # Schema cache metadata — frontend'de "Son Analiz: ..." göstermek için
    schema_cache_updated_at: Optional[int] = None   # Unix timestamp, None = henüz analiz yok

    model_config = ConfigDict(from_attributes=True)


class QueryRequest(BaseModel):
    query: str


# ─────────────────────────────────────────────────────────────────────────────
# Logic Extracted from Routes (To Avoid Circular Dependencies)
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import duckdb
from sqlalchemy.orm import Session
from fastapi import HTTPException
from api.models.data_group import DataGroupModel
from api.routes.mcp import _call_mcp_tool

async def get_data_group_schema_logic(group_id: int, db: Session):
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    if db_group.source_type == "mcp":
        if not db_group.mcp_config:
            raise HTTPException(status_code=400, detail="MCP config is missing for this data group")
        schema_text = await _call_mcp_tool(db_group.mcp_config, "get_schema")
        return {"status": "success", "schema": schema_text}
        
    elif db_group.source_type == "csv":
        schema_lines = []
        for csv_file in db_group.csv_files:
            table_name = os.path.splitext(csv_file.filename)[0].lower()
            table_name = re.sub(r'[^a-z0-9_]', '_', table_name)
            schema_lines.append(f"Table: {table_name}")
            schema_lines.append(f"Description: CSV file with {csv_file.row_count} rows")
            schema_lines.append("Columns:")
            for col in csv_file.columns_schema:
                schema_lines.append(f"- {col['name']} ({col['type']})")
            schema_lines.append("")
        return {"status": "success", "schema": "\n".join(schema_lines)}
    
    raise HTTPException(status_code=400, detail="Unknown source type")

async def run_data_group_query_logic(group_id: int, req: QueryRequest, db: Session):
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    if db_group.source_type == "mcp":
        if not db_group.mcp_config:
            raise HTTPException(status_code=400, detail="MCP config is missing for this data group")
        result_text = await _call_mcp_tool(db_group.mcp_config, "run_sql_query", {"query": req.query})
        return {"status": "success", "result": result_text}
        
    elif db_group.source_type == "csv":
        try:
            con = duckdb.connect(database=':memory:')
            for csv_file in db_group.csv_files:
                table_name = os.path.splitext(csv_file.filename)[0].lower()
                table_name = re.sub(r'[^a-z0-9_]', '_', table_name)
                con.execute(f'CREATE VIEW "{table_name}" AS SELECT * FROM read_csv_auto(\'{csv_file.file_path}\')')
                
            result_df = con.execute(req.query).df()
            
            columns = result_df.columns.tolist()
            output_lines = [f"Columns: {', '.join(columns)}"]
            for row in result_df.itertuples(index=False, name=None):
                output_lines.append(str(row))
            return {"status": "success", "result": "\n".join(output_lines)}
        except Exception as e:
            return {"status": "error", "result": f"SQL Error: {str(e)}"}
            
    raise HTTPException(status_code=400, detail="Unknown source type")
