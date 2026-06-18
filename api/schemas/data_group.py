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
    user_id: Optional[int] = None

    mcp_config: Optional[MCPConfigResponse] = None
    csv_files: Optional[List[CSVFileResponse]] = None

    # Schema cache metadata — frontend'de "Son Analiz: ..." göstermek için
    schema_cache_updated_at: Optional[int] = None   # Unix timestamp, None = henüz analiz yok

    # Domain (sector) pack for this dataset. None/empty = global DOMAIN_PACK.
    domain_pack: Optional[str] = None

    # Domain/dataset description (markdown) used as pipeline context. None = none.
    domain_doc: Optional[str] = None

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

async def get_data_group_schema_logic(group_id: int, db: Session, user_id: int):
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == user_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    if db_group.source_type == "mcp":
        if not db_group.mcp_config:
            raise HTTPException(status_code=400, detail="MCP config is missing for this data group")
        if db_group.mcp_config.schema_cache:
            print("[MCP] schema_cache hit, skipping MCP call")
            return {"status": "success", "schema": db_group.mcp_config.schema_cache}
        print(f"[MCP] schema_cache miss, calling MCP server for get_schema | type={db_group.mcp_config.type} url={getattr(db_group.mcp_config, 'url', None)}")
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

async def run_data_group_query_logic(group_id: int, req: QueryRequest, db: Session, user_id: int):
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id, DataGroupModel.user_id == user_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    if db_group.source_type == "mcp":
        if not db_group.mcp_config:
            raise HTTPException(status_code=400, detail="MCP config is missing for this data group")
        result_text = await _call_mcp_tool(db_group.mcp_config, "run_sql_query", {"query": req.query})
        return {"status": "success", "result": result_text}
        
    elif db_group.source_type == "csv":
        from api.core.storage import storage
        # For each CSV, download to a temp file if on MinIO, then register in DuckDB
        tmp_paths: list[tuple[str, object]] = []  # (local_path, context_manager_or_none)
        context_managers = []
        try:
            con = duckdb.connect(database=':memory:')
            for csv_file in db_group.csv_files:
                table_name = os.path.splitext(csv_file.filename)[0].lower()
                table_name = re.sub(r'[^a-z0-9_]', '_', table_name)

                ctx = storage.download_to_tempfile(csv_file.file_path)
                local_path = ctx.__enter__()
                context_managers.append(ctx)

                con.execute(f'CREATE VIEW "{table_name}" AS SELECT * FROM read_csv_auto(\'{local_path}\')')

            result_df = con.execute(req.query).df()

            columns = result_df.columns.tolist()
            output_lines = [f"Columns: {', '.join(columns)}"]
            for row in result_df.itertuples(index=False, name=None):
                output_lines.append(str(row))
            return {"status": "success", "result": "\n".join(output_lines)}
        except Exception as e:
            return {"status": "error", "result": f"SQL Error: {str(e)}"}
        finally:
            for ctx in context_managers:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
            
    raise HTTPException(status_code=400, detail="Unknown source type")
