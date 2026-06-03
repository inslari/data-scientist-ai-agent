import os
import json
import duckdb
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel

from api.core.database import get_db
from api.models.data_group import DataGroupModel
from api.schemas.data_group import DataGroupResponse
from api.routes.mcp import _call_mcp_tool

router = APIRouter(prefix="/data-groups", tags=["Data Groups"])

class QueryRequest(BaseModel):
    query: str

@router.get("", response_model=List[DataGroupResponse])
def get_data_groups(db: Session = Depends(get_db)):
    """
    Returns a list of all data groups, including their associated MCP config or CSV files.
    """
    return db.query(DataGroupModel).all()

@router.post("/{group_id}/schema")
async def get_data_group_schema(group_id: int, db: Session = Depends(get_db)):
    """
    Returns the database schema for the specified data group.
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    if db_group.source_type == "mcp":
        if not db_group.mcp_config:
            raise HTTPException(status_code=400, detail="MCP config is missing for this data group")
        schema_text = await _call_mcp_tool(db_group.mcp_config, "get_schema")
        return {"status": "success", "schema": schema_text}
        
    elif db_group.source_type == "csv":
        # Compile schema from CSV metadata
        schema_lines = []
        for csv_file in db_group.csv_files:
            table_name = os.path.splitext(csv_file.filename)[0].replace(" ", "_").lower()
            schema_lines.append(f"Table: {table_name}")
            schema_lines.append(f"Description: CSV file with {csv_file.row_count} rows")
            schema_lines.append("Columns:")
            for col in csv_file.columns_schema:
                schema_lines.append(f"- {col['name']} ({col['type']})")
            schema_lines.append("")
        return {"status": "success", "schema": "\n".join(schema_lines)}
    
    raise HTTPException(status_code=400, detail="Unknown source type")

@router.post("/{group_id}/query")
async def run_data_group_query(group_id: int, req: QueryRequest, db: Session = Depends(get_db)):
    """
    Executes a SQL query against the specified data group.
    """
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
            # Setup in-memory DuckDB and create views for CSVs
            con = duckdb.connect(database=':memory:')
            for csv_file in db_group.csv_files:
                table_name = os.path.splitext(csv_file.filename)[0].replace(" ", "_").lower()
                con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_csv_auto('{csv_file.file_path}')")
                
            # Run query
            result_df = con.execute(req.query).df()
            
            # Convert to string format exactly like what MCP returns (tuple format)
            columns = result_df.columns.tolist()
            output_lines = [f"Columns: {', '.join(columns)}"]
            for row in result_df.itertuples(index=False, name=None):
                output_lines.append(str(row))
            return {"status": "success", "result": "\n".join(output_lines)}
        except Exception as e:
            return {"status": "error", "result": f"SQL Error: {str(e)}"}
            
    raise HTTPException(status_code=400, detail="Unknown source type")

@router.delete("/{group_id}")
def delete_data_group(group_id: int, db: Session = Depends(get_db)):
    """
    Deletes a data group.
    """
    db_group = db.query(DataGroupModel).filter(DataGroupModel.id == group_id).first()
    if not db_group:
        raise HTTPException(status_code=404, detail="Data group not found")
        
    # If it's a CSV group, delete physical files
    if db_group.source_type == "csv" and db_group.csv_files:
        for csv_file in db_group.csv_files:
            if os.path.exists(csv_file.file_path):
                try:
                    os.remove(csv_file.file_path)
                except Exception as e:
                    print(f"Failed to delete file {csv_file.file_path}: {e}")
                    
    db.delete(db_group)
    db.commit()
    return {"status": "success", "message": f"Data group with id {group_id} deleted successfully"}
