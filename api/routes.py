from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import httpx
import shutil
import asyncio
import contextlib
import time

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters, get_default_environment
from mcp.client.streamable_http import streamable_http_client

from api.database import get_db, MCPConfigModel
from api.models import (
    MCPConfigCreate, MCPConfigUpdate, MCPConfigResponse,
    ConnectionTestRequest, ConnectionTestResponse,
    MCPToolRequest, MCPSchemaResponse, MCPColumnStatsResponse,
    RefreshCacheResponse, MCPSampleDataRequest, MCPSampleDataResponse
)

router = APIRouter(prefix="/mcp", tags=["MCP Configuration"])

@router.post("/config", response_model=MCPConfigResponse)
def create_mcp_config(config: MCPConfigCreate, db: Session = Depends(get_db)):
    """
    Kullanıcıdan MCP bilgilerini alıp veritabanına kaydeder.
    """
    db_config = db.query(MCPConfigModel).filter(MCPConfigModel.name == config.name).first()
    if db_config:
        raise HTTPException(status_code=400, detail="An MCP config with this name already exists")
    
    new_config = MCPConfigModel(
        name=config.name,
        type=config.type,
        command=config.command,
        args=config.args,
        url=config.url,
        env=config.env
    )
    db.add(new_config)
    db.commit()
    db.refresh(new_config)
    return new_config

@router.get("/config", response_model=list[MCPConfigResponse])
def get_mcp_configs(db: Session = Depends(get_db)):
    """
    Kayıtlı tüm MCP konfigürasyonlarını döner.
    """
    return db.query(MCPConfigModel).all()

@router.put("/config/{config_id}", response_model=MCPConfigResponse)
def update_mcp_config(config_id: int, config_update: MCPConfigUpdate, db: Session = Depends(get_db)):
    """
    Belirtilen ID'ye sahip MCP konfigürasyonunu günceller.
    """
    db_config = db.query(MCPConfigModel).filter(MCPConfigModel.id == config_id).first()
    if not db_config:
        raise HTTPException(status_code=404, detail="MCP config not found")
        
    if config_update.name is not None and config_update.name != db_config.name:
        existing = db.query(MCPConfigModel).filter(MCPConfigModel.name == config_update.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="An MCP config with this new name already exists")
    
    update_data = config_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_config, key, value)
        
    db.commit()
    db.refresh(db_config)
    return db_config

@router.delete("/config/{config_id}")
def delete_mcp_config(config_id: int, db: Session = Depends(get_db)):
    """
    Belirtilen ID'ye sahip MCP konfigürasyonunu siler.
    """
    db_config = db.query(MCPConfigModel).filter(MCPConfigModel.id == config_id).first()
    if not db_config:
        raise HTTPException(status_code=404, detail="MCP config not found")
        
    db.delete(db_config)
    db.commit()
    return {"status": "success", "message": f"MCP config with id {config_id} deleted successfully"}

@router.post("/test-connection", response_model=ConnectionTestResponse)
async def test_mcp_connection(req: ConnectionTestRequest, db: Session = Depends(get_db)):
    """
    Kayıtlı bir MCP konfigürasyonuna bağlanmayı dener ve araçlarını listeler.
    """
    db_config = db.query(MCPConfigModel).filter(MCPConfigModel.id == req.config_id).first()
    if not db_config:
        raise HTTPException(status_code=404, detail="MCP config not found")
    
    async with contextlib.AsyncExitStack() as stack:
        try:
            if db_config.type == "stdio":
                if not db_config.command:
                    raise HTTPException(status_code=400, detail="Command is required for stdio type")
                    
                if not shutil.which(db_config.command):
                    return {"status": "error", "message": f"Command not found in system PATH: {db_config.command}"}
                
                env = get_default_environment()
                if db_config.env:
                    env.update(db_config.env)
                    
                server_params = StdioServerParameters(
                    command=db_config.command,
                    args=db_config.args or [],
                    env=env
                )
                
                try:
                    read, write = await asyncio.wait_for(stack.enter_async_context(stdio_client(server_params)), timeout=10.0)
                except asyncio.TimeoutError:
                    return {"status": "error", "message": "Connection to stdio MCP server timed out."}
                    
            elif db_config.type == "sse":
                if not db_config.url:
                    raise HTTPException(status_code=400, detail="URL is required for sse type")
                    
                try:
                    try:
                        # streamable_http_client returns (read, write, get_session_id_fn)
                        result = await asyncio.wait_for(stack.enter_async_context(streamable_http_client(db_config.url)), timeout=10.0)
                        read, write = result[0], result[1]
                    except Exception:
                        read, write = await asyncio.wait_for(stack.enter_async_context(sse_client(db_config.url)), timeout=10.0)
                except asyncio.TimeoutError:
                    return {"status": "error", "message": "Connection to SSE/HTTP MCP server timed out."}
                except Exception as e:
                    return {"status": "error", "message": f"Connection to MCP server failed: {str(e)}"}
            else:
                return {"status": "error", "message": f"Unknown connection type: {db_config.type}"}
                
            # Initialize session
            session = await stack.enter_async_context(ClientSession(read, write))
            
            try:
                await asyncio.wait_for(session.initialize(), timeout=10.0)
                tools_response = await asyncio.wait_for(session.list_tools(), timeout=10.0)
            except asyncio.TimeoutError:
                return {"status": "error", "message": "MCP session initialization or tool listing timed out."}
                
            tools_list = []
            tool_names = set()
            for t in tools_response.tools:
                tool_names.add(t.name)
                tools_list.append({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema
                })
                

            print(tool_names)   
            # Sistemin çalışması için zorunlu olan araçların kontrolü
            required_tools = {"get_schema", "get_column_stats", "run_sql_query"}
            missing_tools = required_tools - tool_names
            
            if missing_tools:
                return {
                    "status": "error", 
                    "message": f"Connected to MCP server, but it is NOT the correct server. Missing required tools: {', '.join(missing_tools)}",
                    "tools": tools_list
                }
                
            return {
                "status": "success", 
                "message": "Successfully connected and verified. This is the correct MCP server with all required tools.",
                "tools": tools_list
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to connect to MCP server: {str(e)}"}


async def _call_mcp_tool(db_config, tool_name: str, tool_args: dict = None) -> str:
    """
    Verilen MCP konfigürasyonuna bağlanır ve belirtilen tool'u çalıştırır.
    Sonucu ham string olarak döner.
    """
    async with contextlib.AsyncExitStack() as stack:
        if db_config.type == "stdio":
            if not db_config.command or not shutil.which(db_config.command):
                raise ValueError(f"Command not found: {db_config.command}")
            env = get_default_environment()
            if db_config.env:
                env.update(db_config.env)
            server_params = StdioServerParameters(
                command=db_config.command,
                args=db_config.args or [],
                env=env
            )
            read, write = await asyncio.wait_for(
                stack.enter_async_context(stdio_client(server_params)), timeout=15.0
            )
        elif db_config.type == "sse":
            if not db_config.url:
                raise ValueError("URL is required for sse type")
            try:
                result = await asyncio.wait_for(
                    stack.enter_async_context(streamable_http_client(db_config.url)), timeout=15.0
                )
                read, write = result[0], result[1]
            except Exception:
                read, write = await asyncio.wait_for(
                    stack.enter_async_context(sse_client(db_config.url)), timeout=15.0
                )
        else:
            raise ValueError(f"Unknown connection type: {db_config.type}")

        session = await stack.enter_async_context(ClientSession(read, write))
        await asyncio.wait_for(session.initialize(), timeout=10.0)
        result = await asyncio.wait_for(session.call_tool(tool_name, tool_args or {}), timeout=30.0)

        # Tool sonucu genellikle bir liste içinde TextContent nesnesi döner
        if result.content:
            parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return ""


@router.post("/schema", response_model=MCPSchemaResponse)
async def get_mcp_schema(req: MCPToolRequest, db: Session = Depends(get_db)):
    """
    Bağlı MCP sunucusundaki tüm veritabanı tablolarının şemasını (schema) döner.
    MCP sunucusundaki 'get_schema' aracını çalıştırır.
    """
    db_config = db.query(MCPConfigModel).filter(MCPConfigModel.id == req.config_id).first()
    if not db_config:
        raise HTTPException(status_code=404, detail="MCP config not found")
        
    if db_config.schema_cache:
        return {"status": "success", "data": db_config.schema_cache}

    try:
        current_time = int(time.time())
        schema_text = await _call_mcp_tool(db_config, "get_schema")
        
        # Save to DB cache
        db_config.schema_cache = schema_text
        db_config.schema_cache_updated_at = current_time
        db.commit()
        
        return {"status": "success", "data": schema_text}
    except asyncio.TimeoutError:
        return {"status": "error", "message": "MCP tool call timed out."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get schema: {str(e)}"}


@router.post("/column-stats", response_model=MCPColumnStatsResponse)
async def get_mcp_column_stats(req: MCPToolRequest, db: Session = Depends(get_db)):
    """
    Bağlı MCP sunucusundaki tüm tabloların kolon istatistiklerini döner.
    MCP sunucusundaki 'get_column_stats' aracını çalıştırır.
    """
    db_config = db.query(MCPConfigModel).filter(MCPConfigModel.id == req.config_id).first()
    if not db_config:
        raise HTTPException(status_code=404, detail="MCP config not found")
        
    if db_config.column_stats_cache:
        return {"status": "success", "data": db_config.column_stats_cache}

    try:
        current_time = int(time.time())
        stats_text = await _call_mcp_tool(db_config, "get_column_stats")
        
        # Save to DB cache
        db_config.column_stats_cache = stats_text
        db_config.column_stats_cache_updated_at = current_time
        db.commit()
        
        return {"status": "success", "data": stats_text}
    except asyncio.TimeoutError:
        return {"status": "error", "message": "MCP tool call timed out."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get column stats: {str(e)}"}

@router.post("/refresh-stats", response_model=RefreshCacheResponse)
async def refresh_mcp_db_stats(req: MCPToolRequest, db: Session = Depends(get_db)):
    """
    Belirtilen MCP sunucusundaki şema ve kolon istatistiklerini zorla yeniler
    ve veritabanına kaydeder.
    """
    db_config = db.query(MCPConfigModel).filter(MCPConfigModel.id == req.config_id).first()
    if not db_config:
        raise HTTPException(status_code=404, detail="MCP config not found")
        
    try:
        current_time = int(time.time())
        
        # Şemayı çek
        schema_text = await _call_mcp_tool(db_config, "get_schema")
        db_config.schema_cache = schema_text
        db_config.schema_cache_updated_at = current_time
        
        # Kolon istatistiklerini çek
        stats_text = await _call_mcp_tool(db_config, "get_column_stats")
        db_config.column_stats_cache = stats_text
        db_config.column_stats_cache_updated_at = current_time
        
        db.commit()
        
        return {"status": "success", "message": "Cache refreshed successfully."}
    except asyncio.TimeoutError:
        return {"status": "error", "message": "MCP tool call timed out during refresh."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to refresh cache: {str(e)}"}

@router.post("/sample-data", response_model=MCPSampleDataResponse)
async def get_mcp_sample_data(req: MCPSampleDataRequest, db: Session = Depends(get_db)):
    """
    Belirtilen tablodan belirtilen limit kadar örnek veri döner.
    """
    db_config = db.query(MCPConfigModel).filter(MCPConfigModel.id == req.config_id).first()
    if not db_config:
        raise HTTPException(status_code=404, detail="MCP config not found")
        
    try:
        import ast
        
        # İlk olarak standart LIMIT sentaksını deniyoruz
        query = f'SELECT * FROM "{req.table_name}" LIMIT {req.limit}'
        args = {"query": query}
        
        sample_data_text = await _call_mcp_tool(db_config, "run_sql_query", args)
        
        # Eğer syntax hatası dönerse ve TOP içeren MSSQL gibi bir db ihtimali varsa
        if "SQL Error" in sample_data_text and ("syntax" in sample_data_text.lower() or "near" in sample_data_text.lower()):
            query = f'SELECT TOP {req.limit} * FROM "{req.table_name}"'
            args = {"query": query}
            sample_data_text = await _call_mcp_tool(db_config, "run_sql_query", args)
            
        if "SQL Error" in sample_data_text or "Error:" in sample_data_text:
            return {"status": "error", "message": sample_data_text}
            
        # Parse output format:
        # Columns: id, name
        # Returned X rows.
        # Results:
        # (1, 'foo')
        # ...
        lines = sample_data_text.split('\n')
        columns = []
        rows = []
        
        in_results = False
        for line in lines:
            line = line.strip()
            if line.startswith("Columns:"):
                col_str = line.replace("Columns:", "").strip()
                columns = [c.strip() for c in col_str.split(',')]
            elif line.startswith("Results:"):
                in_results = True
            elif in_results and line.startswith('('):
                try:
                    # Parse string tuple into actual tuple
                    row_tuple = ast.literal_eval(line)
                    if isinstance(row_tuple, tuple):
                        row_dict = dict(zip(columns, row_tuple))
                        rows.append(row_dict)
                except Exception:
                    # Fallback for unparseable lines (like datetimes or Decimals)
                    rows.append({"raw_data": line})
                    
        return {
            "status": "success", 
            "data": {
                "columns": columns,
                "rows": rows
            }
        }
            
    except asyncio.TimeoutError:
        return {"status": "error", "message": "MCP tool call timed out."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get sample data: {str(e)}"}
