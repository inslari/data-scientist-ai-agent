from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from fastapi import HTTPException

from api.core.database import SessionLocal
from api.routes.data_group import get_data_group_schema, run_data_group_query, QueryRequest

@tool
async def get_schema(config: RunnableConfig) -> str:
    """
    Fetches the schema of all tables and Database Intelligence Engine analyses.
    Always use this tool BEFORE writing any SQL query!
    """
    data_group_id = config.get("configurable", {}).get("data_group_id")
    if not data_group_id:
        return "API Hatası: data_group_id bulunamadı"

    print(f" 📡 [Dahili İşlem]: Şema çekiliyor... (Data Group: {data_group_id})")
    db = SessionLocal()
    try:
        result = await get_data_group_schema(data_group_id, db)
        return result.get("schema", str(result))
    except HTTPException as e:
        return f"API Hatası: {e.detail}"
    except Exception as e:
        return f"Sistem Hatası: {str(e)}"
    finally:
        db.close()

@tool
async def run_sql_query(query: str, config: RunnableConfig) -> str:
    """
    Executes a SQL query containing only SELECT and returns the results.
    If you get an error, do not panic; analyze the error and correct your query.
    """
    data_group_id = config.get("configurable", {}).get("data_group_id")
    if not data_group_id:
        return "Sistem Hatası: data_group_id bulunamadı"

    print(f" 📡 [Dahili İşlem - SQL]: {query} (Data Group: {data_group_id})")
    db = SessionLocal()
    try:
        req = QueryRequest(query=query)
        result = await run_data_group_query(data_group_id, req, db)
        
        if result.get("status") == "success":
            return result.get("result", str(result))
        else:
            return f"SQL veya Validasyon Hatası: {result.get('result', str(result))}"
            
    except HTTPException as e:
        return f"API Hatası: {e.detail}"
    except Exception as e:
        return f"Sistem Hatası: {str(e)}"
    finally:
        db.close()
