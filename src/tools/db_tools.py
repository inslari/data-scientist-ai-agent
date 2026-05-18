import requests
from langchain_core.tools import tool
from src.core.config import settings

@tool
def get_schema() -> str:
    """
    Fetches the schema of all tables and Database Intelligence Engine analyses.
    Always use this tool BEFORE writing any SQL query!
    """
    print(" 📡 [API İsteği]: Şema çekiliyor...")
    try:
        response = requests.get(f"{settings.API_BASE_URL}/mcp/schema") 
        response.raise_for_status()
        data = response.json()
        return data.get("schema", str(data))
    except Exception as e:
        return f"API Hatası: {str(e)}"

@tool
def run_sql_query(query: str) -> str:
    """
    Executes a SQL query containing only SELECT and returns the results.
    If you get an error, do not panic; analyze the error and correct your query.
    """
    print(f" 📡 [API İsteği - SQL]: {query}")
    try:
        payload = {"query": query}
        response = requests.post(f"{settings.API_BASE_URL}/mcp/query", json=payload)
 
        if response.status_code == 200:
            data = response.json()
            return data.get("result", str(data))
        else:
            return f"SQL veya Validasyon Hatası: {response.text}"
            
    except Exception as e:
        return f"Sistem Hatası: {str(e)}"
