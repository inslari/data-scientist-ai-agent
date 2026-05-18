import os
import sys
import requests
from dotenv import load_dotenv


sys.stdout.reconfigure(encoding='utf-8')

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

load_dotenv()
API_BASE_URL = "http://127.0.0.1:8000"

@tool
def get_schema() -> str:
    """
    Fetches the schema of all tables and Database Intelligence Engine analyses.
    Always use this tool BEFORE writing any SQL query!
    """
    print(" 📡 [API İsteği]: Şema çekiliyor...")
    try:
        response = requests.get(f"{API_BASE_URL}/mcp/schema") 
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
        response = requests.post(f"{API_BASE_URL}/mcp/query", json=payload)
 
        if response.status_code == 200:
            data = response.json()
            return data.get("result", str(data))
        else:
            return f"SQL veya Validasyon Hatası: {response.text}"
            
    except Exception as e:
        return f"Sistem Hatası: {str(e)}"


def run_single_agent():
    """Single agent function to test the new API architecture"""
    tools = [get_schema, run_sql_query]
    
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    system_prompt = (
        "Sen kıdemli bir Veri Mühendisisin. Görevin, sana verilen görevleri veritabanından veri çekerek yanıtlamak. "
        "Cevaplarını her zaman Türkçe ver. "
        "1. Her zaman önce 'get_schema' aracını çağırıp sütun isimlerine bak. "
        "2. Asla tablolarda olmayan bir sütun uydurma. "
        "3. Eğer SQL hatası alırsan paniğe kapılma, hatayı analiz et ve doğru SQL'i tekrar yaz."
    )
    
    agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
    
    print(f"=================================")
    print(f"! Veri Mühendisi Ajan Aktif !    ")
    print(f"=================================\n")
    
    user_input = "order_items tablosunda kaç satır var?"
    print(f"Kullanıcı: {user_input}\n")
    

    for event in agent_executor.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print(value["messages"][-1].content)

if __name__ == "__main__":
    run_single_agent()