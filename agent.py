import asyncio
import os
import sys
from dotenv import load_dotenv

# For Turkish
sys.stdout.reconfigure(encoding='utf-8')

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# MCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

async def run_agent():
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_server.py"]
    )
    
    print("Connecting to the MCP Server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("MCP Connection successful!\n")
            
            @tool
            async def get_schema() -> str:
                """Veritabanındaki tüm tabloların şemasını (sütunları ve tiplerini) getirir. 
                SQL yazmadan ÖNCE her zaman bu aracı kullan!"""
                result = await session.call_tool("get_schema", {})
                return result.content[0].text
                
            @tool
            async def run_sql_query(query: str) -> str:
                """Sadece SELECT içeren bir SQL sorgusunu çalıştırır ve veritabanı sonuçlarını getirir. 
                Eğer hata (SQL Error) dönerse, hatayı oku, NEYİN yanlış olduğunu anla ve SORGUNU DÜZELTEREK tekrar dene!"""
                print(f" [Agent Executing SQL]: {query}")
                result = await session.call_tool("run_sql_query", {"query": query})
                return result.content[0].text

            tools = [get_schema, run_sql_query]
            
            model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
            llm = ChatOpenAI(model=model_name, temperature=0)
            
            system_prompt = (
                "Sen kıdemli bir Veri Mühendisisin. Görevin, sana verilen görevleri veritabanından veri çekerek yanıtlamak. "
                "Cevaplarını her zaman Türkçe ver. "
                "1. Her zaman önce 'get_schema' aracını çağırıp sütun isimlerine bak. "
                "2. Asla tablolarda olmayan bir sütun (örneğin 'maas', 'monthly_spend' vs) uydurma. "
                "3. Eğer SQL hatası alırsan paniğe kapılma, hatayı analiz et ve doğru SQL'i tekrar yaz."
            )
            
            agent_executor = create_react_agent(llm, tools, prompt=system_prompt)
            
            print(f"=================================")
            print(f"!  Data Scientist Agent ({model_name}) Active!  ")
            print(f"=================================\n")
            
            user_input = "Veritabanında hangi tablolar var? Sonra 'customers' tablosundan 3 kişinin yaşını (age) getir. Lütfen önce tablolarda olmayan 'maas' diye bir sütun çekmeyi dene ki hata düzeltme (reflection) yeteneğini görelim."
            print(f"User: {user_input}\n")
            
            async for event in agent_executor.astream(
                {"messages": [("user", user_input)]},
                stream_mode="values"
            ):
                message = event["messages"][-1]
                message.pretty_print()

if __name__ == "__main__":
    asyncio.run(run_agent())
