"Bu bir prompt deneme scriptidir."

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
load_dotenv()

MCP_CONFIG = {
    "database": {
        "url": "http://127.0.0.1:9001/mcp",
        "transport": "streamable_http",
    }
}

async def run_agent(user_input: str):
    client = MultiServerMCPClient(MCP_CONFIG)  # type: ignore[arg-type]
    tools = await client.get_tools()

    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    system_prompt = (
        "Sen kıdemli bir Veri Mühendisisin. Görevin, sana verilen görevleri veritabanından veri çekerek yanıtlamak. "
        "Cevaplarını her zaman Türkçe ver. "
        "1. Her zaman önce 'get_schema' aracını çağırıp sütun isimlerine bak. "
        "2. Asla tablolarda olmayan bir sütun uydurma. "
        "3. Eğer SQL hatası alırsan paniğe kapılma, hatayı analiz et ve doğru SQL'i tekrar yaz."
    )

    agent = create_react_agent(llm, tools, prompt=system_prompt)

    print("=================================")
    print("! Veri Mühendisi Ajan Aktif !    ")
    print("=================================\n")
    print(f"Kullanıcı: {user_input}\n")

    async for event in agent.astream({"messages": [("user", user_input)]}):
        for value in event.values():
            if "messages" in value:
                print(value["messages"][-1].content)

if __name__ == "__main__":
    user_input = "orders tablosundan order_date'in min, max değerini ve 2024-11-16'dan önce (yani reference_date - 180 gün) kaç sipariş olduğunu göster"
    asyncio.run(run_agent(user_input))