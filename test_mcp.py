import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_connection():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    print("Connecting to MCP Server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connection successful!\n")
            
            tools_response = await session.list_tools()
            print("Available MCP Tools:")
            for tool in tools_response.tools:
                print(f" - {tool.name}: {tool.description}")
            
            print("\n=== Test 1: Calling get_schema ===")
            schema_result = await session.call_tool("get_schema", {})
            print(schema_result.content[0].text[:300] + "...\n")
            
            print("=== Test 2: Calling run_sql_query ===")
            sql_query = "SELECT age, monthly_spend, is_churned FROM customers LIMIT 3"
            print(f"Query: {sql_query}")
            sql_result = await session.call_tool("run_sql_query", {"query": sql_query})
            print(sql_result.content[0].text)
            
            print("\nTest Successful! MCP Server can communicate with the database.")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
