import re

with open("src/nodes/sql_node.py", "r") as f:
    content = f.read()

# 1. Update imports
if "RunnableConfig" not in content:
    content = content.replace("from ..schemas.sql_schemas import SQLGenerationResult, ChartConfig", "from ..schemas.sql_schemas import SQLGenerationResult, ChartConfig\nfrom langchain_core.runnables import RunnableConfig")

# Remove mcp_client import
content = content.replace("from ..services.mcp_client import mcp_run_query", 
"""from api.core.database import SessionLocal
from api.routes.data_group import run_data_group_query, QueryRequest
import ast""")

# 2. Update signature
content = content.replace("def sql_node(state: AgentState) -> AgentState:", "async def sql_node(state: AgentState, config: RunnableConfig) -> AgentState:")

# 3. Add data_group_id
content = content.replace('user_query = state["user_query"]',
'''user_query = state["user_query"]
    data_group_id = config.get("configurable", {}).get("data_group_id")
    if not data_group_id:
        return {**state, "error": "data_group_id is missing", "current_step": "sql_node"}
''')

# 4. Replace mcp_run_query
old_exec = "columns, rows = mcp_run_query(sql, target_db_name)"
new_exec = '''db = SessionLocal()
        try:
            req = QueryRequest(query=sql)
            res = await run_data_group_query(data_group_id, req, db)
            raw_text = str(res.get("result", "")) if res.get("status") == "success" else ""
            
            lines = raw_text.strip().splitlines()
            columns = []
            rows = []
            for i, line in enumerate(lines):
                if line.startswith("Columns:"):
                    columns = [c.strip() for c in line.replace("Columns:", "").split(",")]
                elif line.startswith("Results:") or line.startswith("("):
                    data_lines = lines[i + 1:] if line.startswith("Results:") else lines[i:]
                    for dl in data_lines:
                        dl = dl.strip()
                        if not dl or dl.startswith("..."):
                            continue
                        if dl.startswith("(") and dl.endswith(")"):
                            try:
                                rows.append(ast.literal_eval(dl))
                            except Exception:
                                pass
                    break
        finally:
            db.close()'''

content = content.replace(old_exec, new_exec)

with open("src/nodes/sql_node.py", "w") as f:
    f.write(content)
