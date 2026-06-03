with open("api/routes/project.py", "r") as f:
    content = f.read()

if "AsyncSqliteSaver" not in content:
    content = "from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver\n" + content

with open("api/routes/project.py", "w") as f:
    f.write(content)
