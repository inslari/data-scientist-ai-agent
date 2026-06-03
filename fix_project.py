import re

with open("api/routes/project.py", "r") as f:
    content = f.read()

# Add AsyncSqliteSaver import if not exists
if "AsyncSqliteSaver" not in content:
    content = content.replace("from api.routes.mcp import _call_mcp_tool", "from api.routes.mcp import _call_mcp_tool\nfrom langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver")

# We want to keep ProjectStepModel append logic if resuming.
# So we remove the deletion of ProjectStepModel if we are resuming.
# Let's do a simple regex or string replace for run_project_graph_task body.

old_func = """async def run_project_graph_task(project_id: int, query: str, data_group_id: int):
    \"\"\"
    Background task to execute the LangGraph for a specific project.
    \"\"\"
    db = SessionLocal()
    try:
        project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not project:
            return
            
        project.status = "running"
        project.error = None
        # Clear existing steps for a fresh run
        db.query(ProjectStepModel).filter(ProjectStepModel.project_id == project_id).delete()
        db.commit()

        # Compile graph
        app = build_graph()
        initial_state = {"user_query": query}
        config = {"configurable": {"data_group_id": data_group_id}}
        
        final_state = {}
        try:
            async for chunk in app.astream(initial_state, config=config):"""

new_func = """async def run_project_graph_task(project_id: int, query: str, data_group_id: int):
    \"\"\"
    Background task to execute the LangGraph for a specific project.
    \"\"\"
    db = SessionLocal()
    try:
        project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not project:
            return
            
        is_resume = project.status == "failed"
            
        project.status = "running"
        project.error = None
        
        if not is_resume:
            # Clear existing steps for a fresh run
            db.query(ProjectStepModel).filter(ProjectStepModel.project_id == project_id).delete()
        
        db.commit()

        async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as saver:
            # Compile graph
            app = build_graph(checkpointer=saver)
            
            # If resuming, initial_state is None
            initial_state = None if is_resume else {"user_query": query}
            config = {"configurable": {"data_group_id": data_group_id, "thread_id": str(project_id)}}
            
            final_state = {}
            try:
                async for chunk in app.astream(initial_state, config=config):"""

content = content.replace(old_func, new_func)

with open("api/routes/project.py", "w") as f:
    f.write(content)
