import re

with open("api/routes/project.py", "r") as f:
    content = f.read()

# Update run_project_graph_task signature
content = content.replace(
    "async def run_project_graph_task(project_id: int, query: str, data_group_id: int):",
    "async def run_project_graph_task(project_id: int, query: str, data_group_id: int, is_resume: bool = False):"
)

# Remove the inner `is_resume = project.status == "failed"`
old_body = """        project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not project:
            return
            
        is_resume = project.status == "failed"
            
        project.status = "running"
        project.error = None"""
new_body = """        project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not project:
            return
            
        project.status = "running"
        project.error = None"""
content = content.replace(old_body, new_body)

# Update run_project to pass is_resume
old_run = """    project.status = "running"
    project.error = None
    db.commit()
    db.refresh(project)
    
    background_tasks.add_task(run_project_graph_task, project.id, project.query, project.data_group_id)"""
new_run = """    is_resume = project.status == "failed"
    project.status = "running"
    project.error = None
    db.commit()
    db.refresh(project)
    
    background_tasks.add_task(run_project_graph_task, project.id, project.query, project.data_group_id, is_resume)"""
content = content.replace(old_run, new_run)

with open("api/routes/project.py", "w") as f:
    f.write(content)
