import time
import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db, SessionLocal
from api.models.project import ProjectModel, ProjectStepModel
from api.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse, ProjectWithStepsResponse, ProjectStepResponse
from src.graph import build_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

router = APIRouter(prefix="/projects", tags=["Projects"])

def run_project_graph_task(project_id: int, query: str, data_group_id: int, is_resume: bool = False, force_restart: bool = False):
    """
    Synchronous wrapper to run the async graph task in a separate threadpool worker.
    This prevents the main FastAPI event loop from being blocked by synchronous DB calls
    or heavy CPU bound operations inside the graph.
    """
    asyncio.run(_run_project_graph_task_async(project_id, query, data_group_id, is_resume, force_restart))

async def _run_project_graph_task_async(project_id: int, query: str, data_group_id: int, is_resume: bool = False, force_restart: bool = False):
    """
    Background task to execute the LangGraph for a specific project.
    """
    db = SessionLocal()
    try:
        project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not project:
            return
            
        project.status = "running"
        project.error = None
        
        async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as saver:
            # Compile graph
            app = build_graph(checkpointer=saver)
            config = {"configurable": {"data_group_id": data_group_id, "thread_id": str(project_id)}}
            
            initial_state = {"user_query": query}
            if getattr(project, "intent", None):
                initial_state["intent"] = project.intent
            if is_resume:
                try:
                    state_snapshot = await app.aget_state(config)
                    if state_snapshot and getattr(state_snapshot, "values", None):
                        initial_state = None  # Safe to resume
                    else:
                        is_resume = False  # No checkpoint found, fallback to fresh run
                except Exception as e:
                    print(f"Error fetching state snapshot: {e}")
                    is_resume = False

            if force_restart:
                # Clear langgraph checkpoints for a complete fresh start
                await saver.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(project_id),))
                await saver.conn.execute("DELETE FROM writes WHERE thread_id = ?", (str(project_id),))
                await saver.conn.commit()

            if not is_resume:
                # Clear existing steps for a fresh run
                db.query(ProjectStepModel).filter(ProjectStepModel.project_id == project_id).delete()
            db.commit()
            
            final_state = {}
            try:
                async for chunk in app.astream(initial_state, config=config):
                    node_name = list(chunk.keys())[0]
                    print(f"\n{'='*40}\n[NODE END] {node_name} tamamlandı.\n{'='*40}")
                    
                    chunk_data = chunk[node_name]
                    final_state.update(chunk_data)
                    
                    # Sanitize data for JSON storage
                    safe_data = {}
                    for k, v in chunk_data.items():
                        if k == "model_artifact" or "df" in str(type(v)).lower() or "DataFrame" in str(type(v)):
                            continue
                        if isinstance(v, list) and len(v) > 0 and "DataFrame" in str(type(v[0])):
                            continue
                        
                        # Prevent saving massive datasets to SQLite/JSON by truncating to preview size
                        if k in ["sql_results", "sql_query_results", "X_train", "X_test", "y_train", "y_test"]:
                            if isinstance(v, list):
                                safe_data[k] = v[:5]  # Keep only 5 rows for preview
                                continue
                            else:
                                safe_data[k] = "[Büyük veri kısıtlandı]"  # type: ignore
                                continue
                        try:
                            # Test if JSON serializable
                            json.dumps(v)
                            safe_data[k] = v
                        except:
                            safe_data[k] = str(v)  # type: ignore
                    
                    step = ProjectStepModel(
                        project_id=project_id,
                        node_name=node_name,
                        step_data=safe_data,
                        created_at=int(time.time())
                    )
                    db.add(step)
                    db.commit()
                    
                project.status = "completed"
                project.report = final_state.get("report", "")
            
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                project.status = "failed"
                project.error = error_trace
                print(f"Error in graph execution: {error_trace}")
                
            db.commit()
    finally:
        db.close()

@router.post("/{project_id}/run", response_model=ProjectResponse)
def run_project(project_id: int, background_tasks: BackgroundTasks, force_restart: bool = False, db: Session = Depends(get_db)):
    """
    Starts the project analysis graph in the background.
    If force_restart is True, it will ignore checkpoints and start completely from scratch.
    """
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.data_group_id:
        raise HTTPException(status_code=400, detail="Project does not have an associated DataGroup")
    if not project.query:
        raise HTTPException(status_code=400, detail="Project does not have an analysis query")
        
    if force_restart:
        is_resume = False
    else:
        is_resume = project.status == "failed"
        
    project.status = "running"
    project.error = None
    if force_restart:
        project.report = None
    db.commit()
    db.refresh(project)
    
    background_tasks.add_task(run_project_graph_task, project.id, project.query, project.data_group_id, is_resume, force_restart)
    return project

@router.get("/{project_id}/steps", response_model=List[ProjectStepResponse])
def get_project_steps(project_id: int, db: Session = Depends(get_db)):
    """
    Returns the executed graph steps for a project in real-time or from history.
    """
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    steps = db.query(ProjectStepModel).filter(ProjectStepModel.project_id == project_id).order_by(ProjectStepModel.created_at.asc(), ProjectStepModel.id.asc()).all()
    return steps

@router.post("", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """
    Creates a new project.
    """
    new_project = ProjectModel(
        title=project.title,
        description=project.description,
        query=project.query,
        intent=project.intent,
        data_group_id=project.data_group_id,
        created_at=int(time.time()),
        updated_at=int(time.time())
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project

@router.get("", response_model=List[ProjectResponse])
def get_projects(db: Session = Depends(get_db)):
    """
    Returns a list of all projects.
    """
    return db.query(ProjectModel).all()

@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """
    Returns a specific project by id.
    """
    db_project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    return db_project

@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, project: ProjectUpdate, db: Session = Depends(get_db)):
    """
    Updates a specific project.
    """
    db_project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    if project.title is not None:
        db_project.title = project.title
    if project.description is not None:
        db_project.description = project.description
    if project.query is not None:
        db_project.query = project.query
    if project.intent is not None:
        db_project.intent = project.intent
    if project.data_group_id is not None:
        db_project.data_group_id = project.data_group_id
        
    db_project.updated_at = int(time.time())
    db.commit()
    db.refresh(db_project)
    return db_project

@router.delete("/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """
    Deletes a specific project.
    """
    db_project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    db.delete(db_project)
    db.commit()
    return {"status": "success", "message": f"Project with id {project_id} deleted successfully"}
