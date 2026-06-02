import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db
from api.models.project import ProjectModel
from api.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse

router = APIRouter(prefix="/projects", tags=["Projects"])

@router.post("", response_model=ProjectResponse)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """
    Creates a new project.
    """
    new_project = ProjectModel(
        title=project.title,
        description=project.description,
        query=project.query,
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
