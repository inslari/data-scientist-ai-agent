import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from api.core.database import get_db
from api.models.project import ProjectModel, ProjectStepModel
from api.models.job import BackgroundJobModel
from api.schemas.project import ProjectCreate, ProjectUpdate, ProjectResponse, ProjectWithStepsResponse, ProjectStepResponse
from api.core.deps import get_current_user
from api.models.user import UserModel

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post("/{project_id}/run", response_model=ProjectResponse)
def run_project(
    project_id: int,
    force_restart: bool = False,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Enqueues the project analysis graph as a background job.
    Returns immediately — the dedicated worker loop picks it up asynchronously.
    If force_restart is True, it will ignore checkpoints and start completely from scratch.
    """
    project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == current_user.id
    ).first()
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

    # Update project status to 'queued' immediately so the UI can reflect it
    project.status = "queued"
    project.error = None
    if force_restart:
        project.report = None

    # Enqueue the job (Outbox pattern)
    job = BackgroundJobModel(
        project_id=project.id,
        task_type="run_project",
        query=project.query,
        data_group_id=project.data_group_id,
        is_resume=int(is_resume),
        force_restart=int(force_restart),
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(project)
    return project


@router.get("/{project_id}/steps", response_model=List[ProjectStepResponse])
def get_project_steps(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Returns the executed graph steps for a project in real-time or from history.
    """
    project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == current_user.id
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    steps = (
        db.query(ProjectStepModel)
        .filter(ProjectStepModel.project_id == project_id)
        .order_by(ProjectStepModel.created_at.asc(), ProjectStepModel.id.asc())
        .all()
    )
    return steps


@router.post("", response_model=ProjectResponse)
def create_project(
    project: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Creates a new project.
    """
    new_project = ProjectModel(
        title=project.title,
        description=project.description,
        query=project.query,
        intent=project.intent,
        target_col=project.target_col,
        modality=project.modality or None,
        time_col=project.time_col or None,
        data_group_id=project.data_group_id,
        domain_pack=project.domain_pack or None,
        user_id=current_user.id,
        created_at=int(time.time()),
        updated_at=int(time.time()),
    )
    db.add(new_project)
    db.commit()
    db.refresh(new_project)
    return new_project


@router.get("", response_model=List[ProjectResponse])
def get_projects(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Returns a list of all projects.
    """
    return db.query(ProjectModel).filter(ProjectModel.user_id == current_user.id).all()


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Returns a specific project by id.
    """
    db_project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == current_user.id
    ).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")
    return db_project


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: int,
    project: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Updates a specific project.
    """
    db_project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == current_user.id
    ).first()
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
    if project.target_col is not None:
        db_project.target_col = project.target_col
    if project.data_group_id is not None:
        db_project.data_group_id = project.data_group_id
    if "domain_pack" in project.model_fields_set:
        db_project.domain_pack = project.domain_pack or None
    if "modality" in project.model_fields_set:
        db_project.modality = project.modality or None
    if "time_col" in project.model_fields_set:
        db_project.time_col = project.time_col or None

    db_project.updated_at = int(time.time())
    db.commit()
    db.refresh(db_project)
    return db_project


@router.delete("/{project_id}")
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    """
    Deletes a specific project.
    """
    db_project = db.query(ProjectModel).filter(
        ProjectModel.id == project_id, ProjectModel.user_id == current_user.id
    ).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")

    db.delete(db_project)
    db.commit()
    return {"status": "success", "message": f"Project with id {project_id} deleted successfully"}

