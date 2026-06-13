"""
Background Worker (Inbox/Outbox pattern)
=========================================
This module runs a single async task for the entire lifetime of the FastAPI
application.  It polls the `background_jobs` table for `pending` jobs and
executes them one at a time (sequential by default; easy to extend to a
semaphore-based concurrency limit later).

Why sequential?
  - LangGraph graph execution is already heavily I/O-bound (LLM calls).
  - Running multiple projects in parallel multiplies token consumption and
    memory usage; sequential keeps things predictable and safe.
  - A semaphore can be added later if parallel execution becomes a requirement.
"""

import asyncio
import time
import traceback
import json

from api.core.database import SessionLocal
from api.models.job import BackgroundJobModel
from api.models.project import ProjectModel, ProjectStepModel

# How long the worker sleeps between polling cycles when there is nothing to do.
POLL_INTERVAL_SECONDS = 5


async def job_worker_loop() -> None:
    """
    Continuously polls the background_jobs table and executes pending jobs.
    Designed to be run as a single long-lived asyncio.Task.
    """
    print("[Worker] Background job worker started.")
    while True:
        try:
            await _process_next_job()
        except asyncio.CancelledError:
            print("[Worker] Worker task cancelled — shutting down cleanly.")
            break
        except Exception as exc:
            # Never let an unexpected error kill the worker loop.
            print(f"[Worker] Unexpected error in worker loop: {exc}")
            traceback.print_exc()

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _process_next_job() -> None:
    """
    Picks the oldest pending job, marks it as running, and executes it.
    Uses a fresh SessionLocal for each poll to avoid stale state.
    """
    db = SessionLocal()
    try:
        # Pick the oldest pending job (FIFO)
        job: BackgroundJobModel | None = (
            db.query(BackgroundJobModel)
            .filter(BackgroundJobModel.status == "pending")
            .order_by(BackgroundJobModel.created_at.asc())
            .first()
        )

        if job is None:
            return  # Nothing to do; caller will sleep

        print(f"[Worker] Picked up job id={job.id} for project_id={job.project_id}")
        job.status = "running"
        job.updated_at = int(time.time())
        db.commit()

    finally:
        db.close()

    # Execute outside the polling session so DB is not held open
    if job is not None:
        await _execute_job(job)


async def _execute_job(job: BackgroundJobModel) -> None:
    """
    Dispatches a job to the appropriate handler based on task_type.
    Updates the job status when done.
    """
    if job.task_type == "run_project":
        await _run_project_job(job)
    else:
        _mark_job(job.id, "failed", f"Unknown task_type: {job.task_type}")


async def _run_project_job(job: BackgroundJobModel) -> None:
    """
    Executes the LangGraph pipeline for a project, streaming results into
    ProjectStepModel rows and updating ProjectModel.status when finished.
    """
    from src.graph import build_graph
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    project_id = job.project_id
    query = job.query
    data_group_id = job.data_group_id
    is_resume = bool(job.is_resume)
    force_restart = bool(job.force_restart)

    db = SessionLocal()
    try:
        project: ProjectModel | None = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
        if not project:
            print(f"[Worker] Project {project_id} not found; skipping job {job.id}.")
            _mark_job(job.id, "failed", "Project not found")
            return

        project.status = "running"
        project.error = None
        db.commit()
    finally:
        db.close()

    # --- LangGraph execution ---
    error_trace: str | None = None
    try:
        async with AsyncSqliteSaver.from_conn_string("data/checkpoints.sqlite") as saver:
            app = build_graph(checkpointer=saver)
            config = {
                "configurable": {
                    "data_group_id": data_group_id,
                    "thread_id": str(project_id),
                }
            }

            initial_state: dict | None = {"user_query": query}

            if is_resume:
                try:
                    state_snapshot = await app.aget_state(config)
                    if state_snapshot and getattr(state_snapshot, "values", None):
                        initial_state = None  # Resume from checkpoint
                    else:
                        is_resume = False
                except Exception as exc:
                    print(f"[Worker] Could not fetch state snapshot: {exc}")
                    is_resume = False

            if force_restart:
                await saver.conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (str(project_id),)
                )
                await saver.conn.execute(
                    "DELETE FROM writes WHERE thread_id = ?", (str(project_id),)
                )
                await saver.conn.commit()

            # Clear existing steps for a fresh run
            if not is_resume:
                db = SessionLocal()
                try:
                    db.query(ProjectStepModel).filter(ProjectStepModel.project_id == project_id).delete()
                    db.commit()
                finally:
                    db.close()

            final_state: dict = {}

            async for chunk in app.astream(initial_state, config=config):
                node_name = list(chunk.keys())[0]
                print(f"\n{'='*40}\n[Worker][NODE END] {node_name}\n{'='*40}")

                chunk_data: dict = chunk[node_name]
                final_state.update(chunk_data)

                safe_data: dict = {}
                for k, v in chunk_data.items():
                    if k == "model_artifact" or "DataFrame" in str(type(v)):
                        continue
                    if isinstance(v, list) and v and "DataFrame" in str(type(v[0])):
                        continue
                    if k in ["sql_results", "sql_query_results", "X_train", "X_test", "y_train", "y_test"]:
                        if isinstance(v, list):
                            safe_data[k] = v[:5]
                            continue
                        else:
                            safe_data[k] = "[Büyük veri kısıtlandı]"
                            continue
                    try:
                        json.dumps(v)
                        safe_data[k] = v
                    except Exception:
                        safe_data[k] = str(v)

                db = SessionLocal()
                try:
                    step = ProjectStepModel(
                        project_id=project_id,
                        node_name=node_name,
                        step_data=safe_data,
                        created_at=int(time.time()),
                    )
                    db.add(step)
                    db.commit()
                finally:
                    db.close()

            # Mark project completed
            db = SessionLocal()
            try:
                project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
                if project:
                    project.status = "completed"
                    project.report = final_state.get("report", "")
                    db.commit()
            finally:
                db.close()

    except asyncio.CancelledError:
        # Worker is shutting down; leave project as 'running' so startup can resume it
        raise
    except Exception:
        error_trace = traceback.format_exc()
        print(f"[Worker] Graph execution failed for project {project_id}:\n{error_trace}")

        db = SessionLocal()
        try:
            project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
            if project:
                project.status = "failed"
                project.error = error_trace
                db.commit()
        finally:
            db.close()

    _mark_job(job.id, "failed" if error_trace else "completed", error_trace)


def _mark_job(job_id: int, status: str, error: str | None = None) -> None:
    """Updates the job row with a final status."""
    db = SessionLocal()
    try:
        job = db.query(BackgroundJobModel).filter(BackgroundJobModel.id == job_id).first()
        if job:
            job.status = status
            job.error = error
            job.updated_at = int(time.time())
            db.commit()
    finally:
        db.close()
