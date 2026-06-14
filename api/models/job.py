import time
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from api.core.database import Base


class BackgroundJobModel(Base):
    """
    Represents a persistent background job (Outbox pattern).
    API endpoints enqueue jobs here; the worker loop picks them up.

    Status flow:  pending -> running -> completed | failed
    """
    __tablename__ = "background_jobs"

    id = Column(Integer, primary_key=True, index=True)

    # The project this job belongs to
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=True, index=True)

    # What kind of work needs to be done (extendable for future job types)
    task_type = Column(String, default="run_project", nullable=False)

    # Payload fields specific to run_project
    query = Column(Text, nullable=True)
    data_group_id = Column(Integer, nullable=True)
    is_resume = Column(Integer, default=0)       # 0 = False, 1 = True
    force_restart = Column(Integer, default=0)   # 0 = False, 1 = True

    # Lifecycle
    status = Column(String, default="pending", index=True)  # pending, running, completed, failed
    error = Column(Text, nullable=True)

    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))
