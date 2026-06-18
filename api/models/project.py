from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from api.core.database import Base
import time

class ProjectModel(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text, nullable=True)
    query = Column(Text, nullable=True)
    intent = Column(String, nullable=True)
    target_col = Column(String, nullable=True)
    data_group_id = Column(Integer, ForeignKey("data_groups.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)

    # Domain (sector) pack override for this project. NULL/empty = inherit from
    # the data group, then from the global DOMAIN_PACK setting.
    domain_pack = Column(String, nullable=True)

    status = Column(String, default="idle") # idle, running, completed, failed
    report = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))

    data_group = relationship("DataGroupModel", back_populates="projects")
    steps = relationship("ProjectStepModel", back_populates="project", cascade="all, delete-orphan", order_by="ProjectStepModel.created_at")
    user = relationship("UserModel")


class ProjectStepModel(Base):
    __tablename__ = "project_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    node_name = Column(String)
    step_data = Column(JSON, nullable=True)
    created_at = Column(Integer, default=lambda: int(time.time()))
    
    project = relationship("ProjectModel", back_populates="steps")
