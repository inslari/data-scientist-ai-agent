from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from api.core.database import Base
import time

class ProjectModel(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text, nullable=True)
    query = Column(Text, nullable=True)
    data_group_id = Column(Integer, ForeignKey("data_groups.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(Integer, default=lambda: int(time.time()))
    updated_at = Column(Integer, default=lambda: int(time.time()), onupdate=lambda: int(time.time()))

    data_group = relationship("DataGroupModel", back_populates="projects")
