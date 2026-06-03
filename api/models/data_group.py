from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from api.core.database import Base

class DataGroupModel(Base):
    __tablename__ = "data_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    source_type = Column(String)  # "mcp" or "csv"
    created_at = Column(Integer)

    # Relationships
    mcp_config = relationship("MCPConfigModel", back_populates="data_group", cascade="all, delete-orphan", uselist=False)
    csv_files = relationship("CSVFileModel", back_populates="data_group", cascade="all, delete-orphan")
    projects = relationship("ProjectModel", back_populates="data_group")
