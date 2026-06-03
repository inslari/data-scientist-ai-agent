from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship
from api.core.database import Base

class CSVFileModel(Base):
    __tablename__ = "csv_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)
    file_size = Column(Integer)
    row_count = Column(Integer)
    columns_schema = Column(JSON, nullable=True)
    created_at = Column(Integer)
    data_group_id = Column(Integer, ForeignKey("data_groups.id", ondelete="CASCADE"))
    
    data_group = relationship("DataGroupModel", back_populates="csv_files")
