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
    column_stats_cache = Column(JSON, nullable=True)
    created_at = Column(Integer)
    data_group_id = Column(Integer, ForeignKey("data_groups.id", ondelete="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    
    data_group = relationship("DataGroupModel", back_populates="csv_files")
    user = relationship("UserModel")

from sqlalchemy import event
import os
from api.core.storage import storage

@event.listens_for(CSVFileModel, 'after_delete')
def receive_after_delete(mapper, connection, target):
    """
    Listen for the 'after_delete' event and delete the physical file from storage.
    """
    if target.file_path:
        storage.delete(target.file_path)

