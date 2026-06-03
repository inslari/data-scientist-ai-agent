from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship
from api.core.database import Base

class MCPConfigModel(Base):
    __tablename__ = "mcp_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(String)  # "stdio" or "sse"
    command = Column(String, nullable=True)
    args = Column(JSON, nullable=True)
    url = Column(String, nullable=True)
    env = Column(JSON, nullable=True)
    
    # Cache fields
    schema_cache = Column(String, nullable=True)
    schema_cache_updated_at = Column(Integer, nullable=True)  # Unix timestamp
    column_stats_cache = Column(String, nullable=True)
    column_stats_cache_updated_at = Column(Integer, nullable=True)  # Unix timestamp
    data_group_id = Column(Integer, ForeignKey("data_groups.id", ondelete="CASCADE"))
    
    data_group = relationship("DataGroupModel", back_populates="mcp_config")
