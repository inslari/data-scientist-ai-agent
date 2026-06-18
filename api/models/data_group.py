from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship
from api.core.database import Base

class DataGroupModel(Base):
    __tablename__ = "data_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    source_type = Column(String)  # "mcp" or "csv"
    created_at = Column(Integer)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True)

    # LLM schema intelligence cache (db_context_node çıktısı)
    db_intelligence_cache = Column(JSON, nullable=True)    # DatabaseIntelligenceModel.model_dump()
    raw_schema_cache = Column(String, nullable=True)       # Ham schema metni
    schema_cache_updated_at = Column(Integer, nullable=True)  # Unix timestamp

    # Long-term memory for Talk to Data
    long_term_memory = Column(String, nullable=True)

    # Domain (sector) pack for this dataset. Used by Talk-to-Data chat and as the
    # fallback for projects on this data group. NULL/empty = global DOMAIN_PACK.
    domain_pack = Column(String, nullable=True)

    # Free-form domain/dataset description (markdown). Injected into the sector
    # agent and feature-engineering prompts as dataset context. NULL = none.
    domain_doc = Column(Text, nullable=True)

    # Relationships
    mcp_config = relationship("MCPConfigModel", back_populates="data_group", cascade="all, delete-orphan", uselist=False)
    csv_files = relationship("CSVFileModel", back_populates="data_group", cascade="all, delete-orphan")
    projects = relationship("ProjectModel", back_populates="data_group")
    chat_messages = relationship("ChatMessageModel", back_populates="data_group", cascade="all, delete-orphan", order_by="ChatMessageModel.created_at")
    user = relationship("UserModel")


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    data_group_id = Column(Integer, ForeignKey("data_groups.id"), index=True)
    role = Column(String)  # "user" or "assistant"
    content = Column(String)
    created_at = Column(Integer, index=True)
    
    # New columns for visualizations
    sql = Column(String, nullable=True)
    chart = Column(JSON, nullable=True)
    columns = Column(JSON, nullable=True)
    rows = Column(JSON, nullable=True)
    warning = Column(String, nullable=True)

    # Relationship back to DataGroup
    data_group = relationship("DataGroupModel", back_populates="chat_messages")
