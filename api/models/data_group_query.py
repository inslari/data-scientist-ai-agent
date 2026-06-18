from sqlalchemy import Column, Integer, String, Text, ForeignKey
from api.core.database import Base


class DataGroupQueryModel(Base):
    __tablename__ = "data_group_queries"

    id = Column(Integer, primary_key=True, index=True)
    data_group_id = Column(
        Integer,
        ForeignKey("data_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    sql = Column(Text, nullable=False)
    expert = Column(String, nullable=True)
    pack_name = Column(String, nullable=True)
    created_at = Column(Integer, nullable=False)
