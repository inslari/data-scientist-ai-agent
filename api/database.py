import os
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

# Veritabanı dizinini belirle
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Bağlantı dizesi
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'mcp_configs.db')}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

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
# Tabloyu oluştur
Base.metadata.create_all(bind=engine)

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
