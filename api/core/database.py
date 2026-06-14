import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# .env dosyasını yükle
load_dotenv(override=True)

# Veritabanı dizinini belirle (SQLite için fallback)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "csvs"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Bağlantı dizesi — önce env'den oku, yoksa SQLite fallback
# ─────────────────────────────────────────────────────────────────────────────
# .env'de API_DATABASE_URL tanımlanmışsa onu kullan,
# tanımlanmamışsa eski davranışı koru (sqlite)
SQLALCHEMY_DATABASE_URL = os.getenv(
    "API_DATABASE_URL",
    f"sqlite:///{os.path.join(DATA_DIR, 'mcp_configs.db')}"
)

# PostgreSQL geçişi için .env'de şu şekilde tanımlayın:
#   API_DATABASE_URL=postgresql://user:password@localhost:5432/dsagent?options=-csearch_path%3Dapp

# Engine oluştur — SQLite için check_same_thread gerekli, PostgreSQL için değil
_is_sqlite = SQLALCHEMY_DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
