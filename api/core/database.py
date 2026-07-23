import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

# .env dosyasını yükle.
# override=False: GERÇEK ortam değişkenleri .env'i EZMELİDİR. Eskiden override=True idi
# ve bir testin/subprocess'in açıkça verdiği API_DATABASE_URL'i .env'deki remote URL ile
# sessizce ezerek "izole" testlerin remote'a kaçmasına yol açıyordu. .env yalnızca env
# değişkeni HİÇ set edilmemişse fallback olmalı. Ek savunma: api/core/db_safety.py.
load_dotenv(override=False)

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

# pool_pre_ping: her pool checkout'unda bağlantıyı test eder; ölü bağlantıyı
# otomatik yeniler. Özellikle worker gibi uzun süre bekleyen servisler için kritik
# (PostgreSQL idle timeout / server restart sonrası "server closed the connection" hatası).
# pool_recycle: 1800s'den eski bağlantıları havuzdan çıkarır (cloud DB'ler genelde
# 5-10 dk idle sonrası keser; 30 dk yeterince güvenli marj).
_pool_kwargs = (
    {"pool_pre_ping": True, "pool_recycle": 1800}
    if not _is_sqlite
    else {}
)

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=_connect_args, **_pool_kwargs)

# SQLite foreign key'leri VARSAYILAN OLARAK ZORLAMAZ. Bu, ondelete="CASCADE" /
# "SET NULL" tanımlarını sqlite'ta sessizce etkisiz bırakır: passive_deletes=True olan
# ilişkilerde (ör. ai_providers → ai_models) silme sonrası çocuk satırlar öksüz kalır ve
# onlara işaret eden atamalar NULL'a düşmez. Postgres'te doğru, dev sqlite'ında yanlış
# davranan bir ayrım — her bağlantıda PRAGMA açarak ikisini eşitliyoruz.
if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
