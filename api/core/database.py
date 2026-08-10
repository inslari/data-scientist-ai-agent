import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

# URL'in KAYNAĞINI (gerçek env değişkeni mi, .env mi) ancak load_dotenv'DEN ÖNCE tespit
# edebiliriz: .env yüklendikten sonra ikisi de os.environ'da görünür ve ayrım kaybolur.
# Başlangıç logunda gösteriliyor — "neden bu DB'ye bağlandı?" sorusunun ilk cevabı bu.
_URL_FROM_REAL_ENV = os.getenv("API_DATABASE_URL") is not None

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

# ── YAPISAL BAĞLANTI KAPISI (fail-closed) — create_engine'DEN ÖNCE ──
# Remote (prod) app DB'ye ancak açık token'la bağlanılır; workstation'daki ad-hoc koşular
# (worker/test/shadow) .env'deki prod URL'e DEFAULT bağlanamaz. Uyarı değil, SystemExit.
from api.core.db_safety import assert_connection_allowed
assert_connection_allowed(SQLALCHEMY_DATABASE_URL, context="app DB engine (database.py)")

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


# ── BAŞLANGIÇ TEŞHİSİ: hangi DB? ────────────────────────────────────────────────
# "Ayağa kalkan süreç hangi veritabanına bağlandı?" sorusu bu projede tekrar tekrar
# soruldu (yerel worker'ın prod'a kaçması, .env ↔ gerçek env değişkeni önceliği). Cevabı
# tahmin ettirmek yerine her başlangıçta konsola yazıyoruz.
#
# Burada LOG ATMIYORUZ, yalnız dizgiyi üretiyoruz: bu modül `setup_logging()` ÇAĞRILMADAN
# ÖNCE import ediliyor (main.py'de model import'ları üzerinden), o yüzden import anında
# atılan bir INFO handler'sız kalır ve hiç görünmez. Çağrı sahibi süreç (main.py /
# worker_main.py) logging'i kurduktan SONRA çağırır.
def describe_db_target() -> str:
    """Bağlanılan app DB'sini tek satırda, PAROLASIZ özetler.

    Parola asla dizgiye girmez: host/port/db alanları URL nesnesinden tek tek okunur,
    ``render_as_string`` gibi tam-URL üretimi kullanılmaz.
    """
    from sqlalchemy.engine import make_url
    from api.core.db_safety import is_local_target

    u = make_url(SQLALCHEMY_DATABASE_URL)
    backend = u.get_backend_name()

    if backend.startswith("sqlite"):
        where = u.database or ":memory:"
    else:
        host = u.host or "?"
        port = f":{u.port}" if u.port else ""
        where = f"{host}{port}/{u.database or '?'}"

    parts = [f"{backend} · {where}"]

    # Postgres search_path (?options=-csearch_path%3Dapp) — hangi şemaya yazdığımız,
    # aynı DB içinde birden fazla ortam varsa kritik.
    opts = u.query.get("options")
    if isinstance(opts, str) and "search_path=" in opts:
        parts.append(f"şema={opts.split('search_path=', 1)[1].split(' ', 1)[0]}")

    parts.append("LOCAL" if is_local_target(u) else "REMOTE")

    if not _URL_FROM_REAL_ENV and os.getenv("API_DATABASE_URL") is not None:
        parts.append("kaynak=.env")
    elif _URL_FROM_REAL_ENV:
        parts.append("kaynak=env değişkeni")
    else:
        parts.append("kaynak=sqlite fallback (API_DATABASE_URL tanımsız)")

    return " · ".join(parts)

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
