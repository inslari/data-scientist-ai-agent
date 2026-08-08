"""DB güvenlik guard'ı — testlerin/migration'ların yanlışlıkla REMOTE (production)
veritabanına bağlanıp şema/veri değiştirmesini KESİN olarak engeller.

Kök neden: `api/core/database.py` eskiden `load_dotenv(override=True)` çağırıyordu;
bu, bir testin/subprocess'in açıkça verdiği `API_DATABASE_URL`'i `.env`'deki remote
URL ile sessizce EZİYORDU. Sonuç: "izole" sanılan bir test remote'a kaçabiliyordu.

Bu modül ikinci savunma hattıdır: hedef LOCAL değilse (sqlite veya host
localhost/127.0.0.1/::1/boş değilse) işlemi reddeder — ancak operatör bilinçli
olarak `ALLOW_REMOTE_DB_OPS=1` verirse devam eder. Guard bağlantı AÇILMADAN,
herhangi bir DDL/DELETE çalışmadan önce iş parçacığını sonlandırır.
"""
import os
from sqlalchemy import inspect
from sqlalchemy.engine import make_url

ALLOW_ENV = "ALLOW_REMOTE_DB_OPS"          # DDL / destructive (migrate/reset) — bağlantıyı da ima eder
CONNECT_ALLOW_ENV = "ALLOW_REMOTE_DB"      # SALT-bağlantı (prod runtime) — DDL izni VERMEZ
DRIFT_ALLOW_ENV = "ALLOW_SCHEMA_DRIFT"     # startup şema-drift kapısını bypass (acil durum)
_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def is_local_target(url) -> bool:
    """sqlite ya da localhost host'u = LOCAL (her zaman güvenli)."""
    # str(url) parolayı maskeler (SA URL.__str__ hide_password=True) — host/backend korunur,
    # bu yüzden parola hiçbir yere sızmadan host tespiti yapabiliriz.
    u = make_url(str(url))
    if u.get_backend_name().startswith("sqlite"):
        return True
    return (u.host or "").lower() in _LOCAL_HOSTS


def assert_connection_allowed(url, *, context: str = "app DB connection") -> None:
    """BAĞLANTI KAPISI — ``create_engine`` çağrılmadan ÖNCE, fail-closed.

    Yapısal amaç: workstation'daki ad-hoc koşular (worker/test/shadow) `.env`'deki remote
    prod URL'e DEFAULT olarak bağlanamasın. Hedef LOCAL değilse ve ne ``ALLOW_REMOTE_DB=1``
    (salt-bağlantı, prod runtime) ne ``ALLOW_REMOTE_DB_OPS=1`` (DDL; bağlantıyı da ima eder)
    verilmemişse **SystemExit ile durdurur — bağlantı hiç açılmadan**. Uyarı değil, ret.

    İki token AYRI: ``ALLOW_REMOTE_DB`` bağlanır ama DDL yetkisi VERMEZ; ``ALLOW_REMOTE_DB_OPS``
    migration/destructive içindir ve bağlantıyı zaten gerektirir → onu da kabul eder.
    """
    u = make_url(str(url))
    if is_local_target(u):
        return
    if os.getenv(CONNECT_ALLOW_ENV) == "1" or os.getenv(ALLOW_ENV) == "1":
        return
    safe = u.render_as_string(hide_password=True)
    raise SystemExit(
        f"\n[db-safety] REDDEDİLDİ: '{context}' REMOTE bir app DB'ye bağlanmaya çalışıyor:\n"
        f"    {safe}\n"
        f"Workstation koşuları (worker/test/shadow) prod'a DEFAULT bağlanamaz. Seçenekler:\n"
        f"    • Yerel/test    → API_DATABASE_URL=sqlite:///...  (gerçek env değişkeni .env'i ezer)\n"
        f"    • Prod runtime  → {CONNECT_ALLOW_ENV}=1  (yalnız deployment'ta; DDL izni VERMEZ)\n"
        f"    • Migration     → {ALLOW_ENV}=1          (DDL + bağlantı, bilinçli)\n"
    )


# ── MODEL ↔ DB ŞEMA DRIFT KONTROLÜ ──────────────────────────────────────────────
# Bu tur İKİ KEZ aynı sınıftan bug çıktı: predictions (business_key/input_row_sampled/
# input_warning_flags eksik) ve monitoring_summary (Faz 0 kolonları overall_status/... hiç
# migrate edilmemiş). Kök: `create_all` YENİ TABLO yaratır ama mevcut tabloya kolon EKLEMEZ →
# modele kolon eklenip migration yazılmazsa DB geride kalır → o tabloyu SELECT eden her endpoint
# 500 (UndefinedColumn), üstelik SESSİZ (yalnız o yol çağrılınca patlar). Tek tek yama üçüncüyü
# engellemez → yapısal kontrol.

def detect_schema_drift(engine, metadata) -> dict:
    """Her model tablosunun ``__table__.columns``'unu inspector'ın GERÇEK DB kolonlarına karşılaştırır.

    Döner: ``{tablo: {"table_missing": bool, "missing_in_db": [...], "extra_in_db": [...]}}`` —
    YALNIZ drift olan tablolar (temiz tablo raporlanmaz).
      * ``missing_in_db`` = model bekliyor ama DB'de YOK → **FATAL**: o tabloyu SELECT eden sorgu
        (0 satırda bile tüm kolonları listeler) UndefinedColumn ile 500'ler.
      * ``extra_in_db``   = DB'de var ama model bilmiyor → **iyi huylu**: SELECT model kolonlarını
        seçer, fazlalık görünmez (eski/emekli kolon). Yalnız bilgi.
      * ``table_missing`` = tablo hiç yok (create_all'dan önce ya da create_all atlanmışsa).
    """
    insp = inspect(engine)
    real_tables = set(insp.get_table_names())
    drift: dict = {}
    for table_name, table in metadata.tables.items():
        model_cols = {c.name for c in table.columns}
        if table_name not in real_tables:
            drift[table_name] = {"table_missing": True,
                                 "missing_in_db": sorted(model_cols), "extra_in_db": []}
            continue
        db_cols = {c["name"] for c in insp.get_columns(table_name)}
        missing = model_cols - db_cols
        extra = db_cols - model_cols
        if missing or extra:
            drift[table_name] = {"table_missing": False,
                                 "missing_in_db": sorted(missing), "extra_in_db": sorted(extra)}
    return drift


def assert_schema_current(engine, metadata, *, logger=None) -> None:
    """STARTUP FAIL-CLOSED — ``create_all``'dan SONRA çağrılır. Model bekleyip DB'de OLMAYAN kolon
    (o tabloya değen ilk sorguyu 500'leyecek olan) varsa **SystemExit**. ``extra_in_db`` yalnız UYARI
    (SELECT etkilenmez). ``ALLOW_SCHEMA_DRIFT=1`` ile bypass (acil durum — sorumluluğu operatörde).

    Gerekçe (nerede koşmalı → burası): bu drift SESSİZ bir runtime 500'dür — tabloya değen endpoint
    çağrılana dek görünmez (monitoring detay prod'da tam böyle kırıldı). Startup'ta yakalamak onu ANINDA,
    GÜRÜLTÜLÜ, aksiyonlanabilir bir başlatma hatasına çevirir ("kolon X eksik → migrate_db koş"). Test
    tespit EDEMEZ (create_all sqlite'ta şemayı zaten modele eşitler → drift hiç doğmaz); bu bir
    deploy/runtime olgusudur → doğal yeri startup guard'ı + on-demand script (``scripts/check_schema_drift.py``).
    db_safety'nin mevcut fail-closed felsefesiyle (assert_connection_allowed) aynı hat.
    """
    drift = detect_schema_drift(engine, metadata)
    if not drift:
        return
    fatal = {t: d for t, d in drift.items() if d["missing_in_db"]}
    warn = {t: d for t, d in drift.items() if d["extra_in_db"] and not d["missing_in_db"]}
    if warn and logger is not None:
        for t, d in warn.items():
            logger.warning(f"[schema-drift] '{t}': DB'de fazla kolon (iyi huylu, model bilmiyor): "
                           f"{d['extra_in_db']}")
    if not fatal:
        return
    if os.getenv(DRIFT_ALLOW_ENV) == "1":
        # BYPASS SESSİZ OLMAZ — her startup'ta hangi TABLO, hangi KOLONLAR yüzünden bypass edildiği
        # tek tek loglanır (operatör riski görsün; bir 500 kaynağı "açıkça biliniyor" olsun).
        emit = logger.warning if logger is not None else print
        emit(f"[schema-drift] {DRIFT_ALLOW_ENV}=1 — FATAL şema-drift'e RAĞMEN başlatılıyor "
             f"({len(fatal)} tablo). Bu tablolara değen sorgular 500 verebilir:")
        for t, d in fatal.items():
            where = "TABLO YOK" if d["table_missing"] else f"eksik: {d['missing_in_db']}"
            emit(f"[schema-drift]   • {t} → {where}")
        return
    lines = []
    for t, d in fatal.items():
        where = "TABLO YOK" if d["table_missing"] else f"eksik kolon(lar): {d['missing_in_db']}"
        lines.append(f"    • {t}: {where}")
    raise SystemExit(
        "\n[schema-drift] REDDEDİLDİ: model ↔ DB şema kayması (migration eksik) — başlatma durduruldu.\n"
        "Model bu kolonları bekliyor ama DB'de YOKLAR → o tabloyu SELECT eden endpoint 500 verir:\n"
        + "\n".join(lines) +
        "\nÇözüm: migration'ı uygula →  ALLOW_REMOTE_DB_OPS=1 uv run python scripts/migrate_db.py\n"
        f"(Bilinçli bypass — riski operatörde: {DRIFT_ALLOW_ENV}=1)\n"
    )


def assert_local_db(url, *, context: str = "this DB operation") -> None:
    """Hedef remote ise ve ALLOW_REMOTE_DB_OPS=1 verilmemişse SystemExit ile durdurur."""
    u = make_url(str(url))
    safe = u.render_as_string(hide_password=True)
    if is_local_target(u):
        return
    if os.getenv(ALLOW_ENV) == "1":
        print(f"[db-safety] {ALLOW_ENV}=1 — REMOTE hedefe izin veriliyor ({context}): {safe}")
        return
    raise SystemExit(
        f"\n[db-safety] REDDEDİLDİ: '{context}' REMOTE bir veritabanını hedefliyor:\n"
        f"    {safe}\n"
        f"Testler ve migration'lar remote'a dokunmamalı. Yerel sandbox için:\n"
        f"    API_DATABASE_URL=sqlite:///... ver (gerçek env değişkenleri artık .env'i ezer).\n"
        f"Gerçekten remote'da çalıştırmak istiyorsan bilinçli olarak {ALLOW_ENV}=1 ayarla.\n"
    )
