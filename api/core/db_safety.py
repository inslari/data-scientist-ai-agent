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
from sqlalchemy.engine import make_url

ALLOW_ENV = "ALLOW_REMOTE_DB_OPS"
_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def is_local_target(url) -> bool:
    """sqlite ya da localhost host'u = LOCAL (her zaman güvenli)."""
    # str(url) parolayı maskeler (SA URL.__str__ hide_password=True) — host/backend korunur,
    # bu yüzden parola hiçbir yere sızmadan host tespiti yapabiliriz.
    u = make_url(str(url))
    if u.get_backend_name().startswith("sqlite"):
        return True
    return (u.host or "").lower() in _LOCAL_HOSTS


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
