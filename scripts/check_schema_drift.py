"""Model ↔ DB şema drift raporu — SALT OKUNUR (yalnız inspector, hiçbir DDL/DML yok).

Kullanım:
    uv run python scripts/check_schema_drift.py                 # yerel/varsayılan DB
    ALLOW_REMOTE_DB=1 uv run python scripts/check_schema_drift.py   # prod (bilinçli, read-only)

Her modelin __table__.columns'unu inspector'ın gerçek DB kolonlarına karşılaştırır. Çıkış kodu:
FATAL (missing_in_db) varsa 1, yoksa 0 → CI/deploy checklist'inde kapı olarak kullanılabilir.
Yapısal amaç: predictions/monitoring_summary'de iki kez çıkan "modele kolon eklendi, migration yok"
bug sınıfını TEK TEK yamalamak yerine topluca yakala. Startup guard'ı (assert_schema_current) canlı
kapı; bu script deploy ÖNCESİ / ad-hoc prod teşhisi içindir.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.core.database import Base, engine  # noqa: E402

# TÜM modelleri kaydettir (Base.metadata dolsun) — main.py bloğuyla senkron.
import api.models.mcp  # noqa: F401,E402
import api.models.csv  # noqa: F401,E402
import api.models.data_group  # noqa: F401,E402
import api.models.project  # noqa: F401,E402
import api.models.user  # noqa: F401,E402
import api.models.job  # noqa: F401,E402
import api.models.data_group_query  # noqa: F401,E402
import api.models.db_intelligence_cache  # noqa: F401,E402
import api.models.log_entry  # noqa: F401,E402
import api.models.llm_usage  # noqa: F401,E402
import api.models.ai_provider  # noqa: F401,E402
import api.models.ai_model  # noqa: F401,E402
import api.models.ai_role_assignment  # noqa: F401,E402
import api.models.password_reset_request  # noqa: F401,E402
import api.models.artifact  # noqa: F401,E402
import api.models.artifact_event  # noqa: F401,E402
import api.models.prediction  # noqa: F401,E402
import api.models.monitoring_summary  # noqa: F401,E402
import api.models.token_blacklist  # noqa: F401,E402

from api.core.db_safety import detect_schema_drift  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

print(f"DB: {make_url(str(engine.url)).render_as_string(hide_password=True)}")
drift = detect_schema_drift(engine, Base.metadata)

if not drift:
    print("✓ Şema güncel — hiçbir tabloda drift yok.")
    sys.exit(0)

fatal_any = False
for table, d in sorted(drift.items()):
    if d["table_missing"]:
        fatal_any = True
        print(f"🔴 {table}: TABLO YOK (create_all koşmamış)")
        continue
    if d["missing_in_db"]:
        fatal_any = True
        print(f"🔴 {table}: DB'de EKSİK (model bekliyor → SELECT 500'ler): {d['missing_in_db']}")
    if d["extra_in_db"]:
        print(f"🟡 {table}: DB'de FAZLA (model bilmiyor, iyi huylu): {d['extra_in_db']}")

print()
print("🔴 = FATAL (migration eksik → migrate_db koş)  ·  🟡 = iyi huylu (eski kolon)")
sys.exit(1 if fatal_any else 0)
