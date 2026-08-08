"""Katman 2 monitoring aggregation'ı TEK BAŞINA koş — YIKICI GC YOK.

`run_maintenance()` app_logs/background_jobs/checkpoints/intelligence_cache SİLER; pencere kapatmak
(monitoring_summary yazmak) için bunları tetiklemek istemeyiz. Bu script yalnız aggregation'ı çağırır.

Kullanım:
    uv run python scripts/run_monitoring_aggregation.py                    # yerel
    ALLOW_REMOTE_DB=1 uv run python scripts/run_monitoring_aggregation.py  # prod (yazma; DDL yok)

Eşdeğer programatik yüzey: `db_maintenance.run_maintenance(only=MONITORING_SUBTASKS)` ya da doğrudan
`aggregator.run_monitoring_aggregation()`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TÜM modelleri kaydettir (ProjectModel string-ilişkileri mapper konfigürasyonunda çözülsün) — main.py bloğuyla senkron.
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

from src.services.monitoring.aggregator import run_monitoring_aggregation  # noqa: E402

n = run_monitoring_aggregation()
print(f"[monitoring] aggregation tamam — {n} özet satırı yazıldı/güncellendi (GC KOŞMADI).")
