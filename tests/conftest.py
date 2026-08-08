"""Pytest-genel fixture/konfig — ŞU AN tek işi: tüm ORM modellerini kaydettirmek.

Neden gerekli: `DataGroupModel` (ve akrabaları) ilişkileri STRING adla kurar
(`relationship("CSVFileModel")`, `"MCPConfigModel"`, `"ChatMessageModel"`, ...).
SQLAlchemy bu adları ancak ilgili sınıf Base.registry'ye KAYITLIYSA (yani modül
import edildiyse) çözebilir. `main.py`/`worker_main.py` üretimde tüm modelleri
açıkça import ederek bunu sağlar (bkz. CLAUDE.md "import every model"); testler o
yolu atlar.

Bu olmadan, tam suite'in davranışı pytest collection SIRASINA bağlı kalır: bir test
modülü `DataGroupModel`'i kaydedip akrabalarını kaydetmeden `test_db_logging` gibi
bir modül mapper konfigürasyonunu tetiklerse "failed to locate a name" hatası düşer.
conftest.py collection'dan ÖNCE bir kez import edildiği için tüm modelleri burada
toplamak bu sırayı bağımsız kılar (kırılgan import-yan-etkisi değil, kalıcı garanti).

Liste `main.py`'deki model import bloğuyla senkron tutulmalı — yeni bir model
eklenirse buraya da eklenir.

── TAZE SANDBOX DB (2026-07-26) ──
İkinci iş: her koşuda TAZE bir sandbox sqlite ver. Test modülleri tek tek
`os.environ.setdefault("API_DATABASE_URL", "sqlite:///./data/_sbx_<name>.sqlite")`
yapıyor; ama tek pytest sürecinde İLK `setdefault` kazanır → TÜM modüller o TEK DB'yi
paylaşır. Dosya diskte KALICI olduğundan bir şema değişikliği (yeni kolon) sonrası eski
dosya güncel şemayı taşımaz → `create_all` var olan tabloyu ALTER etmez → "no such column"
→ SAHTE regresyon (ortam, kod değil; bu tuzağa `business_key` eklerken düşüldü, `label_backfilled_at`
Float'a çevrilirken de). Çözüm: conftest model import'larından (→ engine yaratımı) ÖNCE
env var'ı GERÇEK-SET eder (setdefault değil → modül setdefault'ları no-op olur), hedef PID-başına
TEMP dosya (her koşu taze, kalıcı-şema sorunu yapısal olarak imkânsız). `test_db_logging` fix'inde
kurulan "conftest = kalıcı garanti" konvansiyonunun devamı.
"""

import os
import pathlib
import tempfile

import pytest

# Model import'larından ÖNCE: api.core.database engine'i ilk import'ta env var'dan kurulur →
# bu satır o import'tan (aşağıdaki model import'ları) önce çalışmalı. PID-başına taze dosya.
_SANDBOX_DB = pathlib.Path(tempfile.gettempdir()) / f"dsagent_test_{os.getpid()}.sqlite"
try:
    _SANDBOX_DB.unlink()                       # önceki koşu artığı → temiz başla (güncel şema)
except FileNotFoundError:
    pass
os.environ["API_DATABASE_URL"] = f"sqlite:///{_SANDBOX_DB.as_posix()}"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_sandbox_db():
    """Session sonu sandbox DB'yi sil (temp'te birikmesin)."""
    yield
    try:
        _SANDBOX_DB.unlink()
    except OSError:
        pass


# Tek amaç: import ederek Base.registry'ye kaydettirmek. Kullanılmadıkları için
# noqa; sıralama main.py ile aynı.
from api.models.mcp import MCPConfigModel  # noqa: F401,E402
from api.models.csv import CSVFileModel  # noqa: F401
from api.models.data_group import DataGroupModel  # noqa: F401
from api.models.project import ProjectModel  # noqa: F401
from api.models.user import UserModel  # noqa: F401
from api.models.job import BackgroundJobModel  # noqa: F401
from api.models.data_group_query import DataGroupQueryModel  # noqa: F401
from api.models.db_intelligence_cache import DbIntelligenceCacheModel  # noqa: F401
from api.models.log_entry import LogEntryModel  # noqa: F401
from api.models.llm_usage import LLMUsageModel  # noqa: F401
from api.models.ai_provider import AiProviderModel  # noqa: F401
from api.models.ai_model import AiModelModel  # noqa: F401
from api.models.ai_role_assignment import AiRoleAssignmentModel  # noqa: F401
from api.models.password_reset_request import PendingPasswordResetModel  # noqa: F401
from api.models.artifact import ArtifactModel  # noqa: F401
from api.models.artifact_event import ArtifactEventModel  # noqa: F401
from api.models.prediction import PredictionModel  # noqa: F401
from api.models.monitoring_summary import MonitoringSummaryModel  # noqa: F401
from api.models.token_blacklist import TokenBlacklistModel  # noqa: F401 (main.py bloğunda yok; auth/deps yolundan gelir)
