import time

from sqlalchemy import Column, Integer, String, Text

from api.core.database import Base


class LogEntryModel(Base):
    """Kalıcı uygulama log kaydı (console + DB logging altyapısının DB tarafı).

    `src/core/log_handlers.py` içindeki asenkron DB handler bu tabloya toplu
    INSERT yapar. Her süreç (worker/api/mcp) kendi `process` etiketiyle yazar;
    `project_id` worker'da işlenen işin proje kimliğidir (yoksa NULL).
    """

    __tablename__ = "app_logs"

    id = Column(Integer, primary_key=True, index=True)

    # Epoch saniye (mevcut model konvansiyonu — bkz. BackgroundJobModel.created_at).
    created_at = Column(Integer, default=lambda: int(time.time()), index=True)

    level = Column(String, index=True)          # "INFO", "WARNING", ...
    levelno = Column(Integer, index=True)        # sayısal seviye (filtre/sorgu için)
    logger = Column(String, index=True)          # record.name (ör. "src.nodes.automl")
    process = Column(String, index=True)         # "worker" | "api" | "mcp"
    project_id = Column(String, nullable=True, index=True)  # aktif işin proje kimliği

    message = Column(Text)
    module = Column(String, nullable=True)
    func = Column(String, nullable=True)
    line = Column(Integer, nullable=True)
    exc_text = Column(Text, nullable=True)       # traceback (varsa)
