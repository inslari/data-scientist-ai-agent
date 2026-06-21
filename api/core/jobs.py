"""
İş kuyruğu (Outbox) yardımcıları.

API süreci ağır işi asla kendisi çalıştırmaz; yalnızca BackgroundJobModel satırı
yazar. Worker (worker_main.py) kuyruğu poll'leyip işi yürütür.
"""
from sqlalchemy.orm import Session

from api.models.job import BackgroundJobModel

from src.core.logging_config import get_logger
logger = get_logger(__name__)


def enqueue_ttd_suggestions(db: Session, data_group_id: int) -> None:
    """
    Bir data group için Talk-to-Data örnek sorularını arka planda üretmek üzere
    worker kuyruğuna iş ekler (best-effort; commit eden çağırandır).
    """
    try:
        db.add(BackgroundJobModel(
            task_type="generate_ttd_suggestions",
            data_group_id=data_group_id,
            status="pending",
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.info(f"[jobs] enqueue_ttd_suggestions başarısız (group {data_group_id}): {str(e)[:150]}")
