"""Offline tests for the async DB logging handler + prune helper.

No real app DB: an isolated in-memory SQLite engine (shared via StaticPool so the
background writer thread sees the same DB) is injected into the handler, and the
real ``LogEntryModel`` table is created on it. Verifies batched writes, level
filtering, project tagging, exception capture and retention pruning.

Run: uv run python -m unittest tests.test_db_logging
"""

import logging
import time
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.models.log_entry import LogEntryModel
from src.core.log_handlers import DBLogQueueHandler, prune_logs
from src.core.logging_config import _ProjectContextFilter, bind_project


def _fresh_backend():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    LogEntryModel.__table__.create(bind=engine, checkfirst=True)
    return engine, sessionmaker(bind=engine)


class _Harness:
    """An isolated logger wired only to a DBLogQueueHandler (no root pollution)."""

    def __init__(self, level=logging.INFO, **kw):
        self.engine, self.Session = _fresh_backend()
        self.handler = DBLogQueueHandler(
            level=level,
            process="test",
            session_factory=self.Session,
            model=LogEntryModel,
            flush_interval=0.2,
            **kw,
        )
        self.handler.addFilter(_ProjectContextFilter())
        self.logger = logging.getLogger(f"test_db_logging.{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = [self.handler]
        self.logger.propagate = False

    def rows(self):
        db = self.Session()
        try:
            return db.query(LogEntryModel).order_by(LogEntryModel.id).all()
        finally:
            db.close()

    def close(self):
        self.handler.stop()


class DBLoggingTest(unittest.TestCase):
    def test_batched_writes(self):
        h = _Harness(batch_size=10)
        try:
            for i in range(25):
                h.logger.info("mesaj %d", i)
            h.close()  # flush kalanları
            rows = h.rows()
            self.assertEqual(len(rows), 25)
            self.assertEqual(rows[0].process, "test")
            self.assertEqual(rows[0].level, "INFO")
            self.assertEqual(rows[0].message, "mesaj 0")
        finally:
            h.close()

    def test_level_filtering(self):
        h = _Harness(level=logging.INFO)
        try:
            h.logger.debug("görünmez")
            h.logger.warning("görünür")
            h.close()
            rows = h.rows()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].level, "WARNING")
        finally:
            h.close()

    def test_project_tagging(self):
        h = _Harness()
        try:
            with bind_project(42):
                h.logger.info("proje içinde")
            h.logger.info("proje dışında")
            h.close()
            by_msg = {r.message: r.project_id for r in h.rows()}
            self.assertEqual(by_msg["proje içinde"], "42")
            self.assertIsNone(by_msg["proje dışında"])
        finally:
            h.close()

    def test_exception_capture(self):
        h = _Harness()
        try:
            try:
                raise ValueError("patladı")
            except ValueError:
                h.logger.exception("hata yakalandı")
            h.close()
            rows = h.rows()
            self.assertEqual(len(rows), 1)
            self.assertIsNotNone(rows[0].exc_text)
            self.assertIn("ValueError", rows[0].exc_text)
            self.assertIn("patladı", rows[0].exc_text)
        finally:
            h.close()

    def test_flush_interval_without_full_batch(self):
        # batch hiç dolmadan, süre eşiğiyle yazılmalı
        h = _Harness(batch_size=1000)
        try:
            h.logger.info("tek satır")
            deadline = time.time() + 3.0
            while time.time() < deadline and not h.rows():
                time.sleep(0.1)
            self.assertEqual(len(h.rows()), 1)
        finally:
            h.close()


class PruneTest(unittest.TestCase):
    def test_prune_removes_old_only(self):
        engine, Session = _fresh_backend()
        now = int(time.time())
        db = Session()
        try:
            db.bulk_insert_mappings(LogEntryModel, [
                {"created_at": now - 40 * 86400, "level": "INFO", "levelno": 20,
                 "logger": "x", "process": "test", "message": "eski"},
                {"created_at": now - 1 * 86400, "level": "INFO", "levelno": 20,
                 "logger": "x", "process": "test", "message": "yeni"},
            ])
            db.commit()
        finally:
            db.close()

        deleted = prune_logs(30, session_factory=Session, model=LogEntryModel)
        self.assertEqual(deleted, 1)

        db = Session()
        try:
            remaining = [r.message for r in db.query(LogEntryModel).all()]
        finally:
            db.close()
        self.assertEqual(remaining, ["yeni"])


if __name__ == "__main__":
    unittest.main()
