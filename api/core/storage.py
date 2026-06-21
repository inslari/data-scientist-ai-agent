"""
api/core/storage.py
-------------------
Central storage service for CSV files.

If the MINIO_ENDPOINT environment variable is set, files are stored in MinIO
(S3-compatible object storage). Otherwise, the service falls back to the local
filesystem (original behaviour — no configuration required).

Usage:
    from api.core.storage import storage

    # Upload
    object_key = storage.upload(file_bytes, "csvs/my-file.csv")

    # Download to a local temp file (context manager — file is deleted on exit)
    with storage.download_to_tempfile(object_key) as tmp_path:
        df = pd.read_csv(tmp_path)

    # Delete
    storage.delete(object_key)

    # Get a presigned download URL (MinIO) or the raw path (local)
    url = storage.get_download_url(object_key)
"""

import os
import tempfile
from contextlib import contextmanager
from typing import Generator

from src.core.logging_config import get_logger
logger = get_logger(__name__)

_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "").strip()
_MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")
_MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")
_MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ds-agent-csvs")
_MINIO_SECURE = os.getenv("MINIO_SECURE", "true").lower() == "true"
_ENVIRONMENT = os.getenv("ENVIRONMENT", "local").lower()

# Presigned URL expiry in seconds (1 hour)
_PRESIGNED_EXPIRY = 3600


class StorageService:
    """Unified file storage: MinIO (S3-compatible) or local filesystem."""

    def __init__(self) -> None:
        self._minio = bool(_MINIO_ENDPOINT) and _ENVIRONMENT in ("prod", "production")
        if self._minio:
            import boto3
            from botocore.client import Config

            # boto3 uses path-style addressing for self-hosted MinIO
            self._s3 = boto3.client(
                "s3",
                endpoint_url=_MINIO_ENDPOINT,
                aws_access_key_id=_MINIO_ACCESS_KEY,
                aws_secret_access_key=_MINIO_SECRET_KEY,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},  # Critical for MinIO
                ),
                use_ssl=_MINIO_SECURE,
            )
            self._bucket = _MINIO_BUCKET
            self._ensure_bucket()
            logger.info(f"[Storage] Using MinIO: {_MINIO_ENDPOINT} / bucket: {self._bucket}")
        else:
            logger.info("[Storage] Using local filesystem (MINIO_ENDPOINT not set).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist (idempotent)."""
        try:
            self._s3.head_bucket(Bucket=self._bucket)
        except Exception:
            self._s3.create_bucket(Bucket=self._bucket)
            logger.info(f"[Storage] Created MinIO bucket: {self._bucket}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(self, file_bytes: bytes, object_key: str) -> str:
        """
        Upload raw bytes to storage.

        Returns:
            object_key (MinIO) or local absolute file path.
        """
        if self._minio:
            import io
            self._s3.upload_fileobj(io.BytesIO(file_bytes), self._bucket, object_key)
            return object_key
        else:
            # Local: object_key is already the absolute path
            os.makedirs(os.path.dirname(object_key), exist_ok=True)
            with open(object_key, "wb") as f:
                f.write(file_bytes)
            return object_key

    @contextmanager
    def download_to_tempfile(self, file_ref: str) -> Generator[str, None, None]:
        """
        Download a file to a temporary local path and yield it.
        The temp file is automatically deleted when the context exits.

        Args:
            file_ref: object key (MinIO) or local absolute path.

        Yields:
            Absolute path to the downloaded temp file.
        """
        if self._minio:
            suffix = os.path.splitext(file_ref)[-1] or ".csv"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
            try:
                self._s3.download_file(self._bucket, file_ref, tmp_path)
                yield tmp_path
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            # Local: file_ref is already a local path — yield directly, no copy needed
            yield file_ref

    @contextmanager
    def get_stream(self, file_ref: str):
        """
        Yields a file-like object containing the file data stream.
        This is useful for reading partial data (like CSV previews) without downloading the entire file.
        """
        if self._minio:
            response = self._s3.get_object(Bucket=self._bucket, Key=file_ref)
            try:
                yield response['Body']
            finally:
                response['Body'].close()
        else:
            with open(file_ref, "rb") as f:
                yield f

    def delete(self, file_ref: str) -> None:
        """
        Delete a file from storage.

        Args:
            file_ref: object key (MinIO) or local absolute path.
        """
        if self._minio:
            try:
                self._s3.delete_object(Bucket=self._bucket, Key=file_ref)
            except Exception as e:
                logger.error(f"[Storage] Warning: failed to delete {file_ref} from MinIO: {e}")
        else:
            if os.path.exists(file_ref):
                try:
                    os.remove(file_ref)
                except OSError as e:
                    logger.error(f"[Storage] Warning: failed to delete local file {file_ref}: {e}")

    def get_download_url(self, file_ref: str, expiry: int = _PRESIGNED_EXPIRY) -> str:
        """
        Return a presigned download URL (MinIO) or the raw local path.

        Args:
            file_ref: object key (MinIO) or local absolute path.
            expiry: URL validity in seconds (MinIO only).
        """
        if self._minio:
            return self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": file_ref},
                ExpiresIn=expiry,
            )
        return file_ref

    @property
    def using_minio(self) -> bool:
        """True when MinIO is configured, False for local filesystem."""
        return self._minio


# Singleton — import and use directly:  from api.core.storage import storage
storage = StorageService()
