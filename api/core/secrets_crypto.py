"""
Symmetric encryption for provider API keys stored at rest in the app DB.

The Fernet key is derived deterministically from the app ``SECRET_KEY`` (the same
secret that signs JWTs), so no extra key management is needed: rotate SECRET_KEY and
stored keys become undecodable (decrypt() returns None → callers fall back to env).
Keys are only ever returned to the UI masked (``mask()``), never in plaintext.
"""
import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from api.core.security import SECRET_KEY


def _fernet() -> Fernet:
    # SHA256(SECRET_KEY) → 32 bytes → urlsafe base64 → valid Fernet key.
    digest = hashlib.sha256(SECRET_KEY.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(plaintext: str) -> str:
    """Encrypt a non-empty secret to a storable token string."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(token: Optional[str]) -> Optional[str]:
    """Decrypt a stored token. Returns None on any failure (missing/rotated key)."""
    if not token:
        return None
    try:
        return _fernet().decrypt(token.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError, TypeError):
        return None


def mask(token: Optional[str]) -> Optional[str]:
    """
    Decrypt then mask for display, e.g. ``sk-1234…wxyz``. Returns None if there is no
    key stored. Never exposes more than the first 3 and last 4 characters.
    """
    plaintext = decrypt(token)
    if not plaintext:
        return None
    if len(plaintext) <= 8:
        return "…" + plaintext[-2:]
    return f"{plaintext[:3]}…{plaintext[-4:]}"
