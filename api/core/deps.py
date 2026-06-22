from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt
from jwt.exceptions import InvalidTokenError
from pydantic import ValidationError

from api.core.database import get_db
from api.core.security import SECRET_KEY, ALGORITHM
from api.models.user import UserModel
from api.models.token_blacklist import TokenBlacklistModel
from api.schemas.auth import TokenData

# JWT Bearer şeması — Swagger (/docs) "Authorize" düğmesinde tek bir token alanı
# gösterir (username/password formu yerine). İstemci `Authorization: Bearer <token>`
# header'ı gönderir. auto_error=False: eksik header'da 403 yerine kendi 401'imizi
# döndürebilelim (frontend interceptor 401'de refresh tetikliyor).
bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token (Bearer)")


def get_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Authorization: Bearer header'ından ham JWT'yi çıkarır."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# Geriye dönük uyumluluk: bazı endpoint'ler (ör. logout) `Depends(oauth2_scheme)`
# ile ham token (str) bekliyor. Aynı davranışı sürdürür.
oauth2_scheme = get_token


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(get_token)
) -> UserModel:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Check if token is blacklisted
    is_blacklisted = db.query(TokenBlacklistModel).filter(TokenBlacklistModel.token == token).first()
    if is_blacklisted:
        raise credentials_exception

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        token_type = payload.get("type")

        if user_id is None or token_type != "access":
            raise credentials_exception

        token_data = TokenData(user_id=int(user_id))
        token_version = payload.get("v")
    except (InvalidTokenError, ValidationError):
        raise credentials_exception

    user = db.query(UserModel).filter(UserModel.id == token_data.user_id).first()
    if user is None:
        raise credentials_exception

    if token_version is not None and user.token_version != token_version:
        raise credentials_exception

    return user
