from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import jwt
from jwt.exceptions import InvalidTokenError
from pydantic import ValidationError

from api.core.database import get_db
from api.core.security import SECRET_KEY, ALGORITHM
from api.models.user import UserModel
from api.models.token_blacklist import TokenBlacklistModel
from api.schemas.auth import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)
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
