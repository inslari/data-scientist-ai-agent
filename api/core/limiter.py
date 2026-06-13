from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
import jwt
from api.core.security import SECRET_KEY, ALGORITHM

def get_user_or_ip(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except Exception:
            pass
    return get_remote_address(request)

limiter = Limiter(key_func=get_user_or_ip, default_limits=["100/minute"])
