from typing import Optional
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings

security = HTTPBearer(auto_error=False)


async def optional_auth(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict | None:
    """Allow local dev without auth while enforcing a real token check in production."""
    if not settings.auth_required:
        return {"user": "local-dev", "role": "admin"}

    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    expected = (settings.auth_token or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is not configured for this environment",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = creds.credentials
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"user": "authenticated-user", "role": "analyst"}
