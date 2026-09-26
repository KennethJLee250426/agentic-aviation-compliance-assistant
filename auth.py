import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings

security = HTTPBearer(auto_error=False)


async def optional_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict | None:
    """Require a configured bearer token when authentication is enabled."""
    if not settings.auth_required:
        return {"user": "local-dev", "role": "admin"}

    expected = settings.auth_token.strip()
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(creds.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"user": "authenticated-user", "role": "analyst"}
