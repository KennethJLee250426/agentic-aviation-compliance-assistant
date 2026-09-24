from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings

security = HTTPBearer(auto_error=False)


async def optional_auth(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict | None:
    """Optional auth dependency used to keep local development easy while enabling
    a production enforcement path. If AUTH_REQUIRED is false, local/dev mode is
    allowed without a token. When enabled, reject missing or malformed tokens."""
    if not settings.auth_required:
        return {"user": "local-dev", "role": "admin"}

    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

# DEVELOPMENT ONLY:
# Replace this token-length check with JWT/OIDC verification
# before setting AUTH_REQUIRED=true in production.
    
    token = creds.credentials
    if not token or len(token) < 8:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"user": "authenticated-user", "role": "analyst"}
