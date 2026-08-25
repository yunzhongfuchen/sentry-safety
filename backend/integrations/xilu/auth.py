"""
西艾氟 OpenAPI 鉴权模块
"""

import os
from typing import Optional
from fastapi import Header, HTTPException, Query, status


def verify_token(
    authorization: Optional[str] = Header(None),
    x_token: Optional[str] = Header(None, alias="X-Token"),
    query_token: Optional[str] = Query(None, alias="token"),
) -> bool:
    expected_token = os.getenv("CVAPI_TOKEN", "").strip()
    if not expected_token:
        return True

    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    elif x_token:
        token = x_token.strip()
    elif query_token:
        token = query_token.strip()

    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Token",
        )
    return True
