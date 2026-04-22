from __future__ import annotations

import base64
import hmac
import hashlib
import os
import time
from typing import Optional

import requests


COGNITO_DOMAINS = {
    "prod":    os.getenv("COGNITO_DOMAIN_PRD", ""),
    "staging": os.getenv("COGNITO_DOMAIN_STG", "https://brandpilot-stg-api-domain.auth.eu-central-1.amazoncognito.com"),
}

COGNITO_CLIENT_ID     = os.getenv("COGNITO_CLIENT_ID", "")
COGNITO_CLIENT_SECRET = os.getenv("COGNITO_CLIENT_SECRET", "")


def _secret_hash(username: str) -> str:
    msg = username + COGNITO_CLIENT_ID
    raw = hmac.new(
        COGNITO_CLIENT_SECRET.encode(),
        msg.encode(),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(raw).decode()


def refresh_access_token(refresh_token: str, env: str = "prod") -> dict:
    """
    Exchange a Cognito refresh_token for a new access_token.
    Returns {"access_token": ..., "expires_in": ...} or raises on failure.
    """
    domain = COGNITO_DOMAINS[env]
    if not domain:
        raise ValueError(f"COGNITO_DOMAIN_{env.upper()} not set")

    r = requests.post(
        f"{domain}/oauth2/token",
        auth=(COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET),
        data={
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=10,
    )
    if not r.ok:
        raise ValueError(f"Token refresh failed ({r.status_code}): {r.text}")
    return r.json()


def is_token_expired(issued_at: Optional[int], expires_in: int = 3600, buffer_s: int = 300) -> bool:
    """True if the token will expire within buffer_s seconds."""
    if issued_at is None:
        return False
    return (time.time() - issued_at + buffer_s) >= expires_in
