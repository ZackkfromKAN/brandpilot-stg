"""
LangGraph Cloud custom auth — validates every request against BrandPilot backend.

Flow:
  1. Extract Bearer token from Authorization header (caller's Cognito access token)
  2. Call GET /profile  → validates token + returns email
  3. Call GET /accounts/{id}/users → check email is an active account member
  4. Allow (return identity) or raise 401/403

User list is cached for 60 s to avoid a BrandPilot round-trip on every single LangGraph
polling tick, while still revoking access within one minute of removal.

LangSmith Studio is exempt (disable_studio_auth: true in langgraph.json) so you can
still inspect runs in the UI without a Cognito token.
"""
from __future__ import annotations

import os
import time

import httpx
from langgraph_sdk import Auth

auth = Auth()

_API = os.getenv("BRANDPILOT_API_STG", "https://brandpilot-stg.com/api")
_ACCOUNT_ID = os.getenv("BRANDPILOT_ACCOUNT_ID", "01KPTNF3WKJ2ASYZA4J6E2V8NS")

# Per-account user-list cache  { account_id: {"emails": set, "at": float} }
_cache: dict = {}
_CACHE_TTL = 60.0


async def _authorized_emails(token: str) -> set[str]:
    """Return the set of authorized email addresses for the account. Cached 60 s."""
    now = time.monotonic()
    entry = _cache.get(_ACCOUNT_ID)
    if entry and (now - entry["at"]) < _CACHE_TTL:
        return entry["emails"]

    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            f"{_API}/accounts/{_ACCOUNT_ID}/users",
            headers={"Authorization": f"Bearer {token}"},
        )

    if not r.is_success:
        return set()

    emails = {u["email"].lower() for u in r.json() if u.get("email")}
    _cache[_ACCOUNT_ID] = {"emails": emails, "at": now}
    return emails


@auth.authenticate
async def authenticate(
    authorization: str | None = None,
) -> Auth.types.MinimalUserDict:
    """Validate the caller's Cognito token and confirm BrandPilot account membership."""

    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authorization header required."
        )

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Bearer token missing."
        )

    # ── 1. Validate token + resolve identity ────────────────────────────────
    async with httpx.AsyncClient(timeout=10) as c:
        profile_resp = await c.get(
            f"{_API}/profile",
            headers={"Authorization": f"Bearer {token}"},
        )

    if profile_resp.status_code == 401:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid or expired token. Please refresh your credentials."
        )
    if not profile_resp.is_success:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Could not validate credentials."
        )

    profile = profile_resp.json()
    email = profile.get("email", "").lower()
    if not email:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Could not determine user identity."
        )

    # ── 2. Check account membership ─────────────────────────────────────────
    if email not in await _authorized_emails(token):
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail=(
                f"{email} is not authorised for this account. "
                "Contact your administrator to request access."
            ),
        )

    display = f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip()
    return {"identity": email, "display_name": display or email}
