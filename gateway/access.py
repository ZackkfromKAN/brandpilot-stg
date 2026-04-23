"""
BrandPilot access check for the MCP Gateway.

For each request the gateway checks: is this user's email in any of the
configured BrandPilot accounts? If yes, which brands do they have access to?

Results are cached 60 s to avoid a round-trip on every tool call while still
revoking access within one minute of removal — identical policy to the
LangGraph custom auth handler.
"""
from __future__ import annotations

import os
import time

import httpx

_API = os.getenv("BRANDPILOT_API_STG", "https://brandpilot-stg.com/api")

# Comma-separated list of account IDs to check.
# Defaults to the Cand'art staging account; override via env for multi-tenant.
_ACCOUNT_IDS: list[str] = [
    a.strip()
    for a in os.getenv(
        "BRANDPILOT_ACCOUNT_IDS",
        os.getenv("BRANDPILOT_ACCOUNT_ID", "01KPTNF3WKJ2ASYZA4J6E2V8NS"),
    ).split(",")
    if a.strip()
]

_CACHE_TTL = 60.0
# email → {"accounts": [...], "at": float}
_cache: dict[str, dict] = {}


async def get_user_access(email: str, cognito_token: str) -> list[dict]:
    """
    Return a list of dicts the email has access to:
        [{"account_id": "...", "brands": [{"brand_id": "...", "brand_name": "..."}]}]

    Returns [] if the email is not in any configured account.
    Result cached 60 s per email address.
    """
    now = time.monotonic()
    cached = _cache.get(email)
    if cached and (now - cached["at"]) < _CACHE_TTL:
        return cached["accounts"]

    accounts = await _fetch_access(email, cognito_token)
    _cache[email] = {"accounts": accounts, "at": now}
    return accounts


async def _fetch_access(email: str, cognito_token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {cognito_token}"}
    result: list[dict] = []

    async with httpx.AsyncClient(base_url=_API, headers=headers, timeout=15) as c:
        for account_id in _ACCOUNT_IDS:
            # ── Check membership ─────────────────────────────────────────────
            try:
                r = await c.get(f"/accounts/{account_id}/users")
                if not r.is_success:
                    continue
                raw = r.json()
                users = raw if isinstance(raw, list) else raw.get("users", [])
                authorized = {u.get("email", "").lower() for u in users if u.get("email")}
                if email not in authorized:
                    continue
            except Exception:
                continue  # treat unreachable account as no-access

            # ── List brands ──────────────────────────────────────────────────
            brands: list[dict] = []
            try:
                rb = await c.get(f"/accounts/{account_id}/brands")
                if rb.is_success:
                    raw_b = rb.json()
                    brand_list = raw_b if isinstance(raw_b, list) else raw_b.get("brands", [])
                    for b in brand_list:
                        bid = b.get("id") or b.get("_id") or b.get("brandId", "")
                        if bid:
                            brands.append({
                                "brand_id":   bid,
                                "brand_name": b.get("name", ""),
                            })
            except Exception:
                pass  # account access confirmed even if brand list fails

            result.append({"account_id": account_id, "brands": brands})

    return result


def invalidate(email: str) -> None:
    """Force a cache miss on next access check for this email."""
    _cache.pop(email, None)
