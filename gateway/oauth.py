"""
OAuth 2.0 + PKCE state machine for the BrandPilot MCP Gateway.

The gateway acts as an OAuth Authorization Server for Claude Desktop while
delegating actual user authentication to AWS Cognito (Google sign-in).

Flow:
  Claude Desktop → GET /oauth/authorize (PKCE)
    → gateway stores pending state, redirects to Cognito
  User logs in to Google via Cognito Hosted UI
  Cognito → GET /oauth/callback
    → gateway exchanges code, fetches email, creates session
    → gateway issues its own auth code, redirects back to Claude Desktop
  Claude Desktop → POST /oauth/token
    → gateway verifies PKCE, returns session_id as access_token
  All subsequent MCP requests: Authorization: Bearer <session_id>
    → gateway looks up session, gets live Cognito token, auto-refreshes when needed
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException

# ── Config ────────────────────────────────────────────────────────────────────

_COGNITO_DOMAIN = os.getenv(
    "COGNITO_DOMAIN_STG",
    "https://brandpilot-stg-api-domain.auth.eu-central-1.amazoncognito.com",
)
_CLIENT_ID      = os.getenv("COGNITO_CLIENT_ID", "")
_CLIENT_SECRET  = os.getenv("COGNITO_CLIENT_SECRET", "")
_GATEWAY_URL    = os.getenv("GATEWAY_URL", "http://localhost:8000")

_REDIRECT_URI   = f"{_GATEWAY_URL}/oauth/callback"

# ── In-memory stores ──────────────────────────────────────────────────────────
# Single-instance only. Extend with Redis when running multiple replicas.

# cognito_state → {client_redirect_uri, client_state, code_challenge,
#                  cognito_verifier, expires_at}
_pending: dict[str, dict] = {}

# gateway auth_code → {session_id, code_challenge, expires_at}
_auth_codes: dict[str, dict] = {}

# session_id (= access_token handed to Claude Desktop) →
#   {email, cognito_access_token, cognito_refresh_token, cognito_expires_at}
_sessions: dict[str, dict] = {}

# gateway refresh_token → session_id
_refresh_map: dict[str, str] = {}


# ── PKCE helper ───────────────────────────────────────────────────────────────

def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


# ── Step 1 — start auth ───────────────────────────────────────────────────────

def begin_auth(
    client_redirect_uri: str,
    client_state: str,
    code_challenge: str,
) -> str:
    """
    Save pending state and return the Cognito authorize URL to redirect the user to.
    """
    cognito_state = secrets.token_urlsafe(24)
    cognito_verifier, cognito_challenge = _pkce_pair()

    _pending[cognito_state] = {
        "client_redirect_uri": client_redirect_uri,
        "client_state":        client_state,
        "code_challenge":      code_challenge,
        "cognito_verifier":    cognito_verifier,
        "expires_at":          time.time() + 600,
    }

    params = {
        "response_type":         "code",
        "client_id":             _CLIENT_ID,
        "redirect_uri":          _REDIRECT_URI,
        "scope":                 "openid email profile",
        "state":                 cognito_state,
        "code_challenge":        cognito_challenge,
        "code_challenge_method": "S256",
    }
    return f"{_COGNITO_DOMAIN}/oauth2/authorize?" + urlencode(params)


# ── Step 2 — Cognito callback ─────────────────────────────────────────────────

async def complete_auth(code: str, cognito_state: str) -> tuple[str, str, str]:
    """
    Exchange the Cognito code for tokens, resolve email, create a gateway session.

    Returns:
        (redirect_url_for_claude_desktop, email, session_id)

    Raises ValueError with a user-friendly message on any failure.
    """
    pending = _pending.pop(cognito_state, None)
    if not pending or pending["expires_at"] < time.time():
        raise ValueError("OAuth state expired or invalid. Please try again.")

    # Exchange code with Cognito
    token_data = {
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  _REDIRECT_URI,
        "client_id":     _CLIENT_ID,
        "code_verifier": pending["cognito_verifier"],
    }
    auth = (_CLIENT_ID, _CLIENT_SECRET) if _CLIENT_SECRET else None

    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(f"{_COGNITO_DOMAIN}/oauth2/token", data=token_data, auth=auth)
    if not r.is_success:
        raise ValueError(f"Cognito token exchange failed ({r.status_code}). Please try again.")

    tokens = r.json()
    cognito_access_token  = tokens["access_token"]
    cognito_refresh_token = tokens.get("refresh_token", "")
    cognito_expires_at    = time.time() + tokens.get("expires_in", 3600)

    # Resolve email from Cognito userinfo
    async with httpx.AsyncClient(timeout=10) as c:
        ui = await c.get(
            f"{_COGNITO_DOMAIN}/oauth2/userInfo",
            headers={"Authorization": f"Bearer {cognito_access_token}"},
        )
    if not ui.is_success:
        raise ValueError("Could not retrieve your profile from Cognito. Please try again.")

    email = ui.json().get("email", "").lower()
    if not email:
        raise ValueError("No email address in your Cognito profile.")

    # Create gateway session
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = {
        "email":                 email,
        "cognito_access_token":  cognito_access_token,
        "cognito_refresh_token": cognito_refresh_token,
        "cognito_expires_at":    cognito_expires_at,
    }

    # Create a short-lived auth code for Claude Desktop to exchange
    auth_code = secrets.token_urlsafe(32)
    _auth_codes[auth_code] = {
        "session_id":     session_id,
        "code_challenge": pending["code_challenge"],
        "expires_at":     time.time() + 300,
    }

    # Redirect back to Claude Desktop's local callback server
    redirect_url = (
        f"{pending['client_redirect_uri']}"
        f"?code={auth_code}&state={pending['client_state']}"
    )
    return redirect_url, email, session_id


# ── Step 3 — token endpoint ───────────────────────────────────────────────────

def exchange_auth_code(code: str, code_verifier: str) -> dict:
    """
    Exchange our auth code for gateway access + refresh tokens.
    Verifies the PKCE code_verifier against the stored code_challenge.
    """
    entry = _auth_codes.pop(code, None)
    if not entry or entry["expires_at"] < time.time():
        raise HTTPException(status_code=400, detail="Invalid or expired authorization code.")

    computed = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    if not hmac.compare_digest(computed, entry["code_challenge"]):
        raise HTTPException(status_code=400, detail="PKCE verification failed.")

    session_id    = entry["session_id"]
    refresh_token = secrets.token_urlsafe(32)
    _refresh_map[refresh_token] = session_id

    return {
        "access_token":  session_id,
        "token_type":    "Bearer",
        "expires_in":    3600,
        "refresh_token": refresh_token,
    }


def do_refresh(refresh_token: str) -> dict:
    """Issue a new access token from a gateway refresh token."""
    session_id = _refresh_map.get(refresh_token)
    if not session_id or session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Invalid or expired refresh token.")
    # The session_id is stable; just return it as a new access_token.
    # The underlying Cognito token refreshes lazily in get_cognito_token().
    return {
        "access_token": session_id,
        "token_type":   "Bearer",
        "expires_in":   3600,
    }


# ── Session helpers ───────────────────────────────────────────────────────────

def get_session(access_token: str) -> dict | None:
    """Return the session dict for this access_token, or None."""
    return _sessions.get(access_token)


def remove_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


async def get_cognito_token(session_id: str) -> str:
    """
    Return a valid Cognito access token for the session, refreshing if < 5 min remain.
    Raises HTTPException(401) if the session is gone or refresh fails.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=401,
            detail="Session not found. Please re-authenticate via Claude Desktop.",
        )

    if time.time() >= session["cognito_expires_at"] - 300:
        refresh_tok = session.get("cognito_refresh_token")
        if not refresh_tok:
            raise HTTPException(
                status_code=401,
                detail="Cognito token expired and no refresh token available. Please re-authenticate.",
            )
        data = {
            "grant_type":    "refresh_token",
            "refresh_token": refresh_tok,
            "client_id":     _CLIENT_ID,
        }
        auth = (_CLIENT_ID, _CLIENT_SECRET) if _CLIENT_SECRET else None
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{_COGNITO_DOMAIN}/oauth2/token", data=data, auth=auth)
        if not r.is_success:
            raise HTTPException(
                status_code=401,
                detail="Token refresh failed. Please re-authenticate via Claude Desktop.",
            )
        tokens = r.json()
        session["cognito_access_token"] = tokens["access_token"]
        session["cognito_expires_at"]   = time.time() + tokens.get("expires_in", 3600)

    return session["cognito_access_token"]
