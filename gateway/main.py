"""
BrandPilot MCP Gateway — FastAPI application entry point.

Exposes:
  /.well-known/oauth-protected-resource   OAuth resource metadata (RFC 9728)
  /.well-known/oauth-authorization-server OAuth AS metadata (RFC 8414)
  /oauth/authorize                         Start Cognito PKCE login
  /oauth/callback                          Cognito callback → issue gateway tokens
  /oauth/token                             Token endpoint for Claude Desktop
  /sse                                     MCP SSE stream (requires Bearer token)
  /messages                                MCP message endpoint (requires Bearer token)
  /health                                  Health check (no auth)
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from . import access, mcp_server, oauth

_GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")

app = FastAPI(title="BrandPilot MCP Gateway", docs_url=None, redoc_url=None)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"ok": True, "service": "brandpilot-mcp-gateway"}


# ── OAuth discovery ────────────────────────────────────────────────────────────

@app.get("/.well-known/oauth-protected-resource")
async def oauth_protected_resource():
    """RFC 9728 — tells Claude Desktop which AS protects this resource."""
    return JSONResponse({
        "resource":              _GATEWAY_URL,
        "authorization_servers": [_GATEWAY_URL],
    })


@app.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server():
    """RFC 8414 — OAuth AS metadata. Claude Desktop reads this to discover endpoints."""
    return JSONResponse({
        "issuer":                                _GATEWAY_URL,
        "authorization_endpoint":                f"{_GATEWAY_URL}/oauth/authorize",
        "token_endpoint":                        f"{_GATEWAY_URL}/oauth/token",
        "response_types_supported":              ["code"],
        "grant_types_supported":                 ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported":      ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    })


# ── OAuth flow ─────────────────────────────────────────────────────────────────

@app.get("/oauth/authorize")
async def oauth_authorize(
    response_type:         str = "code",
    redirect_uri:          str = "",
    state:                 str = "",
    code_challenge:        str = "",
    code_challenge_method: str = "S256",
    client_id:             str = "",  # accepted but not validated — public client
):
    """Step 1 — redirect user to Cognito Hosted UI (Google login)."""
    if response_type != "code":
        raise HTTPException(status_code=400, detail="Only 'code' response_type is supported.")
    if not redirect_uri:
        raise HTTPException(status_code=400, detail="redirect_uri is required.")
    if not code_challenge:
        raise HTTPException(status_code=400, detail="PKCE code_challenge is required.")

    cognito_url = oauth.begin_auth(
        client_redirect_uri=redirect_uri,
        client_state=state,
        code_challenge=code_challenge,
    )
    return RedirectResponse(url=cognito_url, status_code=302)


@app.get("/oauth/callback")
async def oauth_callback(
    code:              str = "",
    state:             str = "",
    error:             str = "",
    error_description: str = "",
):
    """Step 2 — Cognito redirects here after login. Check BrandPilot access, issue tokens."""
    if error:
        return HTMLResponse(
            _html_error(f"Authentication failed: {error}", error_description),
            status_code=400,
        )

    try:
        redirect_url, email, session_id = await oauth.complete_auth(
            code=code, cognito_state=state
        )
    except ValueError as exc:
        return HTMLResponse(_html_error("Authentication error", str(exc)), status_code=400)

    # Gate on BrandPilot account membership before issuing the session
    try:
        cognito_token = await oauth.get_cognito_token(session_id)
        accounts = await access.get_user_access(email, cognito_token)
    except Exception:
        accounts = []

    if not accounts:
        oauth.remove_session(session_id)
        return HTMLResponse(
            _html_error(
                "Access denied",
                f"{email} is not authorised for any BrandPilot account.<br>"
                "Contact your administrator to request access.",
            ),
            status_code=403,
        )

    # All good — redirect to Claude Desktop's local callback to complete the OAuth handshake
    return RedirectResponse(url=redirect_url, status_code=302)


@app.post("/oauth/token")
async def oauth_token(request: Request):
    """Step 3 — Claude Desktop exchanges auth code for access + refresh tokens."""
    form = await request.form()
    grant_type = str(form.get("grant_type", ""))

    if grant_type == "authorization_code":
        code          = str(form.get("code", ""))
        code_verifier = str(form.get("code_verifier", ""))
        if not code or not code_verifier:
            raise HTTPException(status_code=400, detail="code and code_verifier are required.")
        return JSONResponse(oauth.exchange_auth_code(code, code_verifier))

    if grant_type == "refresh_token":
        refresh_token = str(form.get("refresh_token", ""))
        if not refresh_token:
            raise HTTPException(status_code=400, detail="refresh_token is required.")
        return JSONResponse(oauth.do_refresh(refresh_token))

    raise HTTPException(status_code=400, detail=f"Unsupported grant_type: {grant_type!r}")


# ── MCP endpoints ──────────────────────────────────────────────────────────────

@app.get("/sse")
async def mcp_sse(request: Request):
    """MCP SSE stream — establishes a persistent connection for JSON-RPC responses."""
    access_token = _extract_bearer(request)
    return await mcp_server.sse_handler(request, access_token)


@app.post("/messages")
async def mcp_messages(request: Request, session_id: str = ""):
    """MCP message endpoint — receives JSON-RPC requests, responds via SSE."""
    _extract_bearer(request)  # auth check (session validated inside mcp_server)
    return await mcp_server.message_handler(request, session_id)


# ── Auth helper ────────────────────────────────────────────────────────────────

def _extract_bearer(request: Request) -> str:
    """Extract Bearer token from Authorization header. Raises 401 with OAuth discovery hint."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization required.",
            headers={
                "WWW-Authenticate": (
                    f'Bearer realm="{_GATEWAY_URL}", '
                    f'resource_metadata_uri="{_GATEWAY_URL}/.well-known/oauth-protected-resource"'
                )
            },
        )
    token = auth[7:].strip()
    if not token or not oauth.get_session(token):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session. Please re-authenticate.",
            headers={
                "WWW-Authenticate": (
                    f'Bearer realm="{_GATEWAY_URL}", '
                    f'resource_metadata_uri="{_GATEWAY_URL}/.well-known/oauth-protected-resource"'
                )
            },
        )
    return token


# ── HTML helpers ───────────────────────────────────────────────────────────────

def _html_error(title: str, detail: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><title>BrandPilot — {title}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:520px;margin:80px auto;padding:0 20px}}
h2{{color:#c0392b}}p{{color:#555;line-height:1.6}}</style></head>
<body><h2>{title}</h2><p>{detail}</p></body></html>"""
