#!/usr/bin/env python3
"""
One-time token getter for BrandPilot MCP server.

Opens your browser to the BrandPilot / Cognito login page (Google sign-in supported).
Captures the OAuth2 callback on localhost and exchanges the code for tokens.
Prints the access_token and refresh_token to paste into Claude Desktop config.

Usage:
    python3 scripts/get_token.py

Requirements in .env (or environment):
    COGNITO_CLIENT_ID       — BrandPilot Cognito app client ID
    COGNITO_CLIENT_SECRET   — app client secret (leave empty for public clients)
    COGNITO_DOMAIN_STG      — Cognito hosted-UI domain
    COGNITO_REDIRECT_URI    — must be registered in Cognito (default: http://localhost:3456/callback)
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import urllib.parse
import webbrowser

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

COGNITO_DOMAIN   = os.getenv("COGNITO_DOMAIN_STG", "https://brandpilot-stg-api-domain.auth.eu-central-1.amazoncognito.com")
CLIENT_ID        = os.getenv("COGNITO_CLIENT_ID", "")
CLIENT_SECRET    = os.getenv("COGNITO_CLIENT_SECRET", "")
REDIRECT_URI     = os.getenv("COGNITO_REDIRECT_URI", "http://localhost:3456/callback")
SCOPES           = "openid email profile"

_callback_result: dict = {}


def _pkce_pair() -> tuple[str, str]:
    verifier  = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _callback_result.update(
            code=params.get("code", [None])[0],
            error=params.get("error", [None])[0],
            error_description=params.get("error_description", [""])[0],
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Authentication successful. You can close this tab.</h2>")

    def log_message(self, *_):
        pass  # suppress request logs


def main() -> None:
    if not CLIENT_ID:
        print("ERROR: COGNITO_CLIENT_ID is not set.")
        print("Add it to your .env file or environment and retry.")
        return

    port = int(urllib.parse.urlparse(REDIRECT_URI).port or 3456)
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)

    params = {
        "response_type":         "code",
        "client_id":             CLIENT_ID,
        "redirect_uri":          REDIRECT_URI,
        "scope":                 SCOPES,
        "state":                 state,
        "code_challenge":        challenge,
        "code_challenge_method": "S256",
    }
    auth_url = f"{COGNITO_DOMAIN}/oauth2/authorize?" + urllib.parse.urlencode(params)

    # Start local callback server
    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print(f"\nOpening browser to BrandPilot login...")
    print(f"If the browser does not open, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    thread.join(timeout=120)
    server.server_close()

    if _callback_result.get("error"):
        print(f"\nAuth error: {_callback_result['error']} — {_callback_result['error_description']}")
        return

    code = _callback_result.get("code")
    if not code:
        print("\nNo auth code received. Did you complete the login in the browser?")
        return

    # Exchange code for tokens
    token_data = {
        "grant_type":    "authorization_code",
        "code":          code,
        "redirect_uri":  REDIRECT_URI,
        "client_id":     CLIENT_ID,
        "code_verifier": verifier,
    }
    auth = (CLIENT_ID, CLIENT_SECRET) if CLIENT_SECRET else None

    resp = requests.post(
        f"{COGNITO_DOMAIN}/oauth2/token",
        auth=auth,
        data=token_data,
        timeout=10,
    )
    if not resp.ok:
        print(f"\nToken exchange failed: {resp.status_code} {resp.text}")
        return

    tokens = resp.json()
    access_token  = tokens.get("access_token", "")
    refresh_token = tokens.get("refresh_token", "")
    expires_in    = tokens.get("expires_in", 3600)

    print("\n" + "=" * 70)
    print("TOKENS OBTAINED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nAccess token  (valid {expires_in // 60} min):")
    print(access_token)
    print(f"\nRefresh token (valid ~30 days — keep this secret):")
    print(refresh_token)
    print("\n" + "=" * 70)
    print("Add these to your Claude Desktop MCP server config:")
    print(json.dumps({
        "BRANDPILOT_ACCESS_TOKEN":  access_token,
        "BRANDPILOT_REFRESH_TOKEN": refresh_token,
        "COGNITO_CLIENT_ID":        CLIENT_ID,
    }, indent=2))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
