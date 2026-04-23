from __future__ import annotations

import os
from typing import Any, Optional

import requests


ENVIRONMENTS = {
    "prod":    os.getenv("BRANDPILOT_API_PRD", "https://brandpilot-prd.com/api"),
    "staging": os.getenv("BRANDPILOT_API_STG", "https://brandpilot-stg.com/api"),
}


class BrandPilotError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"BrandPilot API {status}: {message}")


class BrandPilotClient:
    """
    Authenticated client scoped to one (account_id, brand_id) pair.
    Constructed with a Cognito Bearer token from the caller — never
    generates or stores credentials itself.
    """

    def __init__(
        self,
        cognito_token: str,
        account_id: str,
        brand_id: str,
        env: str = "prod",
    ):
        self.base_url   = ENVIRONMENTS[env]
        self.account_id = account_id
        self.brand_id   = brand_id
        self._headers   = {
            "Authorization": f"Bearer {cognito_token}",
            "Content-Type":  "application/json",
        }

    # ── internals ────────────────────────────────────────────────────────────

    def _brand_path(self, path: str = "") -> str:
        return f"/accounts/{self.account_id}/brands/{self.brand_id}{path}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        r = requests.request(method, url, headers=self._headers, timeout=30, **kwargs)
        if not r.ok:
            raise BrandPilotError(r.status_code, r.text[:500])
        if r.content:
            return r.json()
        return {}

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._request("GET", path, params=params)

    def _put(self, path: str, body: dict) -> Any:
        return self._request("PUT", path, json=body)

    def _post(self, path: str, body: Optional[dict] = None) -> Any:
        return self._request("POST", path, json=body or {})

    # ── scope validation ─────────────────────────────────────────────────────

    def validate_scope(self) -> dict:
        """Verify the token can access this brand. Call once at run start."""
        return self._get(self._brand_path())

    # ── brand context reads ──────────────────────────────────────────────────

    def get_brand(self) -> dict:
        return self._get(self._brand_path())

    def get_brand_manual(self) -> dict:
        return self._get(self._brand_path("/brand_manual"))

    def get_passport(self) -> dict:
        return self._get(self._brand_path("/data/passport"))

    def get_markets(self, defined_only: bool = False) -> list:
        params = {"defined": "true"} if defined_only else None
        return self._get(self._brand_path("/markets"), params=params)

    def get_market_personas(self, market_id: str) -> list:
        return self._get(self._brand_path(f"/markets/{market_id}/marketpersonas"))

    def get_reviews(self) -> dict:
        return self._get(self._brand_path("/data/reviews"))

    def get_news(self) -> dict:
        return self._get(self._brand_path("/data/news"))

    def get_pdfs(self) -> list:
        return self._get(self._brand_path("/pdfs"))

    # ── brand writes ─────────────────────────────────────────────────────────

    def update_brand_manual(self, manual: dict) -> dict:
        return self._put(self._brand_path("/brand_manual"), {"manual": manual})

    def update_passport(self, passport: dict) -> dict:
        return self._put(self._brand_path("/passport"), passport)

    def update_brand(self, data: dict) -> dict:
        return self._put(self._brand_path(), data)

    # ── chat sessions (interaction memory) ───────────────────────────────────

    def create_chat_session(self) -> dict:
        return self._post(self._brand_path("/chatsessions"))

    def get_chat_session(self, session_id: str) -> dict:
        return self._get(self._brand_path(f"/chatsessions/{session_id}"))

    def list_chat_sessions(self) -> list:
        return self._get(self._brand_path("/chatsessions"))

    def save_prospect_run(self, payload: dict) -> dict:
        """
        Persist a prospect run result to BrandPilot via the chat sessions endpoint.
        Payload is stored as-is under the session body. Non-fatal: callers should
        catch BrandPilotError and treat storage failure as a warning, not a crash.
        """
        return self._post(self._brand_path("/chatsessions"), payload)

    # ── account reads ─────────────────────────────────────────────────────────

    def list_account_brands(self) -> list:
        return self._get(f"/accounts/{self.account_id}/brands")

    def get_profile(self) -> dict:
        return self._get("/profile")
