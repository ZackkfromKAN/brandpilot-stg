"""
BrandPilot MCP tool implementations.

Each public method on BrandPilotToolExecutor maps to one MCP tool exposed to Claude.
Sync BrandPilot API calls are offloaded to a thread pool via asyncio.to_thread()
so they never block the FastAPI event loop.
LangSmith @traceable is applied to the underlying sync functions — this works cleanly
with asyncio.to_thread and captures full inputs/outputs in LangSmith.
"""
from __future__ import annotations

import asyncio
from typing import Any

from langsmith import traceable

from core.client import BrandPilotClient, BrandPilotError
from core.memory import BrandMemoryStore
from core.prompts import get_prompt
from core.tools import tavily_extract, tavily_search


class BrandPilotToolExecutor:
    """
    Stateless executor bound to one authenticated MCP session.
    Instantiated once per tool dispatch; discarded after the call.
    """

    def __init__(self, cognito_token: str, email: str, environment: str = "staging"):
        self._token = cognito_token
        self._email = email
        self._env   = environment

    async def initialize_session(
        self, account_id: str, brand_id: str, project_code: str = ""
    ) -> dict:
        return await asyncio.to_thread(
            _sync_initialize_session,
            self._token, account_id, brand_id, self._env, project_code,
        )

    async def web_search(self, query: str, max_results: int = 8) -> list:
        return await asyncio.to_thread(_sync_web_search, query, int(max_results))

    async def extract_pages(self, urls: list) -> dict:
        return await asyncio.to_thread(_sync_extract_pages, list(urls))

    async def save_research_results(
        self, account_id: str, brand_id: str, results: dict
    ) -> dict:
        return await asyncio.to_thread(
            _sync_save_research_results,
            self._token, account_id, brand_id, results, self._env,
        )

    async def update_brand_memory(
        self, account_id: str, brand_id: str, memory_updates: dict
    ) -> dict:
        return await asyncio.to_thread(
            _sync_update_brand_memory,
            self._token, account_id, brand_id, memory_updates, self._env,
        )


# ── Traced sync implementations ───────────────────────────────────────────────
# Standalone functions (not methods) so @traceable attaches cleanly.


@traceable(name="tool__initialize_session")
def _sync_initialize_session(
    cognito_token: str,
    account_id: str,
    brand_id: str,
    environment: str,
    project_code: str = "",
) -> dict:
    """Load brand context, long-term memory, and agent instructions from LangSmith Hub."""
    client = BrandPilotClient(
        cognito_token=cognito_token,
        account_id=account_id,
        brand_id=brand_id,
        env=environment,
    )

    try:
        client.validate_scope()
    except BrandPilotError as exc:
        return {"error": f"Scope validation failed: {exc}", "status": "error"}

    brand_info  = {}
    manual_resp: Any = {}
    passport    = {}
    markets: list = []

    try:
        brand_info = client.get_brand()
    except BrandPilotError:
        pass

    try:
        manual_resp = client.get_brand_manual()
    except BrandPilotError:
        pass

    try:
        passport = client.get_passport()
    except BrandPilotError:
        pass

    try:
        markets = client.get_markets(defined_only=True)
    except BrandPilotError:
        pass

    # Load long-term memory from brand_manual
    memory = BrandMemoryStore()
    memory.load_from_manual(manual_resp)

    # Build clean manual (strip internal _agent_memory key for Claude)
    inner: Any = manual_resp
    if isinstance(manual_resp, dict) and "manual" in manual_resp:
        inner = manual_resp["manual"]
    clean_manual = (
        {k: v for k, v in inner.items() if k != BrandMemoryStore._NAMESPACE}
        if isinstance(inner, dict)
        else {}
    )

    # Fetch brand-specific instructions — project override first, then generic
    agent_instructions = ""
    candidates = (
        [f"{project_code}__agent__brand_context", "brandpilot__agent__brand_context"]
        if project_code
        else ["brandpilot__agent__brand_context"]
    )
    for prompt_name in candidates:
        template, meta = get_prompt(prompt_name, "active")
        if not meta.get("hub_error"):
            agent_instructions = template
            break

    brand_name = (
        brand_info.get("name")
        or brand_info.get("brand_name")
        or clean_manual.get("brand_name", "")
    )

    return {
        "status":             "ok",
        "brand_name":         brand_name,
        "account_id":         account_id,
        "brand_id":           brand_id,
        "brand_context":      {
            "passport": passport,
            "manual":   clean_manual,
            "markets":  markets,
        },
        "memory":             memory.summary_for_prompt(),
        "agent_instructions": agent_instructions,
    }


@traceable(name="tool__web_search")
def _sync_web_search(query: str, max_results: int) -> list:
    return tavily_search(query, max_results)


@traceable(name="tool__extract_pages")
def _sync_extract_pages(urls: list) -> dict:
    return tavily_extract(urls)


@traceable(name="tool__save_research_results")
def _sync_save_research_results(
    cognito_token: str,
    account_id: str,
    brand_id: str,
    results: dict,
    environment: str,
) -> dict:
    """Persist research results to /chatsessions and update brand memory."""
    client = BrandPilotClient(
        cognito_token=cognito_token,
        account_id=account_id,
        brand_id=brand_id,
        env=environment,
    )

    session_id = ""
    try:
        resp = client.save_prospect_run(results)
        session_id = resp.get("id", "")
    except BrandPilotError:
        pass

    # Update brand memory with shortlisted prospects
    prospects = results.get("prospects", results.get("shortlist", []))
    if prospects:
        try:
            manual_resp = client.get_brand_manual()
            memory = BrandMemoryStore()
            memory.load_from_manual(manual_resp)
            memory.record_prospects(prospects)
            memory.record_run(
                queries=results.get(
                    "queries_used", results.get("search_queries", [])
                ),
                candidates_found=results.get("candidates_found", len(prospects)),
                shortlisted_count=len(prospects),
                geography=results.get("geography", ""),
            )
            updated_manual = memory.merge_into_manual(manual_resp)
            client.update_brand_manual(updated_manual)
        except BrandPilotError:
            pass

    return {"status": "saved", "session_id": session_id}


@traceable(name="tool__update_brand_memory")
def _sync_update_brand_memory(
    cognito_token: str,
    account_id: str,
    brand_id: str,
    memory_updates: dict,
    environment: str,
) -> dict:
    """Apply key/value updates to the brand's long-term agent memory."""
    client = BrandPilotClient(
        cognito_token=cognito_token,
        account_id=account_id,
        brand_id=brand_id,
        env=environment,
    )
    try:
        manual_resp = client.get_brand_manual()
        memory = BrandMemoryStore()
        memory.load_from_manual(manual_resp)
        for key, value in memory_updates.items():
            if key == "do_not_target" and isinstance(value, list):
                memory.mark_do_not_target(value)
            else:
                memory._data[key] = value
        updated_manual = memory.merge_into_manual(manual_resp)
        client.update_brand_manual(updated_manual)
        return {"status": "updated"}
    except BrandPilotError as exc:
        return {"error": str(exc), "status": "failed"}
