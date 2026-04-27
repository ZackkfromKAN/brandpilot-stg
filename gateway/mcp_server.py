"""
MCP over HTTP+SSE for the BrandPilot Gateway.

Transport (MCP spec 2024-11-05):
  1. Claude / Claude Desktop → GET /sse            establishes SSE stream
  2. Server                  → event: endpoint     tells client where to POST messages
  3. Claude / Claude Desktop → POST /messages      sends JSON-RPC requests
  4. Server                  → event: message      sends JSON-RPC responses via SSE

All BrandPilot tool calls are started as asyncio tasks so the POST /messages
handler returns 202 immediately. Responses arrive via SSE when the tool completes.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from . import access, oauth

# ── Config ────────────────────────────────────────────────────────────────────

_BRANDPILOT_ENV = os.getenv("BRANDPILOT_ENV", "staging")
_GATEWAY_URL    = os.getenv("GATEWAY_URL", "http://localhost:8000")

# ── MCP session store ─────────────────────────────────────────────────────────
# mcp_session_id → {"queue": asyncio.Queue, "access_token": str, "email": str}
_mcp_sessions: dict[str, dict] = {}


# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "get_my_brands",
        "description": (
            "Returns the BrandPilot accounts and brands this user has access to. "
            "Call this if you don't already have the account_id and brand_id. "
            "Then call initialize_session to load brand context before doing any research."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "initialize_session",
        "description": (
            "ALWAYS call this at the start of every conversation, before any research. "
            "Loads brand context (passport, brand manual, markets), long-term memory "
            "(previously discovered prospects and past search queries to avoid repetition), "
            "and detailed research instructions specific to this brand. "
            "Returns everything needed to conduct focused, non-repetitive research."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account ID — from get_my_brands.",
                },
                "brand_id": {
                    "type": "string",
                    "description": "Brand ID — from get_my_brands.",
                },
                "project_code": {
                    "type": "string",
                    "description": (
                        "Optional internal project code (e.g. 'CAND0000') "
                        "for brand-specific instruction overrides in LangSmith Hub."
                    ),
                },
            },
            "required": ["account_id", "brand_id"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for B2B prospects, market intelligence, or brand research. "
            "Uses advanced web search with detailed snippets. "
            "Run multiple focused queries rather than one broad query for best coverage. "
            "Returns title, URL, content snippet, and relevance score per result."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — be specific and targeted.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 8, max 20).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "extract_pages",
        "description": (
            "Extract full text content from web pages (e.g. company homepages). "
            "Use this to enrich prospect profiles after initial web_search results. "
            "Pass up to 20 URLs at once. "
            "Returns {url: content} for all successfully extracted pages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs to extract content from (max 20).",
                },
            },
            "required": ["urls"],
        },
    },
    {
        "name": "save_research_results",
        "description": (
            "Save completed research results to the BrandPilot backend. "
            "Call this once you have a final shortlist of prospects or research output. "
            "Also updates brand memory so future research avoids the same companies. "
            "Results are stored as a chat session under the brand."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account ID from initialize_session.",
                },
                "brand_id": {
                    "type": "string",
                    "description": "Brand ID from initialize_session.",
                },
                "results": {
                    "type": "object",
                    "description": (
                        "Research output. Should include: "
                        "prospects (list of scored companies with name, url, domain, score, "
                        "score_rationale, why_strong_fit, outreach_angle), "
                        "search_queries (list of queries used), "
                        "candidates_found (int), geography (str), request_text (str)."
                    ),
                },
            },
            "required": ["account_id", "brand_id", "results"],
        },
    },
    {
        "name": "update_brand_memory",
        "description": (
            "Directly update specific fields in the brand's long-term memory. "
            "Use for incremental updates such as marking domains as do-not-target. "
            "For saving a full research run with prospects, prefer save_research_results."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account ID from initialize_session.",
                },
                "brand_id": {
                    "type": "string",
                    "description": "Brand ID from initialize_session.",
                },
                "memory_updates": {
                    "type": "object",
                    "description": (
                        "Key/value pairs to merge into brand memory. "
                        "Pass 'do_not_target' as a list of domains to blacklist."
                    ),
                },
            },
            "required": ["account_id", "brand_id", "memory_updates"],
        },
    },
]


# ── SSE endpoint ──────────────────────────────────────────────────────────────

async def sse_handler(request: Request, access_token: str) -> StreamingResponse:
    """Open an SSE stream for a Claude session."""
    session = oauth.get_session(access_token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session.")

    mcp_session_id = secrets.token_urlsafe(16)
    queue: asyncio.Queue = asyncio.Queue()
    _mcp_sessions[mcp_session_id] = {
        "queue":        queue,
        "access_token": access_token,
        "email":        session["email"],
    }

    messages_url = f"{_GATEWAY_URL}/messages?session_id={mcp_session_id}"

    async def event_stream():
        yield f"event: endpoint\ndata: {messages_url}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = queue.get_nowait()
                    yield f"event: message\ndata: {json.dumps(msg, ensure_ascii=False)}\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.05)
        finally:
            _mcp_sessions.pop(mcp_session_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Message endpoint ──────────────────────────────────────────────────────────

async def message_handler(request: Request, mcp_session_id: str) -> Response:
    """Receive a JSON-RPC message. Processes async; response arrives via SSE."""
    mcp_session = _mcp_sessions.get(mcp_session_id)
    if not mcp_session:
        raise HTTPException(status_code=404, detail="MCP session not found.")

    body = await request.json()
    asyncio.create_task(
        _process(mcp_session, body.get("method", ""), body.get("id"), body)
    )
    return Response(status_code=202)


# ── Message processor ─────────────────────────────────────────────────────────

async def _process(mcp_session: dict, method: str, rid: Any, body: dict) -> None:
    queue        = mcp_session["queue"]
    access_token = mcp_session["access_token"]
    email        = mcp_session["email"]

    try:
        if method == "initialize":
            response = {
                "jsonrpc": "2.0", "id": rid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities":    {"tools": {}},
                    "serverInfo":      {"name": "brandpilot", "version": "2.0.0"},
                },
            }

        elif method in ("notifications/initialized", "notifications/cancelled"):
            return

        elif method == "ping":
            response = {"jsonrpc": "2.0", "id": rid, "result": {}}

        elif method == "tools/list":
            response = {"jsonrpc": "2.0", "id": rid, "result": {"tools": _TOOLS}}

        elif method == "tools/call":
            params    = body.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            text = await _dispatch_tool(tool_name, arguments, access_token, email)
            response = {
                "jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": text}]},
            }

        elif rid is not None:
            response = {
                "jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        else:
            return

    except HTTPException as exc:
        response = {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "content": [{"type": "text", "text": f"Access error: {exc.detail}"}],
                "isError": True,
            },
        }
    except Exception as exc:
        response = {
            "jsonrpc": "2.0", "id": rid,
            "result": {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            },
        }

    await queue.put(response)


# ── Tool dispatch ─────────────────────────────────────────────────────────────

async def _dispatch_tool(
    name: str,
    arguments: dict,
    access_token: str,
    email: str,
) -> str:
    # get_my_brands only needs the access token, not a Cognito token
    if name == "get_my_brands":
        return await _tool_get_my_brands(access_token, email)

    # All other tools need a valid Cognito token for BrandPilot API calls
    try:
        cognito_token = await oauth.get_cognito_token(access_token)
    except HTTPException as exc:
        return json.dumps({"error": f"Authentication error: {exc.detail}"})

    from .tools import BrandPilotToolExecutor
    executor = BrandPilotToolExecutor(
        cognito_token=cognito_token,
        email=email,
        environment=_BRANDPILOT_ENV,
    )

    method_map = {
        "initialize_session":    executor.initialize_session,
        "web_search":            executor.web_search,
        "extract_pages":         executor.extract_pages,
        "save_research_results": executor.save_research_results,
        "update_brand_memory":   executor.update_brand_memory,
    }

    handler = method_map.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = await handler(**arguments)
        return json.dumps(result, ensure_ascii=False, default=str)
    except TypeError as exc:
        return json.dumps({"error": f"Invalid tool arguments: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ── get_my_brands ─────────────────────────────────────────────────────────────

async def _tool_get_my_brands(access_token: str, email: str) -> str:
    try:
        cognito_token = await oauth.get_cognito_token(access_token)
        accounts = await access.get_user_access(email, cognito_token)
    except HTTPException as exc:
        return json.dumps({"error": exc.detail})
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    if not accounts:
        return json.dumps({
            "error": f"No BrandPilot accounts found for {email}.",
            "hint":  "Ask your BrandPilot administrator to add your email to an account.",
        })

    return json.dumps({"email": email, "accounts": accounts}, ensure_ascii=False, indent=2)
