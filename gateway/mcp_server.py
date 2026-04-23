"""
MCP over HTTP+SSE for the BrandPilot Gateway.

Transport (MCP spec 2024-11-05):
  1. Claude Desktop → GET /sse            establishes SSE stream
  2. Server         → event: endpoint     tells client where to POST messages
  3. Claude Desktop → POST /messages      sends JSON-RPC requests
  4. Server         → event: message      sends JSON-RPC responses via SSE

Tool calls that run the LangGraph agent (~5 min) are started as asyncio tasks
so the POST /messages handler returns 202 immediately. The result arrives via
the SSE stream when the agent completes.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
from typing import Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from . import access, oauth

# ── Config ────────────────────────────────────────────────────────────────────

_LANGGRAPH_URL     = os.getenv(
    "LANGGRAPH_URL",
    "https://brandpilot-poc-e8d716ed009d5b75936cb979995338bb.eu.langgraph.app",
)
_LANGGRAPH_API_KEY = os.getenv("LANGGRAPH_API_KEY", "")
_AGENT_ID          = "CAND0000__prospect"
_BRANDPILOT_ENV    = os.getenv("BRANDPILOT_ENV", "staging")
_GATEWAY_URL       = os.getenv("GATEWAY_URL", "http://localhost:8000")

# ── MCP session store ─────────────────────────────────────────────────────────
# mcp_session_id → {"queue": asyncio.Queue, "access_token": str, "email": str}
_mcp_sessions: dict[str, dict] = {}


# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "get_my_brands",
        "description": (
            "Returns the BrandPilot accounts and brands this user has access to. "
            "Always call this first to get the account_id and brand_id required "
            "by run_prospect_research. If the user has access to multiple brands, "
            "ask which one they want to work with before proceeding."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_prospect_research",
        "description": (
            "Run the BrandPilot B2B prospect research agent. "
            "Searches the web, enriches company profiles, scores against the brand's "
            "ideal customer profile, and returns a ranked shortlist of prospects. "
            "Takes ~5 minutes. Optionally drafts personalised outreach emails."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Account ID — get from get_my_brands.",
                },
                "brand_id": {
                    "type": "string",
                    "description": "Brand ID — get from get_my_brands.",
                },
                "brand": {
                    "type": "string",
                    "description": "Brand name (e.g. 'Cand\\'art'). Optional if brand_id is provided.",
                },
                "request_text": {
                    "type": "string",
                    "description": "What kind of prospects to find (e.g. 'Find European distributors for our functional lollipops').",
                },
                "geography": {
                    "type": "string",
                    "description": "Optional geographic scope (e.g. 'BeNeLux', 'Western Europe', 'Global').",
                },
                "prospect_count": {
                    "type": "integer",
                    "description": "Max shortlisted prospects to return (default 10, max 50).",
                    "default": 10,
                },
                "want_outreach": {
                    "type": "boolean",
                    "description": "If true, draft a personalised outreach email per prospect.",
                    "default": False,
                },
                "model": {
                    "type": "string",
                    "description": "Optional LLM override (e.g. 'claude-opus-4-7'). Leave empty for default.",
                },
            },
            "required": ["account_id", "brand_id", "request_text"],
        },
    },
]


# ── SSE endpoint ──────────────────────────────────────────────────────────────

async def sse_handler(request: Request, access_token: str) -> StreamingResponse:
    """
    Open an SSE stream for a Claude Desktop session.
    Sends an `endpoint` event so the client knows where to POST messages.
    """
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
    """
    Receive a JSON-RPC message from Claude Desktop.
    Starts an async task to process it; response arrives via SSE.
    """
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
                    "serverInfo":      {"name": "brandpilot", "version": "1.0.0"},
                },
            }

        elif method in ("notifications/initialized", "notifications/cancelled"):
            return  # no response needed for notifications

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
    if name == "get_my_brands":
        return await _tool_get_my_brands(access_token, email)

    if name == "run_prospect_research":
        return await _tool_run_prospect(arguments, access_token, email)

    return json.dumps({"error": f"Unknown tool: {name}"})


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


async def _tool_run_prospect(
    arguments: dict,
    access_token: str,
    email: str,
) -> str:
    account_id     = arguments.get("account_id", "")
    brand_id       = arguments.get("brand_id", "")
    brand          = arguments.get("brand", "")
    request_text   = arguments.get("request_text", "")
    geography      = arguments.get("geography", "")
    prospect_count = int(arguments.get("prospect_count", 10))
    want_outreach  = bool(arguments.get("want_outreach", False))
    model          = arguments.get("model", "")

    if not account_id or not brand_id:
        return json.dumps({
            "error": "account_id and brand_id are required.",
            "hint":  "Call get_my_brands first to discover your account and brand IDs.",
        })
    if not request_text:
        return json.dumps({"error": "request_text is required."})

    try:
        cognito_token = await oauth.get_cognito_token(access_token)
    except HTTPException as exc:
        return json.dumps({"error": f"Authentication error: {exc.detail}"})

    try:
        result = await _run_langgraph(
            cognito_token  = cognito_token,
            account_id     = account_id,
            brand_id       = brand_id,
            brand          = brand,
            request_text   = request_text,
            geography      = geography,
            prospect_count = prospect_count,
            want_outreach  = want_outreach,
            model          = model,
        )
    except Exception as exc:
        return json.dumps({"error": f"Agent run failed: {exc}"})

    shortlist = result.get("shortlist", [])
    duration  = result.get("duration_s")

    summary: dict = {
        "status":       result.get("status", "unknown"),
        "shortlisted":  len(shortlist),
        "duration_min": round(duration / 60, 1) if duration else None,
        "prospects":    shortlist,
    }
    if result.get("errors"):
        summary["agent_errors"] = result["errors"]

    return json.dumps(summary, ensure_ascii=False, indent=2)


# ── LangGraph proxy ───────────────────────────────────────────────────────────

async def _run_langgraph(
    *,
    cognito_token:  str,
    account_id:     str,
    brand_id:       str,
    brand:          str,
    request_text:   str,
    geography:      str,
    prospect_count: int,
    want_outreach:  bool,
    model:          str,
) -> dict[str, Any]:
    """
    Create a LangGraph thread, start a streaming run, poll until terminal status,
    and return the final state values dict.

    Passes the user's Cognito token to the agent so it can call the BrandPilot
    API with the user's own identity. LangGraph's custom auth handler
    (auth/handler.py) also validates this token independently on every request.
    """
    async with httpx.AsyncClient(
        base_url=_LANGGRAPH_URL,
        headers={
            "x-api-key":     _LANGGRAPH_API_KEY,
            "Authorization": f"Bearer {cognito_token}",
            "Content-Type":  "application/json",
        },
        timeout=660,  # agent runs take ~5 min; give extra headroom
    ) as c:
        # Create thread
        tr = await c.post("/threads", json={})
        tr.raise_for_status()
        thread_id = tr.json()["thread_id"]

        # Build agent input — includes BrandPilot credentials so the agent
        # can call GET /brand_manual, GET /passport, POST /chatsessions, etc.
        agent_input: dict[str, Any] = {
            "brand":          brand,
            "request_text":   request_text,
            "geography":      geography,
            "prospect_count": prospect_count,
            "want_outreach":  want_outreach,
            "cognito_token":  cognito_token,
            "account_id":     account_id,
            "brand_id":       brand_id,
            "environment":    _BRANDPILOT_ENV,
        }
        if model:
            agent_input["model"] = model

        # Start run
        rr = await c.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": _AGENT_ID,
                "input":        agent_input,
                "stream_mode":  "events",
            },
        )
        rr.raise_for_status()
        run_id = rr.json()["run_id"]

        # Poll until terminal status
        while True:
            sr = await c.get(f"/threads/{thread_id}/runs/{run_id}")
            sr.raise_for_status()
            status = sr.json().get("status")
            if status == "success":
                break
            if status in ("error", "timeout", "interrupted"):
                raise RuntimeError(f"Agent run ended with status: {status!r}")
            await asyncio.sleep(5)

        # Fetch final state
        state_r = await c.get(f"/threads/{thread_id}/state")
        state_r.raise_for_status()
        return state_r.json().get("values", {})
