#!/usr/bin/env python3
"""
BrandPilot Prospect Agent — Claude Desktop MCP server.

Each user runs this script locally. It talks to the deployed LangGraph agent
on their behalf, forwarding their Cognito access token for server-side auth.

──────────────────────────────────────────────────────────────────────────────
Claude Desktop config  (~/.config/Claude/claude_desktop_config.json on Mac):

{
  "mcpServers": {
    "brandpilot-prospect": {
      "command": "python3",
      "args": ["/path/to/brandpilot/scripts/brandpilot_mcp.py"],
      "env": {
        "BRANDPILOT_ACCESS_TOKEN":  "<your access token — run get_token.py>",
        "BRANDPILOT_REFRESH_TOKEN": "<your refresh token — run get_token.py>",
        "COGNITO_CLIENT_ID":        "<ask your BrandPilot admin>",
        "COGNITO_DOMAIN_STG":       "https://brandpilot-stg-api-domain.auth.eu-central-1.amazoncognito.com",
        "LANGGRAPH_URL":            "<deployment URL — ask your BrandPilot admin>",
        "LANGGRAPH_API_KEY":        "<LangSmith API key — ask your BrandPilot admin>"
      }
    }
  }
}

BRANDPILOT_ACCESS_TOKEN  : required — valid Cognito access token (1-hour TTL).
BRANDPILOT_REFRESH_TOKEN : optional — enables automatic token refresh so you
                           never need to update the config manually.
COGNITO_CLIENT_ID        : required when BRANDPILOT_REFRESH_TOKEN is set.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx

# ── Configuration ─────────────────────────────────────────────────────────────

_LANGGRAPH_URL     = os.getenv(
    "LANGGRAPH_URL",
    "https://brandpilot-poc-e8d716ed009d5b75936cb979995338bb.eu.langgraph.app",
)
_LANGGRAPH_API_KEY = os.getenv("LANGGRAPH_API_KEY", "")
_AGENT_ID          = "CAND0000__prospect"

_COGNITO_DOMAIN    = os.getenv(
    "COGNITO_DOMAIN_STG",
    "https://brandpilot-stg-api-domain.auth.eu-central-1.amazoncognito.com",
)
_CLIENT_ID         = os.getenv("COGNITO_CLIENT_ID", "")
_CLIENT_SECRET     = os.getenv("COGNITO_CLIENT_SECRET", "")

# ── Token management ──────────────────────────────────────────────────────────

_token_state: dict[str, Any] = {
    "access_token": os.getenv("BRANDPILOT_ACCESS_TOKEN", ""),
    "expires_at":   0.0,  # 0 = unknown, will re-read from env on first call
}


def _token_expires_soon() -> bool:
    """True if the current token is missing or will expire in < 5 minutes."""
    if not _token_state["access_token"]:
        return True
    at = _token_state["expires_at"]
    if at == 0:
        return False  # token was set from env, trust it
    return time.time() >= at - 300


def _refresh_access_token() -> str:
    """
    Exchange the Cognito refresh token for a new access token.
    Requires BRANDPILOT_REFRESH_TOKEN + COGNITO_CLIENT_ID in env.
    Returns the new access token or raises RuntimeError.
    """
    refresh_token = os.getenv("BRANDPILOT_REFRESH_TOKEN", "")
    if not refresh_token:
        raise RuntimeError(
            "Token expired and no BRANDPILOT_REFRESH_TOKEN is configured. "
            "Run scripts/get_token.py to get a new token."
        )
    if not _CLIENT_ID:
        raise RuntimeError("COGNITO_CLIENT_ID is required for token refresh.")

    data = {
        "grant_type":    "refresh_token",
        "refresh_token": refresh_token,
        "client_id":     _CLIENT_ID,
    }
    auth = (_CLIENT_ID, _CLIENT_SECRET) if _CLIENT_SECRET else None

    r = httpx.post(
        f"{_COGNITO_DOMAIN}/oauth2/token",
        auth=auth,
        data=data,
        timeout=10,
    )
    if not r.is_success:
        raise RuntimeError(f"Token refresh failed: {r.status_code} {r.text}")

    payload = r.json()
    _token_state["access_token"] = payload["access_token"]
    _token_state["expires_at"] = time.time() + payload.get("expires_in", 3600)
    return _token_state["access_token"]


def get_token() -> str:
    """Return a valid access token, refreshing automatically if needed."""
    if _token_expires_soon():
        return _refresh_access_token()
    return _token_state["access_token"]


# ── LangGraph agent call ──────────────────────────────────────────────────────

async def run_prospect_agent(
    brand: str,
    request_text: str,
    geography: str = "",
    prospect_count: int = 10,
    want_outreach: bool = False,
    model: str = "",
) -> dict[str, Any]:
    """
    Invoke the CAND0000__prospect agent and stream until completion.
    Returns the final state dict.
    """
    token = get_token()

    async with httpx.AsyncClient(
        base_url=_LANGGRAPH_URL,
        headers={
            "x-api-key":     _LANGGRAPH_API_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        timeout=600,  # prospect runs take ~5 minutes
    ) as c:
        # Create a thread
        thread_resp = await c.post("/threads", json={})
        thread_resp.raise_for_status()
        thread_id = thread_resp.json()["thread_id"]

        # Build input
        agent_input: dict[str, Any] = {
            "brand":           brand,
            "request_text":    request_text,
            "geography":       geography,
            "prospect_count":  prospect_count,
            "want_outreach":   want_outreach,
        }
        if model:
            agent_input["model"] = model

        # Start a streaming run
        run_resp = await c.post(
            f"/threads/{thread_id}/runs",
            json={
                "assistant_id": _AGENT_ID,
                "input":        agent_input,
                "stream_mode":  "events",
            },
        )
        run_resp.raise_for_status()
        run_id = run_resp.json()["run_id"]

        # Poll until terminal status
        while True:
            status_resp = await c.get(f"/threads/{thread_id}/runs/{run_id}")
            status_resp.raise_for_status()
            run = status_resp.json()
            status = run.get("status")

            if status == "success":
                break
            if status in ("error", "timeout", "interrupted"):
                raise RuntimeError(f"Agent run ended with status: {status}")

            await asyncio.sleep(5)

        # Return final thread state
        state_resp = await c.get(f"/threads/{thread_id}/state")
        state_resp.raise_for_status()
        return state_resp.json().get("values", {})


# ── MCP server ────────────────────────────────────────────────────────────────

async def _handle_tool_call(name: str, arguments: dict) -> str:
    if name == "run_prospect_research":
        brand          = arguments.get("brand", "")
        request_text   = arguments.get("request_text", "")
        geography      = arguments.get("geography", "")
        prospect_count = int(arguments.get("prospect_count", 10))
        want_outreach  = bool(arguments.get("want_outreach", False))
        model          = arguments.get("model", "")

        if not brand or not request_text:
            return json.dumps({"error": "brand and request_text are required."})

        result = await run_prospect_agent(
            brand=brand,
            request_text=request_text,
            geography=geography,
            prospect_count=prospect_count,
            want_outreach=want_outreach,
            model=model,
        )

        shortlist  = result.get("shortlist", [])
        status     = result.get("status", "unknown")
        duration   = result.get("duration_seconds")
        outreach   = result.get("outreach", [])

        summary = {
            "status":       status,
            "shortlisted":  len(shortlist),
            "duration_min": round(duration / 60, 1) if duration else None,
            "prospects":    shortlist,
        }
        if outreach:
            summary["outreach"] = outreach

        return json.dumps(summary, ensure_ascii=False, indent=2)

    return json.dumps({"error": f"Unknown tool: {name}"})


_TOOLS = [
    {
        "name": "run_prospect_research",
        "description": (
            "Run the BrandPilot B2B prospect research agent for a brand. "
            "Searches the web, enriches company profiles, scores against the brand's ideal customer "
            "profile, and returns a ranked shortlist of prospects. "
            "Optionally drafts personalised outreach emails for each prospect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "brand": {
                    "type": "string",
                    "description": "Brand name (e.g. 'Cand\\'art')",
                },
                "request_text": {
                    "type": "string",
                    "description": "What kind of prospects to find (e.g. 'Find European distributors and co-development partners for our functional lollipops').",
                },
                "geography": {
                    "type": "string",
                    "description": "Optional geographic scope (e.g. 'BeNeLux', 'Western Europe', 'Global').",
                },
                "prospect_count": {
                    "type": "integer",
                    "description": "Maximum number of shortlisted prospects to return (default 10, max 50).",
                    "default": 10,
                },
                "want_outreach": {
                    "type": "boolean",
                    "description": "If true, draft a personalised outreach email for each shortlisted prospect.",
                    "default": False,
                },
                "model": {
                    "type": "string",
                    "description": "Optional LLM override (e.g. 'claude-opus-4-7'). Leave empty for default.",
                },
            },
            "required": ["brand", "request_text"],
        },
    }
]


async def _stdio_loop() -> None:
    """Minimal MCP stdio transport — reads JSON-RPC from stdin, writes to stdout."""
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_event_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await loop.connect_write_pipe(
        asyncio.BaseProtocol, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

    async def send(obj: dict) -> None:
        msg = json.dumps(obj)
        writer.write(f"Content-Length: {len(msg.encode())}\r\n\r\n{msg}".encode())
        await writer.drain()

    while True:
        # Read headers
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if not line or line == b"\r\n":
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        length = int(headers.get("content-length", 0))
        if not length:
            continue

        body = await reader.readexactly(length)
        req  = json.loads(body)
        rid  = req.get("id")
        method = req.get("method", "")

        if method == "initialize":
            await send({
                "jsonrpc": "2.0", "id": rid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "brandpilot-prospect", "version": "1.0.0"},
                },
            })

        elif method == "tools/list":
            await send({"jsonrpc": "2.0", "id": rid, "result": {"tools": _TOOLS}})

        elif method == "tools/call":
            params = req.get("params", {})
            tool_name = params.get("name", "")
            arguments  = params.get("arguments", {})
            try:
                result_text = await _handle_tool_call(tool_name, arguments)
                await send({
                    "jsonrpc": "2.0", "id": rid,
                    "result": {"content": [{"type": "text", "text": result_text}]},
                })
            except Auth.exceptions.HTTPException as e:
                await send({
                    "jsonrpc": "2.0", "id": rid,
                    "result": {
                        "content": [{"type": "text", "text": f"Access denied: {e.detail}"}],
                        "isError": True,
                    },
                })
            except Exception as e:
                await send({
                    "jsonrpc": "2.0", "id": rid,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {e}"}],
                        "isError": True,
                    },
                })

        elif method == "notifications/initialized":
            pass  # no response needed

        elif rid is not None:
            await send({
                "jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })


if __name__ == "__main__":
    if not _token_state["access_token"]:
        print(
            "ERROR: BRANDPILOT_ACCESS_TOKEN is not set.\n"
            "Run scripts/get_token.py to obtain your token, then add it to your\n"
            "Claude Desktop MCP server config.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _LANGGRAPH_API_KEY:
        print(
            "ERROR: LANGGRAPH_API_KEY is not set.\n"
            "Add it to your Claude Desktop MCP server config.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        asyncio.run(_stdio_loop())
    except KeyboardInterrupt:
        pass
