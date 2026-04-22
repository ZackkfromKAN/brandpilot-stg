# BrandPilot — Claude Code Project Brief

This file is read automatically by Claude Code at session start.
It contains everything needed to continue work without the prior conversation.

---

## What This Project Is

A production-grade multi-tenant AI agent platform for KAN Design's BrandPilot product.
Multiple agents run on a single LangGraph Cloud deployment, each scoped to a client project.
All prompts are managed in LangSmith Hub. Brand data lives in BrandPilot Backend (AWS).

---

## Repository Structure

```
brandpilot/
├── core/                          ← shared kernel (auth, API client, LLM factory, prompts)
│   ├── client.py                  ← BrandPilotClient — all HTTP calls to brandpilot-stg/prd.com/api
│   ├── auth.py                    ← Cognito token refresh helpers
│   ├── llm.py                     ← get_llm() — supports Claude (claude-*) and OpenAI (gpt-*)
│   ├── prompts.py                 ← LangSmith Hub fetcher
│   └── state.py                   ← BaseRunConfig + BrandContext shared by all agents
│
├── projects/
│   ├── CLRT0257/                  ← Colruyt project (FINISHED — Colruyt + Xtra brands)
│   │   └── innovation/            ← brand innovation agent (interview/jtbd/features/journey/sketches)
│   │       ├── agent.py           ← LangGraph graph, exported as `agent`
│   │       ├── nodes.py           ← all node functions
│   │       └── state.py           ← InnovationState (extends BaseRunConfig)
│   │
│   └── CAND0000/                  ← Cand'art project (ACTIVE)
│       ├── README.md              ← has account_id and brand_id for staging
│       └── prospect/              ← B2B prospect research agent (IN PROGRESS — stub only)
│           └── agent.py
│
├── generic/                       ← agents reusable across all clients (placeholder)
│   └── prospect/
│
├── langgraph.json                 ← registers all agents for LangGraph Cloud
└── requirements.txt
```

---

## Agent Naming Convention

`{PROJECT_CODE}__{agent_type}`

Examples:
- `CLRT0257__innovation` — innovation agent for Colruyt project
- `CAND0000__prospect` — prospect agent for Cand'art project
- `generic__prospect` — reusable prospect agent (future)

Registered in `langgraph.json`. Add new agents there when creating them.

---

## LangSmith Prompt Naming Convention

`{PROJECT_CODE}__{agent}__{team}` tagged `active` in LangSmith Hub.

Examples:
- `CLRT0257__innovation__interview`
- `CLRT0257__innovation__jtbd`
- `CAND0000__prospect__search_plan`
- `CAND0000__prospect__score`

Prompts are fetched at runtime via `core/prompts.py`. Changing the `active` tag in LangSmith
rolls out instantly without a redeploy.

---

## BrandPilot Backend

**API:**
- Staging: `https://brandpilot-stg.com/api`
- Production: `https://brandpilot-prd.com/api`

**Auth:** AWS Cognito OAuth 2.0 — Bearer token in Authorization header.
The token comes from the caller (frontend/Postman). The agent never generates tokens itself.
Cognito domain (staging): `https://brandpilot-stg-api-domain.auth.eu-central-1.amazoncognito.com`

**Data hierarchy:** Account → Brand → (passport, brand_manual, markets, pdfs, chatsessions)

**Key endpoints used by agents:**
- `GET /accounts/{accountId}/brands/{brandId}` — validate scope
- `GET /accounts/{accountId}/brands/{brandId}/brand_manual` — read brand manual
- `PUT /accounts/{accountId}/brands/{brandId}/brand_manual` — write brand manual
- `GET /accounts/{accountId}/brands/{brandId}/data/passport` — read passport
- `GET /accounts/{accountId}/brands/{brandId}/markets` — read markets/personas
- `POST /accounts/{accountId}/brands/{brandId}/chatsessions` — create interaction record

---

## CAND0000 — Cand'art (Staging IDs)

- **account_id:** `01KPTNF3WKJ2ASYZA4J6E2V8NS`
- **brand_id:** `01KPTNFJNJV2X6C5N1291K843X`
- Created 2026-04-22 via API

Cand'art context: lolly and hard sugar specialist. Position as a format specialist with
functional differentiation and co-development potential. Key strengths: slow-dissolve formats,
sugar-free/vegan/kosher options, impulse packaging. Interesting where gummies are crowded.

---

## Security Model

- Agent receives `cognito_token` + `account_id` + `brand_id` from the caller in the run payload
- `core/client.py` calls `validate_scope()` on construction (GET brand endpoint) — confirms token can access this brand
- All tool calls use `scope.account_id` / `scope.brand_id` from the validated client — never raw user input
- Secrets (API keys, Cognito client secret) live in `.env` only — never in prompts or code
- Tenant isolation enforced by BrandPilot Backend URL structure + Cognito JWT

---

## LLM Factory

`core/llm.py` — `get_llm(model, temperature, json_mode)` returns the right LangChain object:
- `claude-*` → `ChatAnthropic`
- `gpt-*` → `ChatOpenAI` (with `response_format: json_object` when json_mode=True)
- Default model: `DEFAULT_MODEL` env var (fallback: `gpt-4.1`)

---

## How to Add a New Agent

1. Create `projects/{PROJECT_CODE}/{agent_type}/` with `agent.py`, `nodes.py`, `state.py`
2. `state.py` — define input/state model extending `BaseRunConfig` from `core/state.py`
3. `nodes.py` — implement node functions; use `core/llm.py` for LLM calls, `core/prompts.py` for prompts
4. `agent.py` — wire LangGraph `StateGraph`, export as `agent`
5. Register in `langgraph.json` as `"{PROJECT_CODE}__{agent_type}": "./projects/..."`
6. Create prompts in LangSmith Hub following `{PROJECT_CODE}__{agent}__{team}` convention

---

## Current Status (as of 2026-04-22)

| Agent | Status | Notes |
|---|---|---|
| `CLRT0257__innovation` | Complete | Refactored from original `app/agent.py`. Loads live brand context from API if credentials provided. |
| `CAND0000__prospect` | Stub | Registered in langgraph.json. Nodes not yet implemented. Next thing to build. |

**Next step:** Implement `CAND0000__prospect` nodes — broad web research via Claude, enrich candidates, score, save to BrandPilot via chat sessions.

---

## Key Design Preferences (from project owner)

- Hybrid build: Claude Code for architecture/schemas/security, LangSmith Hub for all prompt text
- Project codes match Dropbox folder names (e.g. CLRT0257, CAND0000)
- One account per client in BrandPilot Backend — no cross-client data
- Agents should be duplicatable across projects — CAND0000__prospect can be cloned later
- Support Claude and GPT models — provider is a config choice, not hardcoded
- ISO-grade tenant isolation: backend enforces it, agent never trusts its own input for scoping
