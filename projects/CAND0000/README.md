# CAND0000 — Cand'art

**Client:** Cand'art
**Project code:** CAND0000 (matches Dropbox)
**Account:** Cand'art (separate account — not under Colruyt)
**Status:** Active

## BrandPilot Backend Setup

Account and brand must be created via the API before running agents:

1. POST `/accounts` — body: `{ "name": "Cand'art", "adminEmail": "..." }`
2. POST `/accounts/{accountId}/brands` — body: `{ "name": "Cand'art", "urls": ["https://candart.be"] }`
3. Save the resulting `account_id` and `brand_id` below once created.

**account_id:** `01KPTNF3WKJ2ASYZA4J6E2V8NS`
**brand_id:** `01KPTNFJNJV2X6C5N1291K843X`
**environment:** staging

## Agents

| Graph ID | Status | Description |
|---|---|---|
| `CAND0000__prospect` | In progress | Deep-research B2B prospect agent |

## LangSmith Prompts

All prompts follow the naming convention `CAND0000__{agent}__{team}`.

| Prompt name | Agent | Team |
|---|---|---|
| `CAND0000__prospect__system` | prospect | system context |
| `CAND0000__prospect__search_plan` | prospect | search query generation |
| `CAND0000__prospect__enrich` | prospect | site enrichment |
| `CAND0000__prospect__score` | prospect | scoring |
| `CAND0000__prospect__outreach_draft` | prospect | outreach email drafting |

## Notes

- Cand'art is a lolly and hard sugar specialist with functional format differentiation
- Position as a format specialist with co-development potential — not generic candy manufacturer
- Key strengths: slow-dissolve formats, sugar-free/vegan/kosher options, impulse packaging
- Can be interesting where gummies are crowded and brands want an original format
