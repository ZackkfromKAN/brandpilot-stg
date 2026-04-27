from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig

from core.llm import get_llm, model_rank, DEFAULT_MODEL
from core.memory import BrandMemoryStore
from core.prompts import get_prompt
from core.tools import domain_from_url, homepage_url, tavily_search, tavily_extract
from .state import ProspectState, AGENT_ID

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore


# ── Default prompts ────────────────────────────────────────────────────────────
# Used when LangSmith Hub has no prompt for a given name.
# These are substantive enough to produce real results without Hub setup.

_DEFAULT_PROMPTS: Dict[str, str] = {

    "search_plan": """\
You are a B2B prospect research strategist. Your task: generate 8-12 diverse web search queries
to find companies that could be ideal B2B customers or distribution partners for the given brand.

Strategy:
- Target distributors, wholesalers, co-packing partners, private label buyers, specialty retailers
- Use industry-specific terminology (technical and trade terms)
- Include local-language queries for non-English geographies (French, Dutch if Belgium)
- Mix broad and niche queries — cast a wide net, not a shallow one
- Focus on commercial companies, not news sites or academic references
- Think about which trade directories, industry portals, and association member lists might list the right companies

Return ONLY valid JSON:
{
  "queries": ["query1", "query2", "..."],
  "target_profile": {
    "company_types": ["distributor", "retailer", "co-packer", "..."],
    "sectors": ["confectionery", "FMCG", "..."],
    "geographies": ["Belgium", "..."],
    "key_needs": ["unique formats", "co-development", "..."]
  },
  "reasoning": "Why these queries will surface the best prospects"
}""",

    "enrich": """\
You are a B2B research analyst. From the search results provided, extract structured company profiles
and make an initial relevance judgment for each candidate.

Be selective:
- KEEP: distributors, wholesalers, retailers, brands, food service operators, co-packers that could
  logically purchase or distribute the client brand's product type
- DISCARD: news articles, Wikipedia, academic papers, consumer review sites, government pages,
  social media profiles, companies in clearly wrong sectors (tech, real estate, healthcare)
- When genuinely uncertain: keep=true, note the uncertainty in discard_reason

Extract what you can from the snippet and URL. If you cannot determine a field, use "" or null.

Return ONLY valid JSON:
{
  "candidates": [
    {
      "name": "Company Name",
      "url": "https://...",
      "sector": "confectionery distribution",
      "company_type": "distributor|retailer|brand|food_service|co-packer|other",
      "country": "Belgium",
      "employees_range": "50-200",
      "products_context": "What they sell or handle",
      "why_potentially_relevant": "Brief reason they could be a fit for the brand",
      "initial_relevance_score": 7,
      "keep": true,
      "discard_reason": null
    }
  ]
}""",

    "score": """\
You are a senior B2B business development director evaluating prospects for the given brand.
Your task: score each enriched candidate and build the strongest shortlist possible.

Scoring rubric (0-100):
- Sector & product fit (0-30): Do they handle products in the same category or adjacent?
- Portfolio gap (0-25): Is there a clear gap in their current range that this brand could fill?
- Commercial scale & reach (0-25): Size, geographic footprint, distribution reach
- Strategic alignment (0-20): Co-development appetite, private label openness, innovation orientation

Rejection threshold: score < 55 → exclude. Be aggressive.
A score of 70+ means a genuinely qualified, worth-pursuing prospect.
Do NOT pad the shortlist to hit a number — quality beats quantity.

For each shortlisted company, cite specific evidence from the research data.
Write the outreach_angle as one concrete sentence on how to open the conversation.

Return ONLY valid JSON:
{
  "shortlist": [
    {
      "name": "Company Name",
      "url": "https://...",
      "domain": "example.com",
      "score": 82,
      "score_rationale": "Specific, evidence-based reasons for this score",
      "sector": "...",
      "company_type": "...",
      "country": "...",
      "employees_range": "...",
      "why_strong_fit": "Detailed explanation with facts from research",
      "evidence": ["Specific fact 1", "Specific fact 2"],
      "outreach_angle": "One sentence: the specific hook for this prospect"
    }
  ],
  "rejected_count": 0,
  "scoring_notes": "Brief notes on what distinguished shortlisted from rejected"
}""",

    "outreach_draft": """\
You are a B2B business development specialist drafting outreach emails for the given brand.

Email requirements per prospect:
- Subject: specific, references a real fact about their business (not generic)
- Body: 150-200 words, structured as:
  1. Opening hook (1 sentence): reference something specific about their business
  2. Value proposition (2-3 sentences): what this brand offers that fills their gap
  3. Proof point (1 sentence): one concrete differentiator (format, certification, volume)
  4. Call to action (1 sentence): low-pressure, specific (sample call, virtual tour, catalogue)
- Tone: professional, direct, no fluff. No "I hope this finds you well."

Return ONLY valid JSON:
{
  "outreach": [
    {
      "name": "Company Name",
      "email_subject": "Specific subject line",
      "email_body": "Full email body text"
    }
  ]
}""",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM output")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Non-JSON output (first 500 chars): {text[:500]}")
    return json.loads(m.group(0))


def _pick_model(
    state: ProspectState,
    prompt_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Model selection priority:
    1. LangSmith Hub  — model attached to the prompt in the Hub UI  (highest)
    2. Run payload    — state.model passed by the caller
    3. DEFAULT_MODEL  — env var fallback
    """
    if prompt_meta:
        m = str(prompt_meta.get("model", "")).strip()
        if m:
            return m
    if isinstance(state.model, str) and state.model.strip():
        return state.model.strip()
    return state.default_model or DEFAULT_MODEL


def _get_llm(
    state: ProspectState,
    prompt_meta: Optional[Dict[str, Any]],
    *,
    json_mode: bool = True,
) -> Any:
    """Resolve the LLM to use: honour Hub model/temperature, then fall back."""
    model_name  = _pick_model(state, prompt_meta)
    temperature = float((prompt_meta or {}).get("temperature", 0.0))
    return get_llm(model_name, temperature=temperature, json_mode=json_mode)


def _get_runtime_ids(config: Optional[RunnableConfig]) -> Dict[str, str]:
    cfg: Dict[str, Any] = {}
    if isinstance(config, dict):
        c = config.get("configurable")
        if isinstance(c, dict):
            cfg = c
    return {
        "thread_id":    str(cfg.get("thread_id") or ""),
        "run_id":       str(cfg.get("run_id") or ""),
        "assistant_id": str(cfg.get("assistant_id") or ""),
    }


def _get_prompt(name: str, tag: str, team_key: str) -> Tuple[str, Dict[str, Any]]:
    system_prompt, meta = get_prompt(name, tag=tag)
    if meta.get("hub_error") and team_key in _DEFAULT_PROMPTS:
        system_prompt = _DEFAULT_PROMPTS[team_key]
        meta["fallback_used"] = True
    return system_prompt, meta


def _render_brand_context(state: ProspectState) -> str:
    """Build the brand context block from API data or fall back to minimal payload info."""
    parts: List[str] = [f"brand={state.brand}"]

    ctx = state.brand_context
    if ctx:
        if ctx.passport:
            parts += ["### brand_passport", json.dumps(ctx.passport, ensure_ascii=False)]
        if ctx.brand_manual:
            parts += ["### brand_manual", json.dumps(ctx.brand_manual, ensure_ascii=False)]
        if ctx.markets:
            # First 3 markets give enough context; full list can be very large
            parts += ["### existing_markets", json.dumps(ctx.markets[:3], ensure_ascii=False)]

    # Always append the prospect request so the LLM knows what it's searching for
    parts += [
        "### prospect_request",
        json.dumps(
            {
                "request_text":          state.request_text,
                "geography":             state.geography or "not specified",
                "target_shortlist_size": state.prospect_count,
            },
            ensure_ascii=False,
        ),
    ]
    return "\n\n".join(parts).strip()


# ── Skip-list: domains we never want in the prospect pool ────────────────────
_SKIP_DOMAINS = frozenset([
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "wikipedia.org", "wikimedia.org",
    "bbc.com", "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
    "techcrunch.com", "forbes.com", "businessinsider.com",
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.nl",
    "ebay.com", "zalando.com", "bol.com",
    "google.com", "bing.com", "yahoo.com",
    "glassdoor.com", "indeed.com", "Monster.com",
    "trustpilot.com", "yelp.com",
    "reddit.com", "quora.com",
])

def _is_skip_domain(url: str) -> bool:
    d = domain_from_url(url)
    return any(d == skip or d.endswith(f".{skip}") for skip in _SKIP_DOMAINS)


# ── Node: load_brand_context ──────────────────────────────────────────────────

def load_brand_context_node(state: ProspectState) -> ProspectState:
    """
    Fetch live brand data from BrandPilot Backend if API credentials are present.
    Also loads the per-brand long-term memory from brand_manual.
    Records start time. Skips silently if credentials are absent.
    """
    state.started_at = _now_iso()

    client = state.api_client()
    if client is None:
        return state

    from core.state import BrandContext
    try:
        brand    = client.validate_scope()
        manual   = client.get_brand_manual()
        passport = client.get_passport()
        markets  = client.get_markets(defined_only=True)

        state.brand_context = BrandContext(
            brand_id=state.brand_id,
            account_id=state.account_id,
            brand_name=brand.get("name", state.brand),
            brand_manual=manual,
            passport=passport,
            markets=markets if isinstance(markets, list) else [],
        )
        if not state.brand:
            state.brand = state.brand_context.brand_name

        # ── Load per-brand long-term memory ───────────────────────────────────
        mem = BrandMemoryStore()
        mem.load_from_manual(manual)
        state.brand_memory = mem.to_dict()

    except Exception as exc:
        state.errors.append(f"brand_context_load_failed: {exc}")
        state.brand_context = None

    return state


# ── Node: search_plan ─────────────────────────────────────────────────────────

def search_plan_node(state: ProspectState) -> ProspectState:
    """
    LLM generates a set of diverse search queries and a target prospect profile
    based on the brand context and the caller's request.
    """
    prompt_name = "prospect__search_plan"
    system_prompt, meta = _get_prompt(prompt_name, state.prompt_tag, "search_plan")
    model_name = _pick_model(state, meta)

    mem = BrandMemoryStore.from_dict(state.brand_memory)
    payload = {
        "task":           "Generate search queries to find B2B prospects for this brand.",
        "brand":          state.brand,
        "request_text":   state.request_text,
        "geography":      state.geography or "not specified",
        "prospect_count": state.prospect_count,
        "date":           _now_iso()[:10],
        "memory":         mem.summary_for_prompt(),
    }

    try:
        llm = get_llm(model_name, temperature=float(meta.get("temperature", 0.0)), json_mode=True)
        resp = llm.invoke([
            {"role": "system", "content": _render_brand_context(state)},
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
        ])
        content = resp.content if hasattr(resp, "content") else str(resp)
        raw = _parse_json(content)

        queries = raw.get("queries", [])
        if isinstance(queries, list):
            state.queries_generated = [str(q) for q in queries if q]
        state.target_profile = raw.get("target_profile", {})

        if model_rank(model_name) > model_rank(state.best_model_used):
            state.best_model_used = model_name

    except Exception as exc:
        state.errors.append(f"search_plan_failed: {exc}")
        # Minimal fallback: one generic query so the run can limp along
        if not state.queries_generated:
            state.queries_generated = [f"{state.brand} B2B distributor {state.geography}".strip()]

    return state


# ── Node: search ──────────────────────────────────────────────────────────────

def search_node(state: ProspectState) -> ProspectState:
    """
    Execute all planned search queries via Tavily.
    Deduplicates by domain and builds a raw candidate pool.
    """
    if not state.queries_generated:
        state.errors.append("search_skipped: no queries from search_plan")
        return state

    seen_domains: set[str] = set()
    pool: List[Dict[str, Any]] = []

    for query in state.queries_generated:
        results = tavily_search(query, max_results=8)
        if not results:
            state.errors.append(f"tavily_empty: {query!r}")

        for r in results:
            url = r.get("url", "")
            if not url:
                continue
            if _is_skip_domain(url):
                continue
            d = domain_from_url(url)
            if d in seen_domains:
                continue
            seen_domains.add(d)
            pool.append({
                "title":        r.get("title", ""),
                "url":          url,
                "domain":       d,
                "snippet":      r.get("content", "")[:600],
                "tavily_score": r.get("score", 0.0),
                "source_query": query,
            })

    # Sort by Tavily relevance score descending, cap pool at 60
    pool.sort(key=lambda c: c.get("tavily_score", 0.0), reverse=True)
    state.raw_candidate_pool = pool[:60]
    state.candidates_found = len(state.raw_candidate_pool)
    state.search_queries_used = list(state.queries_generated)

    return state


# ── Node: enrich ──────────────────────────────────────────────────────────────

def enrich_node(state: ProspectState) -> ProspectState:
    """
    Two-phase enrichment:
    1. LLM extracts structured profiles + initial relevance filter from search snippets.
    2. For survivors, fetch actual homepages and attach content for the scoring step.
    """
    if not state.raw_candidate_pool:
        state.errors.append("enrich_skipped: no raw candidates")
        return state

    prompt_name = "prospect__enrich"
    system_prompt, meta = _get_prompt(prompt_name, state.prompt_tag, "enrich")
    model_name = _pick_model(state, meta)

    # ── Phase 1: LLM profile extraction + initial filter ─────────────────────
    candidates_payload = [
        {
            "name":         c["title"],
            "url":          c["url"],
            "domain":       c["domain"],
            "snippet":      c["snippet"],
            "source_query": c["source_query"],
        }
        for c in state.raw_candidate_pool
    ]

    payload = {
        "task":           "Extract company profiles and filter irrelevant results.",
        "brand":          state.brand,
        "target_profile": state.target_profile,
        "candidates":     candidates_payload,
    }

    kept: List[Dict[str, Any]] = []
    try:
        llm = get_llm(model_name, temperature=float(meta.get("temperature", 0.0)), json_mode=True)
        resp = llm.invoke([
            {"role": "system", "content": _render_brand_context(state)},
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
        ])
        content = resp.content if hasattr(resp, "content") else str(resp)
        raw = _parse_json(content)

        for c in raw.get("candidates", []):
            if not isinstance(c, dict):
                continue
            if c.get("keep", True) is False:
                continue
            # Preserve the original domain/url from our search results, not LLM's guess
            url = c.get("url", "")
            d = domain_from_url(url) if url else ""
            kept.append({
                "name":                    c.get("name", ""),
                "url":                     url,
                "domain":                  d,
                "sector":                  c.get("sector", ""),
                "company_type":            c.get("company_type", ""),
                "country":                 c.get("country", ""),
                "employees_range":         c.get("employees_range", ""),
                "products_context":        c.get("products_context", ""),
                "why_potentially_relevant": c.get("why_potentially_relevant", ""),
                "initial_relevance_score": c.get("initial_relevance_score", 5),
                "homepage_content":        "",  # filled in phase 2
            })

        if model_rank(model_name) > model_rank(state.best_model_used):
            state.best_model_used = model_name

    except Exception as exc:
        state.errors.append(f"enrich_llm_failed: {exc}")
        # Fall through with raw candidates so the run doesn't die completely
        for c in state.raw_candidate_pool[:30]:
            kept.append({
                "name":                    c["title"],
                "url":                     c["url"],
                "domain":                  c["domain"],
                "sector":                  "",
                "company_type":            "",
                "country":                 "",
                "employees_range":         "",
                "products_context":        c["snippet"],
                "why_potentially_relevant": "",
                "initial_relevance_score": 5,
                "homepage_content":        "",
            })

    # ── Phase 2: Homepage extraction for up to 30 surviving candidates ────────
    # Sort by initial_relevance_score so we prioritise the best candidates
    kept.sort(key=lambda c: c.get("initial_relevance_score", 0), reverse=True)
    fetch_targets = kept[:30]

    home_urls = [homepage_url(c["url"]) for c in fetch_targets if c.get("url")]
    home_urls = list(dict.fromkeys(home_urls))  # dedup preserving order

    if home_urls:
        extracted = tavily_extract(home_urls, chars_per_page=3500)
        for c in fetch_targets:
            h = homepage_url(c.get("url", ""))
            c["homepage_content"] = extracted.get(h, "")

    state.enriched_candidate_pool = kept
    return state


# ── Node: score ───────────────────────────────────────────────────────────────

def score_node(state: ProspectState) -> ProspectState:
    """
    LLM scores all enriched candidates against the brand's ideal prospect profile.
    Rejects weak fits, returns a ranked shortlist capped at prospect_count.
    """
    if not state.enriched_candidate_pool:
        state.errors.append("score_skipped: no enriched candidates")
        state.status = "completed_no_results"
        return state

    prompt_name = "prospect__score"
    system_prompt, meta = _get_prompt(prompt_name, state.prompt_tag, "score")
    model_name = _pick_model(state, meta)

    # Build scoring payload — truncate homepage_content to keep prompt manageable
    candidates_for_scoring = [
        {
            "name":                    c.get("name", ""),
            "url":                     c.get("url", ""),
            "domain":                  c.get("domain", ""),
            "sector":                  c.get("sector", ""),
            "company_type":            c.get("company_type", ""),
            "country":                 c.get("country", ""),
            "employees_range":         c.get("employees_range", ""),
            "products_context":        c.get("products_context", ""),
            "why_potentially_relevant": c.get("why_potentially_relevant", ""),
            # Homepage content capped at 3000 chars — enough signal without ballooning context
            "homepage_content":        (c.get("homepage_content") or "")[:3000],
        }
        for c in state.enriched_candidate_pool
    ]

    payload = {
        "task":           "Score each prospect and return the shortlist.",
        "brand":          state.brand,
        "target_profile": state.target_profile,
        "prospect_count": state.prospect_count,
        "candidates":     candidates_for_scoring,
    }

    try:
        llm = get_llm(model_name, temperature=float(meta.get("temperature", 0.0)), json_mode=True)
        resp = llm.invoke([
            {"role": "system", "content": _render_brand_context(state)},
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
        ])
        content = resp.content if hasattr(resp, "content") else str(resp)
        raw = _parse_json(content)

        shortlist = raw.get("shortlist", [])
        if isinstance(shortlist, list):
            # Ensure they're sorted by score descending and capped
            shortlist.sort(key=lambda c: c.get("score", 0), reverse=True)
            state.shortlist = shortlist[: state.prospect_count]
        else:
            state.shortlist = []

        state.candidates_shortlisted = len(state.shortlist)

        if model_rank(model_name) > model_rank(state.best_model_used):
            state.best_model_used = model_name

    except Exception as exc:
        state.errors.append(f"score_failed: {exc}")
        # Return enriched candidates as-is so the run has some output
        state.shortlist = [
            {"name": c.get("name", ""), "url": c.get("url", ""), "score": 0}
            for c in state.enriched_candidate_pool[: state.prospect_count]
        ]
        state.candidates_shortlisted = len(state.shortlist)

    return state


# ── Node: outreach_draft (conditional) ───────────────────────────────────────

def outreach_draft_node(state: ProspectState) -> ProspectState:
    """
    Draft personalised outreach emails for each shortlisted prospect.
    Only runs when state.want_outreach is True.
    Patches outreach_email and email_subject into each shortlist entry.
    """
    if not state.shortlist:
        return state

    prompt_name = "prospect__outreach_draft"
    system_prompt, meta = _get_prompt(prompt_name, state.prompt_tag, "outreach_draft")
    model_name = _pick_model(state, meta)

    payload = {
        "task":      "Draft personalised outreach emails for each shortlisted prospect.",
        "brand":     state.brand,
        "shortlist": [
            {
                "name":           c.get("name", ""),
                "url":            c.get("url", ""),
                "why_strong_fit": c.get("why_strong_fit", ""),
                "outreach_angle": c.get("outreach_angle", ""),
                "evidence":       c.get("evidence", []),
            }
            for c in state.shortlist
        ],
    }

    try:
        # outreach defaults to 0.3 for natural tone; LangSmith Hub overrides if set
        llm = get_llm(model_name, temperature=float(meta.get("temperature", 0.3)), json_mode=True)
        resp = llm.invoke([
            {"role": "system", "content": _render_brand_context(state)},
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
        ])
        content = resp.content if hasattr(resp, "content") else str(resp)
        raw = _parse_json(content)

        outreach_map: Dict[str, Dict[str, str]] = {}
        for o in raw.get("outreach", []):
            name = o.get("name", "")
            if name:
                outreach_map[name] = {
                    "email_subject": o.get("email_subject", ""),
                    "email_body":    o.get("email_body", ""),
                }

        for entry in state.shortlist:
            name = entry.get("name", "")
            if name in outreach_map:
                entry["email_subject"] = outreach_map[name].get("email_subject", "")
                entry["outreach_email"] = outreach_map[name].get("email_body", "")

        if model_rank(model_name) > model_rank(state.best_model_used):
            state.best_model_used = model_name

    except Exception as exc:
        state.errors.append(f"outreach_draft_failed: {exc}")

    return state


# ── Node: save_to_backend ─────────────────────────────────────────────────────

def save_to_backend_node(state: ProspectState) -> ProspectState:
    """
    Two tasks:
    1. Persist prospect run result to BrandPilot via chat sessions.
    2. Update per-brand long-term memory in brand_manual (prospects seen, run history).
    Both are non-fatal: the run completes even if persistence fails.
    """
    client = state.api_client()
    if client is None:
        return state

    # ── 1. Save run to chat sessions ──────────────────────────────────────────
    payload = {
        "type":                   "prospect_run",
        "agent":                  AGENT_ID,
        "brand":                  state.brand,
        "request_text":           state.request_text,
        "geography":              state.geography,
        "run_date":               (state.started_at or _now_iso())[:10],
        "shortlist":              state.shortlist,
        "search_queries_used":    state.search_queries_used,
        "target_profile":         state.target_profile,
        "candidates_found":       state.candidates_found,
        "candidates_shortlisted": state.candidates_shortlisted,
        "errors":                 state.errors,
        "model":                  state.best_model_used,
    }

    try:
        result = client.save_prospect_run(payload)
        session_id = result.get("id") or result.get("session_id") or result.get("_id") or ""
        state.session_id = str(session_id)
    except Exception as exc:
        state.errors.append(f"backend_save_failed: {exc}")

    # ── 2. Update brand memory in brand_manual ────────────────────────────────
    try:
        mem = BrandMemoryStore.from_dict(state.brand_memory)
        mem.record_prospects(state.shortlist)
        mem.record_run(
            queries=state.search_queries_used,
            candidates_found=state.candidates_found,
            shortlisted_count=state.candidates_shortlisted,
            geography=state.geography,
        )
        # GET current manual, merge memory in, PUT back
        raw_manual = client.get_brand_manual()
        updated_manual = mem.merge_into_manual(raw_manual)
        client.update_brand_manual(updated_manual)
        state.brand_memory = mem.to_dict()
    except Exception as exc:
        state.errors.append(f"memory_save_failed: {exc}")

    return state


# ── Node: finalize ────────────────────────────────────────────────────────────

def finalize_node(
    state: ProspectState,
    config: Optional[RunnableConfig] = None,
) -> ProspectState:
    """Set runtime IDs, timestamps, duration, and final status."""
    ids = _get_runtime_ids(config)
    state.thread_id = ids["thread_id"]
    state.run_id    = ids["run_id"]

    state.ended_at = _now_iso()

    try:
        t0 = datetime.fromisoformat(state.started_at.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(state.ended_at.replace("Z", "+00:00"))
        state.duration_s = float((t1 - t0).total_seconds())
    except Exception:
        pass

    state.model = state.best_model_used or state.default_model

    if not state.status or state.status == "pending":
        if state.shortlist:
            state.status = "completed"
        elif state.errors:
            state.status = "completed_with_errors"
        else:
            state.status = "completed_no_results"

    return state
