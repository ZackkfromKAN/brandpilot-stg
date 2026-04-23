#!/usr/bin/env python3
"""
run_prospect.py — invoke CAND0000__prospect directly (no LangGraph Cloud needed)

Usage:
  python3 run_prospect.py
  python3 run_prospect.py --brand "Cand'art" --request "find Belgian confectionery distributors" --count 5
  python3 run_prospect.py --outreach                  # also draft outreach emails
  python3 run_prospect.py --cognito-token TOKEN ...   # loads live brand data from BrandPilot

Environment (in .env):
  ANTHROPIC_API_KEY  — for Claude models
  OPENAI_API_KEY     — for GPT models
  TAVILY_API_KEY     — web research
  DEFAULT_MODEL      — default: claude-sonnet-4-6
"""

import argparse
import json
import os
import sys

# ── load .env before anything else ───────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from dotenv import load_dotenv
load_dotenv(os.path.join(_here, ".env"))


def _check_env(model: str) -> list[str]:
    warnings = []
    if model.startswith("claude") and not os.getenv("ANTHROPIC_API_KEY", "").strip():
        warnings.append("ANTHROPIC_API_KEY not set — Claude calls will fail")
    if model.startswith("gpt") and not os.getenv("OPENAI_API_KEY", "").strip():
        warnings.append("OPENAI_API_KEY not set — GPT calls will fail")
    if not os.getenv("TAVILY_API_KEY", "").strip():
        warnings.append("TAVILY_API_KEY not set — web search will return empty results")
    return warnings


def _print_shortlist(shortlist: list, show_outreach: bool) -> None:
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  SHORTLIST  ({len(shortlist)} prospects)")
    print(sep)
    for i, p in enumerate(shortlist, 1):
        name  = p.get("name", "Unknown")
        url   = p.get("url", "")
        score = p.get("score", 0)
        ctype = p.get("company_type", "")
        country = p.get("country", "")
        emp   = p.get("employees_range", "")
        fit   = p.get("why_strong_fit") or p.get("score_rationale", "")
        evidence = p.get("evidence", [])
        angle = p.get("outreach_angle", "")

        print(f"\n  #{i}  {name}  [score: {score}/100]")
        print(f"      {url}")
        meta = " | ".join(filter(None, [ctype, country, emp]))
        if meta:
            print(f"      {meta}")
        if fit:
            print(f"\n      WHY:  {fit}")
        for ev in evidence[:2]:
            print(f"      ▸ {ev}")
        if angle:
            print(f"      ANGLE: {angle}")

        if show_outreach:
            subj  = p.get("email_subject", "")
            email = p.get("outreach_email", "")
            if email:
                print(f"\n      ── Outreach ──")
                if subj:
                    print(f"      Subject: {subj}")
                for line in email.splitlines():
                    print(f"      {line}")

    print(f"\n{sep}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAND0000__prospect")
    parser.add_argument("--brand",          default="Cand'art",
                        help="Brand name (default: Cand'art)")
    parser.add_argument("--request",        default=(
                            "Find Belgian and Dutch confectionery distributors, wholesalers, "
                            "and specialty food retailers that could be B2B partners for "
                            "a lolly and hard sugar specialist with co-development potential."
                        ))
    parser.add_argument("--geography",      default="Belgium, Netherlands")
    parser.add_argument("--count",          type=int, default=5,
                        help="Target shortlist size (default: 5)")
    parser.add_argument("--outreach",       action="store_true",
                        help="Also draft outreach emails")
    parser.add_argument("--model",          default="",
                        help="LLM model override (default: from DEFAULT_MODEL env)")
    parser.add_argument("--cognito-token",  default="")
    parser.add_argument("--account-id",     default="")
    parser.add_argument("--brand-id",       default="")
    parser.add_argument("--environment",    default="staging")
    parser.add_argument("--output",         default="prospect_output.json",
                        help="Path to save full JSON output")
    args = parser.parse_args()

    model = args.model.strip() or os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6")

    warnings = _check_env(model)
    for w in warnings:
        print(f"  WARN  {w}")

    # ── assemble payload ──────────────────────────────────────────────────────
    payload: dict = {
        "brand":          args.brand,
        "request_text":   args.request,
        "geography":      args.geography,
        "prospect_count": args.count,
        "want_outreach":  args.outreach,
        "model":          model,
    }
    if args.cognito_token:
        payload.update({
            "cognito_token": args.cognito_token,
            "account_id":    args.account_id,
            "brand_id":      args.brand_id,
            "environment":   args.environment,
        })

    # ── print run config ──────────────────────────────────────────────────────
    print("\n  BrandPilot Prospect Agent — CAND0000__prospect")
    print(f"  brand:    {payload['brand']}")
    print(f"  request:  {payload['request_text'][:80]}...")
    print(f"  geo:      {payload['geography'] or 'any'}")
    print(f"  count:    {payload['prospect_count']}")
    print(f"  model:    {model}")
    print(f"  outreach: {payload['want_outreach']}")
    print(f"  api:      {'yes (live brand data)' if args.cognito_token else 'no (web-only mode)'}")
    print()

    # ── import & run ──────────────────────────────────────────────────────────
    from projects.CAND0000.prospect.agent import agent

    # LangGraph auto-instruments every node when LANGSMITH_TRACING=true.
    # run_name, tags, and metadata appear in the LangSmith trace UI.
    run_config = {
        "run_name": f"CAND0000__prospect / {payload['brand']}",
        "tags":     ["CAND0000", "prospect"],
        "metadata": {
            "brand":          payload["brand"],
            "geography":      payload["geography"],
            "prospect_count": payload["prospect_count"],
            "model":          model,
            "environment":    "local",
        },
    }

    print("  Running... (this takes 1-3 minutes)")
    result = agent.invoke(payload, config=run_config)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n  Status:   {result.get('status')}")
    print(f"  Queries:  {len(result.get('search_queries_used', []))} sent")
    print(f"  Pool:     {result.get('candidates_found', 0)} raw candidates found")
    print(f"  Short:    {result.get('candidates_shortlisted', 0)} made the shortlist")
    print(f"  Model:    {result.get('model', '')}")
    print(f"  Time:     {result.get('duration_s', 0):.1f}s")

    errors = result.get("errors", [])
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    ⚠  {e}")

    shortlist = result.get("shortlist", [])
    if shortlist:
        _print_shortlist(shortlist, args.outreach)
    else:
        print("\n  No prospects shortlisted.")

    # ── save output ───────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Full output → {args.output}")


if __name__ == "__main__":
    main()
