"""
CAND0000__prospect — Cand'art B2B prospect research agent

Graph:
  load_brand_context → search_plan → search → enrich → score
    → [outreach_draft if want_outreach] → save_to_backend → finalize

Prompts managed in LangSmith Hub under CAND0000__prospect__{team}.
When Hub prompts are absent, substantive default prompts are used (see nodes.py).

Registered in langgraph.json as "CAND0000__prospect".
"""

from langgraph.graph import StateGraph, START, END

from .state import ProspectState
from .nodes import (
    load_brand_context_node,
    search_plan_node,
    search_node,
    enrich_node,
    score_node,
    outreach_draft_node,
    save_to_backend_node,
    finalize_node,
)


def _route_after_score(state: ProspectState) -> str:
    return "outreach_draft" if state.want_outreach else "save_to_backend"


builder = StateGraph(ProspectState)

builder.add_node("load_brand_context", load_brand_context_node)
builder.add_node("search_plan",        search_plan_node)
builder.add_node("search",             search_node)
builder.add_node("enrich",             enrich_node)
builder.add_node("score",              score_node)
builder.add_node("outreach_draft",     outreach_draft_node)
builder.add_node("save_to_backend",    save_to_backend_node)
builder.add_node("finalize",           finalize_node)

builder.add_edge(START,                "load_brand_context")
builder.add_edge("load_brand_context", "search_plan")
builder.add_edge("search_plan",        "search")
builder.add_edge("search",             "enrich")
builder.add_edge("enrich",             "score")
builder.add_conditional_edges(
    "score",
    _route_after_score,
    {
        "outreach_draft":  "outreach_draft",
        "save_to_backend": "save_to_backend",
    },
)
builder.add_edge("outreach_draft", "save_to_backend")
builder.add_edge("save_to_backend",    "finalize")
builder.add_edge("finalize",           END)

agent = builder.compile()
