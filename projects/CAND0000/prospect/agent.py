"""
CAND0000__prospect — Cand'art B2B prospect research agent

Runs deep web research to find qualified B2B prospects for Cand'art,
loads brand context from BrandPilot Backend, saves results back.

Prompts managed in LangSmith Hub under CAND0000__prospect__{team}.
"""

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import Optional

from core.state import BaseRunConfig


class ProspectInput(BaseRunConfig):
    brand:          str
    request_text:   str
    geography:      str  = ""
    prospect_count: int  = 10
    want_outreach:  bool = False


class ProspectState(ProspectInput):
    shortlist: list = []
    errors:    list = []
    status:    str  = "pending"


def run_graph(state_in: dict, config=None) -> dict:
    raise NotImplementedError("CAND0000__prospect not yet implemented — build in next sprint")


builder = StateGraph(ProspectState)
builder.add_node("run_graph", run_graph)
builder.add_edge(START, "run_graph")
builder.add_edge("run_graph", END)

agent = builder.compile()
