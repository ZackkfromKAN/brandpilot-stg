"""
generic__prospect — multi-client prospect research agent
Status: stub — to be built in next sprint

This agent will:
1. Load brand context from BrandPilot Backend (account_id + brand_id scoped)
2. Run broad web research via Claude
3. Enrich, filter, and score candidates
4. Save shortlist back to BrandPilot Backend via chat sessions
5. Optionally draft outreach emails

Prompt names follow the convention: generic__prospect__{team}
e.g. generic__prospect__system
     generic__prospect__search_plan
     generic__prospect__score
     generic__prospect__outreach_draft
"""

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from core.state import BaseRunConfig


class ProspectInput(BaseRunConfig):
    brand:           str
    request_text:    str
    geography:       str = ""
    prospect_count:  int = 10
    want_outreach:   bool = False


class ProspectState(ProspectInput):
    shortlist: list = []
    errors:    list = []
    status:    str  = "pending"


def run_graph(state_in: dict, config=None) -> dict:
    raise NotImplementedError("generic__prospect agent not yet implemented")


builder = StateGraph(ProspectState)
builder.add_node("run_graph", run_graph)
builder.add_edge(START, "run_graph")
builder.add_edge("run_graph", END)

agent = builder.compile()
