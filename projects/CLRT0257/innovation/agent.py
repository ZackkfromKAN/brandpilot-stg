from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig

from .state import InnovationState, TeamName, InnovationInput
from .nodes import (
    load_brand_context_node,
    apply_team_node,
    finalize_node,
    _tz_now_iso,
    BRUSSELS_TZ,
)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


def _as_state(x: Any) -> InnovationState:
    if isinstance(x, InnovationState):
        return x
    if isinstance(x, dict):
        return InnovationState(**x)
    if isinstance(x, InnovationInput):
        return InnovationState(**x.model_dump())
    raise TypeError(f"Unsupported input type: {type(x)}")


def run_graph(state_in: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    state = _as_state(state_in)

    if not state.started_at:
        state.started_at = _tz_now_iso(BRUSSELS_TZ)

    # Load live brand context from BrandPilot API (no-op if no credentials)
    state = load_brand_context_node(state)

    # Resolve which teams to run
    teams: List[TeamName]
    if state.recipe and len(state.recipe) > 0:
        teams = list(state.recipe)
    elif state.step:
        teams = [state.step]
    else:
        raise ValueError("Provide either 'recipe' (list of teams) or 'step' (single team)")

    # Run each team in sequence
    for team in teams:
        team_node = apply_team_node(team)
        state = team_node(state, config=config)

    state = finalize_node(state, config=config)
    return state.model_dump(exclude_none=False)


# ── LangGraph registration ───────────────────────────────────────────────────

builder = StateGraph(InnovationState)
builder.add_node("run_graph", run_graph)
builder.add_edge(START, "run_graph")
builder.add_edge("run_graph", END)

agent = builder.compile()
