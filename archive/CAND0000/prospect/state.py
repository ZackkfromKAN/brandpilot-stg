from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.state import BaseRunConfig

AGENT_ID = "CAND0000__prospect"


class ProspectInput(BaseRunConfig):
    brand:          str
    request_text:   str
    geography:      str            = ""
    prospect_count: int            = Field(default=10, ge=1, le=50)
    want_outreach:  bool           = False
    model:          Optional[str]  = Field(default=None, exclude=True)
    prompt_tag:     str            = Field(default="active", exclude=True)
    default_model:  str            = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4.1"),
        exclude=True,
    )


class ProspectState(ProspectInput):
    # ── Pipeline intermediate data ─────────────────────────────────────────────
    # Must be regular public fields (NOT PrivateAttr) so LangGraph's checkpointing
    # preserves them across node transitions. PrivateAttr is excluded from
    # model_dump() — the mechanism LangGraph uses to pass state between nodes.
    queries_generated:      List[str]             = Field(default_factory=list)
    raw_candidate_pool:     List[Dict[str, Any]]  = Field(default_factory=list)
    enriched_candidate_pool: List[Dict[str, Any]] = Field(default_factory=list)
    best_model_used:        str                   = ""

    # ── Long-term memory (loaded from brand_manual at run start) ─────────────
    brand_memory:           Dict[str, Any]        = Field(default_factory=dict)

    # ── Outputs ───────────────────────────────────────────────────────────────
    shortlist:              List[Dict[str, Any]]  = Field(default_factory=list)
    search_queries_used:    List[str]             = Field(default_factory=list)
    target_profile:         Dict[str, Any]        = Field(default_factory=dict)
    candidates_found:       int                   = 0
    candidates_shortlisted: int                   = 0
    errors:                 List[str]             = Field(default_factory=list)
    status:                 str                   = "pending"

    # ── Runtime IDs ───────────────────────────────────────────────────────────
    run_id:     str   = ""
    thread_id:  str   = ""
    session_id: str   = ""
    started_at: str   = ""
    ended_at:   str   = ""
    duration_s: float = 0.0
    model:      str   = ""
