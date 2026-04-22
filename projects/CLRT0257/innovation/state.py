from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, PrivateAttr

from core.state import BaseRunConfig


TeamName = Literal["interview", "jtbd", "current", "ideal", "features", "sketches"]

AGENT_ID = "CLRT0257__innovation"


class Assets(BaseModel):
    """
    Optional brand assets passed directly in the payload.
    Superseded by live API data when cognito_token + account_id + brand_id are provided.
    """
    brand_manual_full_md:     Optional[str]           = None
    brand_manual_snippet_md:  Optional[str]           = None
    company_context_full_md:  Optional[str]           = None
    company_context_snippet_md: Optional[str]         = None
    lss_json:                 Optional[Dict[str, Any]] = None
    moka_json:                Optional[Dict[str, Any]] = None
    market_trends_md:         Optional[str]           = None


class InnovationInput(BaseRunConfig):
    # ── required ──────────────────────────────────────────────────────────────
    brand:   str
    persona: Dict[str, Any]  = Field(exclude=True)

    # ── optional run config ───────────────────────────────────────────────────
    account: str = ""
    assets:  Assets = Field(default_factory=Assets, exclude=True)
    recipe:  Optional[List[TeamName]] = Field(default=None, exclude=True)
    step:    Optional[TeamName]       = Field(default=None, exclude=True)

    # ── model selection ───────────────────────────────────────────────────────
    models:        Optional[Dict[str, str]] = Field(default=None, exclude=True)
    model:         Optional[str]            = Field(default=None, exclude=True)
    default_model: str = Field(
        default_factory=lambda: __import__("os").getenv("DEFAULT_MODEL", "gpt-4.1"),
        exclude=True,
    )

    # ── misc ──────────────────────────────────────────────────────────────────
    existing_features:       Optional[Union[str, List[Any]]] = Field(default=None, exclude=True)
    extra_input:             Optional[str]  = Field(default=None, exclude=True)
    output_template_version: str            = Field(default="v1", exclude=True)
    prompt_tag:              str            = Field(default="active", exclude=True)


class InnovationState(InnovationInput):
    _step_outputs: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _best_model:   str = PrivateAttr(default="")

    # ── output fields ─────────────────────────────────────────────────────────
    persona_id: str   = ""
    thread_id:  str   = ""
    run_id:     str   = ""
    started_at: str   = ""
    ended_at:   str   = ""
    duration_s: float = 0.0
    status:     str   = ""
    team:       str   = ""
    model:      str   = ""
    datum:      str   = ""

    q1:  str = ""; a1:  str = ""
    q2:  str = ""; a2:  str = ""
    q3:  str = ""; a3:  str = ""
    q4:  str = ""; a4:  str = ""
    q5:  str = ""; a5:  str = ""
    q6:  str = ""; a6:  str = ""
    q7:  str = ""; a7:  str = ""
    q8:  str = ""; a8:  str = ""
    q9:  str = ""; a9:  str = ""
    q10: str = ""; a10: str = ""

    job1:  str = ""; pain1:  str = ""; gain1:  str = ""
    job2:  str = ""; pain2:  str = ""; gain2:  str = ""
    job3:  str = ""; pain3:  str = ""; gain3:  str = ""
    job4:  str = ""; pain4:  str = ""; gain4:  str = ""
    job5:  str = ""; pain5:  str = ""; gain5:  str = ""
    job6:  str = ""; pain6:  str = ""; gain6:  str = ""
    job7:  str = ""; pain7:  str = ""; gain7:  str = ""
    job8:  str = ""; pain8:  str = ""; gain8:  str = ""
    job9:  str = ""; pain9:  str = ""; gain9:  str = ""
    job10: str = ""; pain10: str = ""; gain10: str = ""

    feature1: str = ""; feature2: str = ""; feature3: str = ""
    feature4: str = ""; feature5: str = ""; feature6: str = ""
    feature7: str = ""; feature8: str = ""; feature9: str = ""
    feature10: str = ""

    title:  str = ""
    journey: str = ""
    step1:  str = ""; step2:  str = ""; step3:  str = ""
    step4:  str = ""; step5:  str = ""; step6:  str = ""
    step7:  str = ""; step8:  str = ""; step9:  str = ""
    step10: str = ""

    visual1:  str = ""; visual2:  str = ""; visual3:  str = ""
    visual4:  str = ""; visual5:  str = ""; visual6:  str = ""
    visual7:  str = ""; visual8:  str = ""; visual9:  str = ""
    visual10: str = ""
