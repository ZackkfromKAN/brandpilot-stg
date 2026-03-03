# app/agent.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

load_dotenv()

TeamName = Literal["interview", "jtbd", "current", "ideal", "features", "sketches"]


class Assets(BaseModel):
    brand_manual_full_md: Optional[str] = None
    brand_manual_snippet_md: Optional[str] = None
    company_context_full_md: Optional[str] = None
    company_context_snippet_md: Optional[str] = None
    lss_json: Optional[Dict[str, Any]] = None
    moka_json: Optional[Dict[str, Any]] = None
    market_trends_md: Optional[str] = None


class RunInput(BaseModel):
    brand: str
    persona: Dict[str, Any]
    team: TeamName
    recipe: Optional[List[TeamName]] = None
    extra_input: Optional[str] = None
    assets: Assets = Field(default_factory=Assets)
    output_template_version: str = "v1"
    prompt_tag: str = "active"
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.2"))


class RunOutput(BaseModel):
    template_version: str
    team: str
    persona_id: str
    output: Dict[str, Any]
    meta: Dict[str, Any]


class GraphState(RunInput):
    step_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    final: Optional[RunOutput] = None
    response: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_s: Optional[float] = None


OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["template_version", "team", "persona_id", "output", "meta"],
    "properties": {
        "template_version": {"type": "string"},
        "team": {"type": "string"},
        "persona_id": {"type": "string"},
        "output": {"type": "object"},
        "meta": {"type": "object"},
    },
    "additionalProperties": True,
}
VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_state(x: Any) -> GraphState:
    if isinstance(x, GraphState):
        return x
    if isinstance(x, dict):
        return GraphState(**x)
    if isinstance(x, RunInput):
        return GraphState(**x.model_dump())
    raise TypeError(f"Unsupported state type: {type(x)}")


def _render_system_context(state: GraphState) -> str:
    a = state.assets
    parts: List[str] = [
        f"brand={state.brand}",
        "### brand_manual_snippet",
        a.brand_manual_snippet_md or "",
        "### company_context_snippet",
        a.company_context_snippet_md or "",
    ]
    if a.brand_manual_full_md:
        parts += ["### brand_manual_full", a.brand_manual_full_md]
    if a.company_context_full_md:
        parts += ["### company_context_full", a.company_context_full_md]
    if a.market_trends_md:
        parts += ["### market_trends", a.market_trends_md]
    if a.lss_json:
        parts += ["### lss_selected", json.dumps(a.lss_json)]
    if a.moka_json:
        parts += ["### moka_selected", json.dumps(a.moka_json)]
    return "\n\n".join(parts).strip()


def _get_runtime_ids(config: Optional[RunnableConfig]) -> Dict[str, Optional[str]]:
    # RunnableConfig is dict-like in practice
    cfg: Dict[str, Any] = {}
    if isinstance(config, dict):
        cfg = (config.get("configurable") or {}) if isinstance(config.get("configurable"), dict) else {}
    return {
        "thread_id": cfg.get("thread_id"),
        "run_id": cfg.get("run_id"),
        "user_id": cfg.get("user_id") or cfg.get("langgraph_auth_user_id"),
    }


def _build_compact_response(
    state: GraphState,
    team: str,
    model_used: str,
    config: Optional[RunnableConfig],
    status: str = "completed",
) -> Dict[str, Any]:
    ids = _get_runtime_ids(config)
    persona_id = None
    if isinstance(state.persona, dict):
        persona_id = state.persona.get("id")

    outputs = dict(state.step_outputs or {})

    return {
        "persona_id": persona_id,
        "thread_id": ids.get("thread_id"),
        "run_id": ids.get("run_id"),
        "status": status,
        "team": team,
        "model": model_used,
        "started_at": state.started_at,
        "ended_at": state.ended_at,
        "duration_s": state.duration_s,
        "outputs": outputs,
    }


def _extract_hub_system_template(pulled_prompt: Any) -> str:
    if isinstance(pulled_prompt, str):
        return pulled_prompt

    # ChatPromptTemplate case
    if hasattr(pulled_prompt, "messages") and getattr(pulled_prompt, "messages"):
        m0 = pulled_prompt.messages[0]
        if hasattr(m0, "prompt") and hasattr(m0.prompt, "template"):
            t = m0.prompt.template
            if isinstance(t, str):
                return t
        return str(m0)

    # PromptTemplate-like
    if hasattr(pulled_prompt, "template") and isinstance(pulled_prompt.template, str):
        return pulled_prompt.template

    return str(pulled_prompt)


def _get_prompt_text(prompt_name: str) -> Tuple[str, Dict[str, Any]]:
    from langsmith import Client

    client = Client()
    meta: Dict[str, Any] = {"prompt_name": prompt_name}

    try:
        # Different langsmith versions accept different kwargs; keep it minimal.
        p = client.pull_prompt(prompt_name)
        meta["hub_type"] = p.__class__.__name__
        return _extract_hub_system_template(p), meta
    except Exception as e:
        meta["hub_error"] = repr(e)
        fallback = (
            "Return ONLY valid JSON (no prose). "
            "Return the inner object with the exact keys requested by the task."
        )
        return fallback, meta


def _best_effort_json_parse(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        return json.loads(candidate)

    raise ValueError(f"Non-JSON model output:\n{text[:2000]}")


def _validate_output(obj: Dict[str, Any]) -> RunOutput:
    errs = sorted(VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    if errs:
        msg = "; ".join([f"{list(er.path)}: {er.message}" for er in errs[:5]])
        raise ValueError(f"Output schema invalid: {msg}")
    return RunOutput(**obj)


def _make_llm(model: str) -> ChatOpenAI:
    # Deterministic
    # Many OpenAI chat models accept this; if not, upstream may ignore it.
    return ChatOpenAI(
        model=model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _call_llm(state: GraphState, team_system_prompt: str, team: TeamName) -> Dict[str, Any]:
    llm = _make_llm(state.model)

    payload = {
        "brand": state.brand,
        "team": team,
        "persona": state.persona,
        "extra_input": state.extra_input,
        "previous": state.step_outputs,
        "output_template_version": state.output_template_version,
    }

    resp = llm.invoke(
        [
            {"role": "system", "content": _render_system_context(state)},
            {"role": "system", "content": team_system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ]
    )

    content = resp.content if hasattr(resp, "content") else str(resp)
    return _best_effort_json_parse(content)


def _run_team_step(state_in: Any, team: TeamName, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    state = _as_state(state_in)

    prompt_name = f"brandpilot_innovation_{team}"
    team_system_prompt, hub_meta = _get_prompt_text(prompt_name)

    # Start time: set once, keep it through recipe
    started_at = state.started_at or _iso_now()

    raw = _call_llm(state, team_system_prompt, team)

    persona_id = ""
    if isinstance(state.persona, dict):
        persona_id = str(state.persona.get("id", ""))

    # Hub prompt returns inner object => wrap it
    if "template_version" not in raw:
        raw = {
            "template_version": state.output_template_version,
            "team": team,
            "persona_id": persona_id,
            "output": raw,
            "meta": {
                "prompt_name": prompt_name,
                "prompt_tag": state.prompt_tag,
                "model": state.model,
                **hub_meta,
            },
        }

    out = _validate_output(raw)

    ended_at = _iso_now()

    duration_s: Optional[float]
    try:
        t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        duration_s = (t1 - t0).total_seconds()
    except Exception:
        duration_s = None

    step_outputs = dict(state.step_outputs or {})
    step_outputs[team] = out.output

    merged_state = GraphState(
        **{
            **state.model_dump(),
            "step_outputs": step_outputs,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": duration_s,
        }
    )

    compact = _build_compact_response(
        merged_state,
        team=str(team),
        model_used=state.model,
        config=config,
        status="completed",
    )

    return {
        "step_outputs": step_outputs,
        "final": out.model_dump(),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_s": duration_s,
        "response": compact,
    }


def run_single(state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    s = _as_state(state)
    return _run_team_step(s, s.team, config=config)


def run_recipe(state: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    s = _as_state(state)
    updates: Dict[str, Any] = {}

    for t in (s.recipe or []):
        merged = s.model_dump()
        merged.update(updates)
        updates = _run_team_step(merged, t, config=config)

    return updates


def choose_path(state: Any) -> str:
    s = _as_state(state)
    return "run_recipe" if (s.recipe and len(s.recipe) > 0) else "run_single"


builder = StateGraph(GraphState)
builder.add_node("run_single", run_single)
builder.add_node("run_recipe", run_recipe)

builder.add_conditional_edges(
    START,
    choose_path,
    {"run_single": "run_single", "run_recipe": "run_recipe"},
)

builder.add_edge("run_single", END)
builder.add_edge("run_recipe", END)

agent = builder.compile()