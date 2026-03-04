# app/agent.py
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, PrivateAttr

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

load_dotenv()

TeamName = Literal["interview", "jtbd", "current", "ideal", "features", "sketches"]
BRUSSELS_TZ = "Europe/Brussels"


def _tz_now_iso(tz_name: str) -> str:
    if ZoneInfo is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.now(ZoneInfo(tz_name)).isoformat()


def _model_rank(name: str) -> int:
    ranks = {
        "gpt-5.2": 300,
        "gpt-5": 290,
        "gpt-4.1": 220,
        "gpt-4o": 200,
        "gpt-4o-mini": 120,
    }
    n = (name or "").strip()
    if n in ranks:
        return ranks[n]
    if n.startswith("gpt-5"):
        return 280
    if n.startswith("gpt-4o"):
        return 180
    if n.startswith("gpt-4"):
        return 160
    return 0


class Assets(BaseModel):
    brand_manual_full_md: Optional[str] = None
    brand_manual_snippet_md: Optional[str] = None
    company_context_full_md: Optional[str] = None
    company_context_snippet_md: Optional[str] = None
    lss_json: Optional[Dict[str, Any]] = None
    moka_json: Optional[Dict[str, Any]] = None
    market_trends_md: Optional[str] = None


class RunInput(BaseModel):
    # user input we DO want to return
    brand: str
    account: str = ""

    # user input we DO NOT want to echo back
    persona: Dict[str, Any] = Field(exclude=True)
    assets: Assets = Field(default_factory=Assets, exclude=True)
    recipe: Optional[List[TeamName]] = Field(default=None, exclude=True)
    step: Optional[TeamName] = Field(default=None, exclude=True)

    # per-agent model override: {"interview":"gpt-4o-mini", ...}
    models: Optional[Dict[str, str]] = Field(default=None, exclude=True)

    # legacy: allow callers to send "model" like before
    model: Optional[str] = Field(default=None, exclude=True)

    # new: allow caller to pass a growing list of already-known ideas/features
    existing_features: Optional[str] = Field(default=None, exclude=True)

    extra_input: Optional[str] = Field(default=None, exclude=True)
    output_template_version: str = Field(default="v1", exclude=True)
    prompt_tag: str = Field(default="active", exclude=True)

    # fallback model when neither "models[team]" nor "model" is set
    default_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.2"),
        exclude=True,
    )


class GraphState(RunInput):
    # Private attrs (NOT pydantic fields)
    _step_outputs: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _best_model: str = PrivateAttr(default="")
    _assistant_id: str = PrivateAttr(default="brandpilot_innovation")

    # =========================
    # OUTPUT FIELDS (IN ORDER)
    # =========================
    persona_id: str = ""
    thread_id: str = ""
    run_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_s: float = 0.0
    status: str = ""
    team: str = ""   # assistant_id
    model: str = ""  # most advanced model used
    datum: str = ""

    # q1,a1..q10,a10
    q1: str = ""
    a1: str = ""
    q2: str = ""
    a2: str = ""
    q3: str = ""
    a3: str = ""
    q4: str = ""
    a4: str = ""
    q5: str = ""
    a5: str = ""
    q6: str = ""
    a6: str = ""
    q7: str = ""
    a7: str = ""
    q8: str = ""
    a8: str = ""
    q9: str = ""
    a9: str = ""
    q10: str = ""
    a10: str = ""

    # job1..job10
    job1: str = ""
    job2: str = ""
    job3: str = ""
    job4: str = ""
    job5: str = ""
    job6: str = ""
    job7: str = ""
    job8: str = ""
    job9: str = ""
    job10: str = ""

    # pain1..pain10
    pain1: str = ""
    pain2: str = ""
    pain3: str = ""
    pain4: str = ""
    pain5: str = ""
    pain6: str = ""
    pain7: str = ""
    pain8: str = ""
    pain9: str = ""
    pain10: str = ""

    # gain1..gain10
    gain1: str = ""
    gain2: str = ""
    gain3: str = ""
    gain4: str = ""
    gain5: str = ""
    gain6: str = ""
    gain7: str = ""
    gain8: str = ""
    gain9: str = ""
    gain10: str = ""

    # feature1..feature10
    feature1: str = ""
    feature2: str = ""
    feature3: str = ""
    feature4: str = ""
    feature5: str = ""
    feature6: str = ""
    feature7: str = ""
    feature8: str = ""
    feature9: str = ""
    feature10: str = ""

    # title, journey, step1..step10
    title: str = ""
    journey: str = ""
    step1: str = ""
    step2: str = ""
    step3: str = ""
    step4: str = ""
    step5: str = ""
    step6: str = ""
    step7: str = ""
    step8: str = ""
    step9: str = ""
    step10: str = ""

    # visual1..visual10
    visual1: str = ""
    visual2: str = ""
    visual3: str = ""
    visual4: str = ""
    visual5: str = ""
    visual6: str = ""
    visual7: str = ""
    visual8: str = ""
    visual9: str = ""
    visual10: str = ""


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
        parts += ["### lss_selected", json.dumps(a.lss_json, ensure_ascii=False)]
    if a.moka_json:
        parts += ["### moka_selected", json.dumps(a.moka_json, ensure_ascii=False)]
    return "\n\n".join(parts).strip()


def _get_runtime_ids(config: Optional[RunnableConfig]) -> Dict[str, str]:
    cfg: Dict[str, Any] = {}
    if isinstance(config, dict):
        c = config.get("configurable")
        if isinstance(c, dict):
            cfg = c
    return {
        "thread_id": str(cfg.get("thread_id") or ""),
        "run_id": str(cfg.get("run_id") or ""),
        "assistant_id": str(cfg.get("assistant_id") or ""),
    }


def _extract_hub_system_template(pulled_prompt: Any) -> str:
    if isinstance(pulled_prompt, str):
        return pulled_prompt
    if hasattr(pulled_prompt, "messages") and getattr(pulled_prompt, "messages"):
        m0 = pulled_prompt.messages[0]
        if hasattr(m0, "prompt") and hasattr(m0.prompt, "template") and isinstance(m0.prompt.template, str):
            return m0.prompt.template
        return str(m0)
    if hasattr(pulled_prompt, "template") and isinstance(pulled_prompt.template, str):
        return pulled_prompt.template
    return str(pulled_prompt)


def _get_prompt_text(prompt_name: str) -> Tuple[str, Dict[str, Any]]:
    from langsmith import Client

    client = Client()
    meta: Dict[str, Any] = {"prompt_name": prompt_name}
    try:
        p = client.pull_prompt(prompt_name)
        meta["hub_type"] = p.__class__.__name__
        return _extract_hub_system_template(p), meta
    except Exception as e:
        meta["hub_error"] = repr(e)
        return (
            "Return ONLY valid JSON (no prose). Return the inner object with the exact keys requested by the task.",
            meta,
        )


def _best_effort_json_parse(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Non-JSON model output:\n{text[:2000]}")
    return json.loads(m.group(0))


def _validate_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    errs = sorted(VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    if errs:
        msg = "; ".join([f"{list(er.path)}: {er.message}" for er in errs[:5]])
        raise ValueError(f"Output schema invalid: {msg}")
    return obj


def _make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def _pick_step_model(state: GraphState, team: TeamName) -> str:
    if isinstance(state.models, dict):
        m = state.models.get(str(team))
        if isinstance(m, str) and m.strip():
            return m.strip()
    if isinstance(state.model, str) and state.model.strip():
        return state.model.strip()
    return state.default_model


def _call_llm(state: GraphState, team_system_prompt: str, team: TeamName, model_name: str) -> Dict[str, Any]:
    llm = _make_llm(model_name)
    payload = {
        "brand": state.brand,
        "team": team,
        "persona": state.persona,
        "extra_input": state.extra_input,
        "existing_features": state.existing_features or "",
        "previous": state._step_outputs,  # continuity
        "output_template_version": state.output_template_version,
    }
    resp = llm.invoke(
        [
            {"role": "system", "content": _render_system_context(state)},
            {"role": "system", "content": team_system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    )
    content = resp.content if hasattr(resp, "content") else str(resp)
    return _best_effort_json_parse(content)


def _wrap_if_needed(
    state: GraphState,
    team: TeamName,
    raw: Dict[str, Any],
    prompt_name: str,
    hub_meta: Dict[str, Any],
    model_used: str,
) -> Dict[str, Any]:
    if "template_version" in raw:
        return raw
    persona_id = ""
    if isinstance(state.persona, dict):
        persona_id = str(state.persona.get("id", ""))
    return {
        "template_version": state.output_template_version,
        "team": team,
        "persona_id": persona_id,
        "output": raw,
        "meta": {
            "prompt_name": prompt_name,
            "prompt_tag": state.prompt_tag,
            "model": model_used,
            **hub_meta,
        },
    }


def _track_best_model(state: GraphState, model_name: str) -> None:
    if not state._best_model or _model_rank(model_name) > _model_rank(state._best_model):
        state._best_model = model_name


def _merge_interview(out: Dict[str, Any], state: GraphState) -> None:
    for i in range(1, 11):
        for key in (f"Q{i}", f"q{i}"):
            v = out.get(key)
            if isinstance(v, str):
                setattr(state, f"q{i}", v)
                break
        for key in (f"A{i}", f"a{i}"):
            v = out.get(key)
            if isinstance(v, str):
                setattr(state, f"a{i}", v)
                break


def _merge_jtbd(out: Dict[str, Any], state: GraphState) -> None:
    for i in range(1, 11):
        v = out.get(f"job{i}")
        if isinstance(v, str):
            setattr(state, f"job{i}", v)
        v = out.get(f"pain{i}")
        if isinstance(v, str):
            setattr(state, f"pain{i}", v)
        v = out.get(f"gain{i}")
        if isinstance(v, str):
            setattr(state, f"gain{i}", v)


def _merge_features(out: Dict[str, Any], state: GraphState) -> None:
    for i in range(1, 11):
        v = out.get(f"feature{i}")
        if isinstance(v, str):
            setattr(state, f"feature{i}", v)


def _merge_journey(out: Dict[str, Any], state: GraphState) -> None:
    v = out.get("title")
    if isinstance(v, str):
        state.title = v
    v = out.get("journey")
    if isinstance(v, str):
        state.journey = v
    for i in range(1, 11):
        v = out.get(f"step{i}")
        if isinstance(v, str):
            setattr(state, f"step{i}", v)


def _merge_sketches(out: Dict[str, Any], state: GraphState) -> None:
    for i in range(1, 11):
        v = out.get(f"visual{i}")
        if isinstance(v, str):
            setattr(state, f"visual{i}", v)


def _apply_team(team: TeamName, state: GraphState) -> None:
    prompt_name = f"brandpilot_innovation_{team}"
    team_system_prompt, hub_meta = _get_prompt_text(prompt_name)

    model_name = _pick_step_model(state, team)

    raw = _call_llm(state, team_system_prompt, team, model_name)
    wrapped = _wrap_if_needed(state, team, raw, prompt_name, hub_meta, model_name)
    wrapped = _validate_output(wrapped)

    output = wrapped.get("output", {})
    if not isinstance(output, dict):
        output = {}

    state._step_outputs[str(team)] = output

    if team == "interview":
        _merge_interview(output, state)
    elif team == "jtbd":
        _merge_jtbd(output, state)
    elif team == "features":
        _merge_features(output, state)
    elif team in ("current", "ideal"):
        _merge_journey(output, state)
    elif team == "sketches":
        _merge_sketches(output, state)

    _track_best_model(state, model_name)


def _finalize_state(state: GraphState, config: Optional[RunnableConfig]) -> None:
    ids = _get_runtime_ids(config)
    state.thread_id = ids.get("thread_id", "")
    state.run_id = ids.get("run_id", "")

    assistant_id = ids.get("assistant_id", "").strip()
    if assistant_id:
        state._assistant_id = assistant_id
    state.team = state._assistant_id

    if isinstance(state.persona, dict):
        state.persona_id = str(state.persona.get("id") or state.persona_id or "")
        d = state.persona.get("datum") or state.persona.get("date") or ""
        state.datum = str(d) if d is not None else state.datum

    if not state.started_at:
        state.started_at = _tz_now_iso(BRUSSELS_TZ)
    if not state.ended_at:
        state.ended_at = _tz_now_iso(BRUSSELS_TZ)

    try:
        t0 = datetime.fromisoformat(state.started_at.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(state.ended_at.replace("Z", "+00:00"))
        state.duration_s = float((t1 - t0).total_seconds())
    except Exception:
        state.duration_s = float(state.duration_s or 0.0)

    state.status = "completed"
    state.model = state._best_model or (state.model or state.default_model)


def run_graph(state_in: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    state = _as_state(state_in)

    if not state.started_at:
        state.started_at = _tz_now_iso(BRUSSELS_TZ)

    teams: List[TeamName]
    if state.recipe and len(state.recipe) > 0:
        teams = list(state.recipe)
    else:
        if not state.step:
            raise ValueError("Missing 'recipe' and missing 'step'")
        teams = [state.step]

    for t in teams:
        _apply_team(t, state)

    state.ended_at = _tz_now_iso(BRUSSELS_TZ)
    _finalize_state(state, config=config)

    return state.model_dump(exclude_none=False)


builder = StateGraph(GraphState)
builder.add_node("run_graph", run_graph)
builder.add_edge(START, "run_graph")
builder.add_edge("run_graph", END)
agent = builder.compile()