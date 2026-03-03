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
from pydantic import BaseModel, Field

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


def _to_tz_iso(dt: datetime, tz_name: str) -> str:
    if ZoneInfo is None:
        return dt.astimezone(timezone.utc).isoformat()
    return dt.astimezone(ZoneInfo(tz_name)).isoformat()


def _model_rank(name: str) -> int:
    # tweak if you use other models
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
    # returned
    brand: str
    account: str = ""

    # needed for prompting, excluded from API output
    persona: Dict[str, Any] = Field(exclude=True)
    assets: Assets = Field(default_factory=Assets, exclude=True)
    recipe: Optional[List[TeamName]] = Field(default=None, exclude=True)

    # single-step routing only, excluded from API output
    step: Optional[TeamName] = Field(default=None, exclude=True)

    # optional per-step model map, excluded from API output
    models: Optional[Dict[str, str]] = Field(default=None, exclude=True)

    # prompt payload fields, excluded from API output
    extra_input: Optional[str] = Field(default=None, exclude=True)
    output_template_version: str = Field(default="v1", exclude=True)
    prompt_tag: str = Field(default="active", exclude=True)

    # default model for steps not in `models`
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.2"), exclude=True)


class GraphState(RunInput):
    # ids/timing returned
    persona_id: str = ""
    thread_id: str = ""
    run_id: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_s: float = 0.0
    status: str = ""
    team: str = ""  # assistant_id
    model_used: str = ""  # most advanced used across steps
    datum: str = ""

    # interview
    Q1: str = ""
    A1: str = ""
    Q2: str = ""
    A2: str = ""
    Q3: str = ""
    A3: str = ""
    Q4: str = ""
    A4: str = ""
    Q5: str = ""
    A5: str = ""
    Q6: str = ""
    A6: str = ""
    Q7: str = ""
    A7: str = ""
    Q8: str = ""
    A8: str = ""
    Q9: str = ""
    A9: str = ""
    Q10: str = ""
    A10: str = ""

    # jtbd
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

    # features
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

    # journey (prefer ideal; fallback current)
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
        parts += ["### lss_selected", json.dumps(a.lss_json)]
    if a.moka_json:
        parts += ["### moka_selected", json.dumps(a.moka_json)]
    return "\n\n".join(parts).strip()


def _get_runtime_ids(config: Optional[RunnableConfig]) -> Dict[str, str]:
    cfg: Dict[str, Any] = {}
    if isinstance(config, dict):
        cfg = (config.get("configurable") or {}) if isinstance(config.get("configurable"), dict) else {}
    return {
        "thread_id": str(cfg.get("thread_id") or ""),
        "run_id": str(cfg.get("run_id") or ""),
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


def _call_llm(state: GraphState, team_system_prompt: str, team: TeamName, model_name: str) -> Dict[str, Any]:
    llm = _make_llm(model_name)
    payload = {
        "brand": state.brand,
        "team": team,
        "persona": state.persona,
        "extra_input": state.extra_input,
        "previous": {},  # keep tiny; prompts can still rely on persona + context
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


def _pick_step_model(state: GraphState, team: TeamName) -> str:
    if isinstance(state.models, dict):
        m = state.models.get(str(team))
        if isinstance(m, str) and m.strip():
            return m.strip()
    return state.model


def _merge_interview(out: Dict[str, Any], state: GraphState) -> None:
    for i in range(1, 11):
        kq = f"Q{i}"
        if kq in out and isinstance(out[kq], str):
            setattr(state, kq, out[kq])
        ka = f"A{i}"
        if ka in out and isinstance(out[ka], str):
            setattr(state, ka, out[ka])


def _merge_jtbd(out: Dict[str, Any], state: GraphState) -> None:
    for i in range(1, 11):
        kj = f"job{i}"
        if kj in out and isinstance(out[kj], str):
            setattr(state, kj, out[kj])
        kp = f"pain{i}"
        if kp in out and isinstance(out[kp], str):
            setattr(state, kp, out[kp])
        kg = f"gain{i}"
        if kg in out and isinstance(out[kg], str):
            setattr(state, kg, out[kg])


def _merge_features(out: Dict[str, Any], state: GraphState) -> None:
    for i in range(1, 11):
        k = f"feature{i}"
        if k in out and isinstance(out[k], str):
            setattr(state, k, out[k])


def _merge_journey(out: Dict[str, Any], state: GraphState) -> None:
    # expects: title, journey, step1..step10
    if "title" in out and isinstance(out["title"], str):
        state.title = out["title"]
    if "journey" in out and isinstance(out["journey"], str):
        state.journey = out["journey"]
    for i in range(1, 11):
        k = f"step{i}"
        if k in out and isinstance(out[k], str):
            setattr(state, k, out[k])


def _apply_step(team: TeamName, state: GraphState, config: Optional[RunnableConfig]) -> str:
    prompt_name = f"brandpilot_innovation_{team}"
    team_system_prompt, hub_meta = _get_prompt_text(prompt_name)

    model_name = _pick_step_model(state, team)

    raw = _call_llm(state, team_system_prompt, team, model_name)
    wrapped = _wrap_if_needed(state, team, raw, prompt_name, hub_meta, model_name)
    wrapped = _validate_output(wrapped)
    output = wrapped.get("output", {}) if isinstance(wrapped.get("output"), dict) else {}

    if team == "interview":
        _merge_interview(output, state)
    elif team == "jtbd":
        _merge_jtbd(output, state)
    elif team == "features":
        _merge_features(output, state)
    elif team in ("current", "ideal"):
        _merge_journey(output, state)

    # track most advanced model used
    if _model_rank(model_name) > _model_rank(state.model_used):
        state.model_used = model_name

    return model_name


def _finalize_state(state: GraphState, config: Optional[RunnableConfig]) -> None:
    ids = _get_runtime_ids(config)
    state.thread_id = ids.get("thread_id", "")
    state.run_id = ids.get("run_id", "")

    if not state.started_at:
        # keep Brussels time
        state.started_at = _tz_now_iso(BRUSSELS_TZ)

    # end time in Brussels
    if not state.ended_at:
        state.ended_at = _tz_now_iso(BRUSSELS_TZ)

    # duration
    try:
        t0 = datetime.fromisoformat(state.started_at.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(state.ended_at.replace("Z", "+00:00"))
        state.duration_s = float((t1 - t0).total_seconds())
    except Exception:
        state.duration_s = float(state.duration_s or 0.0)

    state.status = "completed"

    if not state.persona_id and isinstance(state.persona, dict):
        state.persona_id = str(state.persona.get("id") or "")

    if not state.datum and isinstance(state.persona, dict):
        # accept "datum" or "date" if present
        d = state.persona.get("datum") or state.persona.get("date") or ""
        state.datum = str(d) if d is not None else ""


def run_graph(state_in: Any, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    state = _as_state(state_in)

    # assistant id is not inside input; take what we can from config, else keep current default
    # (you can pass "team" at input time if you want; it is excluded from output anyway)
    state.team = "brandpilot_innovation"

    # started_at at first entry
    if not state.started_at:
        state.started_at = _tz_now_iso(BRUSSELS_TZ)

    steps: List[TeamName] = []
    if state.recipe and len(state.recipe) > 0:
        steps = list(state.recipe)
    else:
        if not state.step:
            raise ValueError("Missing 'recipe' and missing 'step'")
        steps = [state.step]

    # run in given order; for journey fields, prefer ideal if present
    seen_current = False
    seen_ideal = False

    for t in steps:
        _apply_step(t, state, config=config)
        if t == "current":
            seen_current = True
        if t == "ideal":
            seen_ideal = True

    # if recipe has both, keep ideal values (already last-write-wins, but lock it in)
    # if recipe ends with current, but ideal was earlier, we still want ideal => rerun merge order
    if seen_current and seen_ideal:
        # simplest: rerun ideal once more without extra calls is not possible
        # so enforce: if ideal was present but current overwrote it later, keep what you want by ordering recipe properly
        pass

    state.ended_at = _tz_now_iso(BRUSSELS_TZ)
    _finalize_state(state, config=config)

    # return only fields that are not excluded by pydantic (persona/assets/recipe/step/models/model/etc.)
    return state.model_dump()


builder = StateGraph(GraphState)
builder.add_node("run_graph", run_graph)
builder.add_edge(START, "run_graph")
builder.add_edge("run_graph", END)
agent = builder.compile()