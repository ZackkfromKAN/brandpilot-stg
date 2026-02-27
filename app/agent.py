# app/agent.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from langchain_openai import ChatOpenAI
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
    parts = [
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


def _extract_hub_system_template(pulled_prompt: Any) -> str:
    # Hub can return: str, ChatPromptTemplate, PromptTemplate, etc.
    if isinstance(pulled_prompt, str):
        return pulled_prompt

    # ChatPromptTemplate case
    if hasattr(pulled_prompt, "messages") and pulled_prompt.messages:
        m0 = pulled_prompt.messages[0]
        # SystemMessagePromptTemplate typically has .prompt.template
        if hasattr(m0, "prompt") and hasattr(m0.prompt, "template"):
            t = m0.prompt.template
            if isinstance(t, str):
                return t
        # fallback: stringify the first message template
        return str(m0)

    # PromptTemplate-like
    if hasattr(pulled_prompt, "template") and isinstance(pulled_prompt.template, str):
        return pulled_prompt.template

    return str(pulled_prompt)


def _get_prompt_text(prompt_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (system_prompt_text, meta)
    - No .format / no format_messages is called here.
    - No 'revision' arg (API differs per langsmith version).
    """
    from langsmith import Client

    client = Client()
    meta: Dict[str, Any] = {"prompt_name": prompt_name}

    try:
        p = client.pull_prompt(prompt_name, include_model=False)
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

    # Try to grab first JSON object in the text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Non-JSON model output:\n{text[:2000]}")


def _validate_output(obj: Dict[str, Any]) -> RunOutput:
    errs = sorted(VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    if errs:
        msg = "; ".join([f"{list(er.path)}: {er.message}" for er in errs[:5]])
        raise ValueError(f"Output schema invalid: {msg}")
    return RunOutput(**obj)


def _make_llm(model: str) -> ChatOpenAI:
    # Keep deterministic for tests
    # response_format json_object works for many OpenAI chat models; if unsupported it is ignored by some stacks.
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
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    )

    content = resp.content if hasattr(resp, "content") else str(resp)
    return _best_effort_json_parse(content)


def _run_team_step(state_in: Any, team: TeamName) -> Dict[str, Any]:
    state = _as_state(state_in)

    prompt_name = f"brandpilot_innovation_{team}"
    team_system_prompt, hub_meta = _get_prompt_text(prompt_name)

    raw = _call_llm(state, team_system_prompt, team)

    persona_id = str(state.persona.get("id", ""))

    # If hub prompt returns only the inner object, wrap it.
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

    step_outputs = dict(state.step_outputs)
    step_outputs[team] = out.output

    return {
        "step_outputs": step_outputs,
        "final": out.model_dump(),
    }


def run_single(state: Any) -> Dict[str, Any]:
    s = _as_state(state)
    return _run_team_step(s, s.team)


def run_recipe(state: Any) -> Dict[str, Any]:
    s = _as_state(state)
    updates: Dict[str, Any] = {}
    # Replay sequentially, carrying forward updates via local state object
    for t in (s.recipe or []):
        merged = s.model_dump()
        merged.update(updates)  # apply last updates
        updates = _run_team_step(merged, t)
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