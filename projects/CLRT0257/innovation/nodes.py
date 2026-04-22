from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from jsonschema import Draft202012Validator
from langchain_core.runnables import RunnableConfig

from core.llm import get_llm, model_rank, DEFAULT_MODEL
from core.prompts import get_prompt
from .state import InnovationState, TeamName, AGENT_ID

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore

BRUSSELS_TZ = "Europe/Brussels"

OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["template_version", "team", "persona_id", "output", "meta"],
    "properties": {
        "template_version": {"type": "string"},
        "team":             {"type": "string"},
        "persona_id":       {"type": "string"},
        "output":           {"type": "object"},
        "meta":             {"type": "object"},
    },
    "additionalProperties": True,
}
VALIDATOR = Draft202012Validator(OUTPUT_SCHEMA)


# ── helpers ───────────────────────────────────────────────────────────────────

def _tz_now_iso(tz_name: str) -> str:
    if ZoneInfo is None:
        return datetime.now(timezone.utc).isoformat()
    return datetime.now(ZoneInfo(tz_name)).isoformat()


def _normalize_features(x: Optional[Union[str, List[Any]]]) -> Tuple[str, List[str]]:
    if x is None:
        return "", []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return "", []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    items = [str(v).strip() for v in parsed if str(v).strip()]
                    return "\n".join(items), items
            except Exception:
                pass
        return s, [s]
    if isinstance(x, list):
        items = [str(v).strip() for v in x if v is not None and str(v).strip()]
        return "\n".join(items), items
    s = str(x).strip()
    return s, [s] if s else []


def _parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model output")
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Non-JSON output:\n{text[:1000]}")
    return json.loads(m.group(0))


def _validate(obj: Dict[str, Any]) -> Dict[str, Any]:
    errs = sorted(VALIDATOR.iter_errors(obj), key=lambda e: e.path)
    if errs:
        msg = "; ".join([f"{list(e.path)}: {e.message}" for e in errs[:5]])
        raise ValueError(f"Output schema invalid: {msg}")
    return obj


def _pick_model(state: InnovationState, team: TeamName) -> str:
    if isinstance(state.models, dict):
        m = state.models.get(str(team), "").strip()
        if m:
            return m
    if isinstance(state.model, str) and state.model.strip():
        return state.model.strip()
    return state.default_model or DEFAULT_MODEL


def _render_system_context(state: InnovationState) -> str:
    """
    Build the system context block from either live API data or payload assets.
    Live API data takes priority when available.
    """
    parts: List[str] = [f"brand={state.brand}"]

    ctx = state.brand_context
    if ctx:
        if ctx.passport:
            parts += ["### brand_passport", json.dumps(ctx.passport, ensure_ascii=False)]
        if ctx.brand_manual:
            parts += ["### brand_manual", json.dumps(ctx.brand_manual, ensure_ascii=False)]
        if ctx.markets:
            parts += ["### markets", json.dumps(ctx.markets, ensure_ascii=False)]
        return "\n\n".join(parts).strip()

    # Fallback: use payload assets
    a = state.assets
    parts += [
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
        "thread_id":    str(cfg.get("thread_id") or ""),
        "run_id":       str(cfg.get("run_id") or ""),
        "assistant_id": str(cfg.get("assistant_id") or ""),
    }


# ── merge functions ───────────────────────────────────────────────────────────

def _merge_interview(out: Dict, state: InnovationState) -> None:
    for i in range(1, 11):
        for key in (f"Q{i}", f"q{i}"):
            v = out.get(key)
            if isinstance(v, str):
                setattr(state, f"q{i}", v); break
        for key in (f"A{i}", f"a{i}"):
            v = out.get(key)
            if isinstance(v, str):
                setattr(state, f"a{i}", v); break


def _merge_jtbd(out: Dict, state: InnovationState) -> None:
    for i in range(1, 11):
        for k in ("job", "pain", "gain"):
            v = out.get(f"{k}{i}")
            if isinstance(v, str):
                setattr(state, f"{k}{i}", v)


def _merge_features(out: Dict, state: InnovationState) -> None:
    for i in range(1, 11):
        v = out.get(f"feature{i}")
        if isinstance(v, str):
            setattr(state, f"feature{i}", v)


def _merge_journey(out: Dict, state: InnovationState) -> None:
    for k in ("title", "journey"):
        v = out.get(k)
        if isinstance(v, str):
            setattr(state, k, v)
    for i in range(1, 11):
        v = out.get(f"step{i}")
        if isinstance(v, str):
            setattr(state, f"step{i}", v)


def _merge_sketches(out: Dict, state: InnovationState) -> None:
    for i in range(1, 11):
        v = out.get(f"visual{i}")
        if isinstance(v, str):
            setattr(state, f"visual{i}", v)


# ── main nodes ────────────────────────────────────────────────────────────────

def load_brand_context_node(state: InnovationState) -> InnovationState:
    """
    If API credentials are present, fetch live brand data from BrandPilot Backend
    and populate state.brand_context. Skips silently if credentials are absent.
    """
    client = state.api_client()
    if client is None:
        return state

    from core.state import BrandContext
    try:
        brand    = client.validate_scope()
        manual   = client.get_brand_manual()
        passport = client.get_passport()
        markets  = client.get_markets(defined_only=True)

        state.brand_context = BrandContext(
            brand_id=state.brand_id,
            account_id=state.account_id,
            brand_name=brand.get("name", state.brand),
            brand_manual=manual,
            passport=passport,
            markets=markets if isinstance(markets, list) else [],
        )
        if not state.brand:
            state.brand = state.brand_context.brand_name
    except Exception as e:
        # Non-fatal: fall back to assets from payload
        state.brand_context = None

    return state


def apply_team_node(team: TeamName):
    """Factory that returns a node function for a specific innovation team."""

    def node(state: InnovationState, config: Optional[RunnableConfig] = None) -> InnovationState:
        prompt_name = f"CLRT0257__innovation__{team}"
        system_prompt, hub_meta = get_prompt(prompt_name, tag=state.prompt_tag)
        model_name = _pick_model(state, team)

        feat_str, feat_list = _normalize_features(state.existing_features)
        payload = {
            "brand":                    state.brand,
            "team":                     team,
            "persona":                  state.persona,
            "extra_input":              state.extra_input,
            "existing_features":        feat_str,
            "existing_features_list":   feat_list,
            "previous":                 state._step_outputs,
            "output_template_version":  state.output_template_version,
        }

        llm = get_llm(model_name, temperature=0, json_mode=True)
        resp = llm.invoke([
            {"role": "system", "content": _render_system_context(state)},
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(payload, ensure_ascii=False)},
        ])
        content = resp.content if hasattr(resp, "content") else str(resp)

        raw = _parse_json(content)

        persona_id = ""
        if isinstance(state.persona, dict):
            persona_id = str(state.persona.get("id", ""))

        if "template_version" not in raw:
            raw = {
                "template_version": state.output_template_version,
                "team":             team,
                "persona_id":       persona_id,
                "output":           raw,
                "meta":             {"prompt_name": prompt_name, "model": model_name, **hub_meta},
            }

        _validate(raw)
        output = raw.get("output", {})
        if not isinstance(output, dict):
            output = {}

        state._step_outputs[str(team)] = output

        if team == "interview":      _merge_interview(output, state)
        elif team == "jtbd":         _merge_jtbd(output, state)
        elif team == "features":     _merge_features(output, state)
        elif team in ("current", "ideal"): _merge_journey(output, state)
        elif team == "sketches":     _merge_sketches(output, state)

        if not state._best_model or model_rank(model_name) > model_rank(state._best_model):
            state._best_model = model_name

        return state

    node.__name__ = f"team_{team}"
    return node


def finalize_node(state: InnovationState, config: Optional[RunnableConfig] = None) -> InnovationState:
    ids = _get_runtime_ids(config)
    state.thread_id = ids["thread_id"]
    state.run_id    = ids["run_id"]
    state.team      = ids.get("assistant_id") or AGENT_ID

    if isinstance(state.persona, dict):
        state.persona_id = str(state.persona.get("id") or state.persona_id or "")
        d = state.persona.get("datum") or state.persona.get("date") or ""
        state.datum = str(d) if d else state.datum

    if not state.started_at:
        state.started_at = _tz_now_iso(BRUSSELS_TZ)
    state.ended_at = _tz_now_iso(BRUSSELS_TZ)

    try:
        t0 = datetime.fromisoformat(state.started_at.replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(state.ended_at.replace("Z", "+00:00"))
        state.duration_s = float((t1 - t0).total_seconds())
    except Exception:
        pass

    state.status = "completed"
    state.model  = state._best_model or state.default_model

    return state
