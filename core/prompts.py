from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_template(obj: Any) -> str:
    """Pull the raw template string out of whatever pull_prompt returns."""
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "messages") and getattr(obj, "messages"):
        m0 = obj.messages[0]
        if hasattr(m0, "prompt") and hasattr(m0.prompt, "template"):
            return str(m0.prompt.template)
        return str(m0)
    if hasattr(obj, "template"):
        return str(obj.template)
    return str(obj)


def _extract_model_config(chain: Any) -> Tuple[Optional[str], Optional[float]]:
    """
    When pull_prompt(include_model=True) returns a RunnableSequence,
    the last step is the bound model. Pull out model name + temperature.
    Returns (model_name_or_None, temperature_or_None).
    """
    model_obj = None

    if hasattr(chain, "steps") and chain.steps:
        model_obj = chain.steps[-1]
    elif hasattr(chain, "last"):
        model_obj = chain.last

    if model_obj is None:
        return None, None

    # LangChain model objects expose model name via different attributes
    model_name = (
        getattr(model_obj, "model_name", None)
        or getattr(model_obj, "model", None)
    )
    temperature = getattr(model_obj, "temperature", None)

    return (str(model_name) if model_name else None), temperature


# ── Public API ────────────────────────────────────────────────────────────────

def get_prompt(prompt_name: str, tag: str = "active") -> Tuple[str, Dict[str, Any]]:
    """
    Pull a prompt from LangSmith Hub by name+tag.

    Uses include_model=True so that any model configured in the LangSmith UI
    (model name, temperature, etc.) is automatically captured and returned in meta.
    This means you can change the model for any team entirely from the Hub — no
    code change or redeploy required.

    Priority for model selection (consumed by nodes/_pick_model):
        1. meta["model"]       — model attached to this prompt in LangSmith Hub
        2. state.model         — caller override in the run payload
        3. DEFAULT_MODEL env   — fallback

    meta dict keys:
        prompt_name       — the name requested
        prompt_tag        — the tag used
        hub_type          — class name of pulled object
        hub_error         — error repr if pull failed
        hub_model_attached — True when a model was found on the chain
        model             — model ID from Hub (e.g. "claude-sonnet-4-6"), if present
        temperature       — temperature from Hub, if present

    Falls back to a safe JSON instruction if Hub is unreachable or prompt missing.
    Falls back to include_model=False if model instantiation fails (missing API key).

    Naming convention: {PROJECT_CODE}__{agent}__{team}
    e.g.  CAND0000__prospect__score
          CLRT0257__innovation__interview
    """
    from langsmith import Client

    meta: Dict[str, Any] = {"prompt_name": prompt_name, "prompt_tag": tag}

    client = Client()

    # ── Build candidate pull identifiers (priority order) ────────────────────
    # 1. {name}:{tag}  — user-promoted version (e.g. "active")
    # 2. {name}:latest — most recent push (automatic fallback, no UI action needed)
    # 3. {name}        — bare name as last resort before code defaults
    candidates = []
    if tag:
        candidates.append(f"{prompt_name}:{tag}")
    candidates.append(f"{prompt_name}:latest")
    candidates.append(prompt_name)

    # ── Attempt each identifier: first try with model, then without ───────────
    chain = None
    for identifier in candidates:
        try:
            chain = client.pull_prompt(identifier, include_model=True, secrets_from_env=True)
            meta["prompt_tag"] = identifier.split(":", 1)[-1] if ":" in identifier else tag
            break
        except Exception:
            pass
        try:
            chain = client.pull_prompt(identifier)
            meta["prompt_tag"] = identifier.split(":", 1)[-1] if ":" in identifier else tag
            break
        except Exception:
            pass

    if chain is None:
        meta["hub_error"] = f"prompt not found: tried {candidates}"
        return (
            "Return ONLY valid JSON with no prose. "
            "Use the exact output keys specified in the task.",
            meta,
        )

    # ── Extract template and model config ─────────────────────────────────────
    meta["hub_type"] = type(chain).__name__

    # If it's a RunnableSequence (prompt | model), steps[0] is the prompt
    if hasattr(chain, "steps") and len(chain.steps) >= 2:
        template = _extract_template(chain.steps[0])
        model_name, temperature = _extract_model_config(chain)
        if model_name:
            meta["model"] = model_name
            meta["hub_model_attached"] = True
        if temperature is not None:
            meta["temperature"] = temperature
    else:
        # Just a prompt template — no model attached in Hub
        template = _extract_template(chain)
        meta["hub_model_attached"] = False

    return template, meta
