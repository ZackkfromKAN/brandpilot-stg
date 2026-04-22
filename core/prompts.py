from __future__ import annotations

from typing import Any, Dict, Tuple


def _extract_template(pulled: Any) -> str:
    if isinstance(pulled, str):
        return pulled
    if hasattr(pulled, "messages") and getattr(pulled, "messages"):
        m0 = pulled.messages[0]
        if hasattr(m0, "prompt") and hasattr(m0.prompt, "template"):
            return str(m0.prompt.template)
        return str(m0)
    if hasattr(pulled, "template"):
        return str(pulled.template)
    return str(pulled)


def get_prompt(prompt_name: str, tag: str = "active") -> Tuple[str, Dict[str, Any]]:
    """
    Pull a prompt from LangSmith Hub by name.
    Naming convention: {PROJECT_CODE}__{agent}__{team}
    e.g. CLRT0257__innovation__interview
         generic__prospect__score

    Falls back to a safe JSON instruction if the hub call fails.
    """
    from langsmith import Client

    meta: Dict[str, Any] = {"prompt_name": prompt_name, "prompt_tag": tag}
    full_name = f"{prompt_name}:{tag}" if tag else prompt_name

    try:
        client = Client()
        pulled = client.pull_prompt(full_name)
        meta["hub_type"] = pulled.__class__.__name__
        return _extract_template(pulled), meta
    except Exception as e:
        meta["hub_error"] = repr(e)
        return (
            "Return ONLY valid JSON with no prose. Use the exact output keys specified in the task.",
            meta,
        )
