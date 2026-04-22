from __future__ import annotations

import os
from typing import Any


def get_llm(model: str, temperature: float = 0, json_mode: bool = True) -> Any:
    """
    Provider-agnostic LLM factory.
    Detects provider from model name prefix and returns the right LangChain object.
    """
    model = (model or "").strip()

    if model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    # Default: OpenAI
    from langchain_openai import ChatOpenAI
    kwargs: dict = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def model_rank(name: str) -> int:
    """Higher = more capable. Used to track the best model used in a run."""
    ranks = {
        "claude-opus-4-7":    500,
        "claude-sonnet-4-6":  450,
        "claude-haiku-4-5":   400,
        "gpt-5.2":            300,
        "gpt-5":              290,
        "gpt-4.1":            220,
        "gpt-4o":             200,
        "gpt-4o-mini":        120,
    }
    n = (name or "").strip()
    if n in ranks:
        return ranks[n]
    if n.startswith("claude-opus"):   return 480
    if n.startswith("claude-sonnet"): return 440
    if n.startswith("claude-haiku"):  return 390
    if n.startswith("gpt-5"):         return 280
    if n.startswith("gpt-4o"):        return 180
    if n.startswith("gpt-4"):         return 160
    return 0


DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4.1")
