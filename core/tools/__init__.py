from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

import requests


# ── URL helpers ───────────────────────────────────────────────────────────────

def domain_from_url(url: str) -> str:
    """Extract bare domain from any URL, stripping www."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return url


def homepage_url(url: str) -> str:
    """Derive the root homepage URL from any page URL."""
    try:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"
    except Exception:
        return url


# ── Tavily search ─────────────────────────────────────────────────────────────

def tavily_search(query: str, max_results: int = 8) -> list[dict[str, Any]]:
    """
    Execute a web search via Tavily and return structured results.
    Returns [] on any failure — callers must handle empty results gracefully.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    try:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
        )
        return [
            {
                "title":          r.get("title", ""),
                "url":            r.get("url", ""),
                "content":        r.get("content", ""),
                "score":          r.get("score", 0.0),
                "published_date": r.get("published_date", ""),
            }
            for r in response.get("results", [])
        ]
    except Exception:
        return []


# ── Tavily extract ────────────────────────────────────────────────────────────

def tavily_extract(urls: list[str], chars_per_page: int = 4000) -> dict[str, str]:
    """
    Extract full page text from a list of URLs via Tavily.
    Returns {url: content} — URLs that failed to extract are simply absent.
    Batches internally in groups of 20 to respect API limits.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key or not urls:
        return {}
    out: dict[str, str] = {}
    try:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=api_key)
        batch_size = 20
        for i in range(0, len(urls), batch_size):
            batch = urls[i : i + batch_size]
            try:
                response = client.extract(urls=batch)
                for r in response.get("results", []):
                    url = r.get("url", "")
                    content = r.get("raw_content", "")
                    if url and content:
                        out[url] = content[:chars_per_page]
            except Exception:
                continue
    except Exception:
        pass
    return out


# ── Fallback page fetch ───────────────────────────────────────────────────────

def fetch_page_text(url: str, timeout: int = 12, max_chars: int = 4000) -> str:
    """
    Fallback HTML fetch when Tavily extract fails.
    Strips HTML tags and returns plain text, capped at max_chars.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; BrandPilotBot/1.0)",
            "Accept":     "text/html,application/xhtml+xml,text/plain",
        }
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if not r.ok:
            return ""
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def get_homepage_content(url: str, chars_limit: int = 4000) -> str:
    """
    Fetch homepage content: Tavily extract first, direct fetch as fallback.
    """
    extracted = tavily_extract([url], chars_per_page=chars_limit)
    if extracted.get(url):
        return extracted[url]
    return fetch_page_text(url, max_chars=chars_limit)
