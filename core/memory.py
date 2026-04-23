from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BrandMemoryStore:
    """
    Per-brand long-term memory backed by the BrandPilot brand_manual endpoint.

    Conceptually identical to LangChain's InMemoryStore but persistent:
    memory lives under the ``_agent_memory`` key of the brand_manual JSON,
    which is loaded once at run start and written back at run end.

    Usage pattern in agent nodes:
        # load
        mem = BrandMemoryStore()
        mem.load_from_manual(raw_brand_manual)
        state.brand_memory = mem.to_dict()

        # use
        known = BrandMemoryStore.from_dict(state.brand_memory)
        known_domains = known.get_known_domains()

        # save
        mem = BrandMemoryStore.from_dict(state.brand_memory)
        updated_manual = mem.merge_into_manual(raw_brand_manual)
        client.update_brand_manual(updated_manual)
    """

    _NAMESPACE = "_agent_memory"

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BrandMemoryStore":
        obj = cls()
        obj._data = dict(d) if isinstance(d, dict) else {}
        return obj

    def load_from_manual(self, manual: Any) -> None:
        """Hydrate from a brand_manual API response (handles nested wrappers)."""
        if isinstance(manual, dict):
            raw = manual.get("manual", manual)
            mem = raw.get(self._NAMESPACE, {}) if isinstance(raw, dict) else {}
        else:
            mem = {}
        self._data = mem if isinstance(mem, dict) else {}

    def merge_into_manual(self, manual: Any) -> Dict[str, Any]:
        """Return updated brand_manual with agent memory merged in."""
        if isinstance(manual, dict):
            inner = manual.get("manual", manual)
            if "manual" in manual:
                updated_inner = {**inner, self._NAMESPACE: self._data}
                return {**manual, "manual": updated_inner}
            return {**manual, self._NAMESPACE: self._data}
        return {self._NAMESPACE: self._data}

    # ── reads ─────────────────────────────────────────────────────────────────

    def get_known_prospects(self) -> Dict[str, Dict[str, Any]]:
        """Return {domain: record} for all previously seen prospects."""
        return self._data.get("prospects", {})

    def get_known_domains(self) -> set:
        return set(self.get_known_prospects().keys())

    def get_do_not_target(self) -> set:
        return set(self._data.get("do_not_target", []))

    def get_past_queries(self, last_n_runs: int = 3) -> List[str]:
        """Search queries used in the last N runs — avoids repetition."""
        runs = self._data.get("past_runs", [])
        queries: List[str] = []
        for run in runs[-last_n_runs:]:
            queries.extend(run.get("queries", []))
        return list(dict.fromkeys(queries))

    def get_shortlisted_names(self) -> List[str]:
        return [
            v.get("name", k)
            for k, v in self.get_known_prospects().items()
            if v.get("status") == "shortlisted"
        ]

    def summary_for_prompt(self) -> Dict[str, Any]:
        """Compact summary injected into the search_plan prompt to guide the LLM."""
        known = self.get_known_prospects()
        return {
            "previously_shortlisted": self.get_shortlisted_names()[:20],
            "total_prospects_in_memory": len(known),
            "past_queries_used": self.get_past_queries()[:15],
            "note": (
                "Avoid generating queries that would simply rediscover the same companies. "
                "Explore new segments, geographies, or channels not yet covered."
            ) if known else "",
        }

    # ── writes ────────────────────────────────────────────────────────────────

    def record_prospects(self, prospects: List[Dict[str, Any]]) -> None:
        """Upsert shortlisted prospects. Existing records are enriched, not overwritten."""
        if "prospects" not in self._data:
            self._data["prospects"] = {}
        today = _now_iso()[:10]
        for p in prospects:
            domain = p.get("domain", "")
            if not domain:
                continue
            existing = self._data["prospects"].get(domain, {})
            self._data["prospects"][domain] = {
                "name":             p.get("name") or existing.get("name", ""),
                "status":           "shortlisted",
                "score":            p.get("score", existing.get("score", 0)),
                "url":              p.get("url") or existing.get("url", ""),
                "first_seen":       existing.get("first_seen", today),
                "last_seen":        today,
                "times_shortlisted": existing.get("times_shortlisted", 0) + 1,
            }

    def mark_do_not_target(self, domains: List[str]) -> None:
        lst = self._data.setdefault("do_not_target", [])
        for d in domains:
            if d and d not in lst:
                lst.append(d)

    def record_run(
        self,
        *,
        queries: List[str],
        candidates_found: int,
        shortlisted_count: int,
        geography: str = "",
    ) -> None:
        runs = self._data.setdefault("past_runs", [])
        runs.append({
            "date":             _now_iso()[:10],
            "geography":        geography,
            "queries":          queries,
            "candidates_found": candidates_found,
            "shortlisted":      shortlisted_count,
        })
        self._data["past_runs"] = runs[-20:]
        self._data["updated_at"] = _now_iso()
        self._data.setdefault("v", 1)
