"""
SYNTHIA Memory Module — Phase 2
Tracks what SYNTHIA has already searched to avoid redundant queries.
In Phase 5, this will be extended to persistent long-term memory via MCP.
"""

from datetime import datetime


class SearchMemory:
    """
    Short-term memory for a single research session.
    Stores all queries made and their results.
    Prevents SYNTHIA from searching the same thing twice.
    """

    def __init__(self):
        self.searches: list[dict] = []
        self.created_at = datetime.now()

    def add(self, query: str, results: list) -> None:
        """Store a completed search in memory."""
        self.searches.append({
            "query": query.lower().strip(),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })

    def already_searched(self, query: str) -> bool:
        """Check if a very similar query has already been run."""
        query_clean = query.lower().strip()
        for s in self.searches:
            # Exact match
            if s["query"] == query_clean:
                return True
            # High similarity check (80%+ word overlap)
            words_new = set(query_clean.split())
            words_old = set(s["query"].split())
            if len(words_new) > 2 and len(words_old) > 2:
                overlap = len(words_new & words_old) / max(len(words_new), len(words_old))
                if overlap > 0.8:
                    return True
        return False

    def get_all_queries(self) -> list[str]:
        """Return all queries made so far."""
        return [s["query"] for s in self.searches]

    def summary(self) -> str:
        """Return a readable summary of search history."""
        if not self.searches:
            return "No searches yet."
        lines = [f"  [{i+1}] {s['query']}" for i, s in enumerate(self.searches)]
        return "\n".join(lines)

    def __len__(self):
        return len(self.searches)
