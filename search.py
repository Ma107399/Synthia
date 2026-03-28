"""
Web search tool — wraps DuckDuckGo search (no API key needed for basic use).
In Phase 5, this becomes an MCP server tool.
"""

import urllib.request
import urllib.parse
import json
import re


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for a query and return top results.
    Uses DuckDuckGo Instant Answer API (free, no key needed).
    
    Returns:
        List of dicts with 'title', 'url', 'snippet'
    """
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        
        req = urllib.request.Request(url, headers={"User-Agent": "AutoResearch/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        results = []

        # Abstract (top result)
        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Result"),
                "url": data.get("AbstractURL", ""),
                "snippet": data.get("Abstract", "")
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:60],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")
                })

        if not results:
            # Fallback: return the raw query as a note
            return [{"title": "No results", "url": "", "snippet": f"No instant results found for: {query}. Try a more specific query."}]

        return results[:max_results]

    except Exception as e:
        return [{"title": "Search error", "url": "", "snippet": f"Search failed: {str(e)}"}]


if __name__ == "__main__":
    # Quick test
    results = web_search("agentic AI trends 2025")
    for r in results:
        print(f"\n📰 {r['title']}")
        print(f"   {r['url']}")
        print(f"   {r['snippet'][:120]}...")
