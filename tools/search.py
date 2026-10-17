"""
SYNTHIA Search Tool — Phase 2
Primary: Tavily API (built for AI agents, free tier available)
Fallback: DuckDuckGo (no key needed)
Opt-in: Parallel Search MCP (SYNTHIA_SEARCH_PROVIDER=parallel)

Get a free Tavily API key at: https://tavily.com
Add to .env: TAVILY_API_KEY=tvly-xxxxxxxx
"""

import os
import json
import urllib.request
import urllib.parse


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web. Opt into Parallel with SYNTHIA_SEARCH_PROVIDER=parallel.
    Otherwise use Tavily if a key is available, falling back to DuckDuckGo.

    Args:
        query: Search query string
        max_results: Number of results to return

    Returns:
        List of dicts with 'title', 'url', 'content'
    """
    if os.getenv("SYNTHIA_SEARCH_PROVIDER", "").strip().lower() == "parallel":
        from tools.parallel_search import parallel_search
        return parallel_search(query, max_results)

    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        return _tavily_search(query, max_results, tavily_key)
    else:
        print("  ℹ️  No TAVILY_API_KEY found — using DuckDuckGo fallback")
        return _duckduckgo_search(query, max_results)


def _tavily_search(query: str, max_results: int, api_key: str) -> list[dict]:
    """
    Tavily search — returns rich, AI-optimized results with full content.
    Best option for an AI agent.
    """
    try:
        url = "https://api.tavily.com/search"
        payload = json.dumps({
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())

        results = []

        # Include Tavily's AI-generated answer if available
        if data.get("answer"):
            results.append({
                "title": "AI Summary",
                "url": "",
                "content": data["answer"]
            })

        # Include individual search results
        for r in data.get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500]
            })

        return results

    except Exception as e:
        print(f"  ⚠️  Tavily search failed: {e}. Falling back to DuckDuckGo.")
        return _duckduckgo_search(query, max_results)


def _duckduckgo_search(query: str, max_results: int) -> list[dict]:
    """
    DuckDuckGo fallback — free, no key needed.
    Less rich than Tavily but always available.
    """
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

        req = urllib.request.Request(url, headers={"User-Agent": "SYNTHIA-Agent/2.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        results = []

        if data.get("Abstract"):
            results.append({
                "title": data.get("Heading", "Result"),
                "url": data.get("AbstractURL", ""),
                "content": data.get("Abstract", "")
            })

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "content": topic.get("Text", "")
                })

        if not results:
            return [{
                "title": "No results found",
                "url": "",
                "content": f"No results for '{query}'. Try rephrasing the query."
            }]

        return results[:max_results]

    except Exception as e:
        return [{"title": "Search error", "url": "", "content": f"Search failed: {str(e)}"}]


if __name__ == "__main__":
    print("Testing SYNTHIA search tool...\n")
    results = web_search("agentic AI Anthropic 2025")
    for r in results:
        print(f"📰 {r['title']}")
        print(f"   🔗 {r['url']}")
        print(f"   📄 {r['content'][:150]}\n")
