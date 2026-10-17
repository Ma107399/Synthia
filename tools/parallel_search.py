"""Optional anonymous Parallel Search MCP adapter for SYNTHIA's sync tool loop."""

import json
import uuid


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "SYNTHIA-Agent/2.0"
SEARCH_TIMEOUT_SECONDS = 30
# Keep one identifier across searches in this process, including failed calls.
SESSION_ID = uuid.uuid4().hex


def parallel_search(query: str, max_results: int = 5) -> list[dict]:
    """Return the existing title/url/content contract without switching providers."""
    if max_results <= 0:
        return []

    try:
        import anyio
        import httpx  # noqa: F401 -- check optional dependencies before starting
        import mcp  # noqa: F401
    except ImportError:
        return _error("Install Parallel support with: pip install -r requirements-parallel.txt")

    try:
        return anyio.run(_search, query, max_results)
    except TimeoutError:
        return _error("Parallel search timed out. Try again later.")
    except Exception as error:
        # MCP transport failures can be wrapped by an AnyIO task group.
        while getattr(error, "exceptions", None):
            error = error.exceptions[0]
        return _error(f"Parallel search failed: {error}")


def _error(message: str) -> list[dict]:
    return [{"title": "Search error", "url": "", "content": message}]


async def _search(query: str, max_results: int) -> list[dict]:
    import anyio
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    # Bound the entire operation, including handshake, tool call and cleanup.
    with anyio.fail_after(SEARCH_TIMEOUT_SECONDS):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True
        ) as http_client:
            async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    response = await session.call_tool("web_search", {
                        "objective": query,
                        "search_queries": [query],
                        "session_id": SESSION_ID,
                    })

    text = "\n".join(block.text for block in response.content if block.type == "text")
    if response.isError:
        raise RuntimeError(text or "Parallel returned a tool error")
    data = response.structuredContent
    if data is None:
        data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Parallel returned an invalid search response")

    return [{
        "title": result.get("title") or result["url"],
        "url": result["url"],
        "content": "\n".join(result["excerpts"]),
    } for result in data["results"][:max_results]]
