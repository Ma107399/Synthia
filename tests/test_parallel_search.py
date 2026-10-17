"""Exercise provider selection and the real MCP SDK with an HTTP fixture."""

import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anyio
import httpx

from tools import parallel_search as parallel
from tools.search import web_search


SEARCH_DATA = {"search_id": "fixture-search", "session_id": "fixture-session", "results": [
    {"title": "Python release", "url": "https://www.python.org/downloads/",
     "excerpts": ["Release highlights", "Supported versions"]},
    {"title": None, "url": "https://docs.python.org/3/", "excerpts": ["Documentation"]},
]}
EXPECTED = [
    {"title": "Python release", "url": "https://www.python.org/downloads/",
     "content": "Release highlights\nSupported versions"},
    {"title": "https://docs.python.org/3/", "url": "https://docs.python.org/3/",
     "content": "Documentation"},
]


@contextlib.contextmanager
def working_directory(path):
    previous = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)


class ParallelSearchTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {
            "SYNTHIA_SEARCH_PROVIDER": "parallel",
            "TAVILY_API_KEY": "saved-tavily-key",
            "PARALLEL_API_KEY": "must-not-be-sent",
            "ANTHROPIC_API_KEY": "fixture-only",
        }, clear=True))
        self.requests = []
        self.reply = {"content": [{"type": "text", "text": json.dumps(SEARCH_DATA)}],
                      "structuredContent": SEARCH_DATA, "isError": False}
        self.failure = None
        self.hang = False
        self.cancelled = False
        real_client = httpx.AsyncClient

        def client(**kwargs):
            return real_client(**kwargs, transport=httpx.MockTransport(self.handle))

        self.stack.enter_context(patch("httpx.AsyncClient", side_effect=client))
        self.fallback = self.stack.enter_context(patch("tools.search._duckduckgo_search"))
        self.tavily = self.stack.enter_context(patch("tools.search._tavily_search"))

    async def handle(self, request):
        self.requests.append(request)
        self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
        self.assertEqual(request.headers["user-agent"], "SYNTHIA-Agent/2.0")
        self.assertNotIn("authorization", request.headers)
        self.assertNotIn("x-api-key", request.headers)
        packet = json.loads(request.content)
        method = packet["method"]
        if method.startswith("notifications/"):
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": packet["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "Parallel fixture", "version": "1.0"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            if self.failure == "http":
                return httpx.Response(503, text="Unavailable")
            if self.failure == "read_timeout":
                raise httpx.ReadTimeout("fixture timed out", request=request)
            if self.hang:
                try:
                    await anyio.sleep_forever()
                finally:
                    self.cancelled = True
            result = self.reply
        else:
            self.fail(f"Unexpected MCP method: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": packet["id"], "result": result})

    def tool_calls(self):
        return [json.loads(r.content)["params"] for r in self.requests
                if json.loads(r.content)["method"] == "tools/call"]

    def test_selected_provider_normalizes_sources_and_preserves_session(self):
        self.assertEqual(web_search("Python release highlights", 2), EXPECTED)
        self.assertEqual(web_search("Python support policy", 1), EXPECTED[:1])
        calls = self.tool_calls()
        self.assertEqual(calls[0]["name"], "web_search")
        self.assertEqual(calls[0]["arguments"], {
            "objective": "Python release highlights",
            "search_queries": ["Python release highlights"],
            "session_id": parallel.SESSION_ID,
        })
        self.assertEqual(calls[0]["arguments"]["session_id"], calls[1]["arguments"]["session_id"])
        self.fallback.assert_not_called()
        self.tavily.assert_not_called()

    def test_json_text_response_and_empty_results(self):
        del self.reply["structuredContent"]
        self.assertEqual(web_search("Python highlights"), EXPECTED)
        self.reply["content"][0]["text"] = json.dumps({"results": []})
        self.assertEqual(web_search("No matching sources"), [])

    def test_no_results_requested_does_not_send_request(self):
        self.assertEqual(web_search("Python", 0), [])
        self.assertEqual(web_search("Python", -1), [])
        self.assertEqual(self.requests, [])

    def test_tool_error_is_visible_without_fallback(self):
        self.reply = {"content": [{"type": "text", "text": "Free-tier quota reached"}], "isError": True}
        result = web_search("Python release")
        self.assertEqual(result[0]["title"], "Search error")
        self.assertIn("Free-tier quota reached", result[0]["content"])
        self.fallback.assert_not_called()
        self.tavily.assert_not_called()

    def test_invalid_response_is_an_error(self):
        for invalid in ({"unexpected": []}, {"results": [{"url": "missing excerpts"}]}):
            with self.subTest(invalid=invalid):
                self.reply["structuredContent"] = invalid
                self.assertEqual(web_search("Python")[0]["title"], "Search error")
        self.reply.pop("structuredContent")
        self.reply["content"][0]["text"] = "not JSON"
        self.assertEqual(web_search("Python")[0]["title"], "Search error")

    def test_http_and_read_timeout_failures_do_not_switch_providers(self):
        for failure in ("http", "read_timeout"):
            with self.subTest(failure=failure):
                self.failure = failure
                self.assertEqual(web_search("Python")[0]["title"], "Search error")
        self.fallback.assert_not_called()
        self.tavily.assert_not_called()

    def test_deadline_cancels_inflight_http_request(self):
        self.hang = True
        with patch.object(parallel, "SEARCH_TIMEOUT_SECONDS", 0.05):
            result = web_search("Python")
        self.assertIn("timed out", result[0]["content"])
        self.assertTrue(self.cancelled)
        self.fallback.assert_not_called()

    def test_external_cancellation_propagates(self):
        self.hang = True

        async def run():
            with anyio.move_on_after(0.05) as scope:
                await parallel._search("Python", 5)
                self.fail("Cancellation must not become an ordinary search result")
            self.assertTrue(scope.cancelled_caught)

        anyio.run(run)
        self.assertTrue(self.cancelled)

    def test_missing_optional_dependencies_explains_setup(self):
        original = builtins.__import__

        def imports(name, *args, **kwargs):
            if name == "mcp":
                raise ImportError("not installed")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=imports):
            self.assertIn("requirements-parallel.txt", web_search("Python")[0]["content"])
        self.assertEqual(self.requests, [])

    def test_existing_routing_without_optional_dependencies(self):
        os.environ.pop("SYNTHIA_SEARCH_PROVIDER")
        self.tavily.return_value = [{"title": "Tavily", "url": "", "content": "existing"}]
        self.assertEqual(web_search("Python"), self.tavily.return_value)
        self.tavily.assert_called_once_with("Python", 5, "saved-tavily-key")
        os.environ.pop("TAVILY_API_KEY")
        self.fallback.return_value = [{"title": "DuckDuckGo", "url": "", "content": "existing"}]
        self.assertEqual(web_search("Python"), self.fallback.return_value)
        self.fallback.assert_called_once_with("Python", 5)
        self.assertEqual(self.requests, [])

    def test_agent_dispatch_consumes_sources_and_saves_report(self):
        from agent import agent

        def tool(name, inputs, identifier):
            return SimpleNamespace(type="tool_use", name=name, input=inputs, id=identifier)

        def message(**kwargs):
            results = [m for m in kwargs["messages"] if isinstance(m["content"], list)
                       and m["role"] == "user"]
            if not results:
                blocks = [tool("web_search", {"query": "Python release highlights"}, "search-1")]
            elif len(results) == 1:
                sources = json.loads(results[0]["content"][0]["content"])
                self.assertEqual(sources, EXPECTED)
                content = "# Python release\n" + sources[0]["content"] + "\n" + sources[0]["url"]
                blocks = [tool("save_report", {"title": "Python release", "content": content}, "save-1")]
            else:
                return SimpleNamespace(stop_reason="end_turn", content=[])
            return SimpleNamespace(stop_reason="tool_use", content=blocks)

        with tempfile.TemporaryDirectory() as directory, working_directory(directory):
            with patch.object(agent.client.messages, "create", side_effect=message):
                with contextlib.redirect_stdout(io.StringIO()):
                    report = Path(agent.run_agent("Python release", max_iterations=3))
            self.assertTrue(report.is_file())
            self.assertIn("https://www.python.org/downloads/", report.read_text())
            self.assertIn("Release highlights", report.read_text())
        self.assertEqual(len(self.tool_calls()), 1)


if __name__ == "__main__":
    unittest.main()
