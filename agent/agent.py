"""
SYNTHIA — Core Agentic Loop (Phase 2)
Upgraded with:
- Smarter multi-step reasoning
- Search memory (avoids duplicate searches)
- Richer tool definitions
- Structured report generation
"""

import anthropic
import json
from memory.search_memory import SearchMemory
from tools.search import web_search
from tools.file_writer import save_report

client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are SYNTHIA — an autonomous AI research agent built on the Anthropic Claude API.

## Your Mission
Given a research topic, you will autonomously produce a thorough, structured research report.

## Your Process (follow this strictly)
1. PLAN — Break the topic into 3-5 focused sub-questions worth investigating
2. SEARCH — Use web_search for each sub-question. Be specific with queries.
3. REFLECT — After each search, ask: "Do I have enough? What's still missing?"
4. SEARCH MORE — If gaps exist, search again with a refined query
5. SYNTHESIZE — Once you have enough, write a full structured report
6. SAVE — Use save_report to save your final report

## Report Format (always follow this structure)
# [Topic Title]
## Executive Summary
## Key Findings
## Deep Dive (multiple sections based on sub-questions)
## Conclusion
## Sources

## Rules
- Never repeat the same search query twice
- Always cite sources with URLs
- Minimum 3 searches before writing the report
- Be thorough — a good report has depth, not just surface facts
"""

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current, real-world information. Use specific, targeted queries for best results. Returns titles, URLs, and content snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific, targeted search query. Be precise — 'quantum computing IBM 2025 breakthroughs' is better than 'quantum computing'."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_report",
        "description": "Save the completed research report as a markdown file. Only call this when your report is fully written and complete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The full title of the research report"
                },
                "content": {
                    "type": "string",
                    "description": "The complete markdown content of the report, following the required structure"
                }
            },
            "required": ["title", "content"]
        }
    }
]


def run_agent(topic: str, max_iterations: int = 15) -> str:
    """
    SYNTHIA's core agentic loop.
    Think → Act → Observe → Repeat until report is complete.
    """
    print(f"\n🧠 SYNTHIA initializing...")
    print(f"📋 Research topic: {topic}\n")

    memory = SearchMemory()
    messages = [
        {
            "role": "user",
            "content": f"Research this topic thoroughly and produce a detailed report:\n\n**{topic}**\n\nRemember to plan first, then search multiple times, then synthesize."
        }
    ]

    iteration = 0
    report_path = None

    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*40}")
        print(f"  🔄 Iteration {iteration}")
        print(f"{'='*40}")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        print(f"  Stop reason: {response.stop_reason}")

        # Extract any text thinking from Claude
        for block in response.content:
            if hasattr(block, "text") and block.text:
                preview = block.text[:200].replace("\n", " ")
                print(f"  💭 Claude: {preview}...")

        # Done — no more tool calls
        if response.stop_reason == "end_turn":
            print("\n✅ SYNTHIA has finished the research.")
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                print(f"\n  🔧 Tool: {tool_name}")
                print(f"  📥 Input: {json.dumps(tool_input)[:120]}")

                if tool_name == "web_search":
                    query = tool_input["query"]

                    # Check memory — skip duplicate searches
                    if memory.already_searched(query):
                        result = f"[SKIPPED] Already searched for '{query}'. Try a different query."
                        print(f"  ⚠️  Duplicate search skipped.")
                    else:
                        result = web_search(query)
                        memory.add(query, result)
                        print(f"  ✅ Got {len(result)} results")

                elif tool_name == "save_report":
                    result, report_path = save_report(
                        tool_input["title"],
                        tool_input["content"]
                    )
                    print(f"  💾 Report saved: {report_path}")

                else:
                    result = f"Unknown tool: {tool_name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result) if isinstance(result, list) else str(result)
                })

        # Append to message history
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    if not report_path:
        print("\n⚠️  Agent finished without saving a report.")
        return "No report saved."

    return report_path
