"""
AutoResearch Agent — Core agentic loop
Uses Anthropic tool use to autonomously plan, search, and synthesize research reports.
"""

import anthropic
import json
from datetime import datetime
from tools.search import web_search
from tools.file_writer import save_report

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env automatically

SYSTEM_PROMPT = """You are AutoResearch, an autonomous AI research agent.

Your job:
1. Break down any research topic into focused sub-questions
2. Search the web for each sub-question using the web_search tool
3. Synthesize findings into a clear, structured report
4. Save the final report using the save_report tool

Be thorough. Search multiple times if needed. Always cite sources.
When you have enough information, produce a complete report and save it.
"""

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information on a topic. Returns top results with titles, URLs, and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_report",
        "description": "Save the final research report as a markdown file in the outputs folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the research report"
                },
                "content": {
                    "type": "string",
                    "description": "The full markdown content of the report"
                }
            },
            "required": ["title", "content"]
        }
    }
]


def run_agent(topic: str, max_iterations: int = 10) -> str:
    """
    Main agentic loop.
    The agent thinks, calls tools, observes results, and loops until done.
    """
    print(f"\n🤖 AutoResearch Agent starting...")
    print(f"📋 Topic: {topic}\n")

    messages = [
        {"role": "user", "content": f"Research this topic thoroughly and produce a detailed report: {topic}"}
    ]

    iteration = 0
    report_path = None

    while iteration < max_iterations:
        iteration += 1
        print(f"--- Step {iteration} ---")

        # Call Claude
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        print(f"Stop reason: {response.stop_reason}")

        # If Claude is done (no more tool calls), we're finished
        if response.stop_reason == "end_turn":
            print("\n✅ Agent finished.")
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                print(f"🔧 Tool call: {tool_name}({json.dumps(tool_input)[:80]}...)")

                # Execute the tool
                if tool_name == "web_search":
                    result = web_search(tool_input["query"])
                elif tool_name == "save_report":
                    result, report_path = save_report(tool_input["title"], tool_input["content"])
                else:
                    result = f"Unknown tool: {tool_name}"

                print(f"   ↳ Result: {str(result)[:100]}...")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

        # Add assistant response and tool results to message history
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return report_path or "No report saved."


if __name__ == "__main__":
    topic = input("Enter a research topic: ")
    result = run_agent(topic)
    print(f"\n📄 Report saved to: {result}")
