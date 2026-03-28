"""
AutoResearch Agent — Entry point
Run this file to start the agent.
"""

from dotenv import load_dotenv
load_dotenv()

from agent.agent import run_agent

if __name__ == "__main__":
    print("=" * 50)
    print("   AutoResearch — Agentic AI Research Assistant")
    print("   Built with Anthropic Claude API + Tool Use")
    print("=" * 50)

    topic = input("\nWhat would you like me to research? → ")
    if not topic.strip():
        print("No topic entered. Exiting.")
    else:
        report_path = run_agent(topic)
        print(f"\n🎉 Done! Your report is at: {report_path}")
