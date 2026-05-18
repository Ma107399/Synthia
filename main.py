"""
SYNTHIA — Autonomous AI Research Agent
Phase 2: Multi-step tool use + Search Memory

Usage:
    python main.py                      # interactive mode
    python main.py "quantum computing"  # direct topic mode
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from agent.agent import run_agent


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║   ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗██╗█████╗  ║
║   ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██║██╔══██╗ ║
║   ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║██║███████║ ║
║   ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██║██╔══██║ ║
║   ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║██║██║  ██║ ║
║   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ║
║                                                      ║
║        Autonomous AI Research Agent — Phase 2        ║
║        Powered by Anthropic Claude API               ║
╚══════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    print_banner()

    # Accept topic as CLI arg or prompt interactively
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        print(f"📋 Topic from CLI: {topic}")
    else:
        topic = input("  What should SYNTHIA research? → ").strip()

    if not topic:
        print("  ⚠️  No topic entered. Exiting.")
        sys.exit(0)

    report_path = run_agent(topic)

    print(f"\n{'='*54}")
    print(f"  ✅ Research complete!")
    print(f"  📄 Report saved: {report_path}")
    print(f"{'='*54}\n")
