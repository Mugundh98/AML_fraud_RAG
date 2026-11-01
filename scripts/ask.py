"""Ask the agent a question from the terminal and see every tool call it makes.

Run from the project root:  python -m scripts.ask "Which open alerts are High priority?"

If the agent proposes an action (a Slack post, a calendar event) you're asked to approve or reject it here.
"""

import asyncio
import json
import sys

from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub, default_servers
from app.config import PROJECT_ROOT, get_settings
from app.llm import build_router


async def main(question: str) -> None:
    settings = get_settings()
    async with MCPHub(default_servers(settings, PROJECT_ROOT)) as hub:
        for name, status in hub.status.items():
            approval = f", {len(status.approval_tools)} need approval" if status.approval_tools else ""
            print(f"MCP {name}: {status.error or f'{len(status.tools)} tools{approval}'}")
        agent = Agent(hub, build_router(settings))
        result = await agent.answer(question, [])

        print()
        for step in result.tool_calls:
            state = "WAIT" if step.needs_approval else "ok  " if step.ok else "ERR "
            print(f"[round {step.round}] {state} {step.tool}({json.dumps(step.arguments)}) {step.duration_ms} ms")
        print(f"\n{result.answer}\n")
        for s in result.sources:
            print(f"[{s.number}] {s.title} > {s.section}  ({s.source})")
        print(f"\nAnswered by {result.provider}/{result.model}")
        for reason in result.fallback_reasons:
            print(f"Skipped: {reason[:150]}")

        for action in result.pending_actions:
            print(f"\nProposed action: {action.tool} on {action.server}")
            for key, value in action.arguments.items():
                print(f"  {key}: {value}")
            approve = input("Approve? [y/N] ").strip().lower() == "y"
            outcome = await agent.execute(action.id, approve)
            print(f"-> {outcome.status} {outcome.result[:300]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m scripts.ask "your question"')
    asyncio.run(main(" ".join(sys.argv[1:])))
