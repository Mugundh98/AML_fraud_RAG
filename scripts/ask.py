"""Ask the agent a question from the terminal and see every tool call it makes.

Run from the project root:  python -m scripts.ask "Which open alerts are High priority?"
"""

import asyncio
import json
import sys

from app.agent.loop import Agent
from app.agent.mcp_hub import MCPHub, default_servers
from app.config import PROJECT_ROOT, get_settings
from app.llm import build_router


async def main(question: str) -> None:
    async with MCPHub(default_servers(PROJECT_ROOT)) as hub:
        for name, status in hub.status.items():
            print(f"MCP {name}: {status.error or f'{len(status.tools)} tools'}")
        result = await Agent(hub, build_router(get_settings())).answer(question, [])

    print()
    for step in result.tool_calls:
        print(f"[round {step.round}] {'ok ' if step.ok else 'ERR'} {step.tool}({json.dumps(step.arguments)}) "
              f"{step.duration_ms} ms")
    print(f"\n{result.answer}\n")
    for s in result.sources:
        print(f"[{s.number}] {s.title} > {s.section}  ({s.source})")
    print(f"\nAnswered by {result.provider}/{result.model}")
    for reason in result.fallback_reasons:
        print(f"Skipped: {reason[:150]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m scripts.ask "your question"')
    asyncio.run(main(" ".join(sys.argv[1:])))
