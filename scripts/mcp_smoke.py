"""Launch the aml-data MCP server as a subprocess over stdio and exercise it like a real client would.

Run from the project root:  python -m scripts.mcp_smoke
"""

import asyncio
import json
import sys

from mcp import Client, StdioServerParameters

from app.config import PROJECT_ROOT


class ToolFailed(Exception):
    pass


def show(title: str, data, limit: int = 1_200) -> None:
    text = json.dumps(data, indent=2, default=str)
    print(f"\n=== {title} ===\n{text[:limit]}{' …' if len(text) > limit else ''}")


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_servers.aml_data"], cwd=str(PROJECT_ROOT))
    async with Client(params, read_timeout_seconds=120) as client:
        tools = (await client.list_tools()).tools
        print("Tools:", ", ".join(t.name for t in tools))
        print("Resources:", ", ".join(str(r.uri) for r in (await client.list_resources()).resources))
        print("Prompts:", ", ".join(p.name for p in (await client.list_prompts()).prompts))

        async def call(name: str, args: dict):
            result = await client.call_tool(name, args)
            if result.is_error:
                raise ToolFailed(f"{name} failed: {result.content[0].text}")
            return result.structured_content

        try:
            alerts = await call("list_alerts", {"status": "open", "limit": 5})
        except ToolFailed as exc:
            print(f"\n{exc}")
            return
        show("list_alerts(open)", alerts, 800)

        by_rule = {a["rule"]: a for a in (await call("list_alerts", {"limit": 100}))["alerts"]}
        show("get_alert", await call("get_alert", {"alert_id": next(iter(by_rule.values()))["alert_id"]}))
        show("scan_for_structuring", await call("scan_for_structuring", {"days": 180}))
        if "CIRCULAR_FLOW" in by_rule:
            show("find_circular_flows", await call("find_circular_flows", {"account_id": by_rule["CIRCULAR_FLOW"]["account_id"]}))
        if "FAN_IN_FAN_OUT" in by_rule:
            show("get_counterparty_summary", await call("get_counterparty_summary", {"account_id": by_rule["FAN_IN_FAN_OUT"]["account_id"]}))
        if "SHARED_IDENTIFIER" in by_rule:
            show("find_shared_identifiers", await call("find_shared_identifiers", {"customer_id": by_rule["SHARED_IDENTIFIER"]["customer_id"]}))
        show("search_policies", await call("search_policies", {"query": "structuring red flags", "top_k": 2}), 600)


if __name__ == "__main__":
    asyncio.run(main())
