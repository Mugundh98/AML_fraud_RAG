"""MCP client side: connects to one or more MCP servers and exposes their tools to the agent as one list.

Each server is started (stdio) or attached (in-process, for tests) when the hub opens, and stays connected
until it closes, so a chat request doesn't pay for launching a subprocess.
"""

import json
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from app.llm import ToolSpec

log = logging.getLogger(__name__)


@dataclass
class ToolResult:
    ok: bool
    text: str  # what the model sees: JSON for structured results, otherwise the server's text
    data: dict[str, Any] | None = None


@dataclass
class ServerStatus:
    tools: list[str] = field(default_factory=list)
    blocked_tools: list[str] = field(default_factory=list)  # tools that may change something; not offered yet
    error: str | None = None


def default_servers(project_root: Path) -> dict[str, Any]:
    return {
        "aml-data": StdioServerParameters(
            command=sys.executable, args=["-m", "mcp_servers.aml_data"], cwd=str(project_root)
        ),
    }


class MCPHub:
    def __init__(self, servers: dict[str, Any], timeout_seconds: float = 60):
        self.servers = servers
        self.timeout_seconds = timeout_seconds
        self.status: dict[str, ServerStatus] = {}
        self._stack = AsyncExitStack()
        self._clients: dict[str, Client] = {}
        self._tools: dict[str, tuple[str, ToolSpec]] = {}  # tool name -> (server, spec)

    async def __aenter__(self) -> "MCPHub":
        for name, target in self.servers.items():
            try:
                client = await self._stack.enter_async_context(Client(target, read_timeout_seconds=self.timeout_seconds))
                tools = (await client.list_tools()).tools
            except Exception as exc:
                # One broken server shouldn't take the whole assistant down; /health reports it.
                log.error("MCP server %s unavailable: %s", name, exc)
                self.status[name] = ServerStatus(error=str(exc) or type(exc).__name__)
                continue
            self._clients[name] = client
            status = self.status[name] = ServerStatus()
            for tool in tools:
                # Phase 4 adds a confirmation step for tools that change things; until then only
                # tools the server marks read-only are offered to the model.
                if not (tool.annotations and tool.annotations.read_only_hint):
                    status.blocked_tools.append(tool.name)
                    continue
                if tool.name in self._tools:
                    raise ValueError(f"Tool name {tool.name!r} is offered by both {self._tools[tool.name][0]} and {name}")
                self._tools[tool.name] = (name, ToolSpec(tool.name, tool.description or "", tool.input_schema))
                status.tools.append(tool.name)
            log.info("MCP server %s: %d tools", name, len(status.tools))
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._stack.aclose()

    def specs(self) -> list[ToolSpec]:
        return [spec for _, spec in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(False, f"Unknown or unavailable tool: {name}")
        server, _ = self._tools[name]
        try:
            result = await self._clients[server].call_tool(name, arguments)
        except Exception as exc:
            return ToolResult(False, f"Tool call failed: {exc}")
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        if result.is_error:
            return ToolResult(False, text or "Tool returned an error")
        if result.structured_content is not None:
            return ToolResult(True, json.dumps(result.structured_content, default=str), result.structured_content)
        return ToolResult(True, text)
