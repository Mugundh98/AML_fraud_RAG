"""MCP client side: connects to several MCP servers and exposes their tools to the agent as one list.

Each server is started (stdio), reached over HTTP (hosted servers), or attached in-process (tests) when the hub
opens, and stays connected until it closes, so a chat request doesn't pay for launching a subprocess.

Every tool either runs straight away or needs the analyst's approval first. For our own servers we trust their
readOnlyHint annotations. For third-party servers annotations are only a claim, so `auto_run` lists the tools
that may run without approval, and everything else from that server needs it.
"""

import json
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from app.config import Settings
from app.llm import ToolSpec

log = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    target: Any  # what mcp.Client connects to: StdioServerParameters, an HTTP transport, or (tests) an MCPServer
    tools: set[str] | None = None  # offer only these tools; None offers all of them
    auto_run: set[str] | None = None  # tools that run without approval; None trusts the server's readOnlyHint
    # Per tool, the only parameters the model sees (the server uses its defaults for the rest). Every tool
    # definition is sent with every model call, so unused options cost tokens on each step.
    params: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    text: str  # what the model sees: JSON for structured results, otherwise the server's text
    data: dict[str, Any] | None = None


@dataclass
class ServerStatus:
    tools: list[str] = field(default_factory=list)
    approval_tools: list[str] = field(default_factory=list)  # offered, but only run after the analyst approves
    error: str | None = None


@dataclass
class _Tool:
    server: str
    spec: ToolSpec
    needs_approval: bool


def default_servers(settings: Settings, project_root: Path) -> dict[str, ServerConfig]:
    """Our own servers run as subprocesses; Tavily's is hosted. Each one is only added when it's configured."""

    def local(module: str) -> StdioServerParameters:
        return StdioServerParameters(command=sys.executable, args=["-m", module], cwd=str(project_root))

    servers = {"aml-data": ServerConfig(local("mcp_servers.aml_data"))}
    if settings.tavily_api_key:
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        # The key goes in a header, not in the URL, so it can't end up in request logs.
        http = create_mcp_http_client(headers={"Authorization": f"Bearer {settings.tavily_api_key}"})
        servers["tavily"] = ServerConfig(
            streamable_http_client(settings.tavily_mcp_url, http_client=http),
            # Crawl, map and research also exist, but cost many more of the free tier's credits.
            tools={"tavily_search", "tavily_extract"},
            auto_run={"tavily_search", "tavily_extract"},
            # Search has 17 options (images, favicons, country boost...); these are the ones AML research uses.
            params={
                "tavily_search": {"query", "max_results", "topic", "time_range", "include_domains"},
                "tavily_extract": {"urls", "query"},
            },
        )
    if settings.google_client_secret_file.exists():
        servers["google-calendar"] = ServerConfig(local("mcp_servers.google_calendar"))
    if settings.slack_bot_token:
        servers["slack"] = ServerConfig(local("mcp_servers.slack"))
    return servers


def compact_schema(schema: dict[str, Any], keep: set[str] | None = None) -> dict[str, Any]:
    """The same constraints in fewer tokens.

    Drops the labels pydantic adds to every field ("title": "Account Id") and `"default": null`, turns
    `anyOf: [X, {"type": "null"}]` into X (the field is optional anyway: it isn't in `required`), and if
    `keep` is given, removes every other top-level parameter.
    """
    compact = _compact(schema)
    if keep is not None:
        compact["properties"] = {k: v for k, v in compact.get("properties", {}).items() if k in keep}
        if "required" in compact:
            compact["required"] = [r for r in compact["required"] if r in keep]
    return compact


def _compact(node: Any) -> Any:
    if isinstance(node, list):
        return [_compact(n) for n in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if (key == "title" and isinstance(value, str)) or (key == "default" and value is None):
            continue
        if key == "properties" and isinstance(value, dict):
            # Keys here are parameter names (one may well be called "title"), not schema keywords.
            out[key] = {name: _compact(prop) for name, prop in value.items()}
        else:
            out[key] = _compact(value)
    variants = out.get("anyOf")
    if isinstance(variants, list) and len(variants) == 2 and {"type": "null"} in variants:
        other = next(v for v in variants if v != {"type": "null"})
        del out["anyOf"]
        out = {**other, **out}
    return out


class MCPHub:
    def __init__(self, servers: dict[str, ServerConfig | Any], timeout_seconds: float = 60):
        self.servers = {name: s if isinstance(s, ServerConfig) else ServerConfig(s) for name, s in servers.items()}
        self.timeout_seconds = timeout_seconds
        self.status: dict[str, ServerStatus] = {}
        self._stack = AsyncExitStack()
        self._clients: dict[str, Client] = {}
        self._tools: dict[str, _Tool] = {}

    async def __aenter__(self) -> "MCPHub":
        for name, config in self.servers.items():
            try:
                client = await self._stack.enter_async_context(
                    Client(config.target, read_timeout_seconds=self.timeout_seconds)
                )
                tools = (await client.list_tools()).tools
            except Exception as exc:
                # One broken server shouldn't take the whole assistant down; /health reports it.
                log.error("MCP server %s unavailable: %s", name, exc)
                self.status[name] = ServerStatus(error=str(exc) or type(exc).__name__)
                continue
            self._clients[name] = client
            status = self.status[name] = ServerStatus()
            for tool in tools:
                if config.tools is not None and tool.name not in config.tools:
                    continue
                if tool.name in self._tools:
                    raise ValueError(f"Tool {tool.name!r} is offered by both {self._tools[tool.name].server} and {name}")
                if config.auto_run is not None:
                    needs_approval = tool.name not in config.auto_run
                else:
                    needs_approval = not (tool.annotations and tool.annotations.read_only_hint)
                schema = compact_schema(tool.input_schema, config.params.get(tool.name))
                self._tools[tool.name] = _Tool(name, ToolSpec(tool.name, tool.description or "", schema),
                                               needs_approval)
                (status.approval_tools if needs_approval else status.tools).append(tool.name)
            log.info("MCP server %s: %d tools, %d need approval", name, len(status.tools), len(status.approval_tools))
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._stack.aclose()

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def needs_approval(self, name: str) -> bool:
        return name in self._tools and self._tools[name].needs_approval

    def server_of(self, name: str) -> str:
        return self._tools[name].server

    async def call(self, name: str, arguments: dict[str, Any], approved: bool = False) -> ToolResult:
        if name not in self._tools:
            return ToolResult(False, f"Unknown or unavailable tool: {name}")
        tool = self._tools[name]
        if tool.needs_approval and not approved:
            return ToolResult(False, f"{name} needs the analyst's approval before it can run")
        try:
            result = await self._clients[tool.server].call_tool(name, arguments)
        except Exception as exc:
            return ToolResult(False, f"Tool call failed: {exc}")
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        if result.is_error:
            return ToolResult(False, text or "Tool returned an error")
        if result.structured_content is not None:
            return ToolResult(True, json.dumps(result.structured_content, default=str), result.structured_content)
        return ToolResult(True, text)
