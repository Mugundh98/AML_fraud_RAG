"""slack MCP server: post escalations to the compliance team's Slack.

Run over stdio:  python -m mcp_servers.slack

list_channels is read-only. post_message sends a message people will read, so it isn't marked read-only:
the assistant asks the analyst to approve the exact text before it's sent.
"""

import logging
from collections.abc import Callable
from functools import cache
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from app.config import get_settings

API = "https://slack.com/api/"
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)

INSTRUCTIONS = """Post messages to the bank's Slack workspace, e.g. escalations to the compliance channel.
Slack formatting is not Markdown: *bold*, _italic_, `code`, and "• " for bullets."""


def create_server(client: Callable[[], httpx.Client], default_channel: str = "") -> MCPServer:
    server = MCPServer("slack", instructions=INSTRUCTIONS, version="0.4.0")

    def call(method: str, **payload) -> dict:
        # Form encoding works for every Web API method; some (e.g. chat.getPermalink) ignore a JSON body.
        form = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in payload.items()}
        response = client().post(API + method, data=form)
        data = response.json()
        if not data.get("ok"):
            error = data.get("error", f"HTTP {response.status_code}")
            hints = {
                "not_in_channel": "invite the bot to the channel (/invite @AML Assistant)",
                "channel_not_found": "check the channel ID, or invite the bot if it's private",
                "invalid_auth": "check SLACK_BOT_TOKEN",
                "missing_scope": f"the Slack app needs scope {data.get('needed', '?')}; reinstall it after adding",
            }
            raise ToolError(f"Slack error: {error}" + (f" ({hints[error]})" if error in hints else ""))
        return data

    @server.tool(annotations=READ_ONLY)
    def list_channels(limit: int = 100) -> dict[str, Any]:
        """Public channels in the workspace, with their IDs and whether the bot is a member."""
        data = call("conversations.list", types="public_channel", exclude_archived=True, limit=max(1, min(limit, 200)))
        channels = [
            {"id": c["id"], "name": c["name"], "bot_is_member": c.get("is_member", False),
             "is_default": c["id"] == default_channel}
            for c in data.get("channels", [])
        ]
        return {"default_channel": default_channel or None, "count": len(channels), "channels": channels}

    @server.tool(annotations=WRITE)
    def post_message(text: str, channel: str = "") -> dict[str, Any]:
        """Post a message to a Slack channel (ID like C0123ABCD). Leave `channel` empty for the default
        compliance channel. Use Slack formatting. Requires the analyst's approval."""
        target = channel or default_channel
        if not target:
            raise ToolError("No channel given and SLACK_DEFAULT_CHANNEL isn't set")
        if not text.strip():
            raise ToolError("The message is empty")
        sent = call("chat.postMessage", channel=target, text=text)
        result = {"posted": True, "channel": sent["channel"], "ts": sent["ts"]}
        try:
            result["link"] = call("chat.getPermalink", channel=sent["channel"], message_ts=sent["ts"])["permalink"]
        except ToolError:
            pass  # the message is sent; a missing link isn't worth failing over
        return result

    return server


def main() -> None:
    settings = get_settings()

    @cache
    def client() -> httpx.Client:
        if not settings.slack_bot_token:
            raise ToolError("SLACK_BOT_TOKEN isn't set in .env")
        return httpx.Client(headers={"Authorization": f"Bearer {settings.slack_bot_token}"}, timeout=30)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every request URL at INFO; the token is in a header, but keep the log quiet anyway.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    create_server(client, settings.slack_default_channel).run("stdio")


if __name__ == "__main__":
    main()
