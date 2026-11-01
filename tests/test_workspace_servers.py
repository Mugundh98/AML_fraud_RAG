"""The Google Calendar and Slack MCP servers, through a real MCP client, with fake HTTP instead of Google and Slack."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl

import httpx
from mcp import Client

from mcp_servers.google_calendar.server import create_server as calendar_server
from mcp_servers.slack.server import create_server as slack_server

TZ = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 11, 2, 8, 0, tzinfo=TZ)  # Monday 08:00


def call(server, tool, args):
    async def main():
        async with Client(server) as client:
            return await client.call_tool(tool, args)

    return asyncio.run(main())


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeGoogle:
    def __init__(self, events=(), status=200):
        self.events = list(events)
        self.status = status
        self.posted = []

    def get(self, url, params=None, timeout=None):
        if self.status != 200:
            return FakeResponse(self.status, {"error": {"message": "Google Calendar API has not been used in project 1"}})
        return FakeResponse(200, {"items": self.events})

    def post(self, url, json=None, params=None, timeout=None):
        self.posted.append((json, params))
        return FakeResponse(200, {"id": "ev1", "summary": json["summary"], "start": json["start"], "end": json["end"],
                                  "attendees": json["attendees"], "htmlLink": "https://calendar.google.com/ev1"})


def timed(title, start, end, **extra):
    return {"summary": title, "start": {"dateTime": f"2026-11-02T{start}:00+05:30"},
            "end": {"dateTime": f"2026-11-02T{end}:00+05:30"}, **extra}


def test_free_slots_skip_busy_free_and_declined_events():
    google = FakeGoogle([
        timed("Standup", "09:00", "09:30"),
        timed("Lunch (free)", "12:00", "13:00", transparency="transparent"),
        timed("Declined sync", "14:00", "15:00", attendees=[{"self": True, "responseStatus": "declined"}]),
        timed("Case review", "15:00", "17:45"),
        {"summary": "Holiday", "start": {"date": "2026-11-02"}, "end": {"date": "2026-11-03"}},
    ])
    server = calendar_server(lambda: google, now=lambda: NOW)
    result = call(server, "find_free_slots", {"day": "2026-11-02", "duration_minutes": 30}).structured_content

    assert result["free_slots"] == [{"start": "2026-11-02T09:30", "end": "2026-11-02T15:00"}]
    assert result["all_day_events"] == ["Holiday"]


def test_create_event_uses_local_time_and_invites_attendees():
    google = FakeGoogle()
    server = calendar_server(lambda: google, now=lambda: NOW)
    result = call(server, "create_event", {"title": "AL-0002 review", "start": "2026-11-03T15:00",
                                           "duration_minutes": 45, "attendees": ["lead@example.com"]})

    body, params = google.posted[0]
    assert body["start"]["dateTime"] == "2026-11-03T15:00:00+05:30"
    assert body["end"]["dateTime"] == "2026-11-03T15:45:00+05:30"
    assert params == {"sendUpdates": "all"}
    assert result.structured_content["link"] == "https://calendar.google.com/ev1"


def test_create_event_rejects_bad_attendee():
    server = calendar_server(lambda: FakeGoogle(), now=lambda: NOW)
    result = call(server, "create_event", {"title": "x", "start": "2026-11-03T15:00", "attendees": ["not-an-email"]})
    assert result.is_error and "not-an-email" in result.content[0].text


def test_calendar_api_not_enabled_gives_a_hint():
    server = calendar_server(lambda: FakeGoogle(status=403), now=lambda: NOW)
    result = call(server, "list_events", {})
    assert result.is_error and "enable the Google Calendar API" in result.content[0].text


def test_calendar_not_signed_in():
    def no_token():
        raise RuntimeError("Google Calendar isn't connected yet. Run `python -m scripts.google_auth`")

    result = call(calendar_server(no_token, now=lambda: NOW), "list_events", {})
    assert result.is_error and "scripts.google_auth" in result.content[0].text


def slack(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_post_message_goes_to_default_channel_with_link():
    requests = []

    def handler(request):
        body = dict(parse_qsl(request.content.decode()))
        requests.append((request.url.path, body))
        if request.url.path.endswith("chat.postMessage"):
            return httpx.Response(200, json={"ok": True, "channel": body["channel"], "ts": "1.2"})
        return httpx.Response(200, json={"ok": True, "permalink": "https://slack.com/archives/C1/p12"})

    client = slack(handler)
    result = call(slack_server(lambda: client, "C1"), "post_message", {"text": "*AL-0002* escalated"})

    assert requests[0] == ("/api/chat.postMessage", {"channel": "C1", "text": "*AL-0002* escalated"})
    assert requests[1] == ("/api/chat.getPermalink", {"channel": "C1", "message_ts": "1.2"})
    assert result.structured_content == {"posted": True, "channel": "C1", "ts": "1.2",
                                         "link": "https://slack.com/archives/C1/p12"}


def test_slack_errors_explain_the_fix():
    client = slack(lambda request: httpx.Response(200, json={"ok": False, "error": "not_in_channel"}))
    result = call(slack_server(lambda: client, "C1"), "post_message", {"text": "hi"})
    assert result.is_error and "/invite" in result.content[0].text


def test_slack_write_tool_is_not_read_only():
    async def main():
        async with Client(slack_server(lambda: None, "C1")) as client:
            return {t.name: t.annotations.read_only_hint for t in (await client.list_tools()).tools}

    assert asyncio.run(main()) == {"list_channels": True, "post_message": False}
