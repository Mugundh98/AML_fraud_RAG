"""google-calendar MCP server: look at the analyst's calendar and book case review meetings.

Run over stdio:  python -m mcp_servers.google_calendar
Sign in first:   python -m scripts.google_auth

list_events and find_free_slots are read-only. create_event changes the calendar (and emails attendees),
so it isn't marked read-only: the assistant asks the analyst to approve it before it runs.
"""

import logging
import re
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from functools import cache
from typing import Any, Protocol

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from app.config import get_settings

API = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True)
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

INSTRUCTIONS = """The analyst's Google Calendar (primary calendar).
Times are in the analyst's local timezone, written as YYYY-MM-DDTHH:MM.
Use find_free_slots before proposing a meeting time."""


class HttpSession(Protocol):
    """What we need from google.auth's AuthorizedSession (a requests.Session that adds the OAuth token)."""

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> Any: ...

    def post(self, url: str, json: dict | None = None, params: dict | None = None, timeout: float | None = None) -> Any: ...


def create_server(session: Callable[[], HttpSession], now: Callable[[], datetime] = lambda: datetime.now().astimezone()) -> MCPServer:
    server = MCPServer("google-calendar", instructions=INSTRUCTIONS, version="0.4.0")

    def request(method: str, **kwargs) -> dict:
        try:
            response = getattr(session(), method)(API, timeout=30, **kwargs)
        except RuntimeError as exc:  # not signed in, or sign-in expired
            raise ToolError(str(exc)) from exc
        if response.status_code >= 400:
            try:
                message = response.json()["error"]["message"]
            except Exception:
                message = response.text[:300]
            if "has not been used" in message or "is disabled" in message:
                message += " (enable the Google Calendar API for this project in Google Cloud Console)"
            raise ToolError(f"Google Calendar error {response.status_code}: {message}")
        return response.json()

    def events_between(start: datetime, end: datetime, limit: int) -> list[dict]:
        data = request("get", params={
            "timeMin": start.isoformat(), "timeMax": end.isoformat(), "singleEvents": "true",
            "orderBy": "startTime", "maxResults": limit,
        })
        return data.get("items", [])

    @server.tool(annotations=READ_ONLY)
    def list_events(days: int = 7, limit: int = 20) -> dict[str, Any]:
        """Events on the analyst's calendar from now until `days` days ahead."""
        start = now()
        events = events_between(start, start + timedelta(days=max(1, min(days, 60))), max(1, min(limit, 100)))
        return {"from": start.isoformat(timespec="minutes"), "count": len(events), "events": [_event(e) for e in events]}

    @server.tool(annotations=READ_ONLY)
    def find_free_slots(day: str, duration_minutes: int = 30, day_start: str = "09:00", day_end: str = "18:00") -> dict[str, Any]:
        """Free times on one day (YYYY-MM-DD) long enough for a meeting of `duration_minutes`, within working hours.
        All-day events are listed but don't block time; events marked "free" or declined are ignored."""
        tz = now().tzinfo
        try:
            d = date.fromisoformat(day)
            window_start = datetime.combine(d, time.fromisoformat(day_start), tz)
            window_end = datetime.combine(d, time.fromisoformat(day_end), tz)
        except ValueError as exc:
            raise ToolError(f"Use day=YYYY-MM-DD and times as HH:MM ({exc})") from exc
        if d == now().date():  # nothing in the past; start at the next quarter hour
            soon = now() + timedelta(minutes=15 - now().minute % 15)
            window_start = max(window_start, soon.replace(second=0, microsecond=0))
        duration = timedelta(minutes=max(5, min(duration_minutes, 480)))

        busy, all_day = [], []
        for e in events_between(window_start, window_end, 100) if window_start < window_end else []:
            if "date" in e.get("start", {}):
                all_day.append(e.get("summary", "(no title)"))
            elif e.get("transparency") != "transparent" and not _declined(e):
                busy.append((datetime.fromisoformat(e["start"]["dateTime"]), datetime.fromisoformat(e["end"]["dateTime"])))

        slots, cursor = [], window_start
        for b_start, b_end in sorted(busy) + [(window_end, window_end)]:
            if b_start - cursor >= duration:
                slots.append({"start": _local(cursor), "end": _local(b_start)})
            cursor = max(cursor, b_end)
        return {"day": day, "duration_minutes": int(duration.total_seconds() // 60), "free_slots": slots,
                "busy_count": len(busy), "all_day_events": all_day}

    @server.tool(annotations=WRITE)
    def create_event(title: str, start: str, duration_minutes: int = 30, description: str = "",
                     attendees: list[str] | None = None) -> dict[str, Any]:
        """Create an event on the analyst's calendar. `start` is local time, YYYY-MM-DDTHH:MM.
        Attendees (email addresses) receive an invitation email. Requires the analyst's approval."""
        try:
            begins = datetime.fromisoformat(start)
        except ValueError as exc:
            raise ToolError("start must look like 2026-11-03T14:30") from exc
        if begins.tzinfo is None:
            begins = begins.replace(tzinfo=now().tzinfo)
        bad = [a for a in attendees or [] if not EMAIL.match(a)]
        if bad:
            raise ToolError(f"Not email addresses: {', '.join(bad)}")
        ends = begins + timedelta(minutes=max(5, min(duration_minutes, 480)))
        body = {
            "summary": title,
            "description": description,
            "start": {"dateTime": begins.isoformat()},
            "end": {"dateTime": ends.isoformat()},
            "attendees": [{"email": a} for a in attendees or []],
        }
        created = request("post", json=body, params={"sendUpdates": "all" if attendees else "none"})
        return {"created": True, **_event(created)}

    return server


def _event(e: dict) -> dict:
    start, end = e.get("start", {}), e.get("end", {})
    return {
        "id": e.get("id"),
        "title": e.get("summary", "(no title)"),
        "start": start.get("dateTime", start.get("date")),
        "end": end.get("dateTime", end.get("date")),
        "attendees": [a.get("email") for a in e.get("attendees", [])],
        "link": e.get("htmlLink"),
    }


def _declined(e: dict) -> bool:
    return any(a.get("self") and a.get("responseStatus") == "declined" for a in e.get("attendees", []))


def _local(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M")


def main() -> None:
    from mcp_servers.google_calendar.auth import load_credentials

    settings = get_settings()

    # Errors aren't cached, so signing in while the server is running works on the next call.
    @cache
    def session():
        from google.auth.transport.requests import AuthorizedSession

        return AuthorizedSession(load_credentials(settings.google_token_file))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    create_server(session).run("stdio")


if __name__ == "__main__":
    main()
