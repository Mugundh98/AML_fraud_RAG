"""Actions the model proposed that change something outside this app (a Slack post, a calendar event).

The model never runs these itself. Its call is stored here exactly as made, the analyst sees the tool and its
arguments, and only an explicit approval runs it, with those same arguments and no LLM in between. Each action
runs at most once and expires if nobody decides in time.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PendingAction:
    id: str
    tool: str
    server: str
    arguments: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    proposed_by: str = ""


class ApprovalQueue:
    def __init__(self, ttl_seconds: float, clock=time.time):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._actions: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def add(self, tool: str, server: str, arguments: dict[str, Any], proposed_by: str = "") -> PendingAction:
        action = PendingAction(f"act_{uuid.uuid4().hex[:12]}", tool, server, dict(arguments), self.clock(), proposed_by)
        with self._lock:
            self._expire()
            self._actions[action.id] = action
        return action

    def peek(self, action_id: str) -> PendingAction | None:
        """Look at an action without deciding it (e.g. to check the approver's role first)."""
        with self._lock:
            self._expire()
            return self._actions.get(action_id)

    def take(self, action_id: str) -> PendingAction | None:
        """Remove and return the action, so a second click (or a replayed request) can't run it twice."""
        with self._lock:
            self._expire()
            return self._actions.pop(action_id, None)

    def _expire(self) -> None:
        cutoff = self.clock() - self.ttl_seconds
        for action_id in [a.id for a in self._actions.values() if a.created_at < cutoff]:
            del self._actions[action_id]
