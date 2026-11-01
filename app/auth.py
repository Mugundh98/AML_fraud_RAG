"""Users, API keys and roles.

Each analyst has an API key, sent as the X-API-Key header. Only a SHA-256 hash of each key is stored
(secrets/users.json, which git ignores), so the file alone can't be used to sign in. Create users with:

    python -m scripts.create_user --name "Priya Shah" --role senior_analyst

Without a users file the API runs in open mode: everyone is one local user with the lead role. Fine for a
laptop, and /health says so.
"""

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Role = Literal["analyst", "senior_analyst", "lead"]
ROLES: tuple[Role, ...] = ("analyst", "senior_analyst", "lead")

# Tools that change something, and who may use them. Example Bank's policy: Level 1 analysts escalate to
# Level 2, so posting an escalation needs a senior analyst. A write tool not listed here needs a lead.
WRITE_TOOL_ROLES: dict[str, set[Role]] = {
    "create_event": {"analyst", "senior_analyst", "lead"},
    "post_message": {"senior_analyst", "lead"},
}


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    role: Role

    def can_use(self, tool: str, needs_approval: bool) -> bool:
        """Read-only tools are open to every role; tools that change something are limited by role."""
        if not needs_approval:
            return True
        return self.role in WRITE_TOOL_ROLES.get(tool, {"lead"})

    @property
    def can_read_audit(self) -> bool:
        return self.role == "lead"


LOCAL_USER = User("local", "Local user (open mode)", "lead")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class UserDirectory:
    def __init__(self, path: Path):
        self.path = path
        self._users: list[tuple[str, User]] = []
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                self._users.append((row["key_sha256"], User(row["user_id"], row["name"], row["role"])))

    @property
    def open_mode(self) -> bool:
        return not self._users

    def authenticate(self, api_key: str | None) -> User | None:
        if self.open_mode:
            return LOCAL_USER
        if not api_key:
            return None
        digest = hash_key(api_key)
        for stored, user in self._users:
            if hmac.compare_digest(stored, digest):  # constant time, so timing doesn't leak the hash
                return user
        return None

    def add(self, name: str, role: Role) -> tuple[User, str]:
        """Create a user and return them with their new API key. The key is shown once and never stored."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")
        key = f"aml_{secrets.token_urlsafe(24)}"
        user = User(f"u{len(self._users) + 1:03d}", name, role)
        self._users.append((hash_key(key), user))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            [{**asdict(u), "key_sha256": digest} for digest, u in self._users], indent=2
        ), encoding="utf-8")
        return user, key
