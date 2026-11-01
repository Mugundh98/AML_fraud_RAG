"""Reversible pseudonymization: the LLM and Langfuse see <NAME_3>, the analyst sees the real name.

    tool result  {"name": "Jessica Martin", "phone": ...}  ─► model sees {"name": "<NAME_1>", ...}
    model calls  get_customer_profile / find_customers(name="<NAME_1>")  ─► server gets "Jessica Martin"
    model answers "<NAME_1> made 5 deposits..."  ─► analyst reads "Jessica Martin made 5 deposits..."

What counts as PII is decided by field name in structured tool results (customer and account-holder names,
the phone/device/address values behind shared identifiers), plus patterns for emails and phone numbers in
any text. Anything typed by the analyst is checked against the values already known in the conversation.

Limits: a name the analyst types before any tool has returned it goes to the model as typed, and free text
from tools isn't scanned for names. Production systems add a named-entity detector for both.
"""

import re
import threading
from collections import Counter
from typing import Any

NAME_FIELDS = {"name", "customer_name", "owner_name", "counterparty_name", "holder_name"}
NAME_LIST_FIELDS = {"intermediaries"}
SHARED_KINDS = {"HAS_PHONE": "PHONE", "USES_DEVICE": "DEVICE", "REGISTERED_AT": "ADDRESS"}

PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\+\d{1,3}-\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s?\d{3}-\d{4}\b"),
}
TOKEN = re.compile(r"<(NAME|PHONE|DEVICE|ADDRESS|EMAIL)_(\d+)>")

# Every real value pseudonymized in this process. The Langfuse exporter masks these as a last line of
# defence, in case some code path puts real data into a trace.
_known: set[str] = set()
_known_lock = threading.Lock()
_known_pattern: tuple[int, re.Pattern | None] = (0, None)


class Pseudonymizer:
    """One conversation's mapping between real values and placeholders."""

    def __init__(self):
        self._token_of: dict[str, str] = {}
        self._value_of: dict[str, str] = {}
        self._counts: Counter = Counter()

    def token(self, value: str, kind: str) -> str:
        if value not in self._token_of:
            self._counts[kind] += 1
            token = f"<{kind}_{self._counts[kind]}>"
            self._token_of[value], self._value_of[token] = token, value
            with _known_lock:
                _known.add(value)
        return self._token_of[value]

    def redact(self, data: Any) -> Any:
        """Structured tool results: replace PII fields, then scan remaining text.

        Two passes, because the same name often appears in free text before its field does
        ("description": "BILL PAY Pacific Gas", then "counterparty_name": "Pacific Gas"): first every PII
        field anywhere in the result is registered, then all text is scanned with the complete list.
        """
        self._redact(data)
        return self._redact(data)

    def _redact(self, data: Any) -> Any:
        if isinstance(data, dict):
            out = {}
            for key, value in data.items():
                if key in NAME_FIELDS and isinstance(value, str) and value:
                    # Cash moves through branches ("Miami branch"): places, not people, and useful to the model.
                    is_branch = str(data.get("counterparty_id", "")).startswith("BR-")
                    out[key] = value if is_branch else self.token(value, "NAME")
                elif key in NAME_LIST_FIELDS and isinstance(value, list):
                    out[key] = [self.token(v, "NAME") if isinstance(v, str) and v else v for v in value]
                elif key == "value" and data.get("type") in SHARED_KINDS and isinstance(value, str):
                    out[key] = self.token(value, SHARED_KINDS[data["type"]])
                else:
                    out[key] = self._redact(value)
            return out
        if isinstance(data, list):
            return [self._redact(v) for v in data]
        if isinstance(data, str):
            return self.redact_text(data)
        return data

    def redact_text(self, text: str) -> str:
        """Free text (questions, history, unstructured results): known values and PII patterns."""
        for value in sorted(self._token_of, key=len, reverse=True):
            if value in text:
                text = text.replace(value, self._token_of[value])
        for kind, pattern in PATTERNS.items():
            text = pattern.sub(lambda m, k=kind: self.token(m.group(0), k), text)
        return text

    def restore(self, data: Any) -> Any:
        if isinstance(data, str):
            return TOKEN.sub(lambda m: self._value_of.get(m.group(0), m.group(0)), data)
        if isinstance(data, dict):
            return {k: self.restore(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.restore(v) for v in data]
        return data

    def contains_tokens(self, data: Any) -> bool:
        if isinstance(data, str):
            return any(m.group(0) in self._value_of for m in TOKEN.finditer(data))
        if isinstance(data, dict):
            return any(self.contains_tokens(v) for v in data.values())
        if isinstance(data, list):
            return any(self.contains_tokens(v) for v in data)
        return False


class NoPseudonymization(Pseudonymizer):
    """PII_PSEUDONYMIZATION=false: everything passes through unchanged."""

    def redact(self, data: Any) -> Any:
        return data

    def redact_text(self, text: str) -> str:
        return text

    def restore(self, data: Any) -> Any:
        return data


def mask_known(text: str) -> str:
    """Replace every real value pseudonymized so far (used by the trace exporter)."""
    global _known_pattern
    with _known_lock:
        size, pattern = _known_pattern
        if size != len(_known):
            values = sorted(_known, key=len, reverse=True)
            pattern = re.compile("|".join(map(re.escape, values))) if values else None
            _known_pattern = (len(_known), pattern)
    return pattern.sub("[PII]", text) if pattern else text
