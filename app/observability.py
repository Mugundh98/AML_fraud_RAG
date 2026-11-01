"""Langfuse tracing (Python SDK v5, built on OpenTelemetry).

One trace per chat turn:

    answer-question (agent)            input: the question, output: the answer
    ├── plan-next-step (generation)    one per model call: messages in, assistant message out, model, tokens
    ├── search_policies (retriever)    knowledge-base search, with the passages found
    ├── check_structuring (tool)       every other MCP tool call, with arguments and result
    ├── plan-next-step (generation)
    └── ...

Approving an action is its own trace (execute-approved-action) in the same session as the chat.

The app keeps one Langfuse client here instead of calling langfuse.get_client() everywhere: get_client()
returns a disabled client whenever a process holds more than one, which happens in tests.
"""

import logging
import re

from langfuse import Langfuse
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

from app.agent.pii import mask_known
from app.config import Settings

log = logging.getLogger(__name__)

# API keys and tokens that must never reach a trace, even if a tool result or error message contains one.
SECRET = re.compile(
    r"AIza[0-9A-Za-z_-]{20,}"             # Google API key
    r"|gsk_[0-9A-Za-z]{20,}"              # Groq
    r"|tvly-[0-9A-Za-z_-]{10,}"           # Tavily
    r"|xox[abpr]-[0-9A-Za-z-]{10,}"       # Slack bot/user tokens
    r"|xapp-[0-9A-Za-z-]{10,}"            # Slack app token
    r"|sk-lf-[0-9a-f-]{20,}"              # Langfuse secret key
    r"|GOCSPX-[0-9A-Za-z_-]{10,}"         # Google OAuth client secret
    r"|ya29\.[0-9A-Za-z_.-]{20,}"         # Google OAuth access token
    r"|(?i:bearer)\s+[0-9A-Za-z._~+/-]{20,}=*"
)

_client: Langfuse | None = None


def mask_secrets(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
    """Export-stage masking, the last step before data leaves the process: credentials become [REDACTED],
    and any customer PII the app has pseudonymized becomes [PII] (traces should only hold placeholders)."""
    patches = {}
    for identifier, span in params.spans.items():
        changed = {}
        for key, value in span.attributes.items():
            if isinstance(value, str):
                masked = mask_known(SECRET.sub("[REDACTED]", value))
                if masked != value:
                    changed[key] = masked
        if changed:
            patches[identifier] = OtelSpanPatch(set_attributes=changed)
    return MaskOtelSpansResult(span_patches=patches) if patches else None


def init_tracing(settings: Settings, **overrides) -> Langfuse:
    """Create the app's Langfuse client. Tracing stays off (a no-op client) without both keys."""
    global _client
    enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
    _client = Langfuse(
        public_key=settings.langfuse_public_key or "disabled",
        secret_key=settings.langfuse_secret_key or "disabled",
        base_url=settings.langfuse_base_url,
        tracing_enabled=enabled,
        # Keeps local experiments out of production views in Langfuse.
        environment=settings.langfuse_tracing_environment,
        mask_otel_spans=mask_secrets,
        **overrides,
    )
    log.info("Langfuse tracing %s", f"on ({settings.langfuse_base_url})" if enabled else "off: no keys")
    return _client


def langfuse() -> Langfuse:
    """The app's client; a disabled one until init_tracing() runs."""
    global _client
    if _client is None:
        _client = Langfuse(public_key="disabled", secret_key="disabled", tracing_enabled=False)
    return _client
