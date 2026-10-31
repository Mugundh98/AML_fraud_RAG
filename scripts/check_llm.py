"""Ping each configured LLM provider so you can confirm keys and model IDs.

Run from the project root:  python -m scripts.check_llm
"""

from app.config import get_settings
from app.llm import Message
from app.llm.router import build_router


def main() -> None:
    router = build_router(get_settings())
    for provider in router.providers:
        if not provider.is_configured():
            print(f"{provider.name:7} SKIPPED  no API key in .env")
            continue
        try:
            reply = provider.generate("Reply with exactly: OK", [Message("user", "ping")])
            print(f"{provider.name:7} OK       model={provider.model} reply={reply.strip()[:40]!r}")
        except Exception as exc:
            print(f"{provider.name:7} FAILED   model={provider.model} error={exc}")


if __name__ == "__main__":
    main()
