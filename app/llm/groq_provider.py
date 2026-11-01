import json

from app.llm.base import AssistantTurn, Message, RateLimited, ToolCall, ToolSpec, chat_format, retry_after_from


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str, temperature: float, timeout_seconds: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, system: str, messages: list[Message]) -> str:
        return self.chat(system, messages, []).text

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec],
             force_answer: bool = False) -> AssistantTurn:
        from groq import Groq, RateLimitError

        if self._client is None:
            self._client = Groq(api_key=self.api_key, timeout=self.timeout_seconds, max_retries=0)

        kwargs = {}
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
            kwargs["tool_choice"] = "none" if force_answer else "auto"
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self.to_messages(system, messages),
                temperature=self.temperature,
                **kwargs,
            )
        except RateLimitError as exc:
            header = exc.response.headers.get("retry-after") if exc.response is not None else None
            raise RateLimited(str(exc), float(header) if header else retry_after_from(str(exc))) from exc
        reply = response.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=json.loads(c.function.arguments or "{}"), model=self.model)
            for c in reply.tool_calls or []
        ]
        if not reply.content and not calls:
            raise RuntimeError("empty response")
        return AssistantTurn(text=reply.content or "", tool_calls=calls, model=response.model or self.model,
                             usage=_usage(response.usage), reasoning=getattr(reply, "reasoning", None) or "")

    to_messages = staticmethod(chat_format)


def _usage(usage) -> dict[str, int]:
    if usage is None:
        return {}
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
    # Groq's prompt_tokens includes cached tokens; Langfuse buckets must not overlap.
    result = {"input": usage.prompt_tokens - cached, "output": usage.completion_tokens}
    if cached:
        result["cache_read_input_tokens"] = cached
    return result
