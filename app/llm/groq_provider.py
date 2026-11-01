import json

from app.llm.base import AssistantTurn, Message, ToolCall, ToolSpec


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
        from groq import Groq

        if self._client is None:
            self._client = Groq(api_key=self.api_key, timeout=self.timeout_seconds, max_retries=0)

        kwargs = {}
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
            kwargs["tool_choice"] = "none" if force_answer else "auto"
        response = self._client.chat.completions.create(
            model=self.model,
            messages=self.to_messages(system, messages),
            temperature=self.temperature,
            **kwargs,
        )
        reply = response.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=json.loads(c.function.arguments or "{}"), model=self.model)
            for c in reply.tool_calls or []
        ]
        if not reply.content and not calls:
            raise RuntimeError("empty response")
        return AssistantTurn(text=reply.content or "", tool_calls=calls)

    @staticmethod
    def to_messages(system: str, messages: list[Message]) -> list[dict]:
        chat = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "tool":
                chat.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.tool_calls:
                chat.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in m.tool_calls
                    ],
                })
            else:
                chat.append({"role": m.role, "content": m.content})
        return chat
