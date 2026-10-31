from app.llm.base import Message


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
        from groq import Groq

        if self._client is None:
            self._client = Groq(api_key=self.api_key, timeout=self.timeout_seconds, max_retries=0)

        chat = [{"role": "system", "content": system}]
        chat += [{"role": m.role, "content": m.content} for m in messages]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=chat,
            temperature=self.temperature,
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("empty response")
        return text
