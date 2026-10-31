from app.llm.base import Message


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float, timeout_seconds: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._client = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, system: str, messages: list[Message]) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            # No SDK retries: on overload the router moves to the next model straight away.
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(
                    timeout=self.timeout_seconds * 1000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )

        # Gemini calls the assistant role "model".
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
        ]
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.temperature,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        if not response.text:
            raise RuntimeError("empty response (possibly blocked by safety filters)")
        return response.text
