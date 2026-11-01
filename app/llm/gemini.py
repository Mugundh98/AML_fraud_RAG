import uuid

from app.llm.base import AssistantTurn, Message, ToolCall, ToolSpec

# Documented placeholder for tool calls whose real signature we don't have (e.g. made by another model).
SKIP_SIGNATURE = b"skip_thought_signature_validator"


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
        turn = self.chat(system, messages, [])
        return turn.text

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec],
             force_answer: bool = False) -> AssistantTurn:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            # We run tools ourselves (through MCP), so the SDK must not try to call Python functions.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if tools:
            config.tools = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                for t in tools
            ])]
            if force_answer:
                config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
                )
        response = self._get_client().models.generate_content(
            model=self.model, contents=self.to_contents(messages), config=config
        )

        candidate = response.candidates[0] if response.candidates else None
        parts = (candidate.content.parts if candidate and candidate.content else None) or []
        calls = [
            ToolCall(
                id=p.function_call.id or f"call_{uuid.uuid4().hex[:12]}",
                name=p.function_call.name,
                arguments=dict(p.function_call.args or {}),
                signature=p.thought_signature,
                model=self.model,
            )
            for p in parts
            if p.function_call
        ]
        text = "".join(p.text for p in parts if p.text and not p.thought)
        if not text and not calls:
            raise RuntimeError("empty response (possibly blocked by safety filters)")
        return AssistantTurn(text=text, tool_calls=calls)

    def to_contents(self, messages: list[Message]) -> list:
        from google.genai import types

        contents: list[types.Content] = []
        for m in messages:
            if m.role == "tool":
                part = types.Part.from_function_response(name=m.tool_name, response={"result": m.content})
                # Results of parallel calls go back together, in one user turn.
                if contents and contents[-1].role == "user" and contents[-1].parts[-1].function_response:
                    contents[-1].parts.append(part)
                else:
                    contents.append(types.Content(role="user", parts=[part]))
            elif m.role == "assistant":
                # Gemini calls the assistant role "model".
                parts = [types.Part(text=m.content)] if m.content else []
                for i, call in enumerate(m.tool_calls):
                    if call.model == self.model:
                        signature = call.signature
                    else:
                        signature = SKIP_SIGNATURE if i == 0 else None
                    parts.append(types.Part(
                        function_call=types.FunctionCall(name=call.name, args=call.arguments),
                        thought_signature=signature,
                    ))
                contents.append(types.Content(role="model", parts=parts))
            elif contents and contents[-1].role == "user":
                contents[-1].parts.append(types.Part(text=m.content))
            else:
                contents.append(types.Content(role="user", parts=[types.Part(text=m.content)]))
        return contents

    def _get_client(self):
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
        return self._client
