"""Anthropic LLM client implementation."""

import json
from typing import Any, Literal

from anthropic import AsyncAnthropic

from swe_forge.config import Settings
from swe_forge.llm.client import (
    Choice,
    FunctionCall,
    GenerationRequest,
    GenerationResponse,
    LLMClient,
    Message,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    Usage,
)


class AnthropicClient(LLMClient):
    """Anthropic API client implementing the LLMClient interface."""

    def __init__(self, settings: Settings | None = None, api_key: str | None = None):
        """Initialize Anthropic client.

        Args:
            settings: Settings instance for API key. If None, created from env.
            api_key: Optional API key override. Takes precedence over settings.
        """
        if api_key:
            self._api_key = api_key
        elif settings:
            self._api_key = settings.anthropic_api_key
        else:
            from swe_forge.config import Settings

            self._api_key = Settings().anthropic_api_key

        self._client = AsyncAnthropic(api_key=self._api_key)

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert our Message format to Anthropic format.

        Anthropic separates system messages from the messages array.
        Tool result messages need special formatting.

        Returns:
            Tuple of (system_prompt, anthropic_messages)
        """
        system_prompt: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                # Extract system message - Anthropic uses separate parameter
                system_prompt = msg.content
            elif msg.role == "user":
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                # Handle assistant message with potential tool calls
                if msg.tool_calls:
                    content: list[dict[str, Any]] = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.function.name,
                                "input": json.loads(tc.function.arguments)
                                if tc.function.arguments
                                else {},
                            }
                        )
                    anthropic_messages.append({"role": "assistant", "content": content})
                else:
                    anthropic_messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
            elif msg.role == "tool":
                # Tool result message
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert our ToolDefinition format to Anthropic format.

        Anthropic uses 'input_schema' instead of 'parameters'.
        """
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            anthropic_tools.append(
                {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "input_schema": tool.function.parameters,
                }
            )
        return anthropic_tools

    def _convert_tool_choice(
        self, tool_choice: ToolChoice | str | None
    ) -> dict[str, Any] | None:
        """Convert tool_choice to Anthropic format.

        Anthropic supports: {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "..."}
        """
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            if tool_choice == "auto":
                return {"type": "auto"}
            elif tool_choice == "none":
                # Anthropic doesn't have 'none', use auto
                return {"type": "auto"}
            return None
        # ToolChoice object - force specific tool
        return {"type": "tool", "name": tool_choice.function.name}

    def _parse_response(self, response: Any, model: str) -> GenerationResponse:
        """Parse Anthropic response to our GenerationResponse format."""
        # Extract text content and tool uses from response.content
        text_content = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )

        # Build message
        if tool_calls:
            message = Message.assistant_with_tool_calls(text_content, tool_calls)
        else:
            message = Message.assistant(text_content)

        # Map stop_reason to finish_reason
        stop_reason = getattr(response, "stop_reason", "end_turn")
        finish_reason_map: dict[str, str] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }
        finish_reason = finish_reason_map.get(stop_reason, stop_reason or "stop")

        # Build usage
        usage = Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        # Build choice and response
        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )

        return GenerationResponse(
            id=response.id,
            model=model,
            choices=[choice],
            usage=usage,
        )

    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion without tool calling."""
        system_prompt, messages = self._convert_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        else:
            kwargs["max_tokens"] = 4096  # Anthropic requires max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response, request.model)

    async def complete_with_tools(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """Generate a completion with tool calling support."""
        system_prompt, messages = self._convert_messages(request.messages)

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        else:
            kwargs["max_tokens"] = 4096
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        # Convert tools if provided
        if request.tools:
            kwargs["tools"] = self._convert_tools(request.tools)

        # Convert tool_choice if provided
        tool_choice = self._convert_tool_choice(request.tool_choice)
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response, request.model)

    async def stream(self, request: GenerationRequest) -> None:
        """Stream a response for the given request.

        Note: Streaming is optional for MVP. Raises NotImplementedError.
        """
        raise NotImplementedError("Streaming not yet implemented for Anthropic client")
