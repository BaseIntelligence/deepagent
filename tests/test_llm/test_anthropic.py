"""Tests for AnthropicClient implementation."""

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_forge.llm import (
    Choice,
    GenerationRequest,
    GenerationResponse,
    Message,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    Usage,
)
from swe_forge.llm.client import FunctionCall
from swe_forge.llm.anthropic_client import AnthropicClient


@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = "Hello! How can I help you?"


@dataclass
class MockToolUseBlock:
    type: str = "tool_use"
    id: str = "toolu_123"
    name: str = "get_weather"
    input: dict[str, Any] = field(default_factory=lambda: {"city": "NYC"})


@dataclass
class MockUsage:
    input_tokens: int = 100
    output_tokens: int = 50


@dataclass
class MockAnthropicResponse:
    id: str = "msg_123"
    content: list[Any] = field(default_factory=lambda: [MockTextBlock()])
    model: str = "claude-sonnet-4-20250514"
    stop_reason: str = "end_turn"
    usage: MockUsage = field(default_factory=MockUsage)


class TestAnthropicClientInit:
    def test_init_with_api_key(self):
        client = AnthropicClient(api_key="test-key")
        assert client._api_key == "test-key"

    def test_init_with_settings(self):
        from swe_forge.config import Settings

        settings = Settings(
            openrouter_api_key="test",
            github_token="test",
            anthropic_api_key="settings-key",
        )
        client = AnthropicClient(settings=settings)
        assert client._api_key == "settings-key"

    def test_api_key_overrides_settings(self):
        from swe_forge.config import Settings

        settings = Settings(
            openrouter_api_key="test",
            github_token="test",
            anthropic_api_key="settings-key",
        )
        client = AnthropicClient(settings=settings, api_key="override-key")
        assert client._api_key == "override-key"


class TestMessageConversion:
    def test_convert_user_message(self):
        client = AnthropicClient(api_key="test")
        messages = [Message.user("Hello")]
        system, anthropic_messages = client._convert_messages(messages)

        assert system is None
        assert anthropic_messages == [{"role": "user", "content": "Hello"}]

    def test_convert_assistant_message(self):
        client = AnthropicClient(api_key="test")
        messages = [Message.assistant("Hi there")]
        system, anthropic_messages = client._convert_messages(messages)

        assert system is None
        assert anthropic_messages == [{"role": "assistant", "content": "Hi there"}]

    def test_system_message_extracted(self):
        client = AnthropicClient(api_key="test")
        messages = [
            Message.system("You are helpful."),
            Message.user("Hello"),
        ]
        system, anthropic_messages = client._convert_messages(messages)

        assert system == "You are helpful."
        assert len(anthropic_messages) == 1
        assert anthropic_messages[0] == {"role": "user", "content": "Hello"}

    def convert_assistant_with_tool_calls(self):
        client = AnthropicClient(api_key="test")
        tool_call = ToolCall(
            id="toolu_123",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        )
        messages = [Message.assistant_with_tool_calls("Thinking...", [tool_call])]
        system, anthropic_messages = client._convert_messages(messages)

        assert system is None
        assert len(anthropic_messages) == 1
        content = anthropic_messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Thinking..."}
        assert content[1]["type"] == "tool_use"
        assert content[1]["id"] == "toolu_123"
        assert content[1]["name"] == "get_weather"
        assert content[1]["input"] == {"city": "NYC"}

    def test_convert_tool_result_message(self):
        client = AnthropicClient(api_key="test")
        messages = [Message.tool_result("toolu_123", '{"temp": 72}')]
        system, anthropic_messages = client._convert_messages(messages)

        assert system is None
        assert len(anthropic_messages) == 1
        assert anthropic_messages[0]["role"] == "user"
        content = anthropic_messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "toolu_123"
        assert content[0]["content"] == '{"temp": 72}'


class TestToolConversion:
    def test_convert_tools_uses_input_schema(self):
        client = AnthropicClient(api_key="test")
        params = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }
        tools = [ToolDefinition.create("get_weather", "Get weather", params)]
        anthropic_tools = client._convert_tools(tools)

        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "get_weather"
        assert anthropic_tools[0]["description"] == "Get weather"
        assert anthropic_tools[0]["input_schema"] == params

    def test_convert_multiple_tools(self):
        client = AnthropicClient(api_key="test")
        tools = [
            ToolDefinition.create("get_weather", "Get weather", {"type": "object"}),
            ToolDefinition.create("get_time", "Get time", {"type": "object"}),
        ]
        anthropic_tools = client._convert_tools(tools)

        assert len(anthropic_tools) == 2
        assert anthropic_tools[0]["name"] == "get_weather"
        assert anthropic_tools[1]["name"] == "get_time"


class TestToolChoiceConversion:
    def test_none_tool_choice(self):
        client = AnthropicClient(api_key="test")
        result = client._convert_tool_choice(None)
        assert result is None

    def test_auto_tool_choice(self):
        client = AnthropicClient(api_key="test")
        result = client._convert_tool_choice("auto")
        assert result == {"type": "auto"}

    def test_none_string_tool_choice(self):
        client = AnthropicClient(api_key="test")
        result = client._convert_tool_choice("none")
        assert result == {"type": "auto"}

    def test_force_tool_choice(self):
        client = AnthropicClient(api_key="test")
        tool_choice = ToolChoice.force("get_weather")
        result = client._convert_tool_choice(tool_choice)

        assert result == {"type": "tool", "name": "get_weather"}


class TestResponseParsing:
    def test_parse_text_response(self):
        client = AnthropicClient(api_key="test")
        response = MockAnthropicResponse(
            content=[MockTextBlock(text="Hello!")],
        )
        result = client._parse_response(response, "claude-sonnet-4-20250514")

        assert isinstance(result, GenerationResponse)
        assert result.model == "claude-sonnet-4-20250514"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].finish_reason == "stop"

    def test_parse_tool_use_response(self):
        client = AnthropicClient(api_key="test")
        response = MockAnthropicResponse(
            content=[MockToolUseBlock()],
            stop_reason="tool_use",
        )
        result = client._parse_response(response, "claude-sonnet-4-20250514")

        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1

        tool_call = result.choices[0].message.tool_calls[0]
        assert tool_call.id == "toolu_123"
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert args == {"city": "NYC"}

    def test_parse_mixed_response(self):
        client = AnthropicClient(api_key="test")
        response = MockAnthropicResponse(
            content=[
                MockTextBlock(text="Let me check that."),
                MockToolUseBlock(),
            ],
            stop_reason="tool_use",
        )
        result = client._parse_response(response, "claude-sonnet-4-20250514")

        assert result.choices[0].message.content == "Let me check that."
        assert result.choices[0].message.tool_calls is not None

    def test_parse_usage(self):
        client = AnthropicClient(api_key="test")
        response = MockAnthropicResponse(
            usage=MockUsage(input_tokens=200, output_tokens=100),
        )
        result = client._parse_response(response, "claude-sonnet-4-20250514")

        assert result.usage.prompt_tokens == 200
        assert result.usage.completion_tokens == 100
        assert result.usage.total_tokens == 300

    def test_stop_reason_mapping(self):
        client = AnthropicClient(api_key="test")

        test_cases = [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
            ("tool_use", "tool_calls"),
        ]
        for stop_reason, expected_finish in test_cases:
            response = MockAnthropicResponse(stop_reason=stop_reason)
            result = client._parse_response(response, "claude-sonnet-4-20250514")
            assert result.choices[0].finish_reason == expected_finish


class TestComplete:
    @pytest.mark.asyncio
    async def test_complete_basic(self):
        client = AnthropicClient(api_key="test")

        mock_response = MockAnthropicResponse(
            content=[MockTextBlock(text="Hello!")],
        )

        with patch.object(
            client._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            request = GenerationRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("Hello")],
            )
            result = await client.complete(request)

            assert isinstance(result, GenerationResponse)
            assert result.choices[0].message.content == "Hello!"

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"
            assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]
            assert "max_tokens" in call_kwargs

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self):
        client = AnthropicClient(api_key="test")

        mock_response = MockAnthropicResponse()

        with patch.object(
            client._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            request = GenerationRequest(
                model="claude-sonnet-4-20250514",
                messages=[
                    Message.system("Be helpful."),
                    Message.user("Hello"),
                ],
            )
            await client.complete(request)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["system"] == "Be helpful."

    @pytest.mark.asyncio
    async def test_complete_with_parameters(self):
        client = AnthropicClient(api_key="test")

        mock_response = MockAnthropicResponse()

        with patch.object(
            client._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            request = GenerationRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("Hello")],
                temperature=0.7,
                max_tokens=1000,
                top_p=0.9,
            )
            await client.complete(request)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 1000
            assert call_kwargs["top_p"] == 0.9


class TestCompleteWithTools:
    @pytest.mark.asyncio
    async def test_complete_with_tools_basic(self):
        client = AnthropicClient(api_key="test")

        mock_response = MockAnthropicResponse(
            content=[
                MockTextBlock(text="Let me get that."),
                MockToolUseBlock(),
            ],
            stop_reason="tool_use",
        )

        with patch.object(
            client._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            tool = ToolDefinition.create(
                "get_weather",
                "Get weather",
                {"type": "object", "properties": {"city": {"type": "string"}}},
            )
            request = GenerationRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("What's the weather in NYC?")],
                tools=[tool],
            )
            result = await client.complete_with_tools(request)

            call_kwargs = mock_create.call_args.kwargs
            assert "tools" in call_kwargs
            assert len(call_kwargs["tools"]) == 1
            assert call_kwargs["tools"][0]["name"] == "get_weather"
            assert "input_schema" in call_kwargs["tools"][0]

    @pytest.mark.asyncio
    async def test_complete_with_tool_choice(self):
        client = AnthropicClient(api_key="test")

        mock_response = MockAnthropicResponse()

        with patch.object(
            client._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_response

            tool = ToolDefinition.create("get_weather", "Get weather", {})
            request = GenerationRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("Hello")],
                tools=[tool],
                tool_choice=ToolChoice.force("get_weather"),
            )
            await client.complete_with_tools(request)

            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["tool_choice"] == {"type": "tool", "name": "get_weather"}


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_not_implemented(self):
        client = AnthropicClient(api_key="test")
        request = GenerationRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Hello")],
        )

        with pytest.raises(NotImplementedError):
            await client.stream(request)


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_roundtrip_with_tools(self):
        client = AnthropicClient(api_key="test")

        tool_response = MockAnthropicResponse(
            id="msg_tool",
            content=[
                MockTextBlock(text=""),
                MockToolUseBlock(
                    id="toolu_abc", name="get_weather", input={"city": "NYC"}
                ),
            ],
            stop_reason="tool_use",
            usage=MockUsage(input_tokens=50, output_tokens=20),
        )

        followup_response = MockAnthropicResponse(
            id="msg_followup",
            content=[MockTextBlock(text="The weather in NYC is sunny, 72F.")],
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=80, output_tokens=15),
        )

        with patch.object(
            client._client.messages, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = [tool_response, followup_response]

            tool = ToolDefinition.create(
                "get_weather",
                "Get weather for a city",
                {"type": "object", "properties": {"city": {"type": "string"}}},
            )

            request1 = GenerationRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("What's the weather in NYC?")],
                tools=[tool],
            )

            result1 = await client.complete_with_tools(request1)
            assert result1.choices[0].finish_reason == "tool_calls"
            tool_call = result1.choices[0].message.tool_calls[0]
            assert tool_call.function.name == "get_weather"

            tool_result_msg = Message.tool_result(
                tool_call.id, '{"temp": 72, "condition": "sunny"}'
            )

            request2 = GenerationRequest(
                model="claude-sonnet-4-20250514",
                messages=[
                    Message.user("What's the weather in NYC?"),
                    result1.choices[0].message,
                    tool_result_msg,
                ],
            )

            result2 = await client.complete(request2)
            assert (
                result2.choices[0].message.content
                == "The weather in NYC is sunny, 72F."
            )
