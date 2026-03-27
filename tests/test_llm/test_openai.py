"""Tests for OpenAI client implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from swe_forge.llm import (
    GenerationRequest,
    GenerationResponse,
    Message,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)
from swe_forge.llm.client import FunctionCall
from swe_forge.llm.openai_client import OpenAIClient


class MockChatCompletion:
    """Mock OpenAI ChatCompletion response."""

    def __init__(
        self,
        id: str = "chatcmpl-test123",
        model: str = "gpt-4",
        content: str = "Hello!",
        tool_calls: list | None = None,
        finish_reason: str = "stop",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        total_tokens: int = 15,
    ):
        self.id = id
        self.model = model
        self.choices = [
            MockChoice(
                index=0,
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        ]
        self.usage = MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class MockChoice:
    """Mock OpenAI Choice."""

    def __init__(
        self,
        index: int,
        content: str,
        tool_calls: list | None = None,
        finish_reason: str = "stop",
    ):
        self.index = index
        self.message = MockMessage(content=content, tool_calls=tool_calls)
        self.finish_reason = finish_reason


class MockMessage:
    """Mock OpenAI Message."""

    def __init__(self, content: str, tool_calls: list | None = None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class MockToolCall:
    """Mock OpenAI ToolCall."""

    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = MockFunction(name=name, arguments=arguments)


class MockFunction:
    """Mock OpenAI Function."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class MockUsage:
    """Mock OpenAI Usage."""

    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockStreamingChunk:
    """Mock streaming chunk."""

    def __init__(self, content: str | None):
        self.choices = [MockStreamingChoice(content=content)]


class MockStreamingChoice:
    """Mock streaming choice."""

    def __init__(self, content: str | None):
        self.delta = MockDelta(content=content)


class MockDelta:
    """Mock streaming delta."""

    def __init__(self, content: str | None):
        self.content = content


@pytest.fixture
def openai_client():
    """Create OpenAIClient with mocked AsyncOpenAI."""
    with patch("swe_forge.llm.openai_client.AsyncOpenAI") as mock_async_openai:
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client
        client = OpenAIClient(api_key="test-key")
        client._mock_client = mock_client
        yield client


class TestOpenAIClientInit:
    def test_init_with_api_key(self):
        with patch("swe_forge.llm.openai_client.AsyncOpenAI") as mock_async_openai:
            client = OpenAIClient(api_key="test-key")
            mock_async_openai.assert_called_once_with(api_key="test-key", base_url=None)
            assert client._client is not None

    def test_init_with_base_url(self):
        with patch("swe_forge.llm.openai_client.AsyncOpenAI") as mock_async_openai:
            OpenAIClient(api_key="test-key", base_url="https://custom.api")
            mock_async_openai.assert_called_once_with(
                api_key="test-key", base_url="https://custom.api"
            )


class TestOpenAIClientComplete:
    @pytest.mark.asyncio
    async def test_complete_basic(self, openai_client):
        mock_response = MockChatCompletion(content="Hello, world!")
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = GenerationRequest(model="gpt-4", messages=[Message.user("Say hello")])
        response = await openai_client.complete(request)

        assert isinstance(response, GenerationResponse)
        assert response.model == "gpt-4"
        assert response.first_content() == "Hello, world!"
        assert response.usage.prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_complete_with_temperature(self, openai_client):
        mock_response = MockChatCompletion(content="Random response")
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = GenerationRequest(
            model="gpt-4", messages=[Message.user("Hi")], temperature=0.5
        )
        await openai_client.complete(request)

        call_kwargs = openai_client._mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens(self, openai_client):
        mock_response = MockChatCompletion(content="Truncated")
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = GenerationRequest(
            model="gpt-4", messages=[Message.user("Hi")], max_tokens=100
        )
        await openai_client.complete(request)

        call_kwargs = openai_client._mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_with_top_p(self, openai_client):
        mock_response = MockChatCompletion(content="Top-p response")
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = GenerationRequest(
            model="gpt-4", messages=[Message.user("Hi")], top_p=0.9
        )
        await openai_client.complete(request)

        call_kwargs = openai_client._mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self, openai_client):
        mock_response = MockChatCompletion(content="I'm helpful!")
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = GenerationRequest(
            model="gpt-4",
            messages=[
                Message.system("You are helpful."),
                Message.user("Hello"),
            ],
        )
        await openai_client.complete(request)

        call_kwargs = openai_client._mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


class TestOpenAIClientCompleteWithTools:
    @pytest.mark.asyncio
    async def test_complete_with_tools_basic(self, openai_client):
        mock_response = MockChatCompletion(content="Let me check that.")
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        tool = ToolDefinition.create(
            name="get_weather",
            description="Get weather info",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("What's the weather in Tokyo?")],
            tools=[tool],
            tool_choice="auto",
        )
        response = await openai_client.complete_with_tools(request)

        assert isinstance(response, GenerationResponse)
        call_kwargs = openai_client._mock_client.chat.completions.create.call_args
        assert "tools" in call_kwargs.kwargs
        assert call_kwargs.kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_complete_with_tools_returns_tool_call(self, openai_client):
        mock_tool_call = MockToolCall(
            id="call_123", name="get_weather", arguments='{"city": "Tokyo"}'
        )
        mock_response = MockChatCompletion(
            content="", tool_calls=[mock_tool_call], finish_reason="tool_calls"
        )
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        tool = ToolDefinition.create(
            name="get_weather",
            description="Get weather info",
            parameters={"type": "object"},
        )
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("What's the weather?")],
            tools=[tool],
        )
        response = await openai_client.complete_with_tools(request)

        assert len(response.choices) == 1
        msg = response.choices[0].message
        assert msg.content == ""
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_123"
        assert msg.tool_calls[0].function.name == "get_weather"
        assert msg.tool_calls[0].function.arguments == '{"city": "Tokyo"}'

    @pytest.mark.asyncio
    async def test_complete_with_tools_forced(self, openai_client):
        mock_response = MockChatCompletion(content="Forced tool call")
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        tool = ToolDefinition.create(
            name="get_time", description="Get current time", parameters={}
        )
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("What time is it?")],
            tools=[tool],
            tool_choice=ToolChoice.force("get_time"),
        )
        await openai_client.complete_with_tools(request)

        call_kwargs = openai_client._mock_client.chat.completions.create.call_args
        tool_choice_arg = call_kwargs.kwargs["tool_choice"]
        assert tool_choice_arg["type"] == "function"
        assert tool_choice_arg["function"]["name"] == "get_time"

    @pytest.mark.asyncio
    async def test_complete_with_tool_choice_none(self, openai_client):
        mock_response = MockChatCompletion(content="Direct response")
        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hi")],
            tool_choice="none",
        )
        await openai_client.complete_with_tools(request)

        call_kwargs = openai_client._mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["tool_choice"] == "none"


class TestOpenAIClientStream:
    @pytest.mark.asyncio
    async def test_stream_basic(self, openai_client):
        async def mock_stream():
            for chunk_content in ["Hello", ", ", "world", "!"]:
                yield MockStreamingChunk(content=chunk_content)
            yield MockStreamingChunk(content=None)

        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        request = GenerationRequest(model="gpt-4", messages=[Message.user("Say hello")])

        chunks = []
        async for chunk in openai_client.stream(request):
            chunks.append(chunk)

        assert chunks == ["Hello", ", ", "world", "!"]

    @pytest.mark.asyncio
    async def test_stream_with_parameters(self, openai_client):
        async def mock_stream():
            yield MockStreamingChunk(content="test")

        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hi")],
            temperature=0.7,
            max_tokens=50,
            top_p=0.8,
        )

        async for _ in openai_client.stream(request):
            pass

        call_kwargs = openai_client._mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["stream"] is True
        assert call_kwargs.kwargs["temperature"] == 0.7
        assert call_kwargs.kwargs["max_tokens"] == 50
        assert call_kwargs.kwargs["top_p"] == 0.8

    @pytest.mark.asyncio
    async def test_stream_empty_chunks(self, openai_client):
        async def mock_stream():
            yield MockStreamingChunk(content=None)
            yield MockStreamingChunk(content="Hello")
            yield MockStreamingChunk(content=None)

        openai_client._mock_client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        request = GenerationRequest(model="gpt-4", messages=[Message.user("Hi")])

        chunks = []
        async for chunk in openai_client.stream(request):
            chunks.append(chunk)

        assert chunks == ["Hello"]


class TestMessageConversion:
    def test_convert_messages_basic(self, openai_client):
        messages = [
            Message.system("System prompt"),
            Message.user("User message"),
            Message.assistant("Assistant response"),
        ]

        converted = openai_client._convert_messages(messages)

        assert len(converted) == 3
        assert converted[0] == {"role": "system", "content": "System prompt"}
        assert converted[1] == {"role": "user", "content": "User message"}
        assert converted[2] == {"role": "assistant", "content": "Assistant response"}

    def test_convert_messages_with_tool_calls(self, openai_client):
        tool_call = ToolCall(
            id="call_123",
            type="function",
            function=FunctionCall(name="test", arguments='{"arg": "val"}'),
        )
        messages = [Message.assistant_with_tool_calls("", [tool_call])]

        converted = openai_client._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        assert "tool_calls" in converted[0]
        assert converted[0]["tool_calls"][0]["id"] == "call_123"

    def test_convert_messages_with_tool_result(self, openai_client):
        messages = [Message.tool_result("call_123", '{"result": "ok"}')]

        converted = openai_client._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "tool"
        assert converted[0]["content"] == '{"result": "ok"}'
        assert converted[0]["tool_call_id"] == "call_123"


class TestToolConversion:
    def test_convert_tools(self, openai_client):
        tools = [
            ToolDefinition.create(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )
        ]

        converted = openai_client._convert_tools(tools)

        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "get_weather"
        assert converted[0]["function"]["description"] == "Get weather"
        assert "city" in converted[0]["function"]["parameters"]["properties"]

    def test_convert_tool_choice_auto(self, openai_client):
        result = openai_client._convert_tool_choice("auto")
        assert result == "auto"

    def test_convert_tool_choice_none(self, openai_client):
        result = openai_client._convert_tool_choice("none")
        assert result == "none"

    def test_convert_tool_choice_forced(self, openai_client):
        choice = ToolChoice.force("get_weather")
        result = openai_client._convert_tool_choice(choice)

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"


class TestResponseConversion:
    def test_convert_response(self, openai_client):
        mock_response = MockChatCompletion(
            id="test-id",
            model="gpt-4",
            content="Test response",
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        )

        result = openai_client._convert_response(mock_response)

        assert isinstance(result, GenerationResponse)
        assert result.id == "test-id"
        assert result.model == "gpt-4"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Test response"
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 30

    def test_convert_response_with_tool_calls(self, openai_client):
        mock_tool_call = MockToolCall(
            id="tc_123", name="test_func", arguments='{"x": 1}'
        )
        mock_response = MockChatCompletion(
            content="", tool_calls=[mock_tool_call], finish_reason="tool_calls"
        )

        result = openai_client._convert_response(mock_response)

        assert len(result.choices) == 1
        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls is not None
        assert len(result.choices[0].message.tool_calls) == 1
        assert result.choices[0].message.tool_calls[0].id == "tc_123"


class TestOpenAIClientClose:
    @pytest.mark.asyncio
    async def test_close_calls_client_close(self, openai_client):
        openai_client._mock_client.close = AsyncMock()

        await openai_client.close()

        openai_client._mock_client.close.assert_called_once()
