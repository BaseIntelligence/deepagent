"""Tests for OpenRouter client implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import aiohttp

from swe_forge.llm import (
    GenerationRequest,
    Message,
    ToolDefinition,
)
from swe_forge.llm.openrouter import (
    APIError,
    DEFAULT_MODEL,
    MAX_RETRIES,
    OPENROUTER_BASE_URL,
    RateLimitError,
    OpenRouterClient,
)


class TestOpenRouterClientInit:
    def test_init_with_defaults(self):
        client = OpenRouterClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.base_url == OPENROUTER_BASE_URL
        assert client.default_model == DEFAULT_MODEL

    def test_init_with_custom_values(self):
        client = OpenRouterClient(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
            default_model="custom/model",
        )
        assert client.api_key == "test-key"
        assert client.base_url == "https://custom.api.com/v1"
        assert client.default_model == "custom/model"

    def test_api_key_masked_short(self):
        client = OpenRouterClient(api_key="abc")
        assert client.api_key_masked() == "***"

    def test_api_key_masked_normal(self):
        client = OpenRouterClient(api_key="sk-1234567890abcdef")
        assert client.api_key_masked() == "sk-1...cdef"


class TestGetHeaders:
    def test_headers_include_auth(self):
        client = OpenRouterClient(api_key="test-api-key")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer test-api-key"
        assert headers["Content-Type"] == "application/json"
        assert headers["HTTP-Referer"] == "https://swe-forge.local"
        assert headers["X-Title"] == "swe_forge"


class TestBuildRequestBody:
    def test_basic_request(self):
        client = OpenRouterClient(api_key="test-key")
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
        )
        body = client._build_request_body(request)

        assert body["model"] == "gpt-4"
        assert len(body["messages"]) == 1
        assert "temperature" not in body
        assert "max_tokens" not in body

    def test_request_with_all_parameters(self):
        client = OpenRouterClient(api_key="test-key")
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )
        body = client._build_request_body(request)

        assert body["model"] == "gpt-4"
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 1000
        assert body["top_p"] == 0.9

    def test_request_uses_default_model_when_empty(self):
        client = OpenRouterClient(api_key="test-key", default_model="custom-default")
        request = GenerationRequest(model="", messages=[Message.user("Hello")])
        body = client._build_request_body(request)
        assert body["model"] == "custom-default"

    def test_request_uses_default_model_when_default_string(self):
        client = OpenRouterClient(api_key="test-key", default_model="custom-default")
        request = GenerationRequest(model="default", messages=[Message.user("Hello")])
        body = client._build_request_body(request)
        assert body["model"] == "custom-default"

    def test_request_with_tools(self):
        client = OpenRouterClient(api_key="test-key")
        tool = ToolDefinition.create(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
            tools=[tool],
        )
        body = client._build_request_body(request)

        assert "tools" in body
        assert len(body["tools"]) == 1
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "get_weather"

    def test_request_with_tool_choice_string(self):
        client = OpenRouterClient(api_key="test-key")
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
            tool_choice="auto",
        )
        body = client._build_request_body(request)
        assert body["tool_choice"] == "auto"


class TestParseResponse:
    def test_parse_simple_response(self):
        client = OpenRouterClient(api_key="test-key")
        data = {
            "id": "gen-123",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        response = client._parse_response(data)

        assert response.id == "gen-123"
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    def test_parse_response_with_tool_calls(self):
        client = OpenRouterClient(api_key="test-key")
        data = {
            "id": "gen-123",
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        response = client._parse_response(data)

        assert len(response.choices) == 1
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.id == "call_abc"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == '{"city": "NYC"}'
        assert response.choices[0].message.content == '{"city": "NYC"}'

    def test_parse_response_empty_choices(self):
        client = OpenRouterClient(api_key="test-key")
        data = {
            "id": "gen-123",
            "model": "gpt-4",
            "choices": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        response = client._parse_response(data)
        assert len(response.choices) == 0


class TestIsTransientError:
    def test_rate_limit_is_transient(self):
        client = OpenRouterClient(api_key="test-key")
        assert client._is_transient_error(RateLimitError("Rate limited")) is True

    def test_api_error_500_is_transient(self):
        client = OpenRouterClient(api_key="test-key")
        assert client._is_transient_error(APIError(500, "Server error")) is True

    def test_api_error_429_is_transient(self):
        client = OpenRouterClient(api_key="test-key")
        assert client._is_transient_error(APIError(429, "Rate limited")) is True

    def test_api_error_400_is_not_transient(self):
        client = OpenRouterClient(api_key="test-key")
        assert client._is_transient_error(APIError(400, "Bad request")) is False

    def test_api_error_401_is_not_transient(self):
        client = OpenRouterClient(api_key="test-key")
        assert client._is_transient_error(APIError(401, "Unauthorized")) is False

    def test_client_error_is_transient(self):
        client = OpenRouterClient(api_key="test-key")
        assert client._is_transient_error(aiohttp.ClientError()) is True


@pytest.mark.asyncio
class TestComplete:
    async def test_complete_success(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.ok = True
        mock_response.json = AsyncMock(
            return_value={
                "id": "gen-123",
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        mock_session.closed = False

        client = OpenRouterClient(api_key="test-key", session=mock_session)
        client._session = mock_session
        client._owns_session = False

        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
        )

        response = await client.complete(request)

        assert response.id == "gen-123"
        assert response.model == "gpt-4"
        assert response.first_content() == "Hello!"

    async def test_complete_raises_on_rate_limit(self):
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value="Rate limited")

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        mock_session.closed = False

        client = OpenRouterClient(api_key="test-key", session=mock_session)
        client._session = mock_session
        client._owns_session = False

        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
        )

        with pytest.raises(RateLimitError):
            await client.complete(request)

    async def test_complete_raises_on_api_error(self):
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.ok = False
        mock_response.text = AsyncMock(
            return_value='{"error": {"message": "Bad request"}}'
        )

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        mock_session.closed = False

        client = OpenRouterClient(api_key="test-key", session=mock_session)
        client._session = mock_session
        client._owns_session = False

        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
        )

        with pytest.raises(APIError) as exc_info:
            await client.complete(request)

        assert exc_info.value.code == 400


@pytest.mark.asyncio
class TestCompleteWithTools:
    async def test_complete_with_tools_delegates(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.ok = True
        mock_response.json = AsyncMock(
            return_value={
                "id": "gen-123",
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_xyz",
                                    "type": "function",
                                    "function": {
                                        "name": "test_func",
                                        "arguments": '{"arg": "value"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_cm)
        mock_session.closed = False

        client = OpenRouterClient(api_key="test-key", session=mock_session)
        client._session = mock_session
        client._owns_session = False

        tool = ToolDefinition.create(
            name="test_func",
            description="Test function",
            parameters={"type": "object"},
        )
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
            tools=[tool],
        )

        response = await client.complete_with_tools(request)

        assert response.choices[0].message.tool_calls is not None
        assert response.choices[0].message.tool_calls[0].function.name == "test_func"


@pytest.mark.asyncio
class TestStream:
    async def test_stream_not_implemented(self):
        client = OpenRouterClient(api_key="test-key")
        request = GenerationRequest(
            model="gpt-4",
            messages=[Message.user("Hello")],
        )

        with pytest.raises(NotImplementedError) as exc_info:
            await client.stream(request)

        assert "not yet implemented" in str(exc_info.value).lower()


@pytest.mark.asyncio
class TestContextManager:
    async def test_context_manager_closes_session(self):
        client = OpenRouterClient(api_key="test-key")

        async with client as c:
            assert c is client
            assert client._session is not None or client._owns_session

    async def test_explicit_close(self):
        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock()

        client = OpenRouterClient(api_key="test-key", session=mock_session)
        client._owns_session = True

        await client.close()

        mock_session.close.assert_called_once()


class TestExceptions:
    def test_rate_limit_error_message(self):
        error = RateLimitError("Too many requests")
        assert str(error) == "Too many requests"

    def test_api_error_attributes(self):
        error = APIError(404, "Not found")
        assert error.code == 404
        assert error.message == "Not found"
        assert "404" in str(error)
        assert "Not found" in str(error)


class TestConstants:
    def test_openrouter_base_url(self):
        assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"

    def test_default_model(self):
        assert DEFAULT_MODEL == "openai/gpt-4o-mini"

    def test_max_retries(self):
        assert MAX_RETRIES == 3
