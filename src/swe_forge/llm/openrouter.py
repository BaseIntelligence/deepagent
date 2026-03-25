"""OpenRouter provider implementation for LLM requests.

OpenRouter provides a unified API for accessing multiple LLM providers
through a single endpoint, making it ideal for multi-model routing scenarios.
"""

import logging
from typing import Any

import aiohttp
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from swe_forge.llm.client import (
    Choice,
    Choice as ClientChoice,
    FunctionCall,
    GenerationRequest,
    GenerationResponse,
    LLMClient,
    Message,
    ToolCall,
    ToolDefinition,
    Usage,
)

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 300


class RateLimitError(Exception):
    """Exception raised when rate limit is hit (429 response)."""

    pass


class APIError(Exception):
    """Exception raised for API errors."""

    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"API Error {code}: {message}")


class OpenRouterClient(LLMClient):
    """OpenRouter provider for LLM requests.

    This provider implements the LLMClient interface and routes requests
    through OpenRouter's API, which provides access to multiple LLM providers.

    Attributes:
        api_key: OpenRouter API key for authentication.
        base_url: Base URL for the OpenRouter API.
        default_model: Default model to use when none is specified.
        session: aiohttp ClientSession for making requests.

    Example:
        >>> client = OpenRouterClient(api_key="sk-or-...")
        >>> request = GenerationRequest(
        ...     model="anthropic/claude-3-opus",
        ...     messages=[Message.user("Hello!")]
        ... )
        >>> response = await client.complete(request)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        session: aiohttp.ClientSession | None = None,
    ):
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key for authentication.
            base_url: Base URL for the OpenRouter API (for testing).
            default_model: Default model to use when none is specified.
            session: Optional aiohttp ClientSession (will create one if not provided).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self._session = session
        self._owns_session = session is None

    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    def _get_headers(self) -> dict[str, str]:
        """Get headers for OpenRouter API requests."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://swe-forge.local",
            "X-Title": "swe_forge",
        }

    def _build_request_body(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the request body for the OpenRouter API.

        Args:
            request: The generation request.

        Returns:
            A dictionary suitable for JSON serialization.
        """
        model = request.model or self.default_model
        if model == "default" or not model:
            model = self.default_model

        body: dict[str, Any] = {
            "model": model,
            "messages": [msg.model_dump() for msg in request.messages],
        }

        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p

        if request.tools:
            body["tools"] = [
                {"type": tool.type, "function": tool.function.model_dump()}
                for tool in request.tools
            ]

        if request.tool_choice is not None:
            if isinstance(request.tool_choice, str):
                body["tool_choice"] = request.tool_choice
            else:
                body["tool_choice"] = request.tool_choice.model_dump()

        return body

    def _parse_response(self, data: dict[str, Any]) -> GenerationResponse:
        """Parse the OpenRouter API response.

        Args:
            data: The JSON response from the API.

        Returns:
            A GenerationResponse object.
        """
        choices = []
        for choice_data in data.get("choices", []):
            message_data = choice_data.get("message", {})

            tool_calls = None
            if message_data.get("tool_calls"):
                tool_calls = []
                for tc in message_data["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            function=FunctionCall(
                                name=tc["function"]["name"],
                                arguments=tc["function"].get("arguments", "{}"),
                            ),
                        )
                    )

            # Use first tool call arguments as content for structured output backcompat
            content = message_data.get("content", "")
            if not content and tool_calls:
                first_call = tool_calls[0]
                if first_call.function.arguments:
                    content = first_call.function.arguments

            message = Message(
                role=message_data.get("role", "assistant"),
                content=content,
                tool_calls=tool_calls,
            )

            choices.append(
                Choice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason", "stop"),
                )
            )

        usage_data = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return GenerationResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
        )

    def _is_transient_error(self, error: Exception) -> bool:
        """Check if an error is transient and should be retried.

        Args:
            error: The exception to check.

        Returns:
            True if the error is transient, False otherwise.
        """
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIError):
            return error.code >= 500 or error.code == 429
        if isinstance(error, aiohttp.ClientError):
            return True
        return False

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    async def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response for the given request.

        Args:
            request: The generation request containing messages and parameters.

        Returns:
            A GenerationResponse with the model's output.

        Raises:
            RateLimitError: If rate limited and retries exhausted.
            APIError: If the API returns an error.
            aiohttp.ClientError: If there's a network error.
        """
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        body = self._build_request_body(request)

        logger.debug(f"Sending request to OpenRouter: model={body['model']}")

        async with self.session.post(url, json=body, headers=headers) as response:
            status = response.status

            if status == 429:
                error_text = await response.text()
                logger.warning(f"Rate limited by OpenRouter: {error_text}")
                raise RateLimitError(f"Rate limited: {error_text}")

            if not response.ok:
                error_text = await response.text()
                logger.error(f"OpenRouter API error {status}: {error_text}")

                try:
                    import json

                    error_data = json.loads(error_text)
                    if "error" in error_data:
                        message = error_data["error"].get("message", error_text)
                    else:
                        message = error_text
                except json.JSONDecodeError:
                    message = error_text

                raise APIError(status, message)

            data = await response.json()

        return self._parse_response(data)

    async def complete_with_tools(
        self, request: GenerationRequest
    ) -> GenerationResponse:
        """Generate a response with tool calling support.

        This method is similar to complete() but specifically handles
        requests that include tools for function calling.

        Args:
            request: The generation request with tools parameter.

        Returns:
            A GenerationResponse potentially including tool_calls.
        """
        return await self.complete(request)

    async def stream(self, request: GenerationRequest) -> None:
        """Stream a response for the given request.

        Note: This method is not implemented for OpenRouter in this version.
        Streaming support will be added in a future update.

        Args:
            request: The generation request.

        Raises:
            NotImplementedError: Always raised as streaming is not yet implemented.
        """
        raise NotImplementedError(
            "Streaming is not yet implemented for OpenRouter client"
        )

    def api_key_masked(self) -> str:
        """Get the masked API key for debugging.

        Returns:
            A masked version of the API key (e.g., 'sk-o...abc').
        """
        if len(self.api_key) <= 8:
            return "*" * len(self.api_key)
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    async def __aenter__(self) -> "OpenRouterClient":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager and close the session."""
        await self.close()
