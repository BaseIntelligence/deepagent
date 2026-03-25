"""LLM response caching for reducing API calls and latency.

This module provides an in-memory LRU cache for LLM responses, enabling
efficient reuse of responses for identical or similar prompts.

# Usage

```python
from swe_forge.llm.cache import LLMCache
from swe_forge.llm.client import GenerationRequest, Message

cache = LLMCache(max_size=1000)

# Check cache
response = cache.get(request)
if response is None:
    # Make API call and cache result
    response = await client.complete(request)
    cache.set(request, response)

# Get stats
stats = cache.stats()
print(f"Hit rate: {stats.hit_rate():.2%}")
```
"""

import asyncio
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .client import GenerationRequest, GenerationResponse


@dataclass
class CacheStats:
    """Statistics for cache monitoring.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        tokens_saved: Estimated tokens saved from cache hits
    """

    hits: int = 0
    misses: int = 0
    tokens_saved: int = 0

    def hit_rate(self) -> float:
        """Calculate the cache hit rate.

        Returns:
            Hit rate as a value between 0.0 and 1.0
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def total_accesses(self) -> int:
        """Get total cache accesses."""
        return self.hits + self.misses


@dataclass
class CacheEntry:
    """A cached response entry.

    Attributes:
        response: The cached generation response
        created_at: Timestamp when entry was created (reserved for TTL)
    """

    response: GenerationResponse
    created_at: float = field(default_factory=lambda: 0.0)


def _hash_messages(messages: list[Any]) -> str:
    """Hash a list of messages for cache key.

    Args:
        messages: List of Message objects

    Returns:
        SHA256 hash string (truncated to 16 chars for readability)
    """
    # Convert messages to JSON-serializable format
    messages_data = [m.model_dump() for m in messages]
    messages_json = json.dumps(messages_data, sort_keys=True)
    return hashlib.sha256(messages_json.encode()).hexdigest()[:16]


def _make_key(request: GenerationRequest) -> str:
    """Generate a cache key from a generation request.

    The key includes all parameters that affect the response:
    - Model name
    - Messages hash
    - Temperature
    - Max tokens
    - Top-p
    - Tools (if any)
    - Tool choice (if any)

    Args:
        request: The generation request

    Returns:
        A unique string key for caching
    """
    key_data: dict[str, Any] = {
        "model": request.model,
        "messages_hash": _hash_messages(request.messages),
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p,
    }

    # Include tools if present
    if request.tools:
        tools_data = [t.model_dump() for t in request.tools]
        key_data["tools_hash"] = hashlib.sha256(
            json.dumps(tools_data, sort_keys=True).encode()
        ).hexdigest()[:16]

    # Include tool_choice if present
    if request.tool_choice:
        if isinstance(request.tool_choice, str):
            key_data["tool_choice"] = request.tool_choice
        else:
            key_data["tool_choice"] = request.tool_choice.model_dump()

    return json.dumps(key_data, sort_keys=True)


class LLMCache:
    """Thread-safe LRU cache for LLM responses.

    Uses OrderedDict for LRU eviction and asyncio.Lock for thread-safe
    async operations.

    Attributes:
        max_size: Maximum number of entries to store

    Example:
        ```python
        cache = LLMCache(max_size=500)

        # Check cache
        cached = cache.get(request)
        if cached is None:
            response = await client.complete(request)
            cache.set(request, response)

        # Use as async context manager
        async with cache.lock():
            response = cache.get(request)
        ```
    """

    def __init__(self, max_size: int = 1000):
        """Initialize the cache.

        Args:
            max_size: Maximum number of entries (default 1000)
        """
        self._max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    @property
    def max_size(self) -> int:
        """Get the maximum cache size."""
        return self._max_size

    def _evict_oldest(self) -> None:
        """Evict the oldest (LRU) entry from the cache."""
        if self._cache:
            self._cache.popitem(last=False)

    def get(self, request: GenerationRequest) -> GenerationResponse | None:
        """Get a cached response for a request.

        Args:
            request: The generation request to look up

        Returns:
            The cached response, or None if not found
        """
        key = _make_key(request)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._stats.hits += 1

            entry = self._cache[key]
            # Update tokens saved (approximate)
            self._stats.tokens_saved += entry.response.usage.total_tokens

            return entry.response

        self._stats.misses += 1
        return None

    def set(self, request: GenerationRequest, response: GenerationResponse) -> None:
        """Cache a response for a request.

        Args:
            request: The generation request (used for key)
            response: The response to cache
        """
        key = _make_key(request)

        # If key exists, update and move to end
        if key in self._cache:
            self._cache[key] = CacheEntry(response=response)
            self._cache.move_to_end(key)
            return

        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            self._evict_oldest()

        self._cache[key] = CacheEntry(response=response)

    def contains(self, request: GenerationRequest) -> bool:
        """Check if a request is in the cache.

        Args:
            request: The generation request to check

        Returns:
            True if the request is cached
        """
        key = _make_key(request)
        return key in self._cache

    def stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            A CacheStats instance with current statistics
        """
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            tokens_saved=self._stats.tokens_saved,
        )

    def clear(self) -> None:
        """Clear all entries from the cache.

        Statistics are preserved; only the cache entries are removed.
        """
        self._cache.clear()

    def len(self) -> int:
        """Get the number of cached entries."""
        return len(self._cache)

    def is_empty(self) -> bool:
        """Check if the cache is empty."""
        return len(self._cache) == 0

    def lock(self) -> asyncio.Lock:
        """Get the async lock for thread-safe operations.

        Returns:
            The asyncio.Lock for this cache

        Example:
            ```python
            async with cache.lock():
                response = cache.get(request)
                if response is None:
                    response = await client.complete(request)
                    cache.set(request, response)
            ```
        """
        return self._lock

    async def get_or_set(
        self,
        request: GenerationRequest,
        factory: Awaitable[GenerationResponse]
        | Callable[[], Awaitable[GenerationResponse]],
    ) -> GenerationResponse:
        """Get a cached response or compute and cache it.

        This is a convenience method that handles the common pattern of
        checking cache, computing if missing, and caching the result.

        Args:
            request: The generation request
            factory: An async callable that returns the response

        Returns:
            The cached or newly computed response

        Example:
            ```python
            response = await cache.get_or_set(
                request,
                lambda: client.complete(request)
            )
            ```
        """
        async with self._lock:
            response = self.get(request)
            if response is not None:
                return response

            # Call the factory - handle both coroutine and awaitable
            if asyncio.iscoroutine(factory):
                response = await factory
            else:
                response = await factory()

            self.set(request, response)
            return response


def cached_response(
    cache: LLMCache,
) -> Callable[
    [Callable[..., Awaitable[GenerationResponse]]],
    Callable[..., Awaitable[GenerationResponse]],
]:
    """Decorator to cache LLM responses.

    This is a convenience decorator for wrapping LLM client methods.

    Args:
        cache: The LLMCache instance to use

    Returns:
        A decorator function

    Example:
        ```python
        cache = LLMCache(max_size=1000)

        @cached_response(cache)
        async def complete(request: GenerationRequest) -> GenerationResponse:
            return await api_client.complete(request)
        ```
    """
    from functools import wraps

    def decorator(
        func: Callable[..., Awaitable[GenerationResponse]],
    ) -> Callable[..., Awaitable[GenerationResponse]]:
        @wraps(func)
        async def wrapper(
            request: GenerationRequest, *args: Any, **kwargs: Any
        ) -> GenerationResponse:
            response = cache.get(request)
            if response is not None:
                return response

            response = await func(request, *args, **kwargs)
            cache.set(request, response)
            return response

        return wrapper

    return decorator
