"""Tests for LLM response caching."""

import pytest

from swe_forge.llm.cache import (
    CacheStats,
    CacheEntry,
    LLMCache,
    _hash_messages,
    _make_key,
    cached_response,
)
from swe_forge.llm.client import (
    GenerationRequest,
    GenerationResponse,
    Message,
    Choice,
    Usage,
)


def make_response(content: str, model: str = "gpt-4") -> GenerationResponse:
    """Helper to create a generation response."""
    return GenerationResponse(
        id="test-id",
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message.assistant(content),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def make_request(
    content: str = "Hello",
    model: str = "gpt-4",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GenerationRequest:
    """Helper to create a generation request."""
    return GenerationRequest(
        model=model,
        messages=[Message.user(content)],
        temperature=temperature,
        max_tokens=max_tokens,
    )


class TestHashMessages:
    """Tests for _hash_messages function."""

    def test_deterministic_hash(self):
        """Same messages should produce same hash."""
        messages = [Message.user("Hello")]
        hash1 = _hash_messages(messages)
        hash2 = _hash_messages(messages)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_different_messages_different_hash(self):
        """Different messages should produce different hashes."""
        hash1 = _hash_messages([Message.user("Hello")])
        hash2 = _hash_messages([Message.user("World")])
        assert hash1 != hash2

    def test_same_content_different_role_different_hash(self):
        """Same content with different roles should produce different hashes."""
        hash1 = _hash_messages([Message.user("test")])
        hash2 = _hash_messages([Message.system("test")])
        assert hash1 != hash2

    def test_multiple_messages(self):
        """Should handle multiple messages correctly."""
        messages = [
            Message.system("You are helpful"),
            Message.user("Hello"),
        ]
        hash1 = _hash_messages(messages)
        hash2 = _hash_messages(messages)
        assert hash1 == hash2


class TestMakeKey:
    """Tests for _make_key function."""

    def test_same_request_same_key(self):
        """Identical requests should produce same key."""
        req = make_request("Hello")
        key1 = _make_key(req)
        key2 = _make_key(req)
        assert key1 == key2

    def test_different_content_different_key(self):
        """Different content should produce different keys."""
        req1 = make_request("Hello")
        req2 = make_request("World")
        assert _make_key(req1) != _make_key(req2)

    def test_different_model_different_key(self):
        """Different model should produce different keys."""
        req1 = make_request("Hello", model="gpt-4")
        req2 = make_request("Hello", model="gpt-3.5")
        assert _make_key(req1) != _make_key(req2)

    def test_different_temperature_different_key(self):
        """Different temperature should produce different keys."""
        req1 = make_request("Hello", temperature=0.5)
        req2 = make_request("Hello", temperature=0.7)
        assert _make_key(req1) != _make_key(req2)

    def test_different_max_tokens_different_key(self):
        """Different max_tokens should produce different keys."""
        req1 = make_request("Hello", max_tokens=100)
        req2 = make_request("Hello", max_tokens=200)
        assert _make_key(req1) != _make_key(req2)


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_default_values(self):
        """Default stats should be zeros."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.tokens_saved == 0

    def test_hit_rate_no_accesses(self):
        """Hit rate should be 0.0 when no accesses."""
        stats = CacheStats()
        assert stats.hit_rate() == 0.0

    def test_hit_rate_all_misses(self):
        """Hit rate should be 0.0 when all misses."""
        stats = CacheStats(hits=0, misses=10)
        assert stats.hit_rate() == 0.0

    def test_hit_rate_all_hits(self):
        """Hit rate should be 1.0 when all hits."""
        stats = CacheStats(hits=10, misses=0)
        assert stats.hit_rate() == 1.0

    def test_hit_rate_50_percent(self):
        """Hit rate should be 0.5 when half hits."""
        stats = CacheStats(hits=5, misses=5)
        assert stats.hit_rate() == 0.5

    def test_total_accesses(self):
        """Total accesses should be hits + misses."""
        stats = CacheStats(hits=3, misses=7)
        assert stats.total_accesses() == 10


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Should create entry with response."""
        response = make_response("test")
        entry = CacheEntry(response=response)
        assert entry.response == response


class TestLLMCache:
    """Tests for LLMCache class."""

    def test_init_default_size(self):
        """Default max_size should be 1000."""
        cache = LLMCache()
        assert cache.max_size == 1000

    def test_init_custom_size(self):
        """Should accept custom max_size."""
        cache = LLMCache(max_size=100)
        assert cache.max_size == 100

    def test_cache_miss(self):
        """Get should return None for cache miss."""
        cache = LLMCache()
        request = make_request("test")
        assert cache.get(request) is None
        stats = cache.stats()
        assert stats.misses == 1
        assert stats.hits == 0

    def test_cache_set_and_get(self):
        """Set then get should return cached response."""
        cache = LLMCache()
        request = make_request("test")
        response = make_response("cached response")

        cache.set(request, response)
        cached = cache.get(request)

        assert cached is not None
        assert cached == response
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 0

    def test_cache_contains(self):
        """Contains should check if request is cached."""
        cache = LLMCache()
        request = make_request("test")

        assert not cache.contains(request)
        cache.set(request, make_response("test"))
        assert cache.contains(request)

    def test_cache_clear(self):
        """Clear should remove all entries."""
        cache = LLMCache()
        cache.set(make_request("test1"), make_response("r1"))
        cache.set(make_request("test2"), make_response("r2"))

        assert cache.len() == 2
        cache.clear()
        assert cache.is_empty()

    def test_clear_preserves_stats(self):
        """Clear should preserve statistics."""
        cache = LLMCache()
        cache.set(make_request("test"), make_response("r1"))
        cache.get(make_request("test"))

        stats_before = cache.stats()
        assert stats_before.hits == 1

        cache.clear()

        stats_after = cache.stats()
        assert stats_after.hits == 1

    def test_len_and_is_empty(self):
        """Should track cache size correctly."""
        cache = LLMCache()
        assert cache.is_empty()
        assert cache.len() == 0

        cache.set(make_request("test"), make_response("r"))
        assert not cache.is_empty()
        assert cache.len() == 1

    def test_lru_eviction(self):
        """LRU eviction should remove oldest entries."""
        cache = LLMCache(max_size=2)

        cache.set(make_request("first"), make_response("r1"))
        cache.set(make_request("second"), make_response("r2"))
        assert cache.len() == 2

        cache.set(make_request("third"), make_response("r3"))
        assert cache.len() == 2

        assert cache.get(make_request("first")) is None
        assert cache.get(make_request("second")) is not None
        assert cache.get(make_request("third")) is not None

    def test_lru_access_updates_order(self):
        """Accessing an entry should move it to most recent."""
        cache = LLMCache(max_size=2)

        cache.set(make_request("first"), make_response("r1"))
        cache.set(make_request("second"), make_response("r2"))

        cache.get(make_request("first"))

        # Now "second" should be oldest
        cache.set(make_request("third"), make_response("r3"))

        assert cache.get(make_request("first")) is not None
        assert cache.get(make_request("second")) is None
        assert cache.get(make_request("third")) is not None

    def test_update_existing_key(self):
        """Setting same key should update value."""
        cache = LLMCache()
        request = make_request("test")

        cache.set(request, make_response("original"))
        cache.set(request, make_response("updated"))

        cached = cache.get(request)
        assert cached is not None
        assert cached.first_content() == "updated"

    def test_different_requests_different_keys(self):
        """Different requests should not share cache entries."""
        cache = LLMCache()

        cache.set(make_request("hello"), make_response("hello response"))
        cache.set(make_request("world"), make_response("world response"))

        r1 = cache.get(make_request("hello"))
        r2 = cache.get(make_request("world"))

        assert r1 is not None
        assert r2 is not None
        assert r1 != r2

    def test_tokens_saved_tracking(self):
        """Should track tokens saved from cache hits."""
        cache = LLMCache()
        request = make_request("test")

        response = make_response("test")
        cache.set(request, response)

        cached = cache.get(request)
        assert cached is not None

        stats = cache.stats()
        assert stats.tokens_saved == 30

    def test_lock_returns_asyncio_lock(self):
        """lock() should return an asyncio.Lock."""
        import asyncio

        cache = LLMCache()
        lock = cache.lock()
        assert isinstance(lock, asyncio.Lock)


class TestLLMCacheAsync:
    """Tests for async LLMCache operations."""

    @pytest.mark.asyncio
    async def test_get_or_set_cache_hit(self):
        """get_or_set should return cached response on hit."""
        cache = LLMCache()
        request = make_request("test")
        response = make_response("cached")

        cache.set(request, response)

        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return make_response("should not be called")

        result = await cache.get_or_set(request, factory)
        assert result == response
        assert call_count == 0

    @pytest.mark.asyncio
    async def test_get_or_set_cache_miss(self):
        """get_or_set should call factory on miss."""
        cache = LLMCache()
        request = make_request("test")
        response = make_response("from factory")

        async def factory():
            return response

        result = await cache.get_or_set(request, factory())
        assert result == response
        assert cache.get(request) == response

    @pytest.mark.asyncio
    async def test_get_or_set_with_callable(self):
        """get_or_set should work with callable factory."""
        cache = LLMCache()
        request = make_request("test")

        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return make_response("result")

        result = await cache.get_or_set(request, factory)
        assert result is not None
        assert call_count == 1


class TestCachedResponseDecorator:
    """Tests for cached_response decorator."""

    @pytest.mark.asyncio
    async def test_decorator_caches_response(self):
        """Decorator should cache responses."""
        cache = LLMCache()
        call_count = 0

        @cached_response(cache)
        async def complete(request: GenerationRequest) -> GenerationResponse:
            nonlocal call_count
            call_count += 1
            return make_response(f"response-{call_count}")

        request = make_request("test")

        r1 = await complete(request)
        r2 = await complete(request)

        assert call_count == 1
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_decorator_different_requests(self):
        """Decorator should handle different requests separately."""
        cache = LLMCache()

        @cached_response(cache)
        async def complete(request: GenerationRequest) -> GenerationResponse:
            return make_response(f"response-{request.messages[0].content}")

        r1 = await complete(make_request("hello"))
        r2 = await complete(make_request("world"))

        assert r1.first_content() == "response-hello"
        assert r2.first_content() == "response-world"


class TestIntegration:
    """Integration tests for cache with realistic scenarios."""

    def test_conversation_flow(self):
        """Test caching in a typical conversation flow."""
        cache = LLMCache(max_size=100)

        messages = [
            Message.system("You are helpful"),
            Message.user("What is 2+2?"),
        ]
        request = GenerationRequest(model="gpt-4", messages=messages)
        response = make_response("2+2 equals 4")

        cache.set(request, response)
        cached = cache.get(request)

        assert cached is not None
        assert cached.first_content() == "2+2 equals 4"

    def test_different_model_same_messages_different_cache(self):
        """Same messages with different models should not share cache."""
        cache = LLMCache()

        messages = [Message.user("Hello")]

        req1 = GenerationRequest(model="gpt-4", messages=messages)
        req2 = GenerationRequest(model="gpt-3.5-turbo", messages=messages)

        cache.set(req1, make_response("gpt-4 response", model="gpt-4"))
        cache.set(req2, make_response("gpt-3.5 response", model="gpt-3.5-turbo"))

        r1 = cache.get(req1)
        r2 = cache.get(req2)

        assert r1 is not None
        assert r2 is not None
        assert r1.model == "gpt-4"
        assert r2.model == "gpt-3.5-turbo"

    def test_tokens_saved_accumulates(self):
        """Tokens saved should accumulate across hits."""
        cache = LLMCache()

        for i in range(3):
            request = make_request(f"test{i}")
            response = make_response(f"response{i}")
            cache.set(request, response)
            cache.get(request)

        stats = cache.stats()
        assert stats.tokens_saved == 90
        assert stats.hits == 3
