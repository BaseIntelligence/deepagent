"""Tests for concurrency control module."""

import asyncio

import pytest

from swe_forge.swe.concurrency import (
    ConcurrencyConfig,
    PipelineSemaphores,
    rate_limit_delay,
    GITHUB_RATE_LIMIT_DELAY,
    get_docker_semaphore,
    set_docker_containers_limit,
)


class TestConcurrencyConfig:
    """Tests for ConcurrencyConfig dataclass."""

    def test_default_values(self):
        config = ConcurrencyConfig()
        assert config.GH_ARCHIVE_FETCH == 8
        assert config.ENRICHMENT == 20
        assert config.PRECLASSIFY == 25
        assert config.DEEP_PROCESSING == 8
        assert config.DEEP_BACKLOG_MULTIPLIER == 5
        assert config.GITHUB_SEARCH_AUTH == 5
        assert config.GITHUB_SEARCH_UNAUTH == 2
        assert config.DOCKER_CONTAINERS == 8

    def test_custom_values(self):
        config = ConcurrencyConfig(
            GH_ARCHIVE_FETCH=10,
            ENRICHMENT=30,
            PRECLASSIFY=50,
            DEEP_PROCESSING=10,
            DEEP_BACKLOG_MULTIPLIER=3,
            GITHUB_SEARCH_AUTH=10,
            GITHUB_SEARCH_UNAUTH=5,
            DOCKER_CONTAINERS=16,
        )
        assert config.GH_ARCHIVE_FETCH == 10
        assert config.ENRICHMENT == 30
        assert config.PRECLASSIFY == 50
        assert config.DEEP_PROCESSING == 10
        assert config.DEEP_BACKLOG_MULTIPLIER == 3
        assert config.GITHUB_SEARCH_AUTH == 10
        assert config.GITHUB_SEARCH_UNAUTH == 5
        assert config.DOCKER_CONTAINERS == 16


class TestPipelineSemaphores:
    """Tests for PipelineSemaphores class."""

    def test_default_config(self):
        pipeline = PipelineSemaphores()
        assert pipeline.config.GH_ARCHIVE_FETCH == 8

    def test_custom_config(self):
        config = ConcurrencyConfig(ENRICHMENT=10)
        pipeline = PipelineSemaphores(config=config)
        assert pipeline.config.ENRICHMENT == 10

    @pytest.mark.asyncio
    async def test_enrich_semaphore_creation(self):
        pipeline = PipelineSemaphores()
        sem = pipeline.enrich_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 20

    @pytest.mark.asyncio
    async def test_preclassify_semaphore_creation(self):
        pipeline = PipelineSemaphores()
        sem = pipeline.preclassify_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 25

    @pytest.mark.asyncio
    async def test_deep_semaphore_creation(self):
        pipeline = PipelineSemaphores()
        sem = pipeline.deep_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 8

    @pytest.mark.asyncio
    async def test_backlog_semaphore_uses_multiplier(self):
        pipeline = PipelineSemaphores()
        sem = pipeline.backlog_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 8 * 5

    @pytest.mark.asyncio
    async def test_gh_archive_semaphore_creation(self):
        pipeline = PipelineSemaphores()
        sem = pipeline.gh_archive_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 8

    @pytest.mark.asyncio
    async def test_github_search_auth_semaphore(self):
        pipeline = PipelineSemaphores()
        sem = pipeline.github_search_auth_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 5

    @pytest.mark.asyncio
    async def test_github_search_unauth_semaphore(self):
        pipeline = PipelineSemaphores()
        sem = pipeline.github_search_unauth_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 2

    @pytest.mark.asyncio
    async def test_docker_semaphore_creation(self):
        pipeline = PipelineSemaphores()
        sem = pipeline.docker_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 8

    @pytest.mark.asyncio
    async def test_custom_docker_semaphore_limit(self):
        config = ConcurrencyConfig(DOCKER_CONTAINERS=4)
        pipeline = PipelineSemaphores(config=config)
        sem = pipeline.docker_sem
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == 4

    @pytest.mark.asyncio
    async def test_custom_config_affects_semaphore_limits(self):
        config = ConcurrencyConfig(ENRICHMENT=5, DEEP_PROCESSING=2)
        pipeline = PipelineSemaphores(config=config)
        assert pipeline.enrich_sem._value == 5
        assert pipeline.deep_sem._value == 2
        assert pipeline.backlog_sem._value == 2 * 5

    @pytest.mark.asyncio
    async def test_semaphore_limit_enforcement(self):
        config = ConcurrencyConfig(ENRICHMENT=2)
        pipeline = PipelineSemaphores(config=config)
        sem = pipeline.enrich_sem

        acquired = []

        async def acquire_slot(idx: int):
            async with sem:
                acquired.append(idx)
                await asyncio.sleep(0.01)

        tasks = [asyncio.create_task(acquire_slot(i)) for i in range(4)]
        await asyncio.sleep(0.005)
        assert len(acquired) == 2
        await asyncio.gather(*tasks)
        assert len(acquired) == 4

    @pytest.mark.asyncio
    async def test_semaphores_reuse_same_instance(self):
        pipeline = PipelineSemaphores()
        sem1 = pipeline.enrich_sem
        sem2 = pipeline.enrich_sem
        assert sem1 is sem2


class TestRateLimitDelay:
    """Tests for rate_limit_delay function."""

    @pytest.mark.asyncio
    async def test_rate_limit_delay(self):
        import time

        start = time.monotonic()
        await rate_limit_delay()
        elapsed = time.monotonic() - start
        assert elapsed >= GITHUB_RATE_LIMIT_DELAY
        assert elapsed < GITHUB_RATE_LIMIT_DELAY + 0.1

    def test_rate_limit_constant_value(self):
        assert GITHUB_RATE_LIMIT_DELAY == 2.1


class TestConcurrencyLimitEnforcement:
    """Integration tests for concurrency limit enforcement."""

    @pytest.mark.asyncio
    async def test_backlog_multiplier_applied(self):
        config = ConcurrencyConfig(DEEP_PROCESSING=3, DEEP_BACKLOG_MULTIPLIER=4)
        pipeline = PipelineSemaphores(config=config)
        assert pipeline.backlog_sem._value == 12

    @pytest.mark.asyncio
    async def test_multiple_semaphores_independent(self):
        pipeline = PipelineSemaphores()

        async with pipeline.enrich_sem:
            assert pipeline.preclassify_sem._value == 25
            assert pipeline.deep_sem._value == 8

        async with pipeline.preclassify_sem:
            assert pipeline.enrich_sem._value == 20
            assert pipeline.deep_sem._value == 8

    @pytest.mark.asyncio
    async def test_zero_limit_semaphore(self):
        config = ConcurrencyConfig(ENRICHMENT=1)
        pipeline = PipelineSemaphores(config=config)
        sem = pipeline.enrich_sem
        assert sem._value == 1

        async with sem:
            assert sem._value == 0


class TestDockerSemaphore:
    """Tests for global Docker semaphore functions."""

    @pytest.mark.asyncio
    async def test_docker_semaphore_limits_container_creation(self):
        set_docker_containers_limit(2)
        sem = get_docker_semaphore()
        assert sem._value == 2

        acquired_count = 0

        async def acquire_slot():
            nonlocal acquired_count
            await sem.acquire()
            acquired_count += 1

        tasks = [asyncio.create_task(acquire_slot()) for _ in range(4)]

        await asyncio.sleep(0.01)
        assert acquired_count == 2

        for _ in range(2):
            sem.release()
        await asyncio.gather(*tasks)
        assert acquired_count == 4

    @pytest.mark.asyncio
    async def test_docker_semaphore_release_on_error(self):
        set_docker_containers_limit(1)
        sem = get_docker_semaphore()

        await sem.acquire()
        assert sem._value == 0

        try:
            raise RuntimeError("Simulated error")
        except RuntimeError:
            sem.release()

        assert sem._value == 1

    @pytest.mark.asyncio
    async def test_docker_semaphore_configurable(self):
        set_docker_containers_limit(4)
        sem = get_docker_semaphore()
        assert sem._value == 4

        set_docker_containers_limit(8)
        sem = get_docker_semaphore()
        assert sem._value == 8
