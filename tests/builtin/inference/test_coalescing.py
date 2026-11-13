# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-call sharing, local failure, accounting, and query-mode compatibility."""

from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile
from powercontext.builtin.inference import CoalescingEmbeddingModel, EmbeddingResult, InferenceUsage, embed_query


class _Embedding:
    profile = EmbeddingProfile(profile_id="coalescing-regressions", model="offline", dimension=1)

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        self.calls.append(texts)
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) for text in texts),
            usage=InferenceUsage(requests=2, input_tokens=9, output_tokens=4),
        )


def test_late_waiters_share_running_work_even_with_cache_disabled() -> None:
    async def scenario() -> None:
        class Blocked(_Embedding):
            def __init__(self) -> None:
                super().__init__()
                self.started = {text: asyncio.Event() for text in ("alpha", "beta", "gamma")}
                self.release = asyncio.Event()

            async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
                for text in texts:
                    self.started[text].set()
                await self.release.wait()
                return await super().embed(texts)

        base = Blocked()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0, cache_size=0)
        first = asyncio.create_task(model.embed(("alpha", "beta", "alpha")))
        await base.started["beta"].wait()
        late = asyncio.create_task(model.embed(("beta", "gamma")))
        await base.started["gamma"].wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        base.release.set()
        result = await late
        assert result.vectors == ((4.0,), (5.0,))
        assert result.usage == InferenceUsage(requests=4, input_tokens=18, output_tokens=8)
        assert sorted(text for call in base.calls for text in call) == ["alpha", "beta", "gamma"]
        await model.embed(("beta",))
        assert sum(call.count("beta") for call in base.calls) == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_failed_batch_does_not_cancel_independent_shared_work_or_poison_cache() -> None:
    async def scenario() -> None:
        class Flaky(_Embedding):
            def __init__(self) -> None:
                super().__init__()
                self.release_good = asyncio.Event()
                self.failed = False

            async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
                if texts == ("bad",) and not self.failed:
                    self.failed = True
                    self.calls.append(texts)
                    raise RuntimeError("controlled failure")  # noqa: TRY003 - controlled provider fixture
                if texts == ("good",):
                    await self.release_good.wait()
                return await super().embed(texts)

        base = Flaky()
        model = CoalescingEmbeddingModel(base, max_batch_size=1, coalesce_ms=0)
        failed = asyncio.create_task(model.embed(("bad", "good")))
        healthy = asyncio.create_task(model.embed(("good", "good")))
        with pytest.raises(RuntimeError, match="controlled failure"):
            await failed
        assert not healthy.done()
        base.release_good.set()
        assert (await healthy).vectors == ((4.0,), (4.0,))
        assert (await model.embed(("bad", "good"))).vectors == ((3.0,), (4.0,))
        assert base.calls.count(("bad",)) == 2
        assert base.calls.count(("good",)) == 1

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_usage_counts_distinct_equal_metered_batches_and_excludes_cached_texts() -> None:
    async def scenario() -> None:
        base = _Embedding()
        model = CoalescingEmbeddingModel(base, max_batch_size=2, coalesce_ms=0)
        await model.embed(("cached",))
        first, second = await asyncio.gather(
            model.embed(("cached", "a", "b", "c", "a")),
            model.embed(("b", "c", "d", "c")),
        )
        assert first.vectors == ((6.0,), (1.0,), (1.0,), (1.0,), (1.0,))
        assert first.usage == second.usage == InferenceUsage(requests=4, input_tokens=18, output_tokens=8)
        assert base.calls == [("cached",), ("a", "b"), ("c", "d")]
        cached = await model.embed(("a", "d", "cached", "a"))
        assert cached.usage.requests == 0
        assert cached.usage.input_tokens in (None, 0)
        assert len(base.calls) == 3

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_wrong_vector_count_rejects_the_entire_batch_and_can_retry() -> None:
    class WrongOnce(_Embedding):
        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            if not self.calls:
                self.calls.append(texts)
                return EmbeddingResult(vectors=((99.0,),))
            return await super().embed(texts)

    async def scenario() -> None:
        base = WrongOnce()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        results = await asyncio.gather(model.embed(("a", "bb")), model.embed(("bb",)), return_exceptions=True)
        assert all(isinstance(result, ValueError) for result in results)
        assert (await model.embed(("a", "bb"))).vectors == ((1.0,), (2.0,))
        assert base.calls == [("a", "bb"), ("a", "bb")]

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_all_callers_can_cancel_before_dispatch_without_cancelling_shared_work() -> None:
    async def scenario() -> None:
        class Blocked(_Embedding):
            def __init__(self) -> None:
                super().__init__()
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
                self.started.set()
                await self.release.wait()
                return await super().embed(texts)

        base = Blocked()
        model = CoalescingEmbeddingModel(base, coalesce_ms=5)
        tasks = [asyncio.create_task(model.embed(("shared",))) for _ in range(2)]
        await asyncio.sleep(0)
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        await base.started.wait()
        survivor = asyncio.create_task(model.embed(("shared",)))
        base.release.set()
        assert (await survivor).vectors == ((6.0,),)
        assert (await model.embed(("shared",))).usage.requests == 0
        assert base.calls == [("shared",)]

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_distinct_query_embedding_keeps_its_own_cache_and_provider_operation() -> None:
    class Asymmetric(_Embedding):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[tuple[str, ...]] = []

        async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            self.queries.append(texts)
            return EmbeddingResult(vectors=tuple((-float(len(text)),) for text in texts))

    async def scenario() -> None:
        base = Asymmetric()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0)
        assert model.profile is base.profile
        documents, queries = await asyncio.gather(model.embed(("same",)), embed_query(model, ("same",)))
        assert documents.vectors == ((4.0,),)
        assert queries.vectors == ((-4.0,),)
        assert (await model.embed(("same",))).vectors == ((4.0,),)
        assert (await embed_query(model, ("same",))).vectors == ((-4.0,),)
        assert base.calls == base.queries == [("same",)]

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_small_cache_does_not_drop_large_batch_results_or_normalize_text() -> None:
    async def scenario() -> None:
        base = _Embedding()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, max_batch_size=2, cache_size=1)
        result = await model.embed(("é", "é", " é ", "é", "tail"))
        assert result.vectors == ((1.0,), (2.0,), (4.0,), (1.0,), (4.0,))
        assert result.usage.requests == 4
        assert sorted(text for call in base.calls for text in call) == sorted(("é", "é", " é ", "tail"))

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))
