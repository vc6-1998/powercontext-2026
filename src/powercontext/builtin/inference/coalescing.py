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

"""Coalesce Embedding calls while preserving ordering, usage and cancellation."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel, QueryEmbeddingModel, embed_query

_Key = tuple[bool, str]


@dataclass(frozen=True, slots=True, eq=False)
class _BatchUsage:
    # Identity, not equal token counts, determines whether a batch was charged.
    usage: InferenceUsage


@dataclass(frozen=True, slots=True)
class _Embedded:
    vector: EmbeddingVector
    batch: _BatchUsage


def _sum_tokens(values: Iterable[int | None]) -> int | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _observe_failure(future: asyncio.Future[_Embedded]) -> None:
    # All callers can cancel before a shared request fails. Retrieve that error
    # to avoid unobserved-Future warnings; remaining awaiters still receive it.
    if not future.cancelled():
        future.exception()


class CoalescingEmbeddingModel:
    """Share in-flight exact texts and cache successful vectors on one event loop.

    Query/document modes share work for symmetric providers; providers exposing
    a distinct query operation receive separate batches and cache identities.
    Caller cancellation never owns or cancels the shared provider tasks.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")  # noqa: TRY003
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a non-negative integer")  # noqa: TRY003
        if (
            isinstance(coalesce_ms, bool)
            or not isinstance(coalesce_ms, (int, float))
            or not math.isfinite(coalesce_ms)
            or coalesce_ms < 0
        ):
            raise ValueError("coalesce_ms must be a finite non-negative number")  # noqa: TRY003
        self.profile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._delay = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[_Key, EmbeddingVector] = OrderedDict()
        self._inflight: dict[_Key, asyncio.Future[_Embedded]] = {}
        self._pending: dict[_Key, asyncio.Future[_Embedded]] = {}
        self._dispatch_task: asyncio.Task[None] | None = None
        self._batches: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return document vectors in the original order, including duplicates."""

        return await self._embed(texts, query=False)

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Keep asymmetric query vectors separate; reuse symmetric document work."""

        return await self._embed(texts, query=isinstance(self._model, QueryEmbeddingModel))

    async def _embed(self, texts: tuple[str, ...], *, query: bool) -> EmbeddingResult:
        ready: dict[_Key, EmbeddingVector] = {}
        waiting: dict[_Key, asyncio.Future[_Embedded]] = {}
        loop = asyncio.get_running_loop()
        for text in texts:
            key = (query, text)
            if key in self._cache:
                ready[key] = self._cache[key]
                self._cache.move_to_end(key)
            else:
                future = self._inflight.get(key)
                if future is None:
                    future = loop.create_future()
                    future.add_done_callback(_observe_failure)
                    self._inflight[key] = future
                    self._pending[key] = future
                waiting[key] = future
        if self._pending and self._dispatch_task is None:
            self._dispatch_task = asyncio.create_task(self._dispatch())
        # Shield each shared Future, so gather cancellation only cancels this
        # caller's wrappers, including calls waiting on several provider batches.
        resolved = await asyncio.gather(*(asyncio.shield(future) for future in waiting.values()))
        batches: dict[_BatchUsage, None] = {}
        for key, item in zip(waiting, resolved, strict=True):
            ready[key] = item.vector
            batches[item.batch] = None
        usages = tuple(batch.usage for batch in batches)
        return EmbeddingResult(
            vectors=tuple(ready[query, text] for text in texts),
            usage=InferenceUsage(
                requests=sum(usage.requests for usage in usages),
                input_tokens=_sum_tokens(usage.input_tokens for usage in usages),
                output_tokens=_sum_tokens(usage.output_tokens for usage in usages),
            ),
        )

    async def _dispatch(self) -> None:
        try:
            # Even zero yields once, letting calls already ready on this loop
            # register before a synchronous provider can complete its first batch.
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            for key, future in self._pending.items():
                future.cancel()
                self._inflight.pop(key, None)
            self._pending.clear()
            raise
        finally:
            self._dispatch_task = None
        pending, self._pending = self._pending, {}
        for query in (False, True):
            keys = tuple(key for key in pending if key[0] == query)
            for start in range(0, len(keys), self._max_batch_size):
                batch = tuple((key, pending[key]) for key in keys[start : start + self._max_batch_size])
                task = asyncio.create_task(self._run_batch(batch, query=query))
                self._batches.add(task)
                task.add_done_callback(self._batches.discard)

    async def _run_batch(self, batch: tuple[tuple[_Key, asyncio.Future[_Embedded]], ...], *, query: bool) -> None:
        try:
            texts = tuple(key[1] for key, _future in batch)
            result = await embed_query(self._model, texts) if query else await self._model.embed(texts)
            if len(result.vectors) != len(batch):
                raise ValueError("provider vector count does not match its text batch")  # noqa: TRY003, TRY301
            usage = _BatchUsage(result.usage)
            for (key, future), vector in zip(batch, result.vectors, strict=True):
                if self._cache_size:
                    self._cache[key] = vector
                    self._cache.move_to_end(key)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                future.set_result(_Embedded(vector, usage))
        except asyncio.CancelledError:
            for _key, future in batch:
                future.cancel()
            raise
        except Exception as error:
            for _key, future in batch:
                future.set_exception(error)
        finally:
            for key, _future in batch:
                self._inflight.pop(key, None)


__all__ = ["CoalescingEmbeddingModel"]
