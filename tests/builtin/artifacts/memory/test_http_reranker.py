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

"""Additional boundary and complete-pool tests for HTTP reranking."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import HTTPMemoryReranker, MemoryHit


def _hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="m", revision=1),
        entry_id=f"e{rank}",
        entry_version_id=f"v{rank}",
        text=f"Candidate {rank} with Unicode: 数据库",
        score=1 / rank,
        matched_by=("fts",),
    )


def test_complete_pool_stable_order_and_reusable_caller_client() -> None:
    seen: list[int] = []
    candidates = tuple(_hit(rank) for rank in range(1, 174))
    original = tuple(hit.model_dump() for hit in candidates)

    def score(rank: int) -> int:
        return (rank * 31) % 17 - 8

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "数据库"
        batch = body["candidates"]
        assert 1 <= len(batch) <= 7
        for item in batch:
            assert item["text"] == candidates[item["rank"] - 1].text
            seen.append(item["rank"])
        return httpx.Response(
            200, json={"scores": [{"rank": item["rank"], "score": score(item["rank"])} for item in reversed(batch)]}
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker("https://score.invalid", client=client, batch_size=7, policy_id="test-policy")
            expected = tuple(sorted(range(1, 174), key=lambda rank: (-score(rank), rank)))
            for limit in (10, 173):
                decision = await reranker.rerank("数据库", candidates, limit)
                assert decision.selected_ranks == expected[:limit]
                assert decision.usage.requests == 25
                assert not decision.used_fallback
                assert not client.is_closed
            assert reranker.policy_id == "test-policy"

    asyncio.run(scenario())
    assert sorted(seen) == sorted(list(range(1, 174)) * 2)
    assert tuple(hit.model_dump() for hit in candidates) == original


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"scores": None},
        {"scores": {}},
        {"scores": []},
        {"scores": [None]},
        {"scores": [{"rank": 1}]},
        {"scores": [{"score": 0.5}]},
        {"scores": [{"rank": True, "score": 0.5}]},
        {"scores": [{"rank": 1.0, "score": 0.5}]},
        {"scores": [{"rank": "1", "score": 0.5}]},
        {"scores": [{"rank": 1, "score": True}]},
        {"scores": [{"rank": 1, "score": "0.5"}]},
        {"scores": [{"rank": 1, "score": None}]},
        {"scores": [{"rank": 1, "score": -float("inf")}]},
    ],
)
def test_rejects_malformed_response_types(payload: object) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=json.dumps(payload).encode()))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://score.invalid", client=client).rerank("q", (_hit(1),), 1)

    asyncio.run(scenario())


def test_rejects_rank_from_another_batch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        batch = json.loads(request.content)["candidates"]
        return httpx.Response(200, json={"scores": [{"rank": 1, "score": batch[0]["rank"]}]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ValueError):
                await HTTPMemoryReranker("https://score.invalid", client=client, batch_size=1).rerank(
                    "q", (_hit(1), _hit(2)), 1
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("status", [302, 400, 429, 503])
def test_preserves_http_status_errors(status: int) -> None:
    async def scenario() -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(status, text="not a scores response"))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(httpx.HTTPStatusError) as error:
                await HTTPMemoryReranker("https://score.invalid", client=client).rerank("q", (_hit(1),), 1)
            assert error.value.response.status_code == status

    asyncio.run(scenario())


def test_invalid_json_and_parameters_fail_without_closing_client() -> None:
    seen = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        return httpx.Response(200, content=b"not JSON")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            for batch_size in (-1, True, 1.5):
                with pytest.raises(ValueError):
                    HTTPMemoryReranker("https://score.invalid", client=client, batch_size=batch_size)
            reranker = HTTPMemoryReranker("https://score.invalid", client=client)
            for limit in (0, -1, True, 1.5):
                with pytest.raises(ValueError):
                    await reranker.rerank("q", (_hit(1),), limit)
            assert seen == 0
            with pytest.raises(ValueError):
                await reranker.rerank("q", (_hit(1),), 1)
            assert seen == 1
            assert not client.is_closed

    asyncio.run(scenario())
