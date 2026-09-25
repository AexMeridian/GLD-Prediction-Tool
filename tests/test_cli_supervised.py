import asyncio

import pytest

from gold_edge.cli import _supervised


def test_restarts_a_loop_that_crashes_then_stops_when_it_returns():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")

    asyncio.run(_supervised("t", flaky, max_backoff_s=0.0))
    assert calls["n"] == 3


def test_cancellation_is_not_swallowed():
    async def forever():
        await asyncio.sleep(3600)

    async def run():
        task = asyncio.create_task(_supervised("t", forever))
        await asyncio.sleep(0)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
