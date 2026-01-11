"""
Lightweight pytest hooks to support async tests without external plugins.

Provides a fallback event loop runner so `async def` tests execute even when
pytest-asyncio/anyio are unavailable in the environment (common in CI sandboxes).
"""
import asyncio
import inspect
from typing import Any, Dict


def pytest_pyfunc_call(pyfuncitem):  # type: ignore[override]
    """
    Run coroutine tests using a local event loop when pytest lacks async plugins.

    Returns True when the async test was executed so pytest skips its default
    pyfunc execution path; otherwise returns None to let pytest handle sync tests.
    """
    test_obj = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_obj):
        return None

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        kwargs: Dict[str, Any] = dict(pyfuncitem.funcargs)

        # Drop fixtures that only exist in pytest-asyncio environments; they
        # are not needed for a basic event loop runner.
        kwargs.pop("event_loop", None)
        kwargs.pop("event_loop_policy", None)

        loop.run_until_complete(test_obj(**kwargs))
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    return True
