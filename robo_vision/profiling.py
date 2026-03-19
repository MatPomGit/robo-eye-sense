"""Simple profiling utilities for measuring execution times.

Activated by the ``--profile`` CLI flag.  When profiling is disabled the
decorator is a no-op with zero overhead.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger("robo_vision.profiling")

# Global flag – set to True via enable_profiling() when --profile is used.
_profiling_enabled = False


def enable_profiling() -> None:
    """Enable the profiling timer for decorated methods."""
    global _profiling_enabled
    _profiling_enabled = True


def is_profiling_enabled() -> bool:
    """Return ``True`` when profiling is currently active."""
    return _profiling_enabled


F = TypeVar("F", bound=Callable[..., Any])


def profile_method(func: F) -> F:
    """Decorator that logs execution time of the wrapped function.

    When profiling is disabled (the default) the decorator has no
    measurable overhead — it simply returns the original function.

    When profiling is enabled via :func:`enable_profiling`, each call to
    the decorated function is timed and a log message is emitted at
    ``DEBUG`` level.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _profiling_enabled:
            return func(*args, **kwargs)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        logger.info(
            "[PROFILE] %s: %.2f ms",
            func.__qualname__, elapsed,
        )
        return result

    return cast(F, wrapper)
