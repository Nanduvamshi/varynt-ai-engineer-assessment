"""Retry + timeout primitives used across Q1 (LLM) and Q2 (provider routing).

Wraps any callable with exponential backoff and a hard wall-clock timeout.
The whole point of Q5 is to demonstrate this is *applied* in live code paths,
not just described in a doc.
"""

import logging
import time
from functools import wraps
from typing import Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class TransientProviderError(Exception):
    """Retriable failure from an upstream model/provider."""


class PermanentProviderError(Exception):
    """Non-retriable failure (auth, bad request) — fall back, do not retry."""


def with_retry(
    max_attempts: int = 3,
    initial_wait: float = 0.5,
    max_wait: float = 4.0,
    overall_timeout: float = 20.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: retry on TransientProviderError with exponential backoff."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @retry(
            stop=(stop_after_attempt(max_attempts) | stop_after_delay(overall_timeout)),
            wait=wait_exponential(multiplier=initial_wait, max=max_wait),
            retry=retry_if_exception_type(TransientProviderError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def with_fallback(*fallback_fns: Callable[..., T]) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator: if primary raises, try each fallback in order. Last error re-raised."""

    def decorator(primary: Callable[..., T]) -> Callable[..., T]:
        @wraps(primary)
        def wrapper(*args, **kwargs):
            try:
                return primary(*args, **kwargs)
            except Exception as primary_err:
                logger.warning(
                    "Primary %s failed (%s); trying %d fallback(s)",
                    primary.__name__,
                    primary_err,
                    len(fallback_fns),
                )
                last_err = primary_err
                for fb in fallback_fns:
                    try:
                        return fb(*args, **kwargs)
                    except Exception as fb_err:
                        logger.warning("Fallback %s failed: %s", fb.__name__, fb_err)
                        last_err = fb_err
                raise last_err

        return wrapper

    return decorator


def timed(fn: Callable[..., T]) -> Callable[..., T]:
    """Lightweight timing wrapper — useful for the Q5 + Q6 latency story."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("%s took %.1fms", fn.__name__, elapsed_ms)

    return wrapper
