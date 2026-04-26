"""Per-provider circuit breaker.

When an upstream provider (e.g., OpenAI image gen) fails repeatedly we stop
calling it for a cool-off window so we don't pile latency or cost. After the
window we let one probe request through to test recovery.

Built on `pybreaker`. Each provider gets its own breaker instance so a flaky
ElevenLabs deployment doesn't take down the OpenAI image path.
"""

import pybreaker

_BREAKERS: dict[str, pybreaker.CircuitBreaker] = {}


def get_breaker(
    provider_name: str,
    fail_max: int = 5,
    reset_timeout: int = 30,
) -> pybreaker.CircuitBreaker:
    """Return (or create) a circuit breaker keyed by provider name."""
    if provider_name not in _BREAKERS:
        _BREAKERS[provider_name] = pybreaker.CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=reset_timeout,
            name=provider_name,
        )
    return _BREAKERS[provider_name]
