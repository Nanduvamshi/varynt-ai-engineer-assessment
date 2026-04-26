import pybreaker

_BREAKERS: dict[str, pybreaker.CircuitBreaker] = {}


def get_breaker(
    provider_name: str, fail_max: int = 5, reset_timeout: int = 30
) -> pybreaker.CircuitBreaker:
    if provider_name not in _BREAKERS:
        _BREAKERS[provider_name] = pybreaker.CircuitBreaker(
            fail_max=fail_max, reset_timeout=reset_timeout, name=provider_name
        )
    return _BREAKERS[provider_name]
