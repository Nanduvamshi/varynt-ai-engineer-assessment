# Q5 — Fallback & Error Handling

> When models fail or APIs time out, what's our strategy for maintaining user experience?

## Layered defense

```
client request
    -> [timeout 20s wall-clock]
        -> [retry 3x exponential backoff]
            -> [primary provider]   (e.g., OpenAI gpt-4o-mini)
        -> [fallback chain]
            -> secondary provider   (e.g., Anthropic Claude / local model)
            -> cached response      (Redis: prompt-hash -> last good output)
            -> graceful template    (deterministic, never fails)
    -> [circuit breaker]            (per-provider, open after 5 fails / 30s)
```

Each layer has a single responsibility. Retry handles transient blips
(rate limits, 503s). Fallback handles partial outages. Circuit breaker
prevents us from piling latency on a known-down provider. Cache + templates
ensure the user always gets *something* useful.

## Classification of errors

| Error | Action |
|---|---|
| 429 (rate limit) | Retry with backoff. Honor `Retry-After` header. |
| 5xx | Retry. After 3 fails, fall back. |
| Timeout | Retry once. Then fall back. |
| 401/403 | Do **not** retry. Page on-call. Fall back to cache/template. |
| 400 (bad request) | Do **not** retry — input is broken. Return 400 to user with diagnostic. |
| Schema-violation in LLM output | Retry once with a stricter prompt. Then return safe default. |

## Where this lives in code

| Concern | File |
|---|---|
| Retry decorator | `app/q5_resilience/retry.py` (`with_retry`) |
| Fallback chain | `app/q5_resilience/retry.py` (`with_fallback`) |
| Circuit breaker | `app/q5_resilience/circuit_breaker.py` |
| Live use in Q1 | `app/q1_classifier/classifier.py` (`_call_openai`) |
| Live use in Q2 | `app/q2_routing/router.py` (router-level fallback chain) |

The Q2 router deliberately registers a `force_fail=true` flag to demo the
fallback chain end-to-end without simulating an actual outage.

## User experience during degradation

- **Hot path success**: `200 OK` + result
- **Cached fallback**: `200 OK` + result + `X-Source: cache` header
- **Template fallback**: `200 OK` + safe generic content + `X-Source: template`
- **Total failure**: `503 Service Unavailable` with `Retry-After` and a
  user-friendly message ("AI is busy — we'll email you when ready"). Async
  jobs are also requeued for later delivery instead of dropping the request.

## Observability hooks

Every retry, fallback, and breaker open/close emits a structured log line.
In production we'd ship these to Datadog with tags `{provider, error_class,
attempt_n}` so dashboards and alerts attribute outages to the correct
upstream — not "AI broke."
