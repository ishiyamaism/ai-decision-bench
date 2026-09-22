"""Small, secret-safe HTTP retry helper for provider adapters."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

_RETRYABLE_STATUS = {408, 429}
_MAX_RETRY_DELAY_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class HttpOutcome:
    data: dict[str, Any] | None
    error: str | None
    latency_ms: float


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    fallback = min(0.5 * (2**attempt), 5.0)
    if response is None:
        return fallback
    retry_after_ms = response.headers.get("retry-after-ms")
    retry_after = response.headers.get("retry-after")
    try:
        if retry_after_ms is not None:
            return min(float(retry_after_ms) / 1000.0, _MAX_RETRY_DELAY_SECONDS)
        if retry_after is not None:
            return min(float(retry_after), _MAX_RETRY_DELAY_SECONDS)
    except ValueError:
        pass
    return fallback


def _is_retryable(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUS or status_code >= 500


async def post_json(
    client: httpx.AsyncClient,
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
    max_retries: int,
) -> HttpOutcome:
    """POST JSON with finite retries and return only sanitized error categories."""

    started = time.perf_counter()
    for attempt in range(max_retries + 1):
        response: httpx.Response | None = None
        try:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
                follow_redirects=False,
            )
            if response.is_success:
                try:
                    data = response.json()
                except ValueError:
                    return HttpOutcome(None, "invalid_response", _elapsed_ms(started))
                if not isinstance(data, dict):
                    return HttpOutcome(None, "invalid_response", _elapsed_ms(started))
                return HttpOutcome(data, None, _elapsed_ms(started))
            if _is_retryable(response.status_code) and attempt < max_retries:
                await asyncio.sleep(_retry_delay(response, attempt))
                continue
            return HttpOutcome(None, f"http_{response.status_code}", _elapsed_ms(started))
        except httpx.TimeoutException:
            if attempt < max_retries:
                await asyncio.sleep(_retry_delay(None, attempt))
                continue
            return HttpOutcome(None, "timeout", _elapsed_ms(started))
        except httpx.RequestError:
            if attempt < max_retries:
                await asyncio.sleep(_retry_delay(None, attempt))
                continue
            return HttpOutcome(None, "connection_error", _elapsed_ms(started))
    return HttpOutcome(None, "provider_error", _elapsed_ms(started))


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
