"""Explicit, dated pricing assumptions for optional cost estimates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PricingRate:
    provider: str
    model: str
    input_per_million_usd: float
    output_per_million_usd: float
    currency: str
    reference_date: str

    def estimate(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None and output_tokens is None:
            return None
        input_cost = (input_tokens or 0) * self.input_per_million_usd / 1_000_000
        output_cost = (output_tokens or 0) * self.output_per_million_usd / 1_000_000
        return input_cost + output_cost


# Source: https://docs.typesafe.ai/models.md, checked 2026-09-22.
BUILTIN_PRICING: tuple[PricingRate, ...] = (
    PricingRate(
        provider="typesafe",
        model="jev-1.13.0",
        input_per_million_usd=0.042,
        output_per_million_usd=0.0,
        currency="USD",
        reference_date="2026-09-22",
    ),
    PricingRate(
        provider="typesafe",
        model="jev-latest",
        input_per_million_usd=0.042,
        output_per_million_usd=0.0,
        currency="USD",
        reference_date="2026-09-22",
    ),
)


def find_pricing(provider: str, model: str | None) -> PricingRate | None:
    if model is None:
        return None
    return next(
        (rate for rate in BUILTIN_PRICING if (rate.provider, rate.model) == (provider, model)),
        None,
    )
