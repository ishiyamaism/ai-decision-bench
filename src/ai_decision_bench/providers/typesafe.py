"""TypeSafe AI / Jev adapter using the documented System One HTTP API."""

from __future__ import annotations

import os
from typing import Annotated, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_decision_bench.models import DecisionResult, EvaluationCase, JsonScalar
from ai_decision_bench.pricing import PricingRate
from ai_decision_bench.providers._http import post_json

TYPESAFE_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_TYPESAFE_MODEL = "jev-latest"
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class MissingTypeSafeKeyError(RuntimeError):
    """Raised when the documented credential environment variable is absent."""

    def __init__(self) -> None:
        super().__init__("TYPESAFE_API_KEY is not set.")


class _ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str
    choice: str
    probabilities: dict[str, Probability]
    confidence: float = Field(ge=0.0, le=1.0)


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class _TypeSafeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    answers: dict[str, _ChoiceAnswer]
    usage: _Usage


class TypeSafeProvider:
    """Classify with Jev Choice while keeping probability and certainty distinct."""

    name = "typesafe"
    required_env_vars = ("TYPESAFE_API_KEY",)
    benchmark_settings: ClassVar[dict[str, JsonScalar]] = {}

    def __init__(
        self,
        *,
        model: str = DEFAULT_TYPESAFE_MODEL,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        pricing: PricingRate | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.pricing = pricing
        self._api_key = os.environ.get("TYPESAFE_API_KEY")
        if not self._api_key:
            raise MissingTypeSafeKeyError
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    async def decide(self, case: EvaluationCase) -> DecisionResult:
        descriptions = case.task.label_descriptions or {}
        criteria = {label: descriptions.get(label, label) for label in case.task.labels}
        payload = {
            "state": case.input,
            "model": self.model,
            "questions": {
                "classification": {
                    "type": "choice",
                    "instructions": "Select the best classification label for the supplied state.",
                    "criteria": criteria,
                }
            },
        }
        outcome = await post_json(
            self._client,
            url=TYPESAFE_ENDPOINT,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        if outcome.error is not None or outcome.data is None:
            return DecisionResult(latency_ms=outcome.latency_ms, error=outcome.error)
        try:
            response = _TypeSafeResponse.model_validate(outcome.data)
            answer = response.answers["classification"]
            if (
                answer.type != "choice"
                or answer.choice not in case.task.labels
                or answer.choice not in answer.probabilities
            ):
                raise ValueError("unexpected choice answer")
        except (KeyError, ValueError, ValidationError):
            return DecisionResult(latency_ms=outcome.latency_ms, error="invalid_response")
        estimated_cost = (
            self.pricing.estimate(response.usage.input_tokens, response.usage.output_tokens)
            if self.pricing
            else None
        )
        return DecisionResult(
            prediction=answer.choice,
            prediction_probability=answer.probabilities[answer.choice],
            probability_source="native_probability",
            provider_confidence=answer.confidence,
            latency_ms=outcome.latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            estimated_cost_usd=estimated_cost,
            cost_source="estimated" if estimated_cost is not None else "unknown",
            model_identifier=response.model,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
