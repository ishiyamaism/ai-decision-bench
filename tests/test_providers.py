from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx

from ai_decision_bench.models import ClassificationTask, EvaluationCase
from ai_decision_bench.pricing import PricingRate
from ai_decision_bench.providers.openai import OPENAI_ENDPOINT, OpenAIProvider
from ai_decision_bench.providers.typesafe import TYPESAFE_ENDPOINT, TypeSafeProvider


def _case() -> EvaluationCase:
    return EvaluationCase(
        id="case-1",
        input={"text": "Please cancel my subscription."},
        task=ClassificationTask(
            type="classification",
            labels=["billing", "cancellation", "other"],
        ),
        expected="cancellation",
    )


def test_typesafe_adapter_uses_documented_choice_contract() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "classification": {
                        "type": "choice",
                        "choice": "cancellation",
                        "probabilities": {
                            "billing": 0.02,
                            "cancellation": 0.96,
                            "other": 0.02,
                        },
                        "confidence": 0.94,
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 5},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key"}):
        provider = TypeSafeProvider(
            client=client,
            max_retries=0,
            pricing=PricingRate("typesafe", "jev-latest", 0.042, 0, "USD", "2026-09-22"),
        )
        result = asyncio.run(provider.decide(_case()))
    asyncio.run(client.aclose())

    assert str(seen[0].url) == TYPESAFE_ENDPOINT
    assert seen[0].headers["authorization"] == "Bearer test-key"
    payload = json.loads(seen[0].content)
    assert payload["model"] == "jev-latest"
    assert payload["questions"]["classification"]["type"] == "choice"
    assert set(payload["questions"]["classification"]["criteria"]) == {
        "billing",
        "cancellation",
        "other",
    }
    assert result.prediction == "cancellation"
    assert result.confidence == 0.94
    assert result.confidence_source == "native_probability"
    assert result.model_identifier == "jev-1.13.0"
    assert result.estimated_cost_usd is not None


def test_openai_adapter_uses_responses_structured_outputs_without_self_confidence() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "example-model-2026-01-01",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"prediction":"cancellation"}',
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 80, "output_tokens": 8},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider(
            model="example-model",
            client=client,
            max_retries=0,
        )
        result = asyncio.run(provider.decide(_case()))
    asyncio.run(client.aclose())

    assert str(seen[0].url) == OPENAI_ENDPOINT
    payload = json.loads(seen[0].content)
    assert payload["store"] is False
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False
    assert result.prediction == "cancellation"
    assert result.confidence is None
    assert result.confidence_source == "unavailable"


def test_typesafe_adapter_rejects_malformed_response() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"unexpected": "response"})
        )
    )
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-key"}):
        provider = TypeSafeProvider(client=client, max_retries=0)
        result = asyncio.run(provider.decide(_case()))
    asyncio.run(client.aclose())

    assert result.error == "invalid_response"
    assert result.prediction is None


def test_openai_adapter_rejects_incomplete_response() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "model": "example-model",
                    "status": "incomplete",
                    "output": [],
                },
            )
        )
    )
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider(model="example-model", client=client, max_retries=0)
        result = asyncio.run(provider.decide(_case()))
    asyncio.run(client.aclose())

    assert result.error == "invalid_response"
    assert result.prediction is None
