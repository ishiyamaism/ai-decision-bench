"""Shared dataset, provider result, and report models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonScalar = str | bool | int | float | None
ConfidenceSource = Literal[
    "native_probability",
    "provider_reported",
    "model_self_reported",
    "unavailable",
]
CostSource = Literal["reported", "estimated", "unknown"]


class ClassificationTask(BaseModel):
    """A closed-set classification task."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["classification"]
    labels: list[str] = Field(min_length=2)
    label_descriptions: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_labels(self) -> ClassificationTask:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("task labels must be unique")
        if any(not label.strip() for label in self.labels):
            raise ValueError("task labels must not be empty")
        if self.label_descriptions is not None:
            unknown = set(self.label_descriptions) - set(self.labels)
            if unknown:
                raise ValueError("label_descriptions contains an unknown label")
        return self


class EvaluationCase(BaseModel):
    """One labelled example from a JSONL dataset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: dict[str, Any]
    task: ClassificationTask
    expected: str

    @model_validator(mode="after")
    def validate_expected(self) -> EvaluationCase:
        if self.expected not in self.task.labels:
            raise ValueError("expected must be one of task.labels")
        return self


class DecisionResult(BaseModel):
    """Provider-independent result for one decision."""

    model_config = ConfigDict(extra="forbid")

    prediction: JsonScalar = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource = "unavailable"
    latency_ms: float = Field(ge=0.0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reported_cost_usd: float | None = Field(default=None, ge=0.0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    cost_source: CostSource = "unknown"
    model_identifier: str | None = None
    error: str | None = None


class CaseEvaluation(BaseModel):
    """An expected value paired with a normalized provider result."""

    case_id: str
    expected: JsonScalar
    correct: bool
    result: DecisionResult


class LatencyMetrics(BaseModel):
    mean_ms: float | None
    median_ms: float | None
    p95_ms: float | None
    min_ms: float | None
    max_ms: float | None


class ConfidenceBucket(BaseModel):
    range: str
    count: int
    average_confidence: float | None
    accuracy: float | None


class BenchmarkMetrics(BaseModel):
    total: int
    successful: int
    correct: int
    failures: int
    accuracy: float
    failure_rate: float
    latency: LatencyMetrics
    confidence_buckets: list[ConfidenceBucket]
    expected_calibration_error: float | None
    total_reported_cost_usd: float | None
    total_estimated_cost_usd: float | None


class BenchmarkConfig(BaseModel):
    concurrency: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0.0)
    max_retries: int = Field(ge=0)
    input_price_per_million_usd: float | None = Field(default=None, ge=0.0)
    output_price_per_million_usd: float | None = Field(default=None, ge=0.0)
    pricing_reference_date: str | None = None


class BenchmarkReport(BaseModel):
    timestamp: datetime
    provider: str
    model: str | None
    model_identifier: str | None
    dataset: str
    dataset_sha256: str
    benchmark_version: str
    python_version: str
    config: BenchmarkConfig
    metrics: BenchmarkMetrics
    cases: list[CaseEvaluation]


class ComparisonReport(BaseModel):
    timestamp: datetime
    dataset: str
    dataset_sha256: str
    benchmark_version: str
    reports: list[BenchmarkReport]
