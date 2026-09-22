from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_decision_bench.evaluator import DatasetError, dataset_sha256, load_dataset


def test_sample_dataset_is_valid_and_synthetic() -> None:
    path = Path("datasets/sample.jsonl")
    cases = load_dataset(path)

    assert 20 <= len(cases) <= 30
    assert len({case.id for case in cases}) == len(cases)
    assert all(case.expected in case.task.labels for case in cases)
    assert len(dataset_sha256(path)) == 64


def test_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    case = {
        "id": "duplicate",
        "input": {"text": "hello"},
        "task": {"type": "classification", "labels": ["a", "b"]},
        "expected": "a",
    }
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join((json.dumps(case), json.dumps(case))), encoding="utf-8")

    with pytest.raises(DatasetError, match="Duplicate case id"):
        load_dataset(path)


def test_dataset_rejects_expected_value_outside_labels(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "invalid",
                "input": {"text": "hello"},
                "task": {"type": "classification", "labels": ["a", "b"]},
                "expected": "c",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="line 1"):
        load_dataset(path)
