from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import httpx

from ai_decision_bench.cli import _validate_credentials, build_parser, main, render_markdown
from ai_decision_bench.evaluator import run_benchmark
from ai_decision_bench.models import BenchmarkConfig, BenchmarkReport
from ai_decision_bench.providers.typesafe import TypeSafeProvider

SENTINEL = "SUPER_SECRET_SENTINEL_DO_NOT_EXPOSE"


def test_secret_is_absent_from_errors_logs_and_result_files(
    caplog,
    capsys,
) -> None:
    caplog.set_level(logging.DEBUG)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"invalid key {SENTINEL}"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": SENTINEL}):
        provider = TypeSafeProvider(
            client=client,
            max_retries=0,
        )
        report = asyncio.run(
            run_benchmark(
                provider,
                Path("datasets/sample.jsonl"),
                BenchmarkConfig(concurrency=1, timeout_seconds=1, max_retries=0),
            )
        )
    asyncio.run(client.aclose())

    json_result = report.model_dump_json(indent=2)
    markdown_result = render_markdown([report])
    captured = capsys.readouterr()
    combined = "\n".join(
        (
            json_result,
            markdown_result,
            captured.out,
            captured.err,
            caplog.text,
        )
    )
    assert SENTINEL not in combined
    assert "Authorization" not in combined
    assert "Bearer" not in combined
    assert "TYPESAFE_API_KEY" not in json_result
    assert "OPENAI_API_KEY" not in json_result
    assert all(item.result.error == "http_401" for item in report.cases)


def test_missing_key_message_never_contains_environment_value(monkeypatch, capsys) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    exit_code = main(
        [
            "run",
            "--provider",
            "typesafe",
            "--dataset",
            "datasets/sample.jsonl",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Missing credential for provider: typesafe" in captured.err
    assert "Required environment variable: TYPESAFE_API_KEY" in captured.err
    assert SENTINEL not in captured.err


def test_redirect_does_not_forward_authorization_header() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={"location": "https://example.invalid/collect"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with patch.dict("os.environ", {"TYPESAFE_API_KEY": SENTINEL}):
        provider = TypeSafeProvider(client=client, max_retries=0)
        report = asyncio.run(
            run_benchmark(
                provider,
                Path("datasets/sample.jsonl"),
                BenchmarkConfig(concurrency=1, timeout_seconds=1, max_retries=0),
            )
        )
    asyncio.run(client.aclose())

    assert len(requests) == len(report.cases)
    assert all(request.url.host == "api.typesafe.ai" for request in requests)
    assert all(item.result.error == "http_307" for item in report.cases)


def test_unused_provider_credentials_are_not_required(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-test-value")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _validate_credentials(["typesafe"])
    _validate_credentials(["mock"])

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-value")
    _validate_credentials(["openai"])


def test_mock_requires_no_provider_credentials(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _validate_credentials(["mock"])


def test_compare_validates_all_credentials_before_creating_a_provider(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-test-value")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider_creation_started = False

    def fail_if_called(*args, **kwargs):
        nonlocal provider_creation_started
        provider_creation_started = True
        raise AssertionError("provider creation must not begin before credential preflight")

    monkeypatch.setattr("ai_decision_bench.cli._make_provider", fail_if_called)

    exit_code = main(
        [
            "compare",
            "--providers",
            "typesafe,openai",
            "--dataset",
            "datasets/sample.jsonl",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert provider_creation_started is False
    assert "Missing credential for provider: openai" in captured.err
    assert "Required environment variable: OPENAI_API_KEY" in captured.err
    assert "typesafe-test-value" not in captured.err


def test_compare_reports_every_missing_credential_before_provider_creation(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("provider creation must not begin before credential preflight")

    monkeypatch.setattr("ai_decision_bench.cli._make_provider", fail_if_called)

    exit_code = main(
        [
            "compare",
            "--providers",
            "typesafe,openai",
            "--dataset",
            "datasets/sample.jsonl",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Missing credential for provider: typesafe" in captured.err
    assert "Required environment variable: TYPESAFE_API_KEY" in captured.err
    assert "Missing credential for provider: openai" in captured.err
    assert "Required environment variable: OPENAI_API_KEY" in captured.err


def test_cli_has_no_api_key_options_and_unknown_values_are_redacted(capsys) -> None:
    help_text = build_parser().format_help()
    assert "--api-key" not in help_text
    assert "--typesafe-key" not in help_text
    assert "--openai-key" not in help_text

    try:
        main(
            [
                "run",
                "--provider",
                "mock",
                "--dataset",
                "datasets/sample.jsonl",
                "--api-key",
                SENTINEL,
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    captured = capsys.readouterr()
    assert SENTINEL not in captured.err


def test_result_schema_does_not_have_credential_or_raw_response_fields() -> None:
    forbidden = {"api_key", "authorization", "headers", "raw_response", "environment"}

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key.lower() not in forbidden
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    schema = BenchmarkReport.model_json_schema()
    walk(schema)
