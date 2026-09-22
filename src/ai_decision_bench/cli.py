"""Command-line interface for running and comparing providers."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from ai_decision_bench import __version__
from ai_decision_bench.evaluator import DatasetError, dataset_sha256, run_benchmark
from ai_decision_bench.models import BenchmarkConfig, BenchmarkReport, ComparisonReport
from ai_decision_bench.pricing import PricingRate, find_pricing
from ai_decision_bench.providers.base import DecisionProvider
from ai_decision_bench.providers.mock import MockProvider
from ai_decision_bench.providers.openai import (
    DEFAULT_OPENAI_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_REASONING_EFFORT,
    OpenAIProvider,
)
from ai_decision_bench.providers.typesafe import DEFAULT_TYPESAFE_MODEL, TypeSafeProvider

_PROVIDER_CLASSES = {
    MockProvider.name: MockProvider,
    TypeSafeProvider.name: TypeSafeProvider,
    OpenAIProvider.name: OpenAIProvider,
}
_PROVIDERS = tuple(_PROVIDER_CLASSES)


class _SafeArgumentParser(argparse.ArgumentParser):
    """Avoid reflecting unknown command-line values that may be credentials."""

    def error(self, message: str) -> None:
        if message.startswith("unrecognized arguments:"):
            unknown = message.removeprefix("unrecognized arguments:").split()
            option_names = [value for value in unknown if value.startswith("-")]
            rendered = ", ".join(option_names) if option_names else "unknown argument"
            message = f"unrecognized option(s): {rendered}"
        super().error(message)


class MissingCredentialsError(RuntimeError):
    def __init__(self, missing: list[tuple[str, str]]) -> None:
        lines: list[str] = []
        for provider, variable in missing:
            if lines:
                lines.append("")
            lines.extend(
                (
                    f"Missing credential for provider: {provider}",
                    f"Required environment variable: {variable}",
                )
            )
        super().__init__("\n".join(lines))


class _ProgressReporter:
    """Render compact TTY progress without adding a runtime dependency."""

    _BAR_WIDTH = 20

    def __init__(self, provider: str, stream: TextIO | None = None) -> None:
        self.provider = provider
        self.stream = stream or sys.stderr
        self.interactive = self.stream.isatty()
        self.started = time.perf_counter()
        self.active = False
        self.finished = False
        self.previous_width = 0

    def update(self, completed: int, total: int) -> None:
        self.active = True
        ratio = completed / total if total else 1.0
        elapsed = time.perf_counter() - self.started
        filled = round(self._BAR_WIDTH * ratio)
        bar = "#" * filled + "-" * (self._BAR_WIDTH - filled)
        line = (
            f"{self.provider}: [{bar}] Processed {completed}/{total} ({ratio:.0%}) {elapsed:.1f}s"
        )
        if self.interactive:
            padding = " " * max(0, self.previous_width - len(line))
            end = "\n" if completed == total else "\r"
            print(f"\r{line}{padding}", end=end, file=self.stream, flush=True)
            self.previous_width = len(line)
        elif completed in {0, total}:
            print(line, file=self.stream, flush=True)
        self.finished = completed == total

    def close(self) -> None:
        if self.interactive and self.active and not self.finished:
            print(file=self.stream, flush=True)


def _validate_credentials(provider_names: list[str]) -> None:
    """Validate only selected providers, collecting every missing variable first."""

    missing: list[tuple[str, str]] = []
    for provider_name in provider_names:
        provider_class = _PROVIDER_CLASSES[provider_name]
        missing.extend(
            (provider_name, variable)
            for variable in provider_class.required_env_vars
            if not os.environ.get(variable)
        )
    if missing:
        raise MissingCredentialsError(missing)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL dataset path")
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Concurrent requests (default: 1)"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-attempt timeout in seconds"
    )
    parser.add_argument(
        "--max-retries", type=int, default=2, help="Retries after the first attempt"
    )
    parser.add_argument("--output", type=Path, help="Write .json or .md result output")
    parser.add_argument(
        "--input-price-per-million",
        type=float,
        help="Override input token price in USD per million tokens",
    )
    parser.add_argument(
        "--output-price-per-million",
        type=float,
        help="Override output token price in USD per million tokens",
    )
    parser.add_argument(
        "--pricing-reference-date",
        help="Reference date for pricing override (YYYY-MM-DD)",
    )


def _add_openai_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--openai-reasoning-effort",
        default=DEFAULT_OPENAI_REASONING_EFFORT,
        metavar="LEVEL",
        help=f"Responses API reasoning effort (default: {DEFAULT_OPENAI_REASONING_EFFORT})",
    )
    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=DEFAULT_OPENAI_MAX_OUTPUT_TOKENS,
        metavar="TOKENS",
        help=(
            "Responses API output-token cap including reasoning tokens "
            f"(default: {DEFAULT_OPENAI_MAX_OUTPUT_TOKENS})"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="ai-decision-bench",
        description="Benchmark structured AI decision providers on one dataset.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one provider")
    run_parser.add_argument("--provider", choices=_PROVIDERS, required=True)
    run_parser.add_argument("--model", help="Provider model identifier")
    _add_openai_options(run_parser)
    _add_common_options(run_parser)

    compare_parser = subparsers.add_parser("compare", help="Measure multiple providers")
    compare_parser.add_argument(
        "--providers",
        required=True,
        help="Comma-separated providers: mock,typesafe,openai",
    )
    compare_parser.add_argument("--typesafe-model", help="TypeSafe model override")
    compare_parser.add_argument("--openai-model", help="OpenAI model (or use OPENAI_MODEL)")
    _add_openai_options(compare_parser)
    _add_common_options(compare_parser)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be at least 0")
    if not args.openai_reasoning_effort.strip():
        parser.error("--openai-reasoning-effort must not be empty")
    if args.openai_max_output_tokens < 1:
        parser.error("--openai-max-output-tokens must be at least 1")
    prices = (args.input_price_per_million, args.output_price_per_million)
    if any(price is not None and price < 0 for price in prices):
        parser.error("pricing overrides must be non-negative")
    if any(price is not None for price in prices) and not args.pricing_reference_date:
        parser.error("--pricing-reference-date is required with a pricing override")
    if args.output and args.output.suffix.lower() not in {".json", ".md"}:
        parser.error("--output must end in .json or .md")
    if args.command == "run" and args.model and args.model.strip().casefold() == args.provider:
        parser.error(f"--model must be an API model ID, not the provider name '{args.provider}'")
    if args.command == "compare":
        provider_models = (
            ("typesafe", args.typesafe_model, "--typesafe-model"),
            ("openai", args.openai_model, "--openai-model"),
        )
        for provider, model, option in provider_models:
            if model and model.strip().casefold() == provider:
                parser.error(
                    f"{option} must be an API model ID, not the provider name '{provider}'"
                )


def _pricing_for(args: argparse.Namespace, provider: str, model: str | None) -> PricingRate | None:
    if args.input_price_per_million is None and args.output_price_per_million is None:
        return find_pricing(provider, model)
    return PricingRate(
        provider=provider,
        model=model or "unspecified",
        input_per_million_usd=args.input_price_per_million or 0.0,
        output_per_million_usd=args.output_price_per_million or 0.0,
        currency="USD",
        reference_date=args.pricing_reference_date,
    )


def _make_provider(
    name: str,
    *,
    model: str | None,
    args: argparse.Namespace,
) -> DecisionProvider:
    if name == "mock":
        return MockProvider()
    if name == "typesafe":
        resolved_model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", DEFAULT_TYPESAFE_MODEL)
        return TypeSafeProvider(
            model=resolved_model,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
            pricing=_pricing_for(args, name, resolved_model),
        )
    if name == "openai":
        resolved_model = model or os.environ.get("OPENAI_MODEL")
        return OpenAIProvider(
            model=resolved_model,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries,
            reasoning_effort=args.openai_reasoning_effort,
            max_output_tokens=args.openai_max_output_tokens,
            pricing=_pricing_for(args, name, resolved_model),
        )
    raise ValueError("unsupported provider")


def _config(
    args: argparse.Namespace,
    pricing: PricingRate | None,
    provider: DecisionProvider,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
        input_price_per_million_usd=(pricing.input_per_million_usd if pricing else None),
        output_price_per_million_usd=(pricing.output_per_million_usd if pricing else None),
        pricing_reference_date=pricing.reference_date if pricing else None,
        provider_settings=provider.benchmark_settings,
    )


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}ms"


def _format_ece(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_cost(report: BenchmarkReport) -> str:
    metrics = report.metrics
    if metrics.total_reported_cost_usd is not None:
        return f"${metrics.total_reported_cost_usd:.6f} reported"
    if metrics.total_estimated_cost_usd is not None:
        return f"${metrics.total_estimated_cost_usd:.6f} estimated"
    return "n/a"


def render_summary(reports: list[BenchmarkReport]) -> str:
    rows = [
        (
            "Provider",
            "Model",
            "Cases",
            "Accuracy",
            "ECE",
            "ECE cases",
            "Median",
            "P95",
            "Failures",
            "Cost (USD)",
        )
    ]
    for report in reports:
        metrics = report.metrics
        rows.append(
            (
                report.provider,
                report.model_identifier or report.model or "n/a",
                str(metrics.total),
                f"{metrics.accuracy:.1%}",
                _format_ece(metrics.expected_calibration_error),
                f"{metrics.calibration_case_count}/{metrics.total}",
                _format_ms(metrics.latency.median_ms),
                _format_ms(metrics.latency.p95_ms),
                f"{metrics.failures} ({metrics.failure_rate:.1%})",
                _format_cost(report),
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    header = "  ".join(value.ljust(widths[index]) for index, value in enumerate(rows[0]))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows[1:]
    ]
    return "\n".join([header, separator, *body])


def _failure_notices(reports: list[BenchmarkReport]) -> list[str]:
    notices: list[str] = []
    for report in reports:
        metrics = report.metrics
        if metrics.failures == 0:
            continue
        categories = Counter(
            item.result.error for item in report.cases if item.result.error is not None
        )
        category_text = ", ".join(
            f"{category}: {count}" for category, count in sorted(categories.items())
        )
        if metrics.successful == 0:
            notices.append(
                f"error: {report.provider}: all {metrics.total} cases failed; "
                f"no valid decisions were produced ({category_text}). Check the provider "
                "and model configuration."
            )
        else:
            notices.append(
                f"warning: {report.provider}: {metrics.failures} of {metrics.total} cases "
                f"failed ({category_text})."
            )
    return notices


def render_markdown(reports: list[BenchmarkReport]) -> str:
    lines = [
        "# ai-decision-bench results",
        "",
        "| Provider | Model | Cases | Accuracy | ECE | ECE cases | Median | P95 | "
        "Failures | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        metrics = report.metrics
        lines.append(
            "| "
            + " | ".join(
                (
                    report.provider,
                    report.model_identifier or report.model or "n/a",
                    str(metrics.total),
                    f"{metrics.accuracy:.1%}",
                    _format_ece(metrics.expected_calibration_error),
                    f"{metrics.calibration_case_count}/{metrics.total}",
                    _format_ms(metrics.latency.median_ms),
                    _format_ms(metrics.latency.p95_ms),
                    f"{metrics.failures} ({metrics.failure_rate:.1%})",
                    _format_cost(report),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            f"Dataset SHA-256: `{reports[0].dataset_sha256}`",
            "",
            "Failures are included in the accuracy denominator. Latency covers each "
            "complete provider operation, including retries and backoff.",
        )
    )
    return "\n".join(lines) + "\n"


def _write_output(path: Path, payload: BenchmarkReport | ComparisonReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return
    reports = payload.reports if isinstance(payload, ComparisonReport) else [payload]
    path.write_text(render_markdown(reports), encoding="utf-8")


async def _run(args: argparse.Namespace) -> list[BenchmarkReport]:
    _validate_credentials([args.provider])
    provider = _make_provider(args.provider, model=args.model, args=args)
    pricing = getattr(provider, "pricing", None)
    report = await _run_with_progress(provider, args, pricing)
    if args.output:
        _write_output(args.output, report)
    return [report]


async def _run_with_progress(
    provider: DecisionProvider,
    args: argparse.Namespace,
    pricing: PricingRate | None,
) -> BenchmarkReport:
    progress = _ProgressReporter(provider.name)
    try:
        return await run_benchmark(
            provider,
            args.dataset,
            _config(args, pricing, provider),
            progress_callback=progress.update,
        )
    finally:
        progress.close()


def _parse_providers(value: str) -> list[str]:
    providers = [item.strip() for item in value.split(",") if item.strip()]
    if not providers or any(item not in _PROVIDERS for item in providers):
        raise ValueError("--providers must contain only mock, typesafe, or openai")
    if len(set(providers)) != len(providers):
        raise ValueError("--providers must not contain duplicates")
    return providers


async def _compare(args: argparse.Namespace) -> list[BenchmarkReport]:
    providers = _parse_providers(args.providers)
    _validate_credentials(providers)
    reports: list[BenchmarkReport] = []
    for name in providers:
        model = args.typesafe_model if name == "typesafe" else args.openai_model
        provider = _make_provider(name, model=model, args=args)
        pricing = getattr(provider, "pricing", None)
        reports.append(await _run_with_progress(provider, args, pricing))
    if args.output:
        comparison = ComparisonReport(
            timestamp=datetime.now(UTC),
            dataset=reports[0].dataset,
            dataset_sha256=dataset_sha256(args.dataset),
            benchmark_version=__version__,
            reports=reports,
        )
        _write_output(args.output, comparison)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        reports = asyncio.run(_run(args) if args.command == "run" else _compare(args))
    except (DatasetError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_summary(reports))
    if args.output:
        print(f"\nResult written to: {args.output}")
    else:
        print("\nResult not saved. Use --output PATH.json to keep it.")
    notices = _failure_notices(reports)
    if notices:
        print("\n".join(notices), file=sys.stderr)
    return 1 if any(report.metrics.successful == 0 for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
