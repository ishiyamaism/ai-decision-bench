# ai-decision-bench

A small, reproducible benchmark for structured AI decision APIs.

Evaluate structured decisions on your own labelled datasets across multiple providers
using the same metrics and evaluation pipeline. Bring your own dataset and provider
credentials; the bundled synthetic dataset is only a runnable example.

## Why

Accuracy alone does not describe an operational decision API. Confidence calibration,
latency, cost, and failure behavior also affect whether a structured decision is useful
in a real workflow. This harness sends the same source input, label definitions, and
expected answer through each provider adapter, then normalizes the result before
calculating metrics.

This project measures provider behavior. It does not choose a winner, create an overall
score, or recommend a provider.

## Features

- Bring your own JSONL dataset
- Bring your own API keys
- Provider-neutral evaluation interface
- Accuracy and failure-rate measurement
- Confidence buckets and Expected Calibration Error (ECE)
- Mean, median, and p95 latency
- Reported versus estimated cost metadata
- Dataset SHA-256 and reproducible JSON or Markdown results
- Conservative, configurable concurrency, timeout, and retry limits
- Offline MockProvider for tests and smoke runs

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ishiyamaism/ai-decision-bench.git
cd ai-decision-bench
uv sync
uv run pytest
```

Run the local deterministic provider without credentials:

```bash
uv run ai-decision-bench run \
  --provider mock \
  --dataset datasets/sample.jsonl
```

Write a reproducible result:

```bash
uv run ai-decision-bench run \
  --provider mock \
  --dataset datasets/sample.jsonl \
  --output results/mock.json
```

The mock is operational scaffolding, not an AI-performance baseline.

## Use your own dataset

The primary workflow is to point the CLI at a labelled JSONL file you control:

```bash
uv run ai-decision-bench run \
  --provider typesafe \
  --dataset path/to/my-data.jsonl

uv run ai-decision-bench run \
  --provider openai \
  --model YOUR_MODEL_ID \
  --dataset path/to/my-data.jsonl
```

Each non-empty line is one case:

```json
{
  "id": "example-001",
  "input": {"text": "Please cancel my subscription."},
  "task": {
    "type": "classification",
    "labels": ["billing", "technical_support", "account", "sales", "cancellation", "other"]
  },
  "expected": "cancellation"
}
```

`expected` is the human-selected correct label and must appear in `task.labels`. Labels
must be unique. An optional `label_descriptions` object can provide a neutral rubric;
the same rubric is supplied to every provider. Case IDs must be unique within a file.

The v1 task is closed-set classification. See [datasets/README.md](datasets/README.md)
for data guidance. Do not put credentials or private data in a dataset intended for a
public repository.

## Providers

### TypeSafe AI / Jev

Set the API key in the process environment:

```bash
export TYPESAFE_API_KEY="your-key"
uv run ai-decision-bench run \
  --provider typesafe \
  --dataset datasets/sample.jsonl
```

`TypeSafeProvider` uses the official System One HTTP endpoint and a Choice question.
The default model alias is `jev-latest`; use `--model` or `TYPESAFE_DEFAULT_MODEL` to
override it. Jev's returned confidence is marked `native_probability`.

### OpenAI

```bash
export OPENAI_API_KEY="your-key"
uv run ai-decision-bench run \
  --provider openai \
  --model YOUR_MODEL_ID \
  --dataset datasets/sample.jsonl
```

The model is never hard-coded. It can also be set with `OPENAI_MODEL`. The adapter uses
the Responses API with strict JSON Schema Structured Outputs and sets `store=false`.
OpenAI classification confidence is `null`: the harness does not ask a model to report
its own confidence and present that value as a native probability.

### Mock

MockProvider is a deterministic keyword baseline used by CI and local smoke tests. It
makes no network requests and needs no credential.

## Compare providers

With both credentials exported:

```bash
export TYPESAFE_API_KEY="..."
export OPENAI_API_KEY="..."
uv run ai-decision-bench compare \
  --providers typesafe,openai \
  --openai-model gpt-5.6-luna \
  --dataset datasets/sample.jsonl \
  --output results/comparison.json
```

Providers run sequentially so cross-provider traffic does not distort the comparison.
Within each provider, `--concurrency` controls the number of in-flight cases and defaults
to `1`. Each adapter uses finite timeouts and retries:

```bash
--concurrency 1 --timeout 30 --max-retries 2
```

The summary reports measurements only:

```text
mock: [####################] Processed 24/24 (100%) 0.0s

Provider  Model                Cases  Accuracy  ECE     Median  P95    Failures  Cost (USD)
--------  -------------------  -----  --------  ------  ------  -----  --------  ----------
mock      keyword-baseline-v1  24     100.0%    0.1000  0.0ms   0.0ms  0 (0.0%)  n/a
```

Mock values vary by machine and are not real-provider performance results.
Interactive terminals update the progress bar in place. In redirected output, only the
start and completion progress lines are written to stderr; the result table remains on
stdout. `Processed` is the number of completed attempts, not the number of successful
decisions. Partial failures produce a warning; an all-failed provider produces a clear
error and a non-zero exit status.

Provider names are not model IDs. For example, use an OpenAI API model ID with
`--provider openai`; `--model openai` is rejected before any API request starts.

## Metrics

- **Accuracy:** correct cases divided by all cases. Provider failures remain in the
  denominator and therefore count as incorrect.
- **Failure rate:** API errors, timeouts, invalid responses, schema validation errors,
  and provider exceptions divided by all cases.
- **Latency:** wall-clock time from the start of a provider operation until its complete
  response or failure, including retry backoff. Local dataset loading, metric calculation,
  and result-file writing are excluded. Mean, median, p95, minimum, and maximum are saved.
- **Confidence calibration:** successful cases with confidence are grouped into buckets
  with count, average confidence, and observed accuracy.
- **ECE:** Expected Calibration Error summarizes how closely stated confidence matches
  observed accuracy. Lower is better for that provider and dataset.
- **Cost:** provider-reported cost and token-based estimated cost are separate fields.
  Unknown cost stays `null`, not zero.

Confidence semantics are not necessarily equivalent across providers. Native,
provider-reported, model-self-reported, and unavailable confidence sources are recorded
separately. The current OpenAI adapter reports confidence as unavailable.

## Pricing

The TypeSafe `jev-latest` / `jev-1.13.0` input price is a dated built-in assumption from
the official model page, checked 2026-09-22. Output price is recorded as zero for that
dated assumption. OpenAI pricing is model-dependent and is not guessed.

Override pricing explicitly when needed:

```bash
uv run ai-decision-bench run \
  --provider openai \
  --model YOUR_MODEL_ID \
  --dataset datasets/sample.jsonl \
  --input-price-per-million 0.00 \
  --output-price-per-million 0.00 \
  --pricing-reference-date 2026-09-22
```

Replace the example zeros with the documented rates for the selected model. Pricing
assumptions may become outdated; result files retain their reference date.

## Results and reproducibility

Use `--output path.json` or `--output path.md`. JSON includes:

- UTC execution timestamp;
- provider and requested/resolved model identifiers;
- safe dataset name and SHA-256;
- benchmark and Python versions;
- concurrency, timeout, retry, and pricing configuration;
- aggregate metrics; and
- per-case expected value, normalized prediction, confidence metadata, usage, cost, and
  sanitized error category.

Source dataset inputs, HTTP headers, credentials, and raw provider responses are not
written. A dataset hash distinguishes files that have the same name but different
content.

Without `--output`, the summary is displayed but no result file is retained. The CLI
states this explicitly at the end of a run.

Benchmark results depend on model version, API configuration, dataset, execution date,
network conditions, and region.

## Security and BYOK

`ai-decision-bench` uses your own provider credentials locally. API keys are read from
environment variables and sent only to the corresponding provider over HTTPS.

This project does not provide, proxy, collect, store, or transmit users' API keys to a
project-operated service. It has no hosted backend, telemetry, analytics, crash reporter,
database, browser storage, or credential prompt. Provider redirects are not followed.

You only need credentials for the providers you actually run. Mock requires none,
TypeSafe-only runs require only `TYPESAFE_API_KEY`, and OpenAI-only runs require only
`OPENAI_API_KEY`. Before a comparison starts, the CLI checks every selected provider's
declared `required_env_vars` and reports all missing variable names without starting any
provider request.

- Never commit API keys or place credentials in datasets or result files.
- Do not commit private datasets to a public repository.
- Review any result before publishing it, especially case labels and model identifiers.
- Public CI uses MockProvider and does not need real provider credentials.
- `.env` files are ignored; `.env.example` contains empty variable names only. The
  application does not automatically load dotenv files.
- API keys cannot be supplied as CLI arguments and are never collected interactively.

If a key is missing, the CLI names the required environment variable but never prints a
value. Logging and errors use sanitized categories rather than provider response bodies.

## Position in the ecosystem

The [TypeSafe Console Playground](https://console.typesafe.ai/playground) is useful for
trying individual states and typed questions. [TypeSafe Workflow Evals](https://evals.typesafe.ai/)
publish fixed workflow evaluations. The official
[System One Adapter](https://github.com/typesafe-ai/system-one-adapter-python) executes
TypeSafe-style questions through regular LLM providers.

`ai-decision-bench` has a different scope: run a user's labelled dataset through
provider-neutral adapters and calculate the same metrics locally. Design research and the
dependency decision are recorded in [docs/design.md](docs/design.md).

## Limitations

- The sample dataset is small and synthetic. It demonstrates the interface only.
- This is not a benchmark of general AI capability.
- Accuracy changes with dataset selection, labels, and rubric quality.
- Latency depends on network path, provider load, rate limits, and region.
- Pricing changes and estimates are not invoices.
- Confidence can mean different things across providers.
- ECE on a small dataset is noisy.
- The harness and its timing measurements have measurement error.
- A sample run is not sufficient evidence of production suitability.
- Provider adapters intentionally use a shared neutral task statement, not extensive
  provider-specific prompt tuning.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

CI performs these checks without external API access. The CLI version is available with:

```bash
uv run ai-decision-bench --version
```

## API references

- [TypeSafe HTTP API](https://docs.typesafe.ai/api)
- [TypeSafe models and dated pricing](https://docs.typesafe.ai/models)
- [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

## License

[MIT](LICENSE)
