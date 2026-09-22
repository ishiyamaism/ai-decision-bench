# Repository instructions for coding agents

These instructions apply to the entire repository. They are defaults for maintaining
this project, not restrictions on what a fork may become. If a fork owner explicitly
changes the product scope, update the tests and user-facing documentation so the new
behavior is intentional and clear.

## Project intent

`ai-decision-bench` measures structured AI decisions on a user's own labeled data. It
compares observable dimensions such as accuracy, failures, latency, cost, and available
probability calibration. It does not publish a fixed leaderboard, calculate a composite
score, or declare a universal winner.

The project is deliberately small, provider-neutral, local-first, and bring-your-own-key.

## Start here

- Read `README.md` for the supported behavior and CLI contract.
- Read `docs/design.md` before changing provider semantics, result fields, probability
  handling, or the evaluation methodology.
- Use Python 3.12+ and the locked `uv` environment.
- Run the offline smoke test before and after a substantial change:

  ```bash
  uv sync --locked --all-groups
  uv run ai-decision-bench run --provider mock --dataset datasets/sample.jsonl
  ```

## Architecture

- `src/ai_decision_bench/models.py`: shared dataset and result schema.
- `src/ai_decision_bench/evaluator.py`: dataset loading and provider-neutral evaluation.
- `src/ai_decision_bench/metrics.py`: shared metrics.
- `src/ai_decision_bench/providers/`: provider-specific request, response, and error
  normalization.
- `src/ai_decision_bench/cli.py`: CLI parsing, provider registration, credential
  preflight, reporting, and output.
- `tests/`: offline unit and contract tests. Provider tests use mocked HTTP transports.

Keep provider SDK concepts and wire formats inside their adapter. Shared models,
evaluation, and metrics must not depend on a particular vendor.

## Behavioral invariants

Preserve these unless the requested change explicitly redefines the benchmark:

1. Send every compared provider the same case input, closed label set, and neutral label
   descriptions. Do not silently add provider-specific prompt advantages.
2. A failure remains in the accuracy denominator and contributes to the failure rate.
3. `prediction_probability` is the probability assigned to the selected label. Use it
   for ECE only when the provider actually exposes a suitable value.
4. Keep provider-specific certainty in `provider_confidence`. Never relabel it as a
   probability or compare it as if definitions were universal.
5. Represent unavailable probability and cost as `null`, not zero and not a
   model-invented estimate. Keep reported and estimated cost separate.
6. Retain available usage and cost metadata from valid but incomplete or unusable API
   responses, while still marking the decision as failed.
7. Measure provider-operation latency, including retries and backoff, but not local
   dataset loading, metric calculation, or result-file writing.
8. Keep comparisons descriptive. Do not add an overall winner or composite score to the
   maintained project without an explicit scope decision and documented methodology.

## Security and privacy

- Read credentials only from the documented environment variables. Do not accept keys
  as CLI arguments, write them to disk, or print them.
- Do not log request headers, raw provider responses, dataset contents, or exception text
  that may contain sensitive material. Expose stable, sanitized error categories.
- Keep finite timeouts and bounded retries. Do not follow provider redirects.
- Do not add telemetry, analytics, a hosted relay, or remote data storage without an
  explicit product decision and prominent documentation.
- Never commit a real credential, private dataset, or result derived from private data.
- Tests and public CI must not require credentials or make real provider requests.
- Treat case input as untrusted data. Preserve prompt-injection boundaries when adapting
  inputs to a general-purpose model.

## Adding or changing a provider

When adding a provider:

1. Implement the `DecisionProvider` protocol, including `name`, `model`,
   `required_env_vars`, `benchmark_settings`, `decide`, and `aclose`.
2. Normalize the result into `DecisionResult`; do not leak the provider response into
   the shared schema merely for convenience.
3. Reuse the bounded HTTP/retry behavior in `providers/_http.py` where applicable and
   close only clients owned by the adapter.
4. Register the provider and its model/configuration options in the CLI. Credential
   preflight must happen before any provider in a comparison starts.
5. Add offline tests for successful, failed, malformed, incomplete, and credential-
   missing behavior as applicable. Use `httpx.MockTransport` or an equivalent fake.
6. Document configuration, probability semantics, cost semantics, and official API
   references. Date any built-in pricing assumption.

Verify external API behavior against first-party documentation. Avoid guessing current
model IDs, prices, response fields, or probability meanings.

## Data, reports, and compatibility

- The bundled dataset must remain synthetic and safe to publish.
- Preserve input order and unique case IDs in reports.
- Keep machine-specific absolute paths and source input content out of saved reports.
- `results/` is ignored except for its placeholder; do not commit local benchmark output
  by default.
- Treat changes to JSON report fields or dataset validation as compatibility changes.
  Update tests, README examples, design notes, and the package version when appropriate.

## Validation

Before handing off a change, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Add focused tests for changed behavior. Do not weaken or delete a safety assertion just
to make a change pass. If a live-provider check is useful, leave it as an optional manual
step and never assume credentials are available.

