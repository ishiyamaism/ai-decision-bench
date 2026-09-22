# Design notes

## Scope

`ai-decision-bench` is a provider-neutral harness for running a user's labeled JSONL
dataset through interchangeable structured-decision providers. The dataset model,
evaluation loop, metrics, and result schema do not use TypeSafe question or response
types. TypeSafe Choice is confined to `TypeSafeProvider`; OpenAI Structured Outputs is
confined to `OpenAIProvider`.

Each provider declares `required_env_vars`. CLI credential preflight is generic over
that metadata: only selected providers are checked, and comparisons validate the entire
selection before the first provider starts.

The bundled dataset is a CLI and schema example. It is not a fixed leaderboard suite,
and the project does not calculate rankings or an overall score.

## Existing implementations reviewed

The initial design was checked against these implementations and products on
2026-09-22:

- [TypeSafe System One Adapter for Python](https://github.com/typesafe-ai/system-one-adapter-python)
  (`0.2.0`, MIT): TypeSafe-compatible Noul, Choice, and Score execution through OpenAI
  and Anthropic, including native structured output, probability modes, retries,
  validation, usage, and latency.
- [TypeSafe Workflow Evals](https://evals.typesafe.ai/): four provider-run workflow
  evaluations with fixed workflow definitions and consensus reference answers.
- [TypeSafe Console Playground](https://console.typesafe.ai/playground): interactive
  single-request exploration of state and typed questions.
- [JevBench](https://github.com/fstandhartinger/jevbench) (MIT): a versioned benchmark
  suite for Jev-class decision models with fixed tasks, published results, calibration,
  latency, cost, and composite scoring.

## Reused ideas and independent implementation

The System One Adapter's OpenAI path confirmed several choices used here: the Responses
API, strict JSON Schema, `store=false`, explicit untrusted-data framing, completion
checks, token usage capture, and separate transient versus malformed-response handling.
Those protocol choices are implemented locally; source code was not copied. The local
adapter additionally records the exact reasoning effort and output-token cap used by a
run. A valid incomplete or unusable OpenAI response remains a failed decision, while its
available model, token usage, and estimated cost metadata are retained.

The adapter is intentionally not a runtime dependency in v1. Its public abstraction is
TypeSafe-compatible Noul/Choice/Score execution across several answer modes, while this
project currently needs one closed-label classification result. Adding the adapter would
also add the TypeSafe SDK, provider SDK, retry, and related transitive dependencies. The
small local OpenAI adapter keeps an explicit CLI-controlled timeout, does not retain the
adapter's raw diagnostic traces, and normalizes directly to `DecisionResult`. If future
versions add boolean and ordinal tasks or multiple LLM backends, using the official
adapter should be reconsidered.

The TypeSafe HTTP contract is similarly small for v1: one Choice question sent to the
documented System One endpoint. A direct `httpx` adapter keeps credentials, retries, and
normalization visible while leaving TypeSafe concepts outside the core.

## Probability calibration and provider confidence

The normalized schema deliberately separates two concepts:

- `prediction_probability` is the probability assigned to the selected label and is the
  only per-case value used by the calibration buckets and ECE.
- `provider_confidence` is an optional provider-specific certainty statistic. It is
  preserved for inspection but is not assumed to be a probability or used by ECE.

For a TypeSafe Choice answer, `prediction_probability` is
`answer.probabilities[answer.choice]`. TypeSafe documents `answer.confidence` separately
as a statistic calculated from the shape of the full probability distribution, so it is
stored as `provider_confidence`. This preserves both official response values without
conflating their meanings.

Probability-source metadata distinguishes `native_probability`, `provider_reported`,
`model_self_reported`, and `unavailable`. OpenAI classification is intentionally recorded
as `unavailable`; the model is not asked to invent a self-reported probability. The Mock
provider's heuristic certainty is retained only as `provider_confidence`; it does not
pretend to provide a calibrated prediction probability.

Calibration aggregates only successful cases that carry `prediction_probability`.
`calibration_case_count` and `calibration_coverage` expose how much of the run was eligible
for ECE, so an unavailable probability is visible rather than silently treated as zero.
Provider-specific confidence remains available for provider-specific analysis, not as an
interchangeable universal score.

## OpenAI reasoning and consumed usage

The Responses API adapter sends explicit `reasoning.effort` and `max_output_tokens`
values. The CLI defaults to `low` effort for the narrow classification workload and a
25,000-token cap. OpenAI documents that `max_output_tokens` includes both visible output
and reasoning tokens, and its reasoning guide recommends leaving substantial headroom
when first using reasoning models. Both settings are stored in
`config.provider_settings` for reproducibility.

OpenAI `usage.output_tokens` already includes reasoning tokens for billing estimation;
`usage.output_tokens_details.reasoning_tokens` is also retained separately when present.
Aggregate cost sums all available per-case costs, including failed decisions backed by a
valid incomplete or malformed structured response. Transport failures with no usable API
response naturally have no token metadata.
