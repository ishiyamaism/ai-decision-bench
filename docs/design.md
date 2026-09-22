# Design notes

## Scope

`ai-decision-bench` is a provider-neutral harness for running a user's labelled JSONL
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
Those protocol choices are implemented locally; source code was not copied.

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

## Confidence

Confidence metadata distinguishes `native_probability`, `provider_reported`,
`model_self_reported`, and `unavailable`. Jev's native probability-derived confidence is
recorded as `native_probability`. OpenAI classification is intentionally recorded as
`unavailable`; the model is not asked to state its own confidence. Mock confidence is
labelled `provider_reported` and is useful only for exercising the metrics pipeline.

Calibration aggregates only successful cases that carry confidence. Provider confidence
values can have different meanings, so ECE is reported per provider and is not treated as
an interchangeable universal probability score.
