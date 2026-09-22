# Contributing

Forks, experiments, provider adapters, documentation improvements, and AI-assisted
changes are welcome. The repository is MIT licensed: you do not need permission to clone,
modify, or use it for your own purpose, subject to the license terms. You also do not
need to send changes back upstream.

If you do want to contribute a change to this repository, keep it focused and open an
issue or pull request with the motivation and any user-visible tradeoffs.

## Development setup

Requires Python 3.12+ and `uv`.

```bash
git clone https://github.com/ishiyamaism/ai-decision-bench.git
cd ai-decision-bench
uv sync --locked --all-groups
uv run pytest
```

Before submitting a change, run the same offline checks as CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

No provider credential is needed for development or CI. Use MockProvider and mocked HTTP
transports for automated tests; do not make live API calls in the test suite.

## Change guidelines

- Read [AGENTS.md](AGENTS.md), even if you are not using an AI coding agent. It records
  the architecture, benchmark invariants, provider checklist, and security boundaries.
- Keep the core provider-neutral. Provider-specific request and response details belong
  in `src/ai_decision_bench/providers/`.
- Add tests for new behavior and update README or design notes when the public contract
  changes.
- Verify provider claims, response semantics, current model IDs, and pricing against
  first-party documentation. Include a reference date for pricing assumptions.
- Never include real keys, private datasets, raw provider responses, or sensitive result
  files in an issue, commit, test fixture, or pull request.
- Keep dependencies small and explain why a new runtime dependency is necessary.

For a new provider, include credential preflight, bounded timeout and retry behavior,
sanitized failures, resource cleanup, reproducibility metadata, offline contract tests,
and user documentation. If the provider does not expose a selected-label probability,
leave calibration unavailable rather than asking a model to invent confidence.

## AI-assisted changes

You may give the repository to any coding agent or model. Point it to `AGENTS.md` so it
starts with the project's intended semantics and validation commands.

AI assistance does not change the review standard: inspect the diff, verify externally
sourced claims, run the tests, and make sure no credential or private data entered the
prompt, patch, fixtures, logs, or commit history. Disclosure of AI assistance is welcome
when it helps reviewers understand the change, but is not required by this project.

## Pull request checklist

- The change has a clear, limited purpose.
- New behavior has focused offline tests.
- Ruff and pytest pass.
- User-facing behavior and methodology are documented.
- No live credential, private data, or generated local result is included.
- Compatibility and reproducibility effects are called out in the pull request.

