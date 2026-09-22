# Datasets

Datasets use UTF-8 JSON Lines: one independent evaluation case per line. The included
`sample.jsonl` contains synthetic support-routing requests only; it is not derived from users,
customers, or production systems.

Each v1 case has:

- a unique `id`;
- an `input` object sent unchanged to each provider adapter;
- a `classification` task with the same closed label set for every provider; and
- one human-selected `expected` label from that set.

Optional `label_descriptions` provide a neutral rubric shared by every provider. If omitted,
the label itself is used as its description.

Do not commit private, regulated, or credential-bearing data to a public repository. Review
custom datasets before sharing them, even when the benchmark result stores only case IDs and
not source inputs.
