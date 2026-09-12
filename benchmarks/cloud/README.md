# FrequenSolve Cloud benchmarks

This suite turns the release tutorial workloads into standalone Python benchmark
programs. Benchmark execution does not open notebooks and does not require
Jupyter or `nbclient`.

The corpus currently contains 24 cases and 45 Cloud submissions. It covers the
same modeling, site, units, coordinate-system, meshing, survey, trace, ParaView,
imaging, and performance workflows as the tutorials, while omitting local-only
presentation code such as plots and notebook display calls.

## Safety and configuration

The runner never hard-codes a domain, environment, user, account, or credential.
It selects an existing named `type = "aws"` profile from the normal
FrequenSolve `site.toml`. The declared backend is an assertion: the run fails if
the returned Cloud metadata routes a submission to another backend.

Do not use production. Before a live run, confirm the selected non-production
environment is available, the account has sufficient test Credit, and any paid
compute lifecycle has an approved cost/deadline envelope. Infrastructure startup,
shutdown, and provider cleanup remain the responsibility of that environment's
runbook.

Each authored project is placed beneath a deterministic
`cloud-benchmark-<run-and-case>` leaf. Local output defaults to
`.benchmarks/cloud-history/`, which is ignored because it can contain private
simulation identifiers and sanitized logs.

## Run

List the corpus and known-bug skips:

```bash
python -m benchmarks.cloud list
```

Run every case through any configured Cloud profile:

```bash
python -m benchmarks.cloud run \
  --profile tutorial-batch \
  --backend batch

python -m benchmarks.cloud run \
  --profile tutorial-slurm \
  --backend slurm
```

Use repeatable case globs for bounded cohorts:

```bash
python -m benchmarks.cloud run \
  --profile tutorial-slurm \
  --backend slurm \
  --case '01_modeling_basics/*' \
  --case '07_performance/*'
```

`--email` is optional for profiles that need an explicit login identity.
`--run-id` is useful in automation. `--timeout-seconds` is per case, and
`--fail-fast` is opt-in; normal runs continue so one defect does not hide the
rest of the compatibility surface. Repeat `--tag KEY=VALUE` to record context
such as the Cloud/runtime revision, instance shape, or cold/warm state. Tags are
reported with history but do not prevent an intentional cross-revision comparison.

## Outcomes and evidence

Every case is one of:

- `PASS`: the script completed, every expected submission was observed, every
  handle reached a successful terminal state, result-side tutorial operations
  completed, and returned backend metadata matched `--backend`.
- `FAIL`: setup, submission, waiting, terminal status, result validation,
  backend assertion, worker exit, or timeout failed. The case record retains the
  phase, exception, traceback, customer-safe Cloud failure, simulation id, and
  provider evidence when available.
- `SKIP_KNOWN_BUG`: an issue-linked bug applies to this case/backend. Skips are
  visible but excluded from the pass-rate denominator.
- `XFAIL` or `XPASS`: `--probe-known-bugs` ran a skip. A still-failing bug is
  `XFAIL`; an unexpectedly passing skip is `XPASS` and makes the run fail until
  the stale registry entry is removed.

A run reports 100% only when every non-skipped case passes. Its directory
contains:

- `manifest.json`: schema, corpus fingerprint, profile, backend, Python/package,
  and Git identity;
- `cases.jsonl`: durable case and per-submission evidence;
- `cases/<id>/result.json`: detailed result plus stdout/stderr and local work;
- `summary.json` and `summary.md`: counts and p50/p95 performance aggregates.

Top-level metrics include case duration, submission-to-acceptance latency,
wait-to-observed-terminal latency, and a single submit-to-observed-terminal
duration that includes any client work between acceptance and waiting. Best-effort
Cloud diagnostics retain planning, gateway, scheduler queue, solver-active,
packing/projection, frequency, provider attempt, and independent
billing-settlement fields. A missing field stays absent or null; the benchmark
does not manufacture historical timing.

## Compare with history

Compare a candidate with the most recent earlier successful run for the same
named profile, backend, case selection, and exact corpus fingerprint:

```bash
python -m benchmarks.cloud compare \
  --candidate .benchmarks/cloud-history/<run-id>
```

Or name a baseline explicitly:

```bash
python -m benchmarks.cloud compare \
  --candidate .benchmarks/cloud-history/<candidate> \
  --baseline .benchmarks/cloud-history/<baseline>
```

This lets a Batch-only change compare with Batch history without paying to rerun
Slurm, and vice versa. Comparisons refuse different profiles, backends, case
selections, or corpus fingerprints so a similarly named provider on a different
Cloud target cannot become an accidental baseline. The default regression marker
requires both a 20% increase and at least five additional seconds; tune those
values explicitly and add `--fail-on-regression` only for a true performance
gate.

## Refresh from tutorials

The checked-in scripts are generated snapshots. After tutorial source changes:

```bash
python scripts/generate_cloud_benchmark_workloads.py
git diff -- benchmarks/cloud/workloads
```

The manifest binds every script to its exact file SHA-256, semantic AST hash,
generator runtime, and expected dynamic submission count. Historical corpus
compatibility uses the semantic hash, so formatter-only refreshes do not discard
valid timing history. Review generated behavioral changes like hand-written code.
