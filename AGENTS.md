# FrequenSolve — Agent Guide

This repository contains the FrequenSolve Python API for authoring, packaging,
and running FrequenSol finite-element simulations. Agents should read this file
first to orient themselves before changing code, tests, docs, or release assets.

The companion cloud application lives in `FrequenSol/cloud-amplify`. Some
planning issues may be tracked on the same GitHub Project board, but code
changes should be made in the repository that owns the behavior.

## Branch Baseline

Unless the user, issue, or pull request explicitly names a different base
branch, create task branches from the repository default branch, `main`.

The local `v2_sam` branch may be useful for ongoing local development, but do
not assume it is the correct PR base unless the user says so or the issue is
already scoped to that branch.

## Required Worklog Protocol

For every non-trivial task, create or update a local worklog before substantive
work begins.

- At task start, tell the user the full absolute path of the worklog file.
- Prefer `.codex/worklogs/<date>-<task-slug>.md`; this path is ignored by git.
- Record the current branch, worktree path, timestamp, task summary, and active
  assumptions.
- Update the worklog after meaningful progress, branch changes, file edits,
  verification, blockers, and final status.
- Keep worklogs out of commits and PRs unless the user explicitly asks to
  publish them.

## Issue And Project Workflow

Use GitHub Project 4 as the shared cross-repository intake surface when a task
comes from the Cloud SaaS Offering project:

- URL: `https://github.com/orgs/FrequenSol/projects/4`
- Owner: `FrequenSol`
- Project number: `4`
- Title: `Cloud SaaS Offering`

This project can contain issues from multiple repositories. Before starting
implementation, identify the issue's owning repository:

- `FrequenSol/FrequenSolve` issues should be implemented in this repository.
- `FrequenSol/cloud-amplify` issues should be implemented in `cloud-amplify`.
- Cross-repository work should name each required PR target and coordinate
  verification explicitly.

For Project 4 work:

1. Confirm the issue belongs to this repository or is explicitly cross-repo.
2. Ensure the issue is present in Project 4 when the work should survive across
   agent sessions.
3. Move ready work to `Agent In Progress` before implementation starts.
4. Move work to `Human Todo` if implementation cannot proceed with high
   confidence because a human decision, credential, vendor action, or product
   policy is required. Leave a precise issue comment describing the blocker and
   the exact input needed.
5. Move work to `Done` only after the change is complete and verification has
   run, or after the verification gap is explicitly documented.

Current Project 4 ids:

- Project id: `PVT_kwDOCb_rF84BCApw`
- Status field id: `PVTSSF_lADOCb_rF84BCApwzg0WKKg`
- Status option ids:
  - `Indefinitely Delayed`: `199e3d8e`
  - `Upcoming`: `1bf4576d`
  - `Agent Todo`: `f75ad846`
  - `Human Todo`: `4ef4e556`
  - `Agent In Progress`: `47fc9ee4`
  - `Human In Progress`: `6aef6727`
  - `Done`: `98236657`

Project write access requires a GitHub token with the `project` scope. If
`gh project` reports missing scope, run `gh auth refresh -s project` and retry.
If the active environment cannot authenticate GitHub, ask the user before
silently skipping Project updates.

## Issue Organization

Every GitHub issue should have a complete label set before implementation
starts. If a FrequenSolve issue is missing required labels, add them when the
label exists or ask the repo owner before starting if a needed label is absent.

Required labels:

- `component:<name>` — at least one affected component or domain.
- `P0`, `P1`, `P2`, `P3`, or `P4` — exactly one priority label.
- `type:<name>` — exactly one primary work type.

Suggested component labels for this repository:

- `component:python-sdk` — public Python APIs, models, simulation builders, and
  user-facing FrequenSolve Python API behavior.
- `component:solver-contracts` — generated input/output contracts consumed by
  solver builds.
- `component:orchestrator` — local, SLURM, cloud, and execution orchestration.
- `component:cloud` — FrequenSol cloud adapters and API clients.
- `component:hpc` — SLURM, SSH, scheduler, and remote-cluster behavior.
- `component:seismic-io` — ASDF, SEG-Y, trace IO, and related optional extras.
- `component:visualization` — plotting, PyVista, matplotlib, and visual tests.
- `component:docs` — Sphinx docs, tutorials, and documentation publishing.
- `component:packaging` — `pyproject.toml`, build, publish, and dependency
  metadata.
- `component:ci-cd` — GitHub Actions, pre-commit, CI, and release automation.
- `component:tests` — test fixtures, reference images, and test infrastructure.
- `component:devex` — local setup, agent guidance, scripts, and developer tools.

Use the standard priority labels `P0` through `P4`. Choose the lower urgency
when uncertain and document the escalation condition in the issue body.

Use one primary type label: `type:bug`, `type:feature`, `type:enhancement`,
`type:chore`, `type:docs`, `type:refactor`, `type:spike`, or `type:security`.

For agent-created issues, add `source:agent` when available and include enough
context for a future agent to reproduce the problem without relying on chat.

## Development Setup

Install the editable package with the extras needed for the task:

```sh
python -m pip install -e ".[dev,docs,visual]"
```

Use narrower extras when appropriate:

```sh
python -m pip install -e ".[dev]"
python -m pip install -e ".[cloud]"
python -m pip install -e ".[parallel]"
python -m pip install -e ".[hpc]"
```

Do not assume optional solver, cloud, HPC, or visual dependencies are available
unless the selected task requires them and the environment is configured for
them.

## Testing And Verification

The default fast test lane is deterministic and excludes solver, cloud, HPC,
interactive, and visual tests:

```sh
python -m pytest
```

The Makefile `test` target runs the non-integration lane with coverage and
matplotlib baseline options:

```sh
make test
```

Run formatting, linting, packaging, and docs checks when touched by the change:

```sh
pre-commit run --all-files
python -m build
cd docs && make html
```

Marked test lanes are opt-in:

```sh
python -m pytest -m integration
python -m pytest -m cloud
python -m pytest -m hpc
python -m pytest -m visual
```

Do not run cloud, HPC, solver-backed integration, interactive, or visual tests
unless the issue explicitly requires them or the user approves the needed
environment and credentials. Never point tests at production cloud resources or
store credentials in the repo.

For visual changes, reference images live under `tests/reference_images/`.
Generate them intentionally:

```sh
make generate_reference_images
```

Keep test additions focused on the behavior being changed. Prefer deterministic
unit tests for Python modules that can be exercised locally without solver
binaries, network access, credentials, schedulers, or production data.
Do not add pytest tests that read tracked workflow, documentation, source, or
style files and merely assert that expected strings are present. Use executable
behavior tests or the owning validator such as `actionlint`, Sphinx, or a schema
validator.

## Repository Layout

```text
.
├── src/frequensolve/       # Python package source
├── tests/                  # pytest suite and fixtures
├── tests/reference_images/ # pytest-mpl baselines
├── docs/                   # Sphinx documentation
├── examples/               # examples and tutorial material
├── pyproject.toml          # package metadata and tool configuration
└── Makefile                # common test/reference-image entrypoints
```

## PR Review Follow-Up

When you create or update a PR, handle only the first
`chatgpt-codex-connector` review round unless the user explicitly asks for a
deeper review cycle. The first round is the first connector review, top-level
connector comment, or clean connector result that appears after the PR is opened,
marked ready, or manually requested with `@codex review`.

1. After opening the PR or marking it ready, wait for the first
   `chatgpt-codex-connector` review or comment. If no result appears within 30
   minutes, do a final thread-aware readback and document the timeout instead of
   assuming the PR is settled.
2. Inspect unresolved review threads and top-level PR comments. Treat comments
   from `chatgpt-codex-connector` as review feedback that must be evaluated
   against the codebase, not accepted blindly.
3. For each first-round review comment, choose one of two outcomes:
   - If a code or docs change is warranted, patch the existing PR branch, run
     the relevant verification, push, and reply in the review thread when
     appropriate.
   - If the suggestion should not be implemented, leave a concise PR comment
     explaining the technical reason and resolve the thread when appropriate.
4. After pushing first-round fixes, comment `@codex review` once and wait until
   the bot reacts with the eyes emoji so you know the review request was picked
   up. Allow up to 30 minutes for pickup before treating the follow-up request
   as timed out.
5. Do not enter a second implementation cycle by default. If the connector
   leaves more feedback after first-round fixes, summarize the new feedback in
   the PR or worklog and hand it off unless the user explicitly asks you to keep
   going. For obvious security, data-loss, billing, or tenant-isolation risks,
   flag the risk prominently in the handoff.
6. Do not mark the PR handoff complete while first-round bot feedback remains
   unaddressed. "Addressed" means either implemented with verification, or
   explicitly answered with a technical reason for not changing the code. Later
   connector feedback is follow-up work unless the user opts into another review
   round.

## Safety Rules

- Do not commit local worklogs, credentials, generated scratch output, or large
  solver artifacts unless the user explicitly asks for them.
- Do not add production resource identifiers, customer data, secrets, or live
  cloud credentials to tests, examples, docs, or fixtures.
- Prefer structured parsers and repo-native APIs over ad hoc string handling
  when changing simulation contracts, HDF5/JSON metadata, or package metadata.
- Keep public Python APIs backward compatible unless the issue explicitly calls
  for a breaking change.
- Do not broaden optional dependency requirements without checking the impact on
  minimal installs and CI.
- Treat solver-contract changes as cross-component work. Verify generated
  artifacts and document compatibility expectations.
- For cloud/HPC paths, fail safely and give actionable diagnostics when
  credentials, schedulers, solver binaries, or remote services are absent.
- Keep documentation sanitized. Do not publish internal-only support contacts,
  incident details, private infrastructure identifiers, or credentials.
