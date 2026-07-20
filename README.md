# FrequenSolve Python API

[![Coverage](https://codecov.io/gh/FrequenSol/FrequenSolve/branch/v2/graph/badge.svg)](https://app.codecov.io/gh/FrequenSol/FrequenSolve/tree/v2)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](pyproject.toml)

FrequenSolve Python is the authoring and orchestration API for FrequenSol finite-element wave simulation software. It builds solver-ready simulation inputs, manages model and acquisition metadata, reads trace outputs, and provides optional adapters for local, SLURM, and cloud execution.

The commercial solver binaries and backend services are licensed separately. This repository contains the FrequenSolve Python API and lightweight mesh bindings needed to prepare inputs and inspect outputs.

## Installation

FrequenSolve supports Python 3.10 through 3.14 on macOS and Linux.

Create an isolated environment and install the released FrequenSolve Python
API with:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install frequensolve
frequensolve --version
```

Because the repository is public, you can also install from a source checkout
when you want local examples, documentation sources, or editable development:

```bash
python -m pip install -e .
```

Install optional capabilities only when needed:

```bash
python -m pip install "frequensolve[visual]"      # matplotlib and PyVista plotting
python -m pip install "frequensolve[parallel]"    # local Dask execution
python -m pip install "frequensolve[hpc]"         # SSH and SLURM site support
python -m pip install "frequensolve[cloud]"       # FrequenSol Cloud API and S3 access
python -m pip install "frequensolve[seismic-io]"  # SEG-Y/ASDF IO helpers
python -m pip install "frequensolve[fast-fft]"    # pyFFTW acceleration
python -m pip install "frequensolve[inversion]"   # PyLops-compatible FWI operators
```

## Quickstart

```python
import frequensolve as fs

u = fs.ureg

project = fs.Project(name="quickstart", path="./scratch/quickstart")
sim = project.new_simulation(
    name="simple_acoustic",
    physics="acoustic",
    dimension=2,
    units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
)

model = fs.LayeredModel(name="model", dimension=2, x_limits=[0.0, 1.0])
model.add_surface(name="top", depth=0.0 * u.km)
model.add_layer(
    name="layer",
    properties={"Vp": 1.5 * u.km / u.s, "Rho": 2.2 * u.g / u.cm**3},
)
model.add_surface(name="bottom", depth=0.5 * u.km)

sim += model
sim += model.hex_mesh_generator([8, 4])
project.save()
```

The FrequenSolve Python API exports JSON/HDF5 contracts consumed by fast solver builds. Solver execution requires a licensed solver binary or an enabled FrequenSol execution backend.

## Sites And Tutorials

### Stampede3 quick setup

Install the HPC dependencies into the active environment, generate a site
profile, authenticate once, and check it:

```bash
python -m pip install "frequensolve[hpc]"

frequensolve site configure stampede3 \
  --account YOUR_TACC_ALLOCATION \
  --solver /absolute/remote/path/to/FS_seismic
frequensolve site connect
frequensolve site check
```

The configuration command prompts for the TACC username and writes a minimal
`~/.frequensolve/site.toml`. The built-in Stampede3 preset supplies the login
host, launcher, partitions, and standard Intel MPI, PETSc, and parallel-HDF5
runtime modules. Users do not normally need to specify a credential label, SSH
key, hostname, or module list.

`frequensolve site connect` establishes an OpenSSH connection that scripts,
notebooks, `fs.Site()` instances, and transfers can reuse for up to eight
hours. It does not modify `~/.ssh/config`. See
[Share one SSH login across FrequenSolve sessions](docs/guides/ssh-connection-sharing.md)
for the optional generic OpenSSH configuration alternative.

After setup, activate the environment and use the saved default site from any
script or notebook:

```bash
. .venv/bin/activate
frequensolve site connect
python your_script.py
```

```python
import frequensolve as fs

with fs.Site() as site:
    result = site.submit(job).wait()
```

For custom Slurm clusters, local solvers, cloud execution, multiple profiles,
and site-level overrides, see the
[site configuration guide](docs/source/user_guide/site_configuration.rst).
Direct constructors such as `fs.LocalSite(...)` and `fs.AWSSite(...)` remain
available for advanced cases. Scheduler templates and the adaptive runner are
installed with FrequenSolve, so `PYTHONPATH` is not required for site setup.

The tutorial notebooks live in `examples/tutorials`. The local documentation catalog is `docs/source/tutorials/index.rst`, with site-specific examples under `examples/tutorials/02_sites`.

## Development

Create a local development environment from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,docs,visual]"
```

Run deterministic unit tests by default:

```bash
python -m pytest
```

GitHub CI runs unit tests, docs, and package checks on normal PRs and pushes.
It also runs a single Python 3.12 lane with the visual and seismic-IO extras,
and installs both built distribution formats on Ubuntu and macOS. The stable
`Required CI` job aggregates every PR-safe gate.

Run the optional local visual and seismic-IO lane with:

```bash
python -m pip install -e ".[dev,parallel,visual,seismic-io]"
make test-optional-extras
```

The downstream Docker image integration gate is intentionally opt-in because it
spends Docker/GitHub Actions minutes. To run it, manually dispatch the `CI`
workflow with `RUN_DOCKER_IMAGE_INTEGRATION=true`.

Release checks:

```bash
pre-commit run --all-files
git status --short
python -m build
python -m twine check dist/*
```

For releases, use PEP 440 package versions and `v`-prefixed tags. Creating a
release candidate is the paid, manual release gate: it requires successful
`Required CI` evidence for the exact source SHA and calls the pinned
FrequenSolveDockerImage solver workflow before creating any tag. The evidence
JSON and checksum-bound heavy-test archive follow that commit through final
promotion and publication. Release
candidates are published to TestPyPI, and final releases are published to PyPI.
See [RELEASING.md](RELEASING.md) for the maintainer workflow.

Solver, cloud, HPC, and visual tests are marked and must be selected explicitly:

```bash
python -m pytest -m integration
python -m pytest -m cloud
python -m pytest -m hpc
python -m pytest -m visual
```

## Documentation

Build the Sphinx documentation locally with:

```bash
python -m pip install -e ".[docs]"
cd docs
make html
```

Published Python API docs are owned by the `FrequenSol/cloud-amplify`
`docs-site-app`. Use that repository's manual `Publish Python Docs` workflow to
build from a selected FrequenSolve release ref and publish immutable artifacts
under `/python/<version>/`, with `/python/latest/` updated after the versioned
artifact is present. The former `docs/host` Terraform stack in this repository
has been destroyed and removed.

Fast solver contract updates are tracked in `docs/source/fast_solver_api_updates.rst`.

## License And Support

FrequenSolve Python API is open source under the MIT license. The fast solver is licensed separately; for solver access and support, contact support@frequensol.com.
