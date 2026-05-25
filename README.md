# FrequenSolve Python SDK

[![Coverage](https://codecov.io/gh/FrequenSol/FrequenSolve/branch/v2/graph/badge.svg)](https://app.codecov.io/gh/FrequenSol/FrequenSolve/tree/v2)

FrequenSolve Python is the authoring and orchestration SDK for FrequenSol finite-element wave simulation software. It builds solver-ready simulation inputs, manages model and acquisition metadata, reads trace outputs, and provides optional adapters for local, SLURM, and cloud execution.

The commercial solver binaries and backend services are licensed separately. This repository contains the Python SDK and lightweight mesh bindings needed to prepare inputs and inspect outputs.

## Installation

FrequenSolve supports Python 3.10 through 3.14 on macOS and Linux.

```bash
python -m pip install frequensolve
```

Install optional capabilities only when needed:

```bash
python -m pip install "frequensolve[parallel]"    # SLURM/SSH/Dask helpers
python -m pip install "frequensolve[fast-fft]"    # pyFFTW acceleration
python -m pip install "frequensolve[cloud]"       # FrequenSol cloud backend
python -m pip install "frequensolve[seismic-io]"  # SEG-Y/ASDF and seismic I/O
python -m pip install "frequensolve[visual]"      # plotting helpers
python -m pip install "frequensolve[dev,docs]"    # development and docs
```

## Quickstart

```python
from frequensolve.model import ModelSubdomain
from frequensolve.simulation import SeismicSimulation
from frequensolve.units import ureg

layer = ModelSubdomain(
    mesh_block_id=1,
    physics="acoustic",
    properties={
        "vp": 1.5 * ureg.km / ureg.s,
        "rho": 2.2 * ureg.g / ureg.cm**3,
    },
)

sim = SeismicSimulation(name="simple_acoustic")
sim.model.subdomains.append(layer)
sim.save("sim.json")
```

The SDK exports JSON/HDF5 contracts consumed by fast solver builds. Solver execution requires a licensed solver binary or an enabled FrequenSol execution backend.

## Sites And Tutorials

Configure the standard execution site once in `~/.frequensolve/site.toml`, then create it in scripts and notebooks with `fs.Site()`. Direct constructors such as `fs.LocalSite(...)` and `fs.AWSSite(...)` remain available for advanced cases.

Local execution:

```toml
[site]
type = "local"
rel_path = "frequensolve/tutorials"
```

Cloud execution:

```toml
[site]
type = "aws"
domain = "frequensolve.app"
interactive = true
```

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

Run the same non-integration coverage lane used by CI:

```bash
make test
```

Release checks:

```bash
pre-commit run --all-files
python -m build
python -m twine check dist/*
```

PyPI publishing is handled by the `Publish PyPI` GitHub Actions workflow using
trusted publishing. Before the first release, a PyPI project owner must create or
approve the `frequensolve` project publisher for the
`FrequenSol/FrequenSolve` repository, `publish-pypi.yml` workflow, and `pypi`
environment. The workflow only publishes from tag refs, including manual
dispatches, so create the intended release tag before publishing. No PyPI API
token is stored in this repository.

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

FrequenSolve Python SDK is open source under the MIT license. The fast solver is licensed separately; for solver access and support, contact support@frequensol.com.
