# FrequenSolve Python SDK

FrequenSolve Python is the authoring and orchestration SDK for FrequenSol finite-element wave simulation software. It builds solver-ready simulation inputs, manages model and acquisition metadata, reads trace outputs, and provides optional adapters for local, SLURM, and cloud execution.

The commercial solver binaries and backend services are licensed separately. This repository contains the Python SDK and lightweight mesh bindings needed to prepare inputs and inspect outputs.

## Installation

FrequenSolve supports Python 3.10, 3.11, and 3.12 on macOS and Linux.

```bash
python -m pip install frequensolve
```

Install optional capabilities only when needed:

```bash
python -m pip install "frequensolve[hpc]"         # SLURM/SSH/Dask helpers
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

Release checks:

```bash
pre-commit run --all-files
git status --short
python -m build
python -m twine check dist/*
```

For a release, build from a clean tagged commit. Versioneer uses plain PEP 440
tags, so tag releases as `0.2.0`, `0.2.1`, and so on. The GitHub release
workflow builds the sdist and wheel, rejects dirty or untagged versions, and can
publish to TestPyPI from `workflow_dispatch` or PyPI from a published GitHub
Release after Trusted Publishing is configured for the `testpypi` and `pypi`
environments.

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

Fast solver contract updates are tracked in `docs/source/fast_solver_api_updates.rst`.

## License And Support

FrequenSolve Python SDK is open source under the MIT license. The fast solver is licensed separately; for solver access and support, contact support@frequensol.com.
