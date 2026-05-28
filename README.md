# FrequenSolve Python SDK

[![Coverage](https://codecov.io/gh/FrequenSol/FrequenSolve/branch/v2/graph/badge.svg)](https://app.codecov.io/gh/FrequenSol/FrequenSolve/tree/v2)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](pyproject.toml)

FrequenSolve Python is the authoring and orchestration SDK for FrequenSol finite-element wave simulation software. It builds solver-ready simulation inputs, manages model and acquisition metadata, reads trace outputs, and provides optional adapters for local, SLURM, and cloud execution.

The commercial solver binaries and backend services are licensed separately. This repository contains the Python SDK and lightweight mesh bindings needed to prepare inputs and inspect outputs.

## Installation

FrequenSolve supports Python 3.10 through 3.14 on macOS and Linux.

Install the released SDK with:

```bash
python -m pip install frequensolve
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

The SDK exports JSON/HDF5 contracts consumed by fast solver builds. Solver execution requires a licensed solver binary or an enabled FrequenSol execution backend.

## Sites And Tutorials

Configure the standard execution site once in `~/.frequensolve/site.toml`, then create it in scripts and notebooks with `fs.Site()`. On first use, `fs.Site()` creates a starter config at that path and raises an exception asking you to review it; rerun after accepting or editing the profiles. Direct constructors such as `fs.LocalSite(...)` and `fs.AWSSite(...)` remain available for advanced cases.

Starter config:

```toml
default = "cloud"

[sites.cloud]
type = "aws"
domain = "app.frequensol.com"
interactive = true
verbose = true

[sites.local]
type = "local"
shutdown_on_completion = true
verbose = true

[sites.hpc]
type = "stampede3"
rel_path = "scratch/frequensolve_tutorials"
queue = "skx-dev"
nodes = 1
duration = "00:30:00"
procs_per_node = 4
procs_per_task = 1
poll_interval = 10
verbose = true
```

The tutorial notebooks live in `examples/tutorials`. The local documentation catalog is `docs/source/tutorials/index.rst`, with site-specific examples under `examples/tutorials/02_sites`.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test commands,
code style, and pull request expectations. See [RELEASING.md](RELEASING.md) for
maintainer release and PyPI publishing steps, and [CHANGELOG.md](CHANGELOG.md)
for release notes.

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
