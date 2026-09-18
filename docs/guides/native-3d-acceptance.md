# Small native 3D acceptance case

`tests/test_native_3d_workflow.py` owns one intentionally small acoustic scenario
for [SDK #79](https://github.com/FrequenSol/FrequenSolve/issues/79). It uses only
public authoring APIs: a homogeneous 1 km by 1 km by 0.5 km layered model,
`LayeredMeshGenerator`, order 2 DPG, one 2 Hz frequency, two scalar sources and
two collocated pressure receivers. The top face is free; the other five faces
use PML boundaries. The native layered axis follows the model layers rather
than interpreting the root counts as a uniform Cartesian cell count.

The default deterministic lane checks save/load, SDK validation, and the actual
saved simulation/acquisition JSON against the vendored Sauce `a54bdda` schemas.
It does not rewrite generated input to satisfy a schema. The legacy
`model.hex_mesh_generator()` emits `HexMeshGenerator`, which the native runtime
still accepts but the current vendored simulation schema does not list. This
case deliberately uses the public contract-facing `LayeredMeshGenerator`.

## Opt-in local run

Install the SDK with `[dev,parallel]`, then supply an explicitly selected native
executable (or an operator-reviewed wrapper around an existing local image):

```sh
LOCAL_SOLVER_EXECUTABLE=/absolute/path/to/FS_seismic \
  python -m pytest tests/test_native_3d_workflow.py -m integration \
  --basetemp=/absolute/path/to/a/new/3d-evidence-run \
  --junitxml=/absolute/path/to/3d-junit.xml
```

Use a new, disposable `--basetemp`: pytest clears that directory before running.
Do not point it at a repository, existing evidence, or valuable data. No cloud
credentials, remote scheduler, network downloads, or provider records are
required. Native init/task/pack must run; a missing solver fails instead of
skipping. The case uses one worker, one thread and a 512 MB Dask worker limit.
`LocalSite.run` has a 300-second wait timeout; use an external process/container
watchdog as well for a release runner because setup precedes that wait.

The test checks native identity against the run manifest, actual 3D `fs3d_s`
execution, one successful task, convergence and packing. Public result loading
must yield `(frequency=1, source=2, component=1, receiver=2, complex=2)`, the `p`
component, 2 Hz, and expected metre coordinates converted from authored km.
Complex pressure must be finite/nonzero. Reciprocal off-diagonal responses must
agree within relative `1e-3`, an explicit single-precision scientific invariant.
It is not a mesh-convergence or absolute-amplitude accuracy benchmark. Pressure
units are not inferred from the empty DataArray attributes.

`acceptance.json` beside the generated project retains source build identity,
input hashes, warning and development-license state, task summary and measured
reciprocity error. JUnit and the generated project retain contracts, native
logs and run metadata. Do not publish a `RunResult`/`LocalSite` representation:
it may include inherited process environment; share only reviewed receipts.
The normal compatibility check remains enabled. An unreleased SDK's missing
preferred-pair warning is recorded; invalid identity or a declared-pair mismatch
fails. No licensing policy is changed by the test.

## Verification and remaining release gate

A local macOS arm64 SDK/Python 3.10 run on 2026-09-18 used a cached Linux arm64
native image `sha256:e41b97cc977549ac8711d5b7a1bbb030307ee3cc42277278519259d6cbf295f9`.
The native executable reports Sauce commit
`10baec192a14fcd771a77709b73bf4244a74a6f6`, version `v0.1.1-rc.5`, clean build
`10baec192a14-false-20260906T104021Z`, and the MPI Fortran compiler. The container
had network disabled, two CPUs, a 4 GB memory cap and a 512 PID limit; only the
synthetic evidence directory was mounted. Init, task and pack succeeded, with
relative reciprocity error `3.2799103792058304e-5`. This is a cached development
image with `development-unlicensed` identity, not a licensed release acceptance.
No image was pulled/built and no Actions workflow was dispatched.

#79 remains open. Complete the exact FS_MUMPS and DockerImage source identity
chain, installed-wheel Linux execution, current-source contract review,
coverage/artifact retention and required-case registration in the existing
heavy-test evidence manifest before treating this as release acceptance.
The local run is not a substitute for Sauce #45 or SDK #70. No additional CI
trigger or recurring schedule is enabled here; agree a manual release run and
Actions-minute cap before scheduling the full chain.
