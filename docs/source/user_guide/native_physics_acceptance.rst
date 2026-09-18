Native physics acceptance
=========================

Public authoring support and verified native execution are different contracts.
``supported_physics()``, ``physics_aliases()`` and
``supported_dimensions_for_physics()`` in ``frequensolve.util.physics`` expose
what the SDK accepts. They do not promise that every dimension, field, material
or solver build has passed scientific acceptance.

The matrix below records the evidence boundary for SDK
`issue #80 <https://github.com/FrequenSol/FrequenSolve/issues/80>`_.
All rows use the base SDK for authoring. Local execution requires the
``parallel`` extra and an explicitly selected compatible native executable;
``dev`` supplies pytest for these acceptance cases. Plotting is unnecessary.
Selecting an optional site or file format adds that feature's own extras.

.. list-table:: Public formulations and selected native proof
   :header-rows: 1
   :widths: 19 12 31 38

   * - Canonical physics
     - Accepted authoring dimensions
     - Source / receiver / output boundary
     - Selected runtime evidence and remaining owner
   * - ``acoustic``
     - 2, 2.5, 3
     - Scalar source and pressure receiver in the native 3D fixture; component registry also declares velocity.
     - SDK #79 has a bounded local 3D case. Complete release provenance and other dimension breadth remain separate.
   * - ``acoustic_axisym``
     - 2
     - Acoustic component family; no source/receiver recipe is certified by this matrix.
     - Authoring/contract coverage only here; selected SDK-to-native acceptance remains #80.
   * - ``elastic``
     - 2, 2.5, 3
     - This 2D case uses vertical vector forces and vertical velocity receivers. Registry additionally declares stress, strain and pressure.
     - Native 2D init/task/pack and reciprocity case below. Other dimensions/fields remain #80; 3D breadth is #79.
   * - ``elastic_axisym``
     - 2
     - Elastic component family; axisymmetric source/receiver behavior is not inferred from the 2D Cartesian case.
     - No selected full SDK workflow proof here; #80 and Sauce #47 own the remaining matrix.
   * - ``elastic_axisym_torsion``
     - 2
     - Elastic component family declaration is not evidence for torsional directions or amplitudes.
     - No selected full SDK workflow proof here; #80 and Sauce #47.
   * - ``coupled``
     - 2, 2.5, 3
     - Mixed material authoring; elastic component registry. Sources and receivers must match their material domains.
     - Sauce #47 describes representative native acoustic-elastic coverage. Full selected SDK/tutorial and interface proof is separate; #80/#198.
   * - ``coupled_aep``
     - 2, 2.5, 3
     - Registry declares pressure, velocity, fluid flux, stress, strain and solid/fluid displacement. This is not executed cross-domain proof.
     - Complete native acoustic-elastic-poroelastic execution is pending Sauce #47; SDK #80 depends on it.
   * - ``coupled_axisym``
     - 2
     - Elastic component family; mixed-domain/axisymmetric recipes are not certified by this matrix.
     - No selected full SDK workflow proof here; #80 and Sauce #47.
   * - ``coupled_axisym_torsion``
     - 2
     - Elastic component family; torsional/interface behavior remains unverified here.
     - No selected full SDK workflow proof here; #80 and Sauce #47.
   * - ``poroelastic``
     - 2, 2.5, 3
     - Registry declares velocity, fluid flux, stress, pressure, strain and solid/fluid displacement; material and boundary restrictions still apply.
     - Sauce #47 describes representative native forward coverage. Exact SDK result/scientific acceptance remains #80/#198.
   * - ``em``
     - 2, 2.5, 3
     - Registry declares electric and magnetic fields. Mechanical source/receiver recipes cannot be reused as Maxwell acceptance.
     - Complete Maxwell execution and supported source/boundary recipe remain Sauce #47, then SDK #80.

These eleven canonical names cover the public physics catalog. The aliases
``coupled-aep`` and ``coupledaep`` select ``coupled_aep``; ``poro``,
``poro-elastic``, ``poro_elastic`` and ``biot`` select ``poroelastic``;
``electromagnetic``, ``electro-magnetic``, ``electro_magnetic`` and ``maxwell``
select ``em``. Names are case-insensitive. ``axisymmetric=True`` selects the
axisymmetric form for acoustic, elastic or coupled physics and requires a 2D
model. Other axisymmetric families fail normalization. Static and alternate
native formulations are not additional public top-level names in this catalog;
Sauce owns their native capability decision.

The ``SolverConfig`` API accepts single/double precision and final/all solve
selection, with additional solver-facing settings. The bounded native fixtures
select single precision, final mesh, DPG, MUMPS and one grid. The recorded
executables are ``fs2d_s`` for this elastic case and ``fs3d_s`` for #79's acoustic
case. This is not acceptance of double precision, every coarse-solver option,
multigrid, super patches, MPI distribution or every physics/precision combination.
A serialized solver option is not proof of an available installed capability.

Non-acoustic ownership and current restrictions
-----------------------------------------------

`Sauce #47 <https://github.com/FrequenSol/Sauce/issues/47>`_ owns native Maxwell,
coupled-AEP and its full native variant table. SDK #80 owns the public Python
path and numerical/result checks. `SDK #70 <https://github.com/FrequenSol/FrequenSolve/issues/70>`_
owns required scientific evidence registration. Unknown or unverified rows
remain open requirements; this matrix does not turn them into supported release
claims or silently remove existing APIs. Retain a linked product/native decision
or implement validation before advertising an unsupported recipe. The VTR
output/tutorial runtime failure tracked in #198 and Sauce #112 remains separate.

Bounded elastic case
--------------------

``tests/test_native_elastic_workflow.py`` builds a homogeneous isotropic 2D model
through public Project, Simulation, LayeredModel, LayeredMeshGenerator,
Acquisition and FrequencyDomainJob APIs. The domain is 1 km wide and 0.5 km deep,
with Vp 2 km/s, Vs 1 km/s and density 2.2 g/cm³. A free top and PML on the other
three sides bound the model. Two vertical 1 N point forces at (0.25, 0.15) km and
(0.7, 0.3) km share positions with vertical velocity receivers. The asymmetric
positions avoid satisfying reciprocity merely through reflection symmetry.
One 2 Hz frequency uses order 4, four elements per wavelength and the native
layered mesh with root counts [16, 1]. The layered axis follows model layers.

The default deterministic lane validates save/load and the actual saved
simulation/acquisition payloads against the vendored Sauce schemas. The opt-in
native test requires init, task and pack; it checks the native identity against
the manifest, 2D elastic workflow, convergence, one successful task and no failed
tasks. Public packed-result loading must return one frequency, two sources, one
``v_z`` component, two receivers and real/imaginary axes. Survey coordinates must
be normalized from km to metres. The complex response must be finite and nonzero.
The relative off-diagonal reciprocity error must be below ``1e-3``.

That bound is a selected single-precision invariant, not a universal error bound
or proof of absolute amplitude accuracy. Packed DataArray unit attributes are
currently empty, so the test does not invent velocity-output units. Complete
field/unit metadata acceptance remains #80.

Calibration retained on September 18, 2026 used the same cached native build:

.. list-table:: Fixture refinement
   :header-rows: 1

   * - Order / EPW / horizontal root count
     - Relative reciprocity error
     - Outcome against unchanged 1e-3 bound
   * - 2 / 2 / 8
     - 0.2139243484
     - Failed; not selected as the acceptance fixture
   * - 4 / 4 / 16
     - 0.0003358182
     - Passed
   * - 6 / 6 / 24
     - 0.0000437113
     - Passed; independent further-refinement diagnostic

The failure was retained rather than weakening the threshold. These results
support the selected fixture resolution; they do not establish a formal
mesh-convergence order or replace validation against a trusted analytic solution.

Running and retaining evidence
------------------------------

Install ``.[dev,parallel]`` into the selected test environment and run explicitly:

.. code-block:: sh

   LOCAL_SOLVER_EXECUTABLE=/absolute/path/to/FS_seismic \
     python -m pytest tests/test_native_elastic_workflow.py -m integration \
     --basetemp=/absolute/path/to/a/new/elastic-evidence-run \
     --junitxml=/absolute/path/to/elastic-junit.xml

Use a new disposable ``--basetemp``; pytest clears it. The test fails when the
solver is missing, instead of quietly skipping. It uses one worker/thread with
a 512 MB Dask worker limit, a 300-second run wait and a 420-second test deadline.
Use an external process/container watchdog to bound setup as well. No remote
scheduler, cloud credentials or network access is needed. The fixture does not
create grid visualization output or mutate billing/provider state.

The compatibility check stays enabled. An unreleased SDK's missing preferred-pair
warning is retained, but invalid native identity and declared-pair mismatch fail.
``acceptance.json`` retains the SDK version, native build/license/workflow,
input hashes, response values and reciprocity metric/limit. Retain the generated
project (contracts, logs and run manifest), JUnit, exact SDK source/wheel digest,
image digest and dependency resolution beside it. Never publish a LocalSite or
RunResult representation, which may contain inherited process environment.

The local diagnostic used cached Linux arm64 image
``sha256:e41b97cc977549ac8711d5b7a1bbb030307ee3cc42277278519259d6cbf295f9`` and
Sauce ``10baec192a14fcd771a77709b73bf4244a74a6f6`` (clean build,
``development-unlicensed``). The installed SDK wheel resolved NumPy 2.4.6 and
Bokeh 3.9.2 with the image's xarray/Dask 2025 dependencies. Network was disabled;
the container had two CPUs, 4 GB memory, 512 PIDs and an external nine-minute
watchdog. Each native test finished in under ten seconds. No image pull/build
or Actions workflow was requested.

This is a local development-image result, not a licensed customer release. #80
stays open for the additional native family, precise field/unit semantics, full
FS_MUMPS/DockerImage source chain, reviewed support decisions and required-case
manifest integration. The existing heavy release route owns full provenance and
artifact retention. Agree the release trigger and Actions-minute cap before
running that chain; this change adds no automatic or scheduled native execution.
