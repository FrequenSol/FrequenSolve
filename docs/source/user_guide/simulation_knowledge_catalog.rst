Simulation Knowledge Catalog
============================

FrequenSolve includes a versioned, machine-readable catalog for tools that
explain supported simulation setup without guessing from prose or calling a
remote service. The catalog is installed with the Python package and loads
entirely offline:

.. code-block:: python

   import frequensolve as fs

   catalog = fs.load_simulation_knowledge()
   print(catalog.identities.package_version)
   print(catalog.identities.catalog_version)
   print(catalog.identities.preferred_frequensolver_release)

The combined identity also names the Sauce-owned public simulation,
acquisition, and job contracts referenced by this package release, including
their pinned Sauce source revision. This lets an agent report which package,
catalog, and solver-contract declarations its answer comes from.

Lookup and Explanation
----------------------

Physics lookup accepts the same canonical names and aliases as the authoring
API:

.. code-block:: python

   acoustic = catalog.lookup_physics("acoustic")
   poroelastic = catalog.lookup_physics("biot")

   print(acoustic.output_components)
   print(acoustic.supported_dimensions)
   print(acoustic.required_properties)
   print(poroelastic.id)

Each physics entry reports the dimensions accepted by the public simulation
configuration. ``required_properties`` is populated only when FrequenSolve has
a cataloged public material profile. Coupled formulations report
``property_requirements="domain-specific"`` because every layer owns its
material family. Entries marked ``not-cataloged`` must not be used for
automatic material generation.

Public API and Glossary
-----------------------

The catalog also names the supported top-level Python symbols and defines the
simulation terms an agent should use:

.. code-block:: python

   project_api = catalog.lookup_public_api("Project")
   vtk_api = catalog.lookup_public_api("ParaViewOutput")
   source_encoding = catalog.lookup_glossary("source encoding")
   remote_files = catalog.lookup_glossary("cluster-resident file")

Public API entries provide a stable id, exact ``frequensolve.<symbol>`` import
path, kind, category, aliases, summary, and related glossary ids. Glossary
entries provide a stable id, term, aliases, definition, and related public API
ids. Both lookup methods are case-insensitive and reject unknown terms instead
of guessing.

Authoring rules expose the same public constraints used by constructors and
preflight validation:

.. code-block:: python

   boundaries = catalog.lookup_authoring_rule("boundary_conditions")
   solver = catalog.lookup_authoring_rule("solver")
   frequencies = catalog.lookup_authoring_rule("frequencies")
   acquisition = catalog.lookup_authoring_rule("acquisition")
   files = catalog.lookup_authoring_rule("file_references")
   outputs = catalog.lookup_authoring_rule("outputs")

These typed records cover supported dimensions, documented boundary names and
PML defaults, discretization ownership, public solver defaults, frequency
compatibility, HDF5 dense source-encoding layout, local-versus-remote file
handling, and output constraints including exact ParaView model-surface
selectors. Additional solver or discretization fields are marked
``contract-dependent``; an agent should not invent them.

The catalog covers every stable diagnostic emitted by this package release's
validators. Each explanation carries a stable code, severity, object path,
plain-language explanation, and remediation:

.. code-block:: python

   item = catalog.explain_validation("field.unsupported")
   print(item.explanation)
   print(item.remediation)

Package-created validation reports enforce this registry, so a new internal
code or severity change cannot silently drift away from the packaged catalog.
Application-created :class:`frequensolve.validation.ValidationReport` objects
may still use application-defined codes.

Catalog entries explain diagnostics. They do not replace the installed package
validators. Always validate the actual object before saving or submitting it:

.. code-block:: python

   report = fs.validate_job(job)
   for issue in report.issues:
       print(issue.code, issue.message, issue.hint)

Guided Starter Scenario
-----------------------

The first fully curated path is a known-small 2D acoustic setup:

.. code-block:: python

   starter = catalog.get_starter_scenario("known-small-2d-acoustic")
   print(starter.setup)
   print(starter.limitations)

``starter.setup`` contains every authoring section used by the tested
:doc:`../quickstart`: project, simulation, layered model, mesh, boundaries,
acquisition, discretization, solver settings, and one frequency-domain job.
It is structured input for an agent or setup tool, not an instruction to run
the solver automatically. Catalog loading rejects mismatched scenario physics
or dimensions, unsupported formulation dimensions, broken example links,
missing required setup structure, and job types other than the supported
``FrequencyDomainJob`` starter.

All currently supported physics are available for lookup. Guided generation is
limited to this acoustic scenario until each additional path has its own
evaluation and tests.

Safety Boundary
---------------

Loading the catalog never runs FrequenSolver, calls a Cloud API, reads customer
data, or checks remote files. It also cannot perform Cloud-only authorization,
quota, storage-existence, or deployment checks. Those checks belong to the
service handling submission.
