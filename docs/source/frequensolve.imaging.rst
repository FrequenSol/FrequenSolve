Imaging
=======

``frequensolve.imaging`` is the imaging and inversion authoring API built on
the Sauce imaging contracts. Import it as ``from frequensolve import imaging
as im``; every public name is also re-exported at the package root. The
conceptual guide is :doc:`user_guide/imaging`.

Control Space Classes
---------------------

.. automodule:: frequensolve.imaging.controls
   :members:
   :show-inheritance:
   :noindex:

Observed Data And Data Vectors
------------------------------

.. automodule:: frequensolve.imaging.data
   :members:
   :show-inheritance:
   :noindex:

Misfit
------

.. automodule:: frequensolve.imaging.misfit
   :members:
   :show-inheritance:
   :noindex:

Problem And Linearization
-------------------------

.. automodule:: frequensolve.imaging.problem
   :members:
   :show-inheritance:
   :noindex:

Operators
---------

.. automodule:: frequensolve.imaging.operators
   :members:
   :show-inheritance:
   :noindex:

Penalties, Smoothing And Preconditioners
----------------------------------------

.. automodule:: frequensolve.imaging.regularization
   :members:
   :show-inheritance:
   :noindex:

Workflows
---------

.. automodule:: frequensolve.imaging.workflows
   :members:
   :show-inheritance:
   :noindex:

Results
-------

.. automodule:: frequensolve.imaging.results
   :members:
   :show-inheritance:
   :noindex:

Extension
---------

.. automodule:: frequensolve.imaging.extension
   :members:
   :show-inheritance:
   :noindex:

Sauce Job Classes
-----------------

.. automodule:: frequensolve.imaging.jobs
   :members:
   :show-inheritance:
   :noindex:

Artifact Readers
----------------

The readers and writers of the Sauce imaging artifacts
(``fs-control-vector-1``, ``fs-control-state-1``, ``fs-control-registry-1``,
``fs-extension-vector-1``, ``fs-objective-report-1``, ``fs-objective-balance-1``,
``fs-extension-solve-1`` and the Cartesian image files) are re-exported from
``frequensolve.imaging`` under their public names: ``ControlVectorFile``,
``ControlStateFile``, ``ControlRegistryManifest``, ``RegistryBlock``,
``ExtensionVectorFile``, ``ExtensionVectorField``, ``ObjectiveReport``,
``ObjectiveTermReport``, ``BalanceArtifact``, ``BalanceTerm``,
``ExtensionSolveReport``, ``ReducedNormalReport``, ``ImageSet``,
``SmoothingConfig``, ``pack_support_mask``, ``unpack_support_mask``,
``qualified_block_name`` and ``unqualified_block_name``. ``ImageSet``,
``ObjectiveReport`` and ``ExtensionSolveReport`` are documented under
`Results`_; the remaining readers are contract-level helpers whose fields
follow the schema they are named after.
