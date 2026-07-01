Changelog
=========

All notable changes to FrequenSolve will be documented in this file.

Unreleased
----------

Added
~~~~~
- Added Python authoring support for borehole-level 3D annular padding.

Changed
~~~~~~~
- Switched CI and release documentation to the ``pyproject.toml``/setuptools
  build path.
- Added GitHub Actions release publishing for TestPyPI and PyPI via Trusted
  Publishing.
- Moved orchestration helper modules into ``frequensolve.orchestrator.utils``.
- Renamed the documented Dask/SLURM optional extra from ``parallel`` to
  ``hpc`` while keeping ``parallel`` as an install alias.

Removed
~~~~~~~
- Removed the unused utility modules for legacy inputs and unused functionality.

[0.1.1] - 2024-03-21
--------------------

Added in 0.1.1
~~~~~~~~~~~~~~
- Initial public release
- Basic seismic modeling capabilities
- Support for acoustic wave propagation
- Python API documentation

Changed in 0.1.1
~~~~~~~~~~~~~~~~
- Improved mesh generation performance
- Enhanced documentation

Fixed in 0.1.1
~~~~~~~~~~~~~~
- Various bug fixes and improvements

[0.1.0] - 2024-02-15
--------------------

Added in 0.1.0
~~~~~~~~~~~~~~
- Core functionality
- Basic documentation
- Initial test suite
