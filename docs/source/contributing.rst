Contributing
============

FrequenSolve is a Python SDK with optional local, cloud, HPC, visual, and docs
dependencies. Keep changes small, test the relevant marked lanes explicitly,
and keep public examples aligned with the current project-owned simulation API.

Development Setup
-----------------

FrequenSolve supports Python 3.10 through 3.14. For local development, create a
virtual environment from the repository root:

.. code-block:: bash

   python -m venv .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev,parallel,cloud,visual]"

Use the ``docs`` extra when you only need documentation dependencies:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Code Style
----------

- Use type hints for public function arguments and return values.
- Document public classes and functions with Google-style docstrings.
- Run formatting and lint hooks before opening a pull request:

  .. code-block:: bash

     pre-commit run --all-files

Testing
-------

Run the deterministic non-integration lane used by CI:

.. code-block:: bash

   make test

The default ``pytest`` configuration also excludes tests that require solvers,
external services, credentials, schedulers, manual input, or visual baselines:

.. code-block:: bash

   python -m pytest

Select marked lanes explicitly when you have the required environment:

.. code-block:: bash

   python -m pytest -m integration
   python -m pytest -m cloud
   python -m pytest -m hpc
   python -m pytest -m visual

Regenerate visual baselines only when the rendered output is intentionally
changed:

.. code-block:: bash

   make generate_reference_images

Pull Request Process
--------------------

1. Create a focused branch for the change.
2. Add or update tests for behavior changes.
3. Update docs and examples when public API or workflow guidance changes.
4. Run ``make test`` and any affected marked lanes locally.
5. Open a pull request and let the CI matrix verify Python 3.10 through 3.14.

Release and deployment workflows are handled by GitHub Actions. Do not add PyPI
tokens, cloud credentials, or solver licenses to the repository.

Documentation
-------------

Build the documentation locally:

.. code-block:: bash

   cd docs
   make html

Published Python docs are staged by the ``FrequenSol/cloud-amplify``
``docs-site-app`` through its manual ``Publish Python Docs`` workflow. The old
``docs/host`` Terraform project has been removed from this repository.
