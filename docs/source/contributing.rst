Contributing
============

We welcome contributions to FrequenSolve! This document will guide you through the process.

Development Setup
-----------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/your-username/frequensolve.git
      cd frequensolve

3. Install development dependencies:

   .. code-block:: bash

      python -m venv .venv
      . .venv/bin/activate
      python -m pip install -e ".[dev,docs,hpc]"

Code Style
----------

- We follow PEP 8 guidelines
- Use type hints for function arguments and return values
- Document classes and functions using Google-style docstrings
- Keep line length to 100 characters or less

Testing
-------

Run the test suite:

.. code-block:: bash

   python -m pytest

Pull Request Process
--------------------

1. Create a new branch for your feature
2. Write tests for new functionality
3. Update documentation as needed
4. Submit a pull request
5. Ensure CI checks pass

Documentation
-------------

Build the documentation locally:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   cd docs
   make html

Release Process
---------------

Releases are built with the PEP 517 backend declared in ``pyproject.toml``.
Create a clean release tag, build and check the distributions, then publish via
the GitHub release workflow:

.. code-block:: bash

   git status --short
   git tag 0.2.0
   python -m build
   python -m twine check dist/*

Use plain PEP 440 tags such as ``0.2.0`` because Versioneer is configured with
an empty tag prefix. The release workflow requires a clean tagged version before
uploading to TestPyPI or PyPI.
