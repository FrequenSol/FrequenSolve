.. _frequensolve-user-directory:

FrequenSolve User Directory
===========================

FrequenSolve stores user-local configuration and cloud cache files under the
:term:`FrequenSolve user directory`. By default, that directory is:

.. code-block:: text

   ~/.frequensolve/

The package uses this directory for state that belongs to a user account or
workstation, not to a :term:`project`. Project :term:`JSON`, :term:`simulation`
files, :term:`job` files, logs, and solver results stay in the project
directories you create.

Changing the Directory
----------------------

Set :term:`FREQUENSOLVE_HOME` before starting Python to move the package user
directory:

.. code-block:: bash

   export FREQUENSOLVE_HOME=/path/to/frequensolve-user-storage

With that setting, the default site config moves to
``$FREQUENSOLVE_HOME/site.toml`` and cloud cache files move under
``$FREQUENSOLVE_HOME/cloud/``. The value may include ``~`` and is expanded by
the Python package.

:term:`FREQUENSOLVE_SITE_CONFIG` is a more specific override for the
:term:`site configuration file` only:

.. code-block:: bash

   export FREQUENSOLVE_SITE_CONFIG=/path/to/site.toml

Use :term:`FREQUENSOLVE_SITE_CONFIG` or ``fs.Site(config_path=...)`` when only
the site profile should move. Use :term:`FREQUENSOLVE_HOME` when all
FrequenSolve user-local state should move together.

Directory Contents
------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Path
     - Purpose
   * - ``site.toml``
     - User-editable execution :term:`site configuration file` read by
       ``fs.Site()``.
   * - Operating-system keyring
     - HPC passwords and SSH-key passphrases saved after a successful login.
       This state is managed by macOS Keychain, Windows Credential Manager, or
       a supported Linux Secret Service rather than a file in this directory.
   * - ``cloud/credentials``
     - Cached FrequenSol Cloud :term:`Cognito` tokens written by ``AWSSite``.
       This file is sensitive and is written with owner-only permissions.
   * - ``cloud/config_<domain>.json``
     - Cached public cloud configuration fetched from a domain such as
       ``app.frequensol.com``. Domain names are made filename-safe by replacing
       ``:`` and ``/`` with ``_``.

Only ``site.toml`` is intended for normal manual editing. Cloud credential and
configuration cache files are managed by the :term:`Python API`; delete them to force
a fresh login or fresh domain configuration fetch, but do not hand-edit token
values.

The site file and cloud JSON cache are intentionally different. ``site.toml``
records the user's execution profiles. ``cloud/config_<domain>.json`` is an
AWS-specific cache fetched from ``<domain>/api/config.json`` and may be
regenerated at any time; it is not a site-profile file.

The ``site.toml`` file format is documented separately in
:doc:`site_configuration`. Start there when you need to configure a cloud,
local, or :term:`HPC` execution backend.

Security and Cleanup
--------------------

Do not commit files from ``~/.frequensolve`` or any custom
``FREQUENSOLVE_HOME`` directory. These files are user-local and may contain
credentials or environment-specific paths.

Do not put passwords, SSH-key passphrases, or two-factor codes in
``site.toml``. Headless systems without an available keyring can use an SSH
agent, prompt on each session, or provide the documented process environment
variables for automation. FrequenSolve never writes a plaintext credential
fallback.

To force a fresh cloud login, delete ``cloud/credentials``. To force a fresh
domain configuration fetch, delete the relevant ``cloud/config_<domain>.json``
file. Deleting ``site.toml`` causes the next default ``fs.Site()`` call to
write a new starter config.
