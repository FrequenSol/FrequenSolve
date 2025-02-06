Installation
============

FrequenSolve can be installed using Poetry, which handles dependencies and virtual environments automatically.

Prerequisites
-------------

- Python 3.8 or higher
- Poetry (recommended) or pip
- C++ compiler (for mesh module compilation)

Using Poetry (Recommended)
--------------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/frequensol/frequensolve.git
      cd frequensolve

2. Install with Poetry:

   .. code-block:: bash

      poetry install

   This will create a virtual environment and install all dependencies.

3. Activate the virtual environment:

   .. code-block:: bash

      poetry shell

Using pip
---------

If you prefer using pip directly:

.. code-block:: bash

   pip install frequensolve

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

   poetry install --with=dev

Verification
------------

To verify your installation:

.. code-block:: python

   import frequensolve
   print(frequensolve.__version__)
