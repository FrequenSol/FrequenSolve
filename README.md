# FrequenSolve (Python API)

FrequenSolve is a fast frequency-domain finite element solver for seismic wave propagation.
It can be used for both time-domain and frequency-domain simulations.
This library provides a Python API for setting up and running the FrequenSolve solver.

> [!WARNING]
> This is a work in progress, the API is not yet stable and will change frequently.

## Features

- High-performance finite element modeling tool (currently oriented toward seismic modeling and RF plasma propagation)
- Support for time-domain and frequency-domain simulations
- Flexible model building and meshing capabilities
- Integration with popular data formats and visualization tools

## Installation

### Prerequisites

- A Unix-like operating system (Linux, macOS, etc.); ***FrequenSolve doesn't currently support Windows.***
- Python >=3.10

> [!NOTE]
> FrequenSolve has tools to automate ParaView visualization. Some of these tools can be run
> with ParaView's interanl Python build (pvpython), but for full integration with FrequenSolve
> you'll need to ensure that the Python version used for the installation is the same as the
> one used for building FrequenSolve.

> [!TIP]
> If you've already built FrequenSolve with a different Python version, you can create a new
> Poetry virtual environment with the desired Python version and rebuild FrequenSolve. Assuming you
> have Python 3.10 installed, you can do the following:
>
> ```console
> $ poetry env use python3.10         # Create a new virtual environment with Python 3.10
> $ poetry install                    # Install the dependencies in the virtual environment
> $ poetry env list                   # List the virtual environments
> ```
>
> To toggle between virtual environments, run `poetry env use <python-version>`.

<!-- - The FrequenSolve Python API has a rich set of tools for model building, parallel visualization, etc.
   that can be used on their own, free of charge. However, running simulations requires
   state-of-the-art solver libraries developed by FrequenSol, LLC.; these can either be licensed on
   your own on-site hardware or run on the cloud. For more information on how to obtain a license or
   run on the cloud, see ***TODO***. -->

### Installing with Poetry

1. Install Poetry (instructions at https://python-poetry.org/docs/#installation)

2. Clone the repository:

   ```console
   $ git clone git@github.com:FrequenSol/FrequenSolve.git
   $ cd frequensolve
   ```

3. Install the package and its dependencies using Poetry:

   ```console
   $ poetry install
   ```


4. Create the virtual environment with Python 3.10:
   ```console
   poetry env use 3.10
   ```


5. Run a shell with the virtual environment activated::

   ```console
   $ poetry shell
   ```

> [!WARNING]
> If you encounter the error 'The command "shell" does not exist.', install the shell plugin:
>
> ```console
> $ poetry self add poetry-plugin-shell
> ```

> [!TIP]
> Exit the virtual environment with:
>
> ```console
> $ exit
> ```

### Installing ParaView (optional)

ParaView is recommended for visualizing simulation results. Download and install (ParaView 5.13 recommended) from the official ParaView website: https://www.paraview.org/download/

Some components of this package (particularly visualization tools) rely on Python modules that are installed with ParaView. You'll need to add the ParaView Python library location to your Python path. The location varies by operating system. On MacOS, the path is likely something similar to `/Applications/ParaView-5.13.0.app/Contents/Python/`
You can add this to your Python path by setting the PYTHONPATH environment variable:

```bash
# Linux/macOS
export PYTHONPATH="/path/to/paraview/python:$PYTHONPATH"
```

## Building Documentation

> [!Note]
> You'll need the development dependencies installed to build the documentation install them with:
>
> ```console
> $ poetry install --with dev
> ```

The project documentation can be built using Sphinx. I'll work on hosting it on readthedocs or similar; for now you can build it on your local machine by:

1. Ensure you have the dependencies installed (via Poetry using `poetry install --with=dev`) and activate the poetry virtual environment (`poetry shell`) or, if you do not wish to activate the virtual environment, prepend `poetry run ` to the following commands.

2. Navigate to the `docs` directory:

   ```console
   $ cd docs
   ```

3. Build the documentation:

   ```console
   $ make html
   ```

The generated documentation will be available in `docs/build/html` directory, to open in your default
 web browser **in macOS**:

```console
$ open docs/build/html/index.html
```

For linux, substitute `open` with `xdg-open`.

## Usage

For more detailed usage instructions and examples, please refer to the documentation.

## Contributing

Contributions to the FrequenSolve Python API are welcome! If you find a bug, have a feature request, or want to contribute code, please open an issue or submit a pull request on the GitHub repository.

## License

This Python library will likely be open-sourced under a permissive open-source license (MIT, BSD-3, etc.) once it is a bit more well-developed.

## Contact

For questions or inquiries, please contact Jacob Badger (jacob.badger@frequensol.com).
