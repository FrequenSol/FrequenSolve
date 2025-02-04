# NOTE: I'm not quite finished with the initial version yet. Feel free to explore, offer suggestions, and contribute if you wish, but the code is going to be changing quickly over the next few days. I'll send out instructions on how to install the FrequenSolve executable, etc. in the next few days.

# FrequenSolve (Python API)

FrequenSolve is a fast frequency-domain finite element solver for seismic wave propagation. It can be used for both time-domain and frequency-domain simulations. This library provides a Python API for setting up and running (running isn't quite ready yet) with the FrequenSolve solver.

## Features

- High-performance finite element modeling tool (currently oriented toward seismic modeling and RF plasma propagation)
- Support for time-domain and frequency-domain simulations
- Flexible model building and meshing capabilities
- Integration with popular data formats and visualization tools

## Installation

### Prerequisites

- Python 3.10 (recommended for compatibility with the python package shipped with ParaView 5.13)
- FrequenSolve executable (required for running simulations)

### Installing with Poetry

1. Install Poetry (instructions at https://python-poetry.org/docs/#installation)

2. Clone the repository:
   ```
   git clone git@github.com:FrequenSol/FrequenSolve.git
   cd frequensolve
   ```

3. Install the package and its dependencies using Poetry:
   ```
   poetry install
   ```

4. Create the virtual environment with Python 3.10:
   ```
   poetry env use 3.10
   ```


5. Run a shell with the virtual environment activated::
   ```
   poetry shell
   ```

   If you encounter the error 'The command "shell" does not exist.', install the shell plugin:
   
   ```
   poetry self add poetry-plugin-shell
   ```

### Installing ParaView (optional)

ParaView is recommended for visualizing simulation results. Download and install (ParaView 5.13 recommended) from the official ParaView website: https://www.paraview.org/download/

Some components of this package (particularly visualization tools) rely on Python modules that are installed with ParaView. You'll need to add the ParaView Python library location to your Python path. The location varies by operating system. On MacOS, the path is likely something similar to `/Applications/ParaView-5.13.0.app/Contents/Python/`
You can add this to your Python path by setting the PYTHONPATH environment variable:

```bash
# Linux/macOS
export PYTHONPATH="/path/to/paraview/python:$PYTHONPATH"
```


## Building Documentation

The project documentation can be built using Sphinx. I'll work on hosting it on readthedocs or similar; for now you can build it on your local machine by:

1. Ensure you have the dependencies installed (via Poetry using `poetry install --with=dev`) and activate the poetry virtual environment (`poetry shell`) or, if you do not wish to activate the virtual environment, prepend `poetry run ` to the following commands.

2. Navigate to the `docs` directory:
   ```
   cd docs
   ```

3. Build the documentation:
   ```
   make html
   ```

The generated documentation will be available in the `docs/_build/html` directory.

## Usage

For more detailed usage instructions and examples, please refer to the documentation.

## Contributing

Contributions to the FrequenSolve Python API are welcome! If you find a bug, have a feature request, or want to contribute code, please open an issue or submit a pull request on the GitHub repository.

## License

This Python library will likely be open-sourced under a permissive open-source license (MIT, BSD-3, etc.) once it is a bit more well-developed.

## Contact

For questions or inquiries, please contact Jacob Badger (jacob.badger@frequensol.com).
