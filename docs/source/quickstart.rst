Quickstart
==========

This guide will help you get started with FrequenSolve quickly.

Basic Usage
-----------

Here's a minimal example of setting up a seismic simulation:

.. code-block:: python

   from frequensolve.project import Project
   from frequensolve.simulation import SeismicSimulation

   # Create a new project
   project = Project(
       name="my_simulation",
       path="my_simulation",
       log_level="INFO",
       log_file="my_simulation/frequensolve.log",
   )

   # Configure a simulation
   sim = SeismicSimulation(
       dimension=2,
       physics="acoustic"
   )

   # Add the simulation to the project
   project += sim

   # Run the simulation
   project.run()

Example: Simple 2D Model
------------------------

Here's a more complete example with a layered model:

.. code-block:: python

   from frequensolve.project import Project
   from frequensolve.seismic.layered_model import LayeredModel
   from frequensolve.simulation import SeismicSimulation

   # Create model
   model = LayeredModel(dimension=2, x_limits=[0.0, 4.0])
   model.add_layer(
       name="surface",
       properties={"Vp": 1.5, "Vs": 0.0, "Rho": 1.0}
   )

   # Create simulation
   sim = SeismicSimulation(
       dimension=2,
       physics="acoustic",
       model=model
   )

   # Add source and receivers
   sim.add_source(x=2.0, z=0.1)
   sim.add_receiver_line(
       x_start=0.0,
       x_end=4.0,
       z=0.0,
       spacing=0.1
   )

   # Run simulation
   project = Project(name="example", path="example")
   project += sim
   project.run()

Next Steps
----------

- Check out the :doc:`tutorials/index` for more detailed examples
- Read the :doc:`user_guide/index` for in-depth explanations
- Browse the API reference for detailed documentation

Running Jobs
------------

Sites submit jobs asynchronously and return an awaitable run handle:

.. code-block:: python

   from frequensolve.orchestrator.sites.local import LocalSite

   site = LocalSite()
   run = site.submit(job)
   result = run.wait()

For SLURM systems, configure batch defaults on the site and call
``submit`` the same way:

.. code-block:: python

   from frequensolve.orchestrator.sites.hpc import SlurmRunConfig
   from frequensolve.orchestrator.sites.stampede3 import Stampede3Site

   site = Stampede3Site(
       "projects/example",
       verbose=True,
       run_config=SlurmRunConfig(
           queue="skx",
           nodes=4,
           duration="02:00:00",
           procs_per_node=8,
           procs_per_task=4,
       ),
   )

   run = site.submit(job)
   result = run.wait()

Set ``verbose=True`` on AWS or SLURM sites to print status updates while jobs
are submitted and polled. Project-level logging can be configured when the
project is created:

.. code-block:: python

   project = Project(
       name="example",
       path="example",
       log_level="DEBUG",
       log_file="example/frequensolve.log",
       log_to_console=False,
   )

If you want to run inside an interactive SLURM allocation first, the allocation
uses the same handle lifecycle:

.. code-block:: python

   allocation = site.provision(nodes=1, tasks=8, duration="00:30:00")
   allocation.wait()

   run = site.submit(job, mode="attached")
   result = run.wait()
   traces = run.traces(upscale=4)
