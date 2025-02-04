Quickstart
==========

This guide will help you get started with FrequenSolve quickly.

Basic Usage
-----------

Here's a minimal example of setting up a seismic simulation:

.. code-block:: python

   from frequensolve import Project, SeismicSimulation
   
   # Create a new project
   project = Project("my_simulation")
   
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

   from frequensolve import Project, SeismicSimulation, LayeredModel
   
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
   project = Project("example")
   project += sim
   project.run()

Next Steps
----------

- Check out the :doc:`tutorials/index` for more detailed examples
- Read the :doc:`user_guide/index` for in-depth explanations
- Browse the API reference for detailed documentation
