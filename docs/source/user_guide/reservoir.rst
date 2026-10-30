Reservoir model authoring
=========================

The ``fs.ReservoirModel`` family authors the standalone
``fs-reservoir-model-1`` input contract for Sauce's compositional HDG reservoir
application. It is separate from seismic ``Simulation`` and ``Job`` workflows:
there is currently no ``ReservoirJob`` or ``Site.run(model)`` adapter. Execution
requires the matching dedicated Sauce reservoir driver. The SDK's domain
builder currently describes rectangular 2D domains.

All reservoir classes are available through ``import frequensolve as fs``.
Build a model from ``ReservoirFluid``, ``ReservoirRock``, ``CoreyStoneII``,
``ReservoirDomain``, wells, injection streams, schedule phases, and an
``InitialState``. Existing contract documents can also be loaded and edited::

   import json
   from pathlib import Path
   import frequensolve as fs

   model = fs.ReservoirModel.from_fs(
       json.loads(Path("reservoir-input.json").read_text())
   )
   print(model.end_time)  # total scheduled duration in days
   Path("reservoir-export.json").write_text(json.dumps(model.to_fs(), indent=2))

Physical values accept plain numbers in contract units, Pint quantities, or
``{"value": ..., "units": ...}`` mappings. Times default to days and injection
rates to m²/day per unit thickness. For example, a 24-hour phase contributes
one day to ``model.end_time``::

   phase = fs.SchedulePhase(duration=fs.Q_(24, "hour"), rate=0)

Most physical quantities preserve their authored units on export. Component
critical constants are converted to the numeric SI scalars required by the
contract: K, Pa, kg/mol, and m³/mol. Thus a component molecular weight of
``fs.Q_(16.043, "g/mol")`` is exported as approximately ``0.016043`` kg/mol.

Permeability is the ``kappa`` property of one ``reservoir:compositional`` material
subdomain; it may be a scalar, file-backed property, or expression. Serialized
input is validated against the matching contract by the dedicated executor.
Authoring a document does not launch a simulation.
