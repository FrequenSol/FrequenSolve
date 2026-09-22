Inversion Toolkit
=================

``frequensolve.inversion`` is the generic optimizer toolkit that
:doc:`frequensolve.imaging` builds on: bounded L-BFGS and inexact Newton-CG
minimizers, frequency and Laplace continuation schedules, restartable
optimization history and checkpoints, inverse-Hessian preconditioning
estimates, and derivative checks. It is independent of the solver and can be
used with any real-valued objective.

Optimizers
----------

.. automodule:: frequensolve.inversion.optimization
   :members:
   :show-inheritance:
   :noindex:

Continuation Schedules
----------------------

.. automodule:: frequensolve.inversion.continuation
   :members:
   :show-inheritance:
   :noindex:

History, Checkpoints And Results
--------------------------------

.. automodule:: frequensolve.inversion.history
   :members:
   :show-inheritance:
   :noindex:

Inverse-Hessian Preconditioning
-------------------------------

.. automodule:: frequensolve.inversion.preconditioning
   :members:
   :show-inheritance:
   :noindex:

Derivative Checks
-----------------

.. automodule:: frequensolve.inversion.validation
   :members:
   :show-inheritance:
   :noindex:

Real Data Layouts
-----------------

.. automodule:: frequensolve.inversion.data
   :members:
   :show-inheritance:
   :noindex:

Least-Squares Adapters
----------------------

.. automodule:: frequensolve.inversion.least_squares
   :members:
   :show-inheritance:
   :noindex:
