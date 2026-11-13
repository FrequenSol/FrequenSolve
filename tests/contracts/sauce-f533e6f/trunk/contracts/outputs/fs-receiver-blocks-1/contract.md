# Receiver-probe local parameter blocks

HDF5 `/schema = fs-receiver-blocks-1` stores local Gauss–Newton curvature from
shared receiver probes. Request `fwi_operator.receiver_diagonal.blocks` with
`controls` (at least two active material names) and optional `output` (default
`receiver_blocks.h5`). The filename receives the standard frequency-task suffix.
Waveform and phase-derivative material linearizations are supported. Selected
controls must share the same meshed material basis, transform, global coefficient
IDs and ownership; incompatible layouts fail before probe accumulation.

`/control_spaces/<name>` records each native basis identity. For every ordered
pair of selected names, `/blocks/<row>/<column>` is a float64 vector in global
material coefficient order. Both triangles are present and identical. Diagonal
entries exactly match the corresponding `receiver_diagonal` output. Values have
objective units divided by the row and column parameter units; neither physical
parameter scaling nor inverse damping is applied to the artifact.

Products are formed after constraints and MPI reduction, using the same probe
normalization, real/complex metric, robust IRLS weights and source-sampling
correction as the diagonal. Cross-parameter retention adds no PDE solves and
stores only local products, not complete probe columns. Each sample contributes
a positive semidefinite outer product. Blocks omit coupling between distinct
material coefficients. Clients add frequency parts using objective weights;
physical scaling, damping and inversion are separate operations.
