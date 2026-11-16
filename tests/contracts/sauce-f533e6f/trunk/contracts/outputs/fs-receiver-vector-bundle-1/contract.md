# Receiver vector collection, version 1

A collection binds multiple receiver groups for one frequency and acquisition.
`groups` references one `fs-receiver-vector-1` manifest per unique receiver group,
with its complete file hash. Relative member paths resolve against the collection.
All groups must be covered exactly once. The single-group action retains the
existing `fs-receiver-vector-1` format.

`source_state` and its checksum bind the whole state collection, and
`state_fingerprint` is that collection's fingerprint. Every member vector is
also bound to its individual group state. Dual collection members can be in any
order; names must match the state collection exactly. Tangents retain state order.

Native `receiver_vjp` loads all groups' independent base/df duals and adds their
receiver loads before the shared adjoint hierarchy. It restores only one copy of
the forward checkpoints. The resulting material covector binds the collection's
state fingerprint. Multiple groups do not multiply the PDE solve count.
