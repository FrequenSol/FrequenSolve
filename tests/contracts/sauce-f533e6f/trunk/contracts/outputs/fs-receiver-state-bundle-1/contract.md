# Receiver state collection, version 1

A collection binds multiple receiver groups for one frequency and acquisition.
`groups` references one `fs-receiver-state-1` manifest per unique receiver group,
with its complete file hash. Relative member paths resolve against the collection.
All groups must be covered exactly once. The single-group action retains the
existing `fs-receiver-state-1` format.

The collection's `state_fingerprint` hashes its canonical JSON excluding that
field. Members share the candidate, acquisition, resolved execution context,
control registry, frequency, source derivative policy, channel and partition.
Retained field checkpoints are written once and referenced by every member.
The group order matches the configured imaging receiver groups. Receiver values,
physical row keys, units and observation identities remain group-local.

Native `receiver_linearize` performs one shared forward hierarchy, then samples
all groups. `receiver_jvp` similarly samples every group after shared Born solves.
