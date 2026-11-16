Time-domain API contract snapshot from Sauce trunk.

These schemas pin the public Time block, including Radau3 as the default wave
scheme, explicit Newmark and Gauss4 selection, output schedules and source
configuration. Other schemas retain their existing snapshot.

The Time schema was refreshed from Sauce `rez` `ea58e6eaf` on 2026-11-05
to adopt `output`/`trace_output` `shard_samples` and `rate_operator` (receiver
traces only), the Radau3 method and `pml_frequency`. The fragment retains its
earlier snapshot.
