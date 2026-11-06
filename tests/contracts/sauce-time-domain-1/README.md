Time-domain API contract snapshot from Sauce trunk (base 46643a324 plus the coordinated TransientJob changes).

These schemas pin the public Time block, including explicit output times and intervals.

The Time schema was refreshed from Sauce 5e2ca528a on 2026-10-30 to adopt
Gauss4 method selection; other schemas retain their existing snapshot.

The Time schema was refreshed again from Sauce `rez` `ea58e6eaf` on 2026-11-05
to adopt `output`/`trace_output` `shard_samples` and `rate_operator` (receiver
traces only), the Radau3 method and `pml_frequency`. The fragment retains its
earlier snapshot.
