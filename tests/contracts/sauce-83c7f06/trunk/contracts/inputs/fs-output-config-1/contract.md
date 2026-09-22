# FS Output Configuration Contract v1

Status: initial
Visibility: public
Schema id: `fs-output-config-1`

## Summary

Output configuration describes solver-produced visualization, trace, and
field outputs. This contract currently focuses on the Paraview manager surface.

## Required Behavior

- `Outputs` may contain a `ParaView` block.
- `Outputs/traces/path` is the preferred receiver trace output directory.
- `Outputs/receivers/path` remains accepted as a legacy fallback.
- `ParaView` may be a legacy object, a legacy top-level list of objects, or a
  structured object with a list under `items`.
- `ParaView/execute_on` may name setup, solve, adapt, output, or cleanup phases.
- Output paths are resolved relative to `FS_env%ResultPath` unless a subsystem
  explicitly documents absolute behavior.
- `Outputs/Units` may define default output units for exported geometry,
  field dimensions, named fields, material-property dimensions, and named
  material properties. Per-component or per-item
  `units` remain the highest-precedence output-unit setting.
- When `Outputs/Units` is omitted, the built-in output defaults are equivalent
  to `geometry = m` plus field dimension defaults of `pressure = Pa`,
  `stress = Pa`, `velocity = mm/s`, and `density = g/cc`. Material properties
  additionally default velocity-model outputs such as `Vp` and `Vs` to `m/s`
  and stress-like model outputs to `MPa`.
- Top-level `Units/defaults/<key>` describes input/default unit conventions and
  does not override output units. Use `Outputs/Units` for all output-unit
  policy.
- `Outputs/Units/dimensions/<key>` overrides field, trace, and wavefield
  dimension defaults only.
- `Outputs/Units/fields/dimensions/<key>` and
  `Outputs/Units/attributes/dimensions/<key>` are explicit aliases for field
  dimension defaults.
- `Outputs/Units/fields/<field>` applies to receiver-plan-backed trace and
  ParaView field outputs when a component does not set `units` directly.
- `Outputs/Units/properties/<property>` applies to ParaView material-property
  outputs. Info outputs such as `Domain` and `Order` remain unitless unless an
  item explicitly sets units.
- `Outputs/Units/properties/dimensions/<key>` overrides dimension fallbacks for
  known material-property outputs without changing field or trace units.
- Known material-property outputs such as `Vp`, `Vs`, `Rho`, poroelastic
  moduli, and EM material properties fall back to material-property dimension
  defaults when no explicit property unit is configured.
- `Outputs/Units/geometry` sets the length unit for exported trace,
  acquisition, and visualization coordinates. Angular coordinate axes remain
  unscaled.
- ParaView geometry arrays record the exported coordinate unit in file-format
  metadata: VTU/VTR XML arrays carry `fs_units` and `UNITS_LABEL`, and XDMF
  HDF5 geometry datasets carry a `units` attribute.
- Receiver trace files are written as `traces_<task>.h5` when
  `Outputs/traces/path` is present. Legacy `receivers_<task>.h5` files remain
  supported for older receiver output configuration.
- `writer.format` currently accepts `vtu`, `vtr`, `xmf`, and related legacy
  names handled by the output manager.
- `target.kind` currently defaults to `surface`.
- ParaView element upscaling is configured only at
  `target/mesh/upscale`. The value is an integer from `0` through `2` and is
  valid only for `volume` and `surface` targets. A top-level ParaView
  `upscale` field is invalid; there is no precedence or compatibility alias.
- ParaView output is solution-backed in the public API.
- ParaView field names resolve through receiver plans; poroelastic trace
  unknowns `poroelastic:p_hat`, `poroelastic:u_hat`, `poroelastic:q_hat`, and
  `poroelastic:t_hat` are available as solution-backed point fields without
  requiring `post_process`. Reconstructed post-processing is currently
  available only when every active physics is acoustic or elastic DPG. If any
  active physics is unsupported, initialization emits one warning and disables
  post-processing for all observation outputs.
- `Outputs/wavefields` is a list of `WavefieldOutput` objects.
- A `WavefieldOutput` samples a grid through either `field`, the legacy
  `fields` shorthand, or an explicit receiver `device`. `device` is mutually
  exclusive with `field` and `fields`; use `device/components` when multiple
  named components should be written from one wavefield output.
- A `forward_df` wavefield output may request `instantaneous_traveltime`. Sauce
  then writes `<field>_instantaneous_traveltime` in seconds using
  `-Im(conjg(u) * du/df) / (2*pi*(abs(u)^2 + epsilon^2))`. The minus sign
  converts Sauce's `exp(+i*omega*t)` frequency-phase slope to positive arrival
  time. The optional
  `relative_amplitude_floor` sets `epsilon` relative to each component/source
  RMS amplitude; it defaults to zero.

## Trace HDF5 Schema

Trace output is split into shared metadata, payload shards, and a packed public
product. Frequency-domain jobs use frequency shards. Transient jobs emit one
deterministically named `step_<12-digit-step>.h5` shard for every scheduled
sample; each transient shard stores `/sample_axis = "time"`, `/time_step`, and
`/time` alongside the receiver payload. The final time step is always sampled.

- `--init` writes one shared `trace_metadata.h5` file for each trace-style
  output family. If a frequency task starts and this metadata file is missing,
  the task may create the same metadata file before writing its payload shard.
  Existing metadata with stale or invalid provenance is an error. This file uses
  schema `fs_trace_metadata_v1`, records provenance for the active job and
  simulation inputs, and stores the survey catalogs plus dataset metadata needed
  to interpret payload shards.
- Frequency tasks write payload-only shard files under `shards/f_*.h5` with
  schema `fs_trace_payload_shard_v1`. Payload shards include `/frequency`,
  `/laplace`, `/task_id`, `/trace_metadata_file`, and the receiver payload
  datasets. They do not duplicate survey catalogs, dense trace identity tables,
  or dataset attributes that are stable across frequencies.
- Nonzero Laplace components are included in shard names as
  `f_<frequency>_s_<laplace>_hz.h5`, so damping sweeps at one real frequency do
  not overwrite one another. Zero-Laplace files retain the legacy
  `f_<frequency>_hz.h5` spelling.
- Transient tasks currently expose their time-step shards directly. Packing a
  time-indexed public trace artifact is reserved for a subsequent contract
  revision; frequency packing must not consume transient shards.
- `--pack` writes the public trace artifact, normally `traces.h5`, using schema
  `fs-traces-packed-1`. `traces.h5` is the consumer-facing trace store; unpacked
  shards and `trace_metadata.h5` are transient implementation details.
- Dense receiver payloads remain available at `/<receiver_group_name>` inside a
  payload shard and at `/trace_data/<receiver_group_name>/<dataset_number>`
  inside the packed artifact.
- Dense receiver dataset attributes such as `dims`, `component`, `units`,
  `dimension`, `layout_kind`, and compact axis attributes are defined once in
  `trace_metadata.h5` and restored onto packed payload datasets by `--pack`.
- `/survey/schema_version` stores `fs_seismic_trace_store_v1` in
  `trace_metadata.h5` and `fs-traces-packed-1` in the packed artifact.
- `/survey/layout_kind` stores `dense_trace_v1`, `sparse_trace_v1`, or
  `mixed_trace_v1`.
- `/survey/sources` stores `source_id`, `field_record`, and `coordinates`.
  Source names are exposed as the logical `/survey/sources/source_name` column,
  either as an explicit string dataset or as generated-name metadata on
  `source_id`.
- `/survey/source_statistics` is present for low-rank source-covariance RHSs.
  Rows join to `/survey/sources/source_id` and carry `factor_set`, one-based
  `factor_index`, fixed `factor_rank`, source spectral conventions,
  factorization provenance, and `response_role = receiver_response_factor`.
  The corresponding complex trace values are the physical response factor
  `Z = H L`; consumers may form PSD or selected CSD across all rows in a factor
  set without materializing a dense receiver covariance.
- `/survey/receiver_groups/_catalog` stores compact receiver-group identity:
  `group_name`, `dataset_path`, `layout_kind`, `receiver_count`,
  `component_count`, and `source_count`.
- Dense receiver groups store group-local receiver metadata under
  `/survey/receiver_groups/<group>/receivers`, with local `receiver_id`,
  logical `receiver_name`, and `coordinates`. Receiver ids are local to the
  group and do not imply a global receiver numbering across distinct
  configurations.
- Dense receiver groups may store group-local component metadata under
  `/survey/receiver_groups/<group>/components`.
- `/survey/receivers` and `/survey/components` may be present for
  sparse/imported layouts that already carry stable numeric layout identifiers.
  Receiver coordinates, when known, are stored under
  `/survey/receivers/coordinates`.
- Catalog name columns may be compactly represented when every name is generated
  from existing integer catalog fields. In that case the logical string dataset
  is omitted and the integer dataset carries `name_encoding` attributes.
  `integer_suffix_v1` reconstructs
  `name_prefix // name_separator // zero_padded_integer`, where the integer
  comes from `name_integer_dataset` and padding width is
  `name_integer_width`. `group_integer_suffix_v1` reconstructs receiver names
  from `name_group_name_dataset`, `name_group_dataset`, and
  `name_integer_dataset`. When an explicit string dataset exists, consumers
  must prefer it over generated-name metadata.
- Dense receiver groups do not write Cartesian per-trace identity tables. Their
  trace identity is defined by dataset shape, axis order, compact axis
  attributes, and the receiver-group catalog.
- Sparse receiver groups write numeric per-group trace identity tables under
  `/survey/receiver_groups/<group>/traces`. These tables include `trace_id`,
  `source_id`, `receiver_id`, `component`, and `weight`.
- Sparse trace stores write flat trace identity tables under `/survey/traces`
  with `trace_id`, `source_id`, `receiver_id`, `component`, and `weight`. When
  all traces share the same component set, the store may instead write
  `layout_encoding = aligned_components_v1`, per-source/receiver rows under
  `/survey/trace_nodes`, and component definitions under `/survey/components`.
- Sparse trace `weight` is stored as `real(4)`. A value of `0.0` marks the
  trace inactive.
- Trace output does not persist internal receiver-point ranges, sample maps, or
  per-trace names. In particular, `point_first`, `point_last`, `n_points`,
  `sample_id`, `receiver_position_id`, `component_id`, `channel_number`,
  `field_record`, `source_name`, `receiver_name`, and `component_name` are not
  written in per-trace tables. Field records live on `/survey/sources`; names
  live only in the source, receiver, and component catalogs.
- Each frequency task publishes its exact trace metadata and payload records in
  `_fs_run/tasks/task_NNNNNN/result.json`. The `--pack` pass publishes an
  immutable packed-segment manifest from
  `_fs_run/operations/pack/result.json`; consumers do not infer filenames or
  discover shards recursively.


## Compatibility

This schema accepts both legacy compact Paraview configuration and the newer
structured writer/target/item forms, except that `upscale` has a single
canonical location at `target/mesh/upscale`. Trace output accepts both the
preferred `traces` block and the legacy `receivers` block during the API
transition.
