# FS Coordinate System Contract v1

Status: initial
Visibility: public
Schema id: `fs-coordinate-system-1`

## Summary

Coordinate contracts describe named coordinate systems and coordinate-aware
values used by simulation, acquisition, source, receiver, and output blocks.

## Required Behavior

- `global_coordinate_system` defines the default global coordinate system when
  present.
- `coordinate_systems` is a list of additional named coordinate systems.
- Coordinate values may be raw arrays or `{ value, units, system }` objects.
- Current coordinate readers support `cartesian`, `cylindrical`, `spherical`,
  `geographic`, and surface-relative systems. Coordinate-system type may be
  supplied as legacy `type` values or `_type` names such as
  `SurfaceCoordinateSystem`.
- Coordinate systems declare their valid dimension names through `axes`.
  A string list names active axes by position. An object list may name axes by
  inherited `direction = "x" | "y" | "z"` and may specify an `origin`.
  When `inherit_axes` is true, object axes may be partial. Properties and grids
  using a coordinate system must use those declared dimension names.
- Surface-relative coordinate systems require `surface` and use inherited
  `x`, `y`, and `z` directions. `inherit_axes` defaults to true, so directions
  not overridden by `axes` keep their default names. A surface `z` axis may set
  `positive = "up" | "down"` and may set an `origin`; that origin is measured
  as an offset relative to the surface. The inherited surface `z` direction is
  positive down by default. `normal = "up" | "down"` remains
  accepted as a deprecated top-level spelling for old inputs. `up` and `down`
  are orientations, not axis names. A deprecated surface xarray dimension named
  `up` is accepted as a compatibility alias for the surface `z` direction and
  should be migrated to a declared axis name. Surface systems infer `ndim` from
  the solver build when omitted. Reduced coordinate systems may still use
  `fixed_axis` and `fixed_value` for non-surface reductions.
- Axis alignment defaults are `x = east`, `y = north`, `z = down`.
- Axis alignment vectors must be orthogonal and right-handed.
- `ndim` may be `2` or `3`. Reduced two-dimensional non-surface coordinate
  systems require `fixed_axis`; `fixed_value` defaults to zero when omitted.
- `CoordDB_t` is mutable until sealed. Adding a system with an existing name
  replaces that system; after sealing, additions are invalid.
- Points carry a coordinate-system id and local coordinates. Vector and tensor
  reads are interpreted at a point in the active coordinate system.
- Vector values may be raw arrays, `{ value, units }` objects, or
  `{ direction }` objects using the active coordinate system. Direction objects
  do not carry their own coordinate-system selector.
- Units are non-dimensionalized through `units_m`; geographic longitude and
  latitude are treated as dimensionless angular quantities and depth/radius
  quantities use length dimensions.


## Compatibility

Legacy raw arrays remain accepted for many coordinate values. Prefer object form
with explicit `value`, `units`, and `system` in new inputs.
