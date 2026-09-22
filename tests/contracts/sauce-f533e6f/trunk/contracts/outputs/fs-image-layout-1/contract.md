# Image product layout v1

Status: initial. Visibility: public. Schema: `fs-image-layout-1`.

This describes the output of a compiled imaging condition. It does not add job
kinds or implement a lag/offset/angle imaging kernel. Each packed HDF5 image dataset
stores the JSON descriptor in its one-string `image_layout` attribute. Generic
`fs-sharded-array-1` image manifests store one descriptor per image at
`arrays/images/attributes/layouts/<zero-based image index>`.

## Layout and units

`channels` and `axes` are ordered arrays. Counts must agree with their entries. Width is the channel count times
the product of axis lengths. Columns vary fastest by channel, then by the first
axis, then subsequent axes. Spatial point order remains the existing regular-grid
order. HDF5 presents `[column, point]` in Python and `(point,column)` in Fortran.
The schema checks structure; the native reader also checks counts, product width,
unique names, finite coordinates and strict coordinate ordering.

Channel `role` is 1=value, 2=forward illumination, 3=adjoint illumination or
4=auxiliary. It is distinct from `value_role`: 1=primal, 2=dual, 3=diagonal.
The latter uses the existing value-frame conversion rules. Empty channel units
inherit the image property unit. Use `1` for dimensionless values. Axis units
are explicit, and coordinates are physical values in those units. Axes are not
nondimensionalized with the background frequency.

The complete per-column value frame remains authoritative for numeric scaling,
space and pairing. Component labels expand as `name[axis=1]`, with one-based
axis indices. Shard descriptors additionally carry `frame` with expanded
components, units, dimensions and solver-to-coordinate scales. The manifest's
existing value-frame space/pairing IDs remain per image. Stored images use
canonical solver representation; they are not optimizer-coordinate covectors.

## Compatibility and operations

Missing layout metadata means the established numerator, forward illumination,
adjoint illumination layout with dual/diagonal/diagonal value roles. Existing
conditions return this layout by default. Old readers may only consume these
three-column products; they must not assume new products have three columns.

Source-batch updates and frequency stacking require identical layouts, including
axis coordinates and value roles. A metadata-only consumer adopts its first
stored layout, then rejects incompatible products. No implicit axis resampling
or unit conversion of axis coordinates is performed.

Shard payloads use `[image, packed-column, point]` and a shared packed-column
extent equal to the largest image width. Columns beyond an individual width are
zero padding. Only each descriptor's width belongs to that product. Full-product
stacked output writes separate, unpadded datasets under `/image/products` (or
`/incremental/products`) in the stack HDF5 file. Each publication replaces that
owned group, removing obsolete product datasets while preserving sibling groups.

Scalar material output keeps its existing scalar value-frame metadata and omits
this packed-product descriptor. Scalar material smoothing and incremental material
updates require the standard
three-channel semantic layout. Extra axes or different channel roles fail before
consumption; an explicit selection/reduction contract is future work. Spectral
composition adds to channels marked value and preserves auxiliary/illumination
channels. Its physical derivative rules still depend on the condition's supported
kernel, not on layout metadata alone.
