# FS Units Contract v1

Status: initial
Visibility: public
Schema id: `fs-units-1`

## Summary

The units contract describes the runtime dimension, unit-expression, reference
scale, and default-output-unit behavior provided by `units_m`. The system is
used by simulation readers, coordinate readers, material/model readers, and
output writers to convert user values into solver non-dimensional values and
back into declared output units.

## Required Behavior

- Dimensions are canonical maps from base symbols to real exponents. Current
  base symbols are `L`, `T`, `M`, `I`, and `Theta` (temperature).
- Dimension arithmetic is additive on exponents. Near-zero exponents are
  removed, and equality uses the runtime dimension tolerance.
- Unit databases are mutable until sealed. Adding a duplicate unit key replaces
  that key; adding an alias requires the canonical key to already exist.
- Unit expressions may contain registered unit keys, SI prefixes, numeric
  factors, multiplication, division, exponentiation with `^` or `**`, and
  parentheses. Whitespace is ignored. The parser normalizes common Unicode
  forms such as micro and middle-dot to ASCII equivalents.
- `non_dimensionalize(scales, db, x, unit_str)` converts a user value into the
  current non-dimensional solver scale. `dimensionalize` performs the inverse.
- Runtime seismic scale initialization reads `disable_scaling`, `scaling`, `f0`,
  `length_scale`, `time_scale`, and `mass_scale` from the active simulation JSON.
  When scaling is disabled, all base scales are `1`. `scaling: "robust"` is an
  opt-in simulation-initialization path: the first unit pass uses legacy defaults,
  material ranges are probed, and final explicit scales are then installed before
  the final model initialization.
- Output unit defaults are configured under `Outputs/Units` (or under `Units`
  when an `Outputs` object is passed directly to an output subsystem). Top-level
  simulation `Units/defaults/<key>` is not an output-unit override.
- The built-in output geometry default is `m`. The output `dimensions` map
  updates field, trace, and wavefield dimension defaults; `fields/<field>` is
  validated against receiver-plan dimensions when the field is bound;
  `properties/<property>` is parsed as a unit expression and applied to material
  property values; known material properties fall back to
  `properties/dimensions`; and `geometry` must be a length unit.
- Missing scale bases, unknown unit tokens, incompatible default units, invalid
  aliases, or non-positive scale factors are validation errors.

## Runtime Defaults

The initial output defaults are:

- `geometry`: `m`
- `length`: `km`
- `time`: `s`
- `mass`: `kg`
- `frequency`: `Hz`
- field `velocity`: `mm/s`
- `density`: `g/cc`
- field `temperature`: `K`
- field `pressure`: `Pa`
- field `stress`: `Pa`
- material-property `velocity`: `m/s`
- material-property `pressure`: `MPa`
- material-property `stress`: `MPa`
- `strain`: `1`
- `force`: `kN`
- `moment`: `kN*km`
- `attenuation`: `1/m`
- `wavenumber`: `1/m`
- `conductivity`: `S/m`
- `permittivity`: `F/m`
- `permeability`: `H/m`
- `efield`: `V/m`
- `bfield`: `T`


## Compatibility

Existing simulation documents may omit the `Units` block. New inputs should use
explicit unit strings for dimensional quantities whenever values are not already
in the solver's active non-dimensional units.

## Thermal units

Absolute temperature uses `K` (`Kelvin`, `kelvin`, `degK`), `degC`
(`Celsius`, `celsius`, `°C`), `degF` (`Fahrenheit`, `fahrenheit`, `°F`), and
`degR` (`Rankine`, `rankine`, `°R`). The Kelvin reference multiplier is one.
Conversions preserve the affine origin: 0 degC = 273.15 K = 32 degF. Use
`convert_units(db, x, source_unit, target_unit)` for direct physical conversion;
`non_dimensionalize` and `dimensionalize` also preserve offsets. A multiplicative
`unit_conversion_factor` rejects conversions requiring an origin shift.

Temperature differences use `K`, `delta_K`, `delta_degC`, `delta_degF`, or
`delta_degR`; a difference of 9 delta_degF equals 5 K. Affine units cannot be
prefixed, multiplied, divided, or exponentiated. Material coefficient expressions
therefore use `W/m/K`, `W/m/delta_degF`, `J/m^3/K`, or `J/kg/K`. `J` and `W`
support ordinary SI prefixes.

Temperature receiver and wavefield output defaults to K. Configure
`Outputs/Units/fields/temperature` or `Outputs/Units/dimensions/temperature`
with `degC` or `degF` for absolute output, or a delta unit for temperature rise.
Affine HDF5 value frames add an optional `value_frame_coordinate_offset` vector
to the primal map: `coordinate = solver_to_coordinate * solver + offset`.
Missing offsets mean zero; dual values and differences use only the scale.

## Conductive EM reference scaling

Set root `em_reference_conductivity` to a positive finite representative conductivity
in S/m (for example 0.01 for 100 Ohm*m earth). The EM initializer then uses
`epsilon_ref = sigma_ref/(2*pi*f_ref)`, `mu_ref = mu0`, and `E_ref = 1 V/m`.
Here `f_ref` is the positive physical frequency selected for runtime scaling,
including `Mesh/adapt/f_low` or `f_adapt`. All spectral evaluations within this
basis hold the reference fixed. At `f_ref`, a matching earth has solver
effective permittivity approximately -i and permeability 1. Air retains its
physical displacement-current contrast.

`f0` (default 10; 1 is useful for MT) or `time_scale` sets the time reference.
Length follows from `1/sqrt(mu_ref*epsilon_ref)`, and the mass/current references
preserve both constitutive equations and SI field conversion. Named EM units
(`V`, `T`, `Wb`, `H`, `Ohm`, `F`, `S`) are mass-consistent with `kg`, `N`, `J`,
and `W`: `V*A` equals `W`, and a base-unit spelling such as `kg*m*s^-3*A^-1`
converts identically to `V/m`. Magnetic receiver
output remains mu0 H in T. This option cannot be combined with `length_scale`,
`mass_scale`, or `disable_scaling: true`. Omission preserves vacuum EM scaling.
These controls belong at the simulation root, not under `Units`.
