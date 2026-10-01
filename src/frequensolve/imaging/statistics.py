# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Statistical imaging declarations and typed Sauce covariance results.

Sauce forms covariance factors, mesh prior transfers and grid projections.
Gaussian prior terms and covariance actions reuse fixed diagonals and stored
factors, so they mirror Sauce's ``covariance_fields`` formulas in NumPy
instead of starting a solver process per evaluation.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np
import xarray as xr
from scipy.sparse import csr_matrix, hstack

from .controls import ControlSpace, ControlState, ControlVector, _BindContext
from .curvature import CurvatureResult, _finite, _integer
from .regularization import BoundRegularization, Regularization, _SymmetricModelOperator

# Hard links are unavailable across filesystems or on filesystems without them.
_UNLINKABLE = frozenset(
    {errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP, errno.EOPNOTSUPP}
)


def _publish_artifact(source: Path, target: Path) -> None:
    """Hard-link an immutable artifact, copying only where a link is impossible.

    Factor and mesh files are never modified after they are recorded (readers
    verify their digests), so a saved result shares their storage instead of
    duplicating up to controls-by-rank bytes. The link is published atomically.
    """
    if target.exists() and os.path.samefile(source, target):
        return
    pending = target.with_name(f".{target.name}.{uuid.uuid4().hex}")
    try:
        try:
            os.link(source, pending)
        except OSError as error:
            if error.errno not in _UNLINKABLE:
                raise
            shutil.copy2(source, pending)
        os.replace(pending, target)
    finally:
        pending.unlink(missing_ok=True)


def _gaussian_terms(
    point: np.ndarray, reference: np.ndarray, std: np.ndarray, diagonal: np.ndarray
) -> tuple[float, np.ndarray]:
    """Value and gradient of Sauce ``gaussian_prior`` (``diagonal = 1/std**2``)."""
    if not (np.isfinite(point).all() and np.isfinite(reference).all()):
        raise ValueError("Gaussian prior coordinates must be finite")
    delta = point - reference
    with np.errstate(over="ignore"):
        gradient = delta * diagonal
        value = 0.5 * float(np.sum((delta / std) ** 2))
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Gaussian prior evaluation overflow")
    return value, gradient


class _Covariance(_SymmetricModelOperator):
    """``C = S (B0 + V diag(eigenvalues) V^T) S`` with ``S = diag(prior_std)``.

    Mirrors Sauce ``covariance_fields::covariance_action`` on factors that are
    read once and checked against their recorded dataset digests in memory;
    blocks of directions use two dense products.
    """

    def __init__(self, factors: CurvatureResult, space: ControlSpace) -> None:
        rank = factors.metadata.get("rank")
        with h5py.File(factors.path, "r") as h5:
            stored = {name for name in ("eigenvalues", "modes") if name in h5}
        if stored not in (set(), {"eigenvalues", "modes"}) or (
            not stored and int(rank or 0) > 0
        ):
            raise ValueError("Covariance factors are incomplete")
        arrays = factors.read_verified(
            "base_inverse_diagonal", "prior_std", *sorted(stored)
        )
        self.base = arrays["base_inverse_diagonal"]
        self.scale = arrays["prior_std"]
        self.eigenvalues = arrays.get("eigenvalues", np.empty(0))
        self.modes = arrays.get("modes", np.empty((0, self.base.size)))
        if rank is not None and int(rank) != self.eigenvalues.size:
            raise ValueError("Covariance rank disagrees with stored factors")
        if (
            self.base.shape != (space.size,)
            or self.scale.shape != self.base.shape
            or self.modes.shape != (self.eigenvalues.size, self.base.size)
        ):
            raise ValueError("Covariance factor shapes disagree with the control space")
        if not all(
            _finite(a) for a in (self.base, self.scale, self.modes, self.eigenvalues)
        ):
            raise ValueError("Covariance factors must be finite")
        if np.any(self.base <= 0) or np.any(self.scale < 0):
            raise ValueError("Invalid covariance baseline or coordinate scale")
        super().__init__(self._apply, space)

    def _apply(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Covariance directions must be finite")
        block = values.reshape(self.base.size, -1)
        with np.errstate(over="ignore", invalid="ignore"):
            direction = self.scale[:, None] * block
            actions = self.base[:, None] * direction
            if self.eigenvalues.size:
                coefficients = self.eigenvalues[:, None] * (self.modes @ direction)
                actions += self.modes.T @ coefficients
            actions *= self.scale[:, None]
        if not np.isfinite(actions).all():
            raise ValueError("Covariance action overflow")
        return actions.reshape(values.shape)

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        return self._apply(X)


def material_units(space: ControlSpace, block: Any) -> Optional[str]:
    """Resolve property units separately from the block's geometric units."""
    if (
        getattr(space, "simulation", None) is None
        or block.prop is None
        or block.subdomain is None
    ):
        return None
    from frequensolve.model.property import canonical_property_name

    prop = (
        _BindContext(getattr(space, "simulation", None))
        .subdomain(block.subdomain)
        .properties[canonical_property_name(block.prop)]
    )
    return prop.units


def mesh_descriptor(space: ControlSpace, block: Any) -> dict:
    """Find and verify the frozen artifact behind one material mesh control."""
    simulation = getattr(space, "simulation", None)
    if simulation is None:
        raise ValueError("A mesh control needs its bound simulation")
    definition = getattr(space, "mesh_property_spaces")[block.control.space]
    path = Path(definition.artifact).expanduser()
    if not path.is_absolute():
        path = Path(simulation.project_path) / path
    path = path.resolve()
    names = [sub.name for sub in simulation.model.subdomains]
    material = names.index(block.subdomain) + 1
    with h5py.File(path) as h5:
        group = h5["property_space"]
        identity = group["identity"][()].decode().rstrip(" \0")
        descriptor = dict(
            schema="sauce-parameterized-property-identity-1",
            control=f"{identity}/material/{material}",
            transform=block.transform,
        )
        expected = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        if expected != block.basis_identity:
            raise ValueError("Property mesh differs from the statistical control basis")
        if "roots" not in group:
            raise ValueError(
                "Native statistical operations require a readable property-space artifact"
            )
        roots = sorted(int(key) for key in group["roots"])
        metres = (
            None
            if "metres_per_native_unit" not in group
            else float(np.asarray(group["metres_per_native_unit"]).reshape(-1)[0])
        )
        if metres is not None and (not np.isfinite(metres) or metres <= 0):
            raise ValueError("Invalid property-mesh geometry units")
    return dict(
        path=str(path),
        identity=identity,
        material=material,
        roots=np.asarray(roots, dtype=np.int32),
        size=block.size,
        metres_per_native_unit=metres,
    )


def mesh_grid_sampling(native: Any, descriptor: dict, grid: Any, block: Any) -> tuple:
    """Ask Sauce for a hanging-node-aware field projection on a labelled grid."""
    from frequensolve.geometry.grids import CartesianGrid
    from frequensolve.units import ureg

    if not isinstance(grid, CartesianGrid) or set(grid.dims) != (
        {"x", "z"} if len(grid.dims) == 2 else {"x", "y", "z"}
    ):
        raise ValueError("Mesh uncertainty needs a global Cartesian x/z or x/y/z grid")
    if (grid.system or "global") != "global":
        raise ValueError("Mesh uncertainty requires a global Cartesian grid")
    coords = {
        dim: np.linspace(grid.x0[i], grid.x1[i], grid.n[i])
        for i, dim in enumerate(grid.dims)
    }
    field = xr.DataArray(np.zeros(grid.shape), dims=grid.dims[::-1], coords=coords)
    factor = float((1 * ureg(grid.units or "m")).to("m").magnitude)
    order = ["x", "z"] if len(grid.dims) == 2 else ["x", "y", "z"]
    points = np.column_stack(
        [
            field.coords[dim].broadcast_like(field).values.ravel() * factor
            for dim in order
        ]
    )
    for dim in grid.dims:
        field.coords[dim].attrs["units"] = grid.units or "m"
    sampling, valid = native.mesh_sampling(descriptor, points)
    return field, sampling, valid.reshape(field.shape)


def _linear_materials(space: ControlSpace) -> None:
    for name in space.blocks:
        block = space.block(name)
        if (
            not name.startswith("model.")
            or block.complex
            or block.transform != "identity"
        ):
            raise ValueError(
                "Statistical imaging currently requires real identity-transform material controls"
            )


@dataclass(frozen=True)
class GaussianPrior(Regularization):
    """Gaussian prior on real additive material fields.

    ``reference`` is a ControlState. ``std`` is a positive scalar, array or
    block mapping; quantities are converted to the material property's units.
    This prior contributes to the fitted objective, gradient and Hessian.
    On meshes, the default penalty is volume-lumped and normalized by the
    complete material volume. ``std`` then specifies a physical field scale,
    not a nodal marginal: nodal variance is std**2 / measure_weight. Mean and
    scale fields are lifted from the reference basis at every refinement.
    ``mesh_measure="coefficients"`` instead defines independent nodal priors
    whose strength depends on resolution. Neither policy carries a posterior
    across stages. Statistical mesh operations require linear root geometry.
    Only mesh transfers run in Sauce; penalty evaluations use NumPy.
    """

    reference: ControlState
    std: Any
    mesh_measure: str = "volume"

    def __post_init__(self) -> None:
        if self.mesh_measure not in {"volume", "coefficients"}:
            raise ValueError("mesh_measure must be volume or coefficients")

    def bind(
        self, space: ControlSpace, *, problem: Any = None, linearization: Any = None
    ) -> BoundGaussianPrior:
        if problem is None:
            raise ValueError(
                "GaussianPrior needs a problem's Sauce execution site; bind through FWI"
            )
        baseline = linearization.state if linearization is not None else problem.state
        meshed = any(space.block(name).kind == "mesh" for name in space.blocks)
        native = problem.backend.curvature() if meshed else None
        return BoundGaussianPrior(self, space, native, baseline=baseline)


class BoundGaussianPrior(BoundRegularization):
    """One Gaussian prior on a stage's active space.

    Mesh means and scales are lifted by Sauce once at binding. Value, gradient
    and the constant curvature diagonal ``1/std**2`` then follow Sauce's
    ``gaussian_prior`` formula in NumPy on every evaluation.
    """

    def __init__(
        self,
        specification: GaussianPrior,
        space: ControlSpace,
        native: Any,
        *,
        baseline: Optional[ControlState] = None,
    ) -> None:
        super().__init__(specification, space)
        _linear_materials(space)
        if not isinstance(specification.reference, ControlState):
            raise TypeError("GaussianPrior.reference must be a ControlState")
        self.reference = np.empty(space.size)
        self.measure_weights = {}
        self.std = np.empty(space.size)
        self.units = {}
        frozen_points, frozen_means, frozen_scales = [], [], []
        full_baseline = None
        if any(not space._mask_of(space.block(k)).all() for k in space.blocks):
            if baseline is None:
                raise ValueError(
                    "A Gaussian prior with frozen controls needs the full stage state"
                )
            full_baseline = baseline.vector(space.without_support()).values
        for name in space.blocks:
            block = space.block(name)
            unit = material_units(space, block)
            self.units[name] = unit
            value = specification.std
            if isinstance(value, Mapping):
                keys = [key for key in (name, block.address, block.key) if key in value]
                if not keys:
                    raise ValueError(
                        f"Gaussian prior is missing std for {block.address!r}"
                    )
                key = keys[0]
                value = value[key]
            if hasattr(value, "to"):
                if unit is None:
                    raise ValueError(
                        f"Property units are required for {block.address!r}"
                    )
                value = value.to(unit).magnitude
            if np.iscomplexobj(value):
                raise ValueError("Gaussian prior scale must be real")
            source_space = specification.reference.space.without_support()
            source = source_space.block(name)
            source_values = specification.reference.values[
                source_space.full_slices[name]
            ]
            array = (
                np.broadcast_to(np.asarray(value, dtype=float), source.shape)
                .ravel(order="F")
                .copy()
            )
            if not np.isfinite(array).all() or np.any(array <= 0):
                raise ValueError("Gaussian prior std must be positive finite")
            if block.kind == "mesh":
                lifted = native.mesh_prior(
                    mesh_descriptor(source_space, source),
                    mesh_descriptor(space, block),
                    source_values,
                    array,
                )
                mean = lifted.read("reference")
                scale = lifted.read("prior_std")
                measure = lifted.read("measure_weights")
                if specification.mesh_measure == "coefficients":
                    scale = scale * np.sqrt(measure)
                self.measure_weights[name] = measure
            elif source_space.restrict(name).equivalent(
                space.without_support().restrict(name)
            ):
                mean, scale = source_values, array
            else:
                # Basis fields are transferred, never covariance marginals.
                old = source_space.restrict(name)
                new = space.without_support().restrict(name)
                mean = old.transfer_to(new, ControlVector(source_values, old)).values
                scale = old.transfer_to(new, ControlVector(array, old)).values
                if np.any(scale <= 0):
                    raise ValueError(
                        "Transferred Gaussian scale is not positive; declare a prior on the target basis"
                    )
            active = space._mask_of(block)
            self.reference[space.slices[name]] = mean[active]
            self.std[space.slices[name]] = scale[active]
            if not active.all():
                assert full_baseline is not None
                frozen_points.append(full_baseline[space.full_slices[name]][~active])
                frozen_means.append(mean[~active])
                frozen_scales.append(scale[~active])
        if isinstance(specification.std, Mapping):
            # Other stage blocks may be present, but spelling errors must be visible.
            known = {
                key
                for name in specification.reference.space.blocks
                for key in (
                    name,
                    specification.reference.space.block(name).address,
                    specification.reference.space.block(name).key,
                )
            }
            unknown = set(specification.std) - known
            if unknown:
                raise ValueError(f"Unknown Gaussian prior blocks: {sorted(unknown)}")
        if not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Gaussian prior std must be positive finite")
        with np.errstate(over="ignore", divide="ignore"):
            self._diagonal = 1.0 / self.std**2
        if not np.isfinite(self._diagonal).all():
            raise ValueError("Gaussian prior std is too small for float64 curvature")
        self._diagonal.flags.writeable = False
        self.native = native
        self.frozen_value = 0.0
        frozen_identity = b""
        if frozen_points:
            fixed = np.concatenate(frozen_points)
            fixed_mean = np.concatenate(frozen_means)
            fixed_std = np.concatenate(frozen_scales)
            frozen_identity = (
                fixed.tobytes() + fixed_mean.tobytes() + fixed_std.tobytes()
            )
            with np.errstate(over="ignore", divide="ignore"):
                fixed_diagonal = 1.0 / fixed_std**2
            self.frozen_value = _gaussian_terms(
                fixed, fixed_mean, fixed_std, fixed_diagonal
            )[0]
        self.identity = hashlib.sha256(
            self.reference.tobytes()
            + self.std.tobytes()
            + frozen_identity
            + json.dumps(
                [
                    (space.block(k).name, space.block(k).basis_identity)
                    for k in space.blocks
                ]
            ).encode()
        ).hexdigest()

    def _terms(self, v: Any) -> tuple[float, np.ndarray]:
        return _gaussian_terms(
            self._values(v), self.reference, self.std, self._diagonal
        )

    def value(self, v: Any) -> float:
        return self.frozen_value + self._terms(v)[0]

    def gradient(self, v: Any) -> ControlVector:
        return self._wrap(self._terms(v)[1])

    def curvature_diagonal(self, v: Any = None) -> np.ndarray:
        if v is not None and not np.isfinite(self._values(v)).all():
            raise ValueError("Gaussian prior coordinates must be finite")
        return self._diagonal.copy()

    def hessian_operator(self, v: Any = None) -> Any:
        diagonal = self.curvature_diagonal(v)
        # Applying the already evaluated diagonal is optimizer bookkeeping.
        return _SymmetricModelOperator(lambda x: diagonal * x, self.space)


@dataclass(frozen=True)
class BFGSUncertainty:
    """Capture complete fixed-metric L-BFGS history and compress it in Sauce.

    Requires an explicit GaussianPrior and Misfit.l2(noise_std=...). By default
    only the final stage produces uncertainty; use stages="all" for every stage.
    Rank defaults to the possible full correction rank (two per secant pair).
    """

    rank: Optional[int] = None
    stages: str = "final"
    curvature_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.rank is not None:
            _integer(self.rank, "rank")
        if (
            not np.isfinite(self.curvature_tolerance)
            or not np.finfo(float).eps <= self.curvature_tolerance < 1
        ):
            raise ValueError(
                "curvature_tolerance must be at least float64 epsilon and less than one"
            )
        if self.stages not in {"final", "all"}:
            raise ValueError("uncertainty stages must be 'final' or 'all'")


class UncertaintyResult:
    """A posterior approximation with labelled marginals and covariance actions.

    ``covariance`` applies the factors in memory; grid projections and mesh
    sampling run in Sauce through ``native``. Factor datasets are checked
    against the digests Sauce recorded when it wrote them, as they are read.
    """

    def __init__(
        self,
        factors: CurvatureResult,
        point: ControlVector,
        *,
        native: Any = None,
        units: Optional[Mapping[str, Optional[str]]] = None,
        meshes: Optional[dict] = None,
        provenance: Optional[dict] = None,
    ):
        self._covariance: Optional[_Covariance] = None
        self.factors = factors
        self.point = point
        self.space = point.space
        self.native = native
        self.units = dict(units or {})
        self._provenance = json.dumps(provenance or {}, sort_keys=True, allow_nan=False)
        self.meshes = dict(
            meshes
            or {
                k: mesh_descriptor(self.space, self.space.block(k))
                for k in self.space.blocks
                if self.space.block(k).kind == "mesh"
            }
        )
        with h5py.File(factors.path, "r") as h5:
            shape = h5["variance"].shape  # Metadata only; no marginals are read.
        if shape != (self.space.size,):
            raise ValueError("Uncertainty artifact does not match its control space")

    def _native(self) -> Any:
        if self.native is None:
            raise ValueError(
                "Grid projection needs a problem; pass problem to FWIResult.load"
            )
        return self.native

    @property
    def provenance(self) -> dict:
        """Stage identities and whether transferred curvature was refreshed."""
        return json.loads(self._provenance)

    @property
    def covariance(self) -> Any:
        """Covariance operator on the active coefficients (``@``, ``matmat``)."""
        if self._covariance is None:
            self._covariance = _Covariance(self.factors, self.space)
        return self._covariance

    def _field(self, key: str, grid: Any, dataset: str) -> xr.DataArray:
        block = self.space.block(key)
        mask = self.space._mask_of(block)
        section = self.space.slices[block.name]
        if grid is None:
            values = np.full(block.size, np.nan)
            values[mask] = self.factors.read_verified(dataset)[dataset][section]
            dims = block.dims or ("coefficient",)
            coords = block.coords or {"coefficient": np.arange(block.size)}
            field = xr.DataArray(
                values.reshape(block.shape, order="F"), dims=dims, coords=coords
            )
            for dim in block.dims:
                if block.units:
                    field.coords[dim].attrs["units"] = block.units
        else:
            if block.kind == "mesh":
                field, sampling, valid = mesh_grid_sampling(
                    self._native(), self.meshes[block.name], grid, block
                )
            else:
                field, sampling, valid = self.point._grid_sampling(grid, key)
            valid &= (
                np.asarray(abs(sampling[:, ~mask]).sum(axis=1)).reshape(field.shape)
                == 0
            )
            active = sampling[:, mask]
            projection = hstack(
                (
                    csr_matrix((active.shape[0], section.start)),
                    active,
                    csr_matrix((active.shape[0], self.space.size - section.stop)),
                ),
                format="csr",
            )
            output = self._native().covariance(self.factors, projection=projection)
            field.data = np.where(
                valid, output.read(dataset).reshape(field.shape), np.nan
            )
        field.name = f"{block.address}_{dataset}"
        unit = self.units.get(block.name)
        field.attrs.update(
            block=block.name,
            representation="posterior_approximation",
            units=(f"({unit})**2" if dataset == "variance" and unit else unit)
            or "dimensionless",
            method="full_history_bfgs",
        )
        return field

    def std(self, block: str, *, grid: Any = None) -> xr.DataArray:
        """Return posterior standard deviations with coordinates, units and masks."""
        return self._field(block, grid, "standard_deviation")

    def variance(self, block: str, *, grid: Any = None) -> xr.DataArray:
        """Return diag(T C T.T) on a grid, including coefficient correlations."""
        return self._field(block, grid, "variance")

    def save(self, path: Any) -> Path:
        """Publish the result; factor and mesh files are hard-linked, not copied.

        Coefficient arrays (the point and support masks) are HDF5 datasets in
        ``uncertainty.h5``; ``uncertainty.json`` holds only the layout.
        """
        directory = Path(path).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        _publish_artifact(self.factors.path, directory / "covariance.h5")
        meshes = {}
        for key, descriptor in self.meshes.items():
            target_mesh = directory / f"{key}.mesh.h5"
            _publish_artifact(Path(descriptor["path"]), target_mesh)
            meshes[key] = {
                **descriptor,
                "path": target_mesh.name,
                "roots": descriptor["roots"].tolist(),
            }
        support = self.space.support_masks()
        with h5py.File(directory / "uncertainty.h5", "w") as h5:
            h5.create_dataset("point", data=self.point.values)
            group = h5.create_group("support")
            for position, mask in enumerate(support.values()):
                group.create_dataset(str(position), data=mask)
        (directory / "uncertainty.json").write_text(
            json.dumps(
                dict(
                    schema="fs-uncertainty-2",
                    metadata=self.factors.metadata,
                    blocks=list(self.space.blocks),
                    sizes=self.space.sizes,
                    arrays="uncertainty.h5",
                    support=list(support),
                    units=self.units,
                    provenance=self.provenance,
                    meshes=meshes,
                    basis={
                        k: self.space.block(k).basis_identity for k in self.space.blocks
                    },
                    layouts=[
                        dict(
                            name=b.name,
                            size=b.size,
                            kind=b.kind,
                            dims=b.dims,
                            coords=(
                                None
                                if b.coords is None
                                else {
                                    k: np.asarray(v).tolist()
                                    for k, v in b.coords.items()
                                }
                            ),
                            units=b.units,
                            basis_identity=b.basis_identity,
                            control_type=type(b.control).__name__,
                            control=None if b.control is None else b.control.to_fs(),
                        )
                        for b in self.space.resolved_blocks
                    ],
                ),
                indent=2,
            )
            + "\n"
        )
        return directory

    @classmethod
    def load(
        cls, path: Any, space: ControlSpace, *, native: Any = None
    ) -> UncertaintyResult:
        directory = Path(path)
        record = json.loads((directory / "uncertainty.json").read_text())
        schema = record.get("schema")
        if schema == "fs-uncertainty-2":
            # Version 1 inlined the point and support masks as JSON lists.
            with h5py.File(directory / record["arrays"], "r") as h5:
                record["point"] = h5["point"][()]
                group = h5["support"]
                record["support"] = {
                    name: group[str(position)][()]
                    for position, name in enumerate(record["support"])
                }
        elif schema != "fs-uncertainty-1":
            raise ValueError("Unsupported uncertainty result schema")
        from frequensolve.model.parameterization import (
            BSplineControl,
            HatControl,
            MeshControl,
            TensorHatControl,
        )

        constructors = {
            cls.__name__: cls
            for cls in (HatControl, BSplineControl, TensorHatControl, MeshControl)
        }
        selected = space.restrict(record["blocks"])
        blocks = []
        for layout in record.get("layouts", []):
            template = selected.block(layout["name"])
            control = layout["control"]
            if control is not None:
                cls_control = constructors.get(layout["control_type"])
                if cls_control is None:
                    raise ValueError("Unsupported saved statistical control type")
                control = cls_control.from_fs(control)
            blocks.append(
                replace(
                    template,
                    size=layout["size"],
                    kind=layout["kind"],
                    dims=tuple(layout["dims"]),
                    coords=layout["coords"],
                    units=layout["units"],
                    basis_identity=layout["basis_identity"],
                    control=control,
                    baseline=None,
                    lower=-np.inf,
                    upper=np.inf,
                )
            )
        if blocks:
            selected = selected._clone(blocks, {})
        space = selected.with_support(record["support"])
        if space.sizes != record["sizes"]:
            raise ValueError("Saved uncertainty control layout changed")
        if record.get("basis", {}) != {
            k: space.block(k).basis_identity for k in space.blocks
        }:
            raise ValueError("Saved uncertainty material basis changed")
        meshes = {
            k: {
                **v,
                "path": str(directory / v["path"]),
                "roots": np.asarray(v["roots"], dtype=np.int32),
            }
            for k, v in record.get("meshes", {}).items()
        }
        # A saved stage owns its original mesh. Rebind the cloned space to the
        # copied artifact so subsequent projections and prior declarations use
        # this stage's basis even when the supplied problem has been refined.
        if meshes:
            definitions = dict(getattr(space, "_property_spaces", {}))
            for name, descriptor in meshes.items():
                block = space.block(name)
                if block.kind != "mesh":
                    raise ValueError("Saved mesh does not match a mesh control")
                with h5py.File(descriptor["path"]) as h5:
                    identity = h5["property_space/identity"][()].decode().rstrip(" \0")
                if identity != descriptor["identity"]:
                    raise ValueError("Saved uncertainty mesh identity changed")
                definition = definitions.get(block.control.space)
                if definition is not None:
                    definitions[block.control.space] = replace(
                        definition, artifact=descriptor["path"]
                    )
            setattr(space, "_property_spaces", definitions)
        artifact = directory / "covariance.h5"
        with h5py.File(artifact) as h5:
            if json.loads(h5["metadata"][()]) != record["metadata"]:
                raise ValueError("Saved uncertainty artifact identity changed")
        # Every factor dataset is checked against the recorded output digests when read.
        factors = CurvatureResult(artifact, record["metadata"])
        return cls(
            factors,
            ControlVector(record["point"], space),
            native=native,
            units=record["units"],
            meshes=meshes,
            provenance=record.get("provenance", {}),
        )
