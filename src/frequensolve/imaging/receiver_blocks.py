"""Local cross-parameter curvature retained from shared receiver probes."""

from dataclasses import dataclass
from typing import Mapping

import h5py
import numpy as np

from .controls import ControlSpace, ControlVector


def block_probe_request(request, space):
    """Resolve public control aliases before sending the native probe request."""
    if not isinstance(request, Mapping):
        raise TypeError("receiver_blocks must be a mapping")
    unknown = set(request) - {"controls", "probes", "source_samples", "seed"}
    if unknown:
        raise ValueError(f"unsupported receiver_blocks options: {sorted(unknown)}")
    names = request.get("controls")
    if not isinstance(names, (list, tuple)) or len(names) < 2:
        raise ValueError("receiver_blocks requires at least two controls")
    blocks = [space.block(name) for name in names]
    if any(b.kind != "mesh" for b in blocks):
        raise ValueError("receiver_blocks requires meshed material controls")
    names = [b.name.removeprefix("model.") for b in blocks]
    if len(set(names)) != len(names):
        raise ValueError("receiver_blocks controls must be distinct")
    identities = {b.basis_identity for b in blocks if b.basis_identity}
    if len(identities) > 1:
        raise ValueError(
            "receiver_blocks controls must share a material basis and transform"
        )
    return {
        **{k: v for k, v in request.items() if k != "controls"},
        "blocks": {"controls": names},
    }


@dataclass(frozen=True)
class ReceiverBlocks:
    """Positive local Gauss–Newton matrices in native parameter coordinates.

    ``matrices[node, row, column]`` has objective units divided by the two
    parameter units. ``controls`` fixes the row/column order; ``control_spaces``
    records the frozen basis identities. Frequency weights are already applied.
    These blocks omit coupling between distinct material coefficients.
    """

    controls: tuple[str, ...]
    matrices: np.ndarray
    control_spaces: dict[str, str]
    space: ControlSpace

    @classmethod
    def from_linearization(cls, linearization):
        """Read and validate every frequency part, then add weighted matrices."""
        request = (linearization.job.receiver_diagonal or {}).get("blocks")
        if request is None:
            raise ValueError("linearize did not request receiver_blocks")
        names = tuple(request["controls"])
        blocks = [linearization.space.block(name) for name in names]
        identities = {name: block.basis_identity for name, block in zip(names, blocks)}
        if len({b.size for b in blocks}) != 1 or len(set(identities.values())) != 1:
            raise ValueError("receiver blocks require identical material bases")
        shape = (blocks[0].size, len(names), len(names))
        total = np.zeros(shape)
        for task, weight in enumerate(linearization.frequency_weights, 1):
            if weight < 0:
                raise ValueError(
                    "receiver blocks require nonnegative frequency weights"
                )
            path = linearization.job.receiver_blocks_file(task)
            with h5py.File(path) as h:
                if _text(h["schema"]) != "fs-receiver-blocks-1":
                    raise ValueError(f"Unsupported receiver-block schema in {path}")
                if set(h["blocks"]) != set(names) or set(h["control_spaces"]) != set(
                    names
                ):
                    raise ValueError(f"Receiver-block control names differ in {path}")
                values = np.empty(shape)
                for i, name in enumerate(names):
                    if _text(h["control_spaces"][name]) != identities[name]:
                        raise ValueError(f"Receiver-block basis mismatch for {name}")
                    if set(h["blocks"][name]) != set(names):
                        raise ValueError(f"Incomplete receiver-block row {name}")
                    for j, other in enumerate(names):
                        column = np.asarray(h["blocks"][name][other])
                        if column.shape != (shape[0],):
                            raise ValueError("Receiver-block coefficient count differs")
                        values[:, i, j] = column
            if not np.isfinite(values).all() or not np.array_equal(
                values, values.transpose(0, 2, 1)
            ):
                raise ValueError("Receiver-block matrices must be finite and symmetric")
            if np.any(np.diagonal(values, axis1=1, axis2=2) < 0):
                raise ValueError("Receiver-block diagonal must be nonnegative")
            factors = np.array(
                [linearization.task_factors[task - 1].get(b.name, 1.0) for b in blocks]
            )
            total += (
                float(weight) * values * factors[None, :, None] * factors[None, None, :]
            )
        total.setflags(write=False)
        return cls(names, total, identities, linearization.space)

    @property
    def correlations(self):
        """Return dimensionless coupling; unilluminated entries are zero."""
        d = np.sqrt(np.maximum(np.diagonal(self.matrices, axis1=1, axis2=2), 0))
        denominator = d[:, :, None] * d[:, None, :]
        return np.divide(
            self.matrices,
            denominator,
            out=np.zeros_like(self.matrices),
            where=denominator > 0,
        )

    def solve(self, covector, *, damping=0.01, scales=None):
        """Apply a damped local inverse and return an update in native units.

        ``scales`` maps control names to positive physical perturbation scales
        (one per parameter); omitting it uses one native unit. For ``dm=S dz``,
        solve ``(S H S + lambda I) dz = S g`` and return ``S dz``.
        ``lambda`` is ``damping`` times the largest diagonal of ``S H S`` over
        the model. This is numerical preconditioning, not a chosen step length.
        Selected controls must cover the active space with matching support
        masks. Other layouts can use :attr:`matrices` directly.
        """
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("damping must be finite and positive")
        vector = (
            covector
            if isinstance(covector, ControlVector)
            else ControlVector(covector, self.space)
        )
        if not vector.space.equivalent(self.space):
            raise ValueError("covector belongs to a different control space")
        names = [self.space.block(n).name for n in self.controls]
        if set(names) != set(self.space.blocks):
            raise ValueError(
                "block solve requires the selected controls to cover the active space"
            )
        scale = np.ones(len(names))
        if scales is not None:
            if not isinstance(scales, Mapping):
                raise TypeError("scales must map control names to positive values")
            resolved = {self.space.block(n).name: v for n, v in scales.items()}
            if set(resolved) != set(names) or len(resolved) != len(scales):
                raise ValueError(
                    "scales must specify each selected control exactly once"
                )
            scale = np.asarray([resolved[n] for n in names], dtype=float)
            if (
                scale.shape != (len(names),)
                or not np.isfinite(scale).all()
                or np.any(scale <= 0)
            ):
                raise ValueError("scales must be finite positive scalars")
        lookup = np.full(self.space.full_size, -1, dtype=int)
        lookup[self.space.active_indices] = np.arange(self.space.size)
        indices = np.column_stack([lookup[self.space.full_slices[n]] for n in names])
        active = indices >= 0
        if not np.all(active == active[:, :1]):
            raise ValueError(
                "block solve requires matching support masks; use matrices for other layouts"
            )
        selected = active[:, 0]
        matrix = self.matrices[selected] * scale[None, :, None] * scale[None, None, :]
        if not len(matrix):
            return self.space.zeros()
        ridge = damping * np.max(np.diagonal(matrix, axis1=1, axis2=2))
        if ridge == 0:
            return self.space.zeros()
        matrix = matrix + ridge * np.eye(len(names))[None, :, :]
        rhs = vector.values[indices[selected]] * scale
        # Explicit final RHS axis works with NumPy 1.x and 2.x broadcasting.
        update = np.linalg.solve(matrix, rhs[..., None])[..., 0] * scale
        result = np.zeros(self.space.size)
        result[indices[selected]] = update
        return ControlVector(result, self.space)


def _text(dataset):
    value = dataset[()]
    return (value.decode() if isinstance(value, bytes) else str(value)).strip()
