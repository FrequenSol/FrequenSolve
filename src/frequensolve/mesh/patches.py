# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Whole-root patch authoring and native geometry preparation."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from frequensolve.mesh.boundary_conditions import BoundaryCondition
from frequensolve.units import is_quantity, ureg

__all__ = ["Patch", "PatchSet", "PreparedPatchSet"]


def _ids(values, label):
    values = tuple(values)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
        for value in values
    ):
        raise ValueError(f"{label} must contain positive integer IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must contain unique IDs")
    return tuple(sorted(int(value) for value in values))


def _metres(value, label, shape=()):
    if not is_quantity(value):
        raise ValueError(
            f"{label} requires explicit length units, such as 2 * fs.ureg.km"
        )
    try:
        result = np.asarray(value.to("m").magnitude, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must have length units") from exc
    if (shape is not None and result.shape != shape) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


@dataclass(frozen=True, kw_only=True)
class Patch:
    """Explicit one-based parent root and physical source IDs."""

    name: str
    roots: tuple[int, ...]
    sources: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A patch requires a nonempty name")
        object.__setattr__(self, "roots", _ids(self.roots, "roots"))
        object.__setattr__(self, "sources", _ids(self.sources, "sources"))


def _bisect_sources(coordinates, size):
    """Split along the longest coordinate extent, breaking ties by source ID."""
    groups = []

    def split(ids):
        if len(ids) <= size:
            groups.append(tuple(int(i + 1) for i in sorted(ids)))
            return
        axis = int(np.argmax(np.ptp(coordinates[ids], axis=0)))
        order = ids[np.lexsort((ids, coordinates[ids, axis]))]
        middle = len(order) // 2
        split(order[:middle])
        split(order[middle:])

    split(np.arange(len(coordinates)))
    return tuple(groups)


def _global_coordinates(values, units, system, dimension, label):
    if system not in (None, "global"):
        raise NotImplementedError(
            f"{label} uses coordinate system {system!r}; materialize global coordinates before preparing patches"
        )
    result = np.asarray(ureg.Quantity(values, units).to("m").magnitude, dtype=float)
    if (
        result.ndim != 2
        or result.shape[1] != dimension
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(
            f"{label} must contain finite {dimension}-dimensional coordinates"
        )
    return result


def _acquisition_coordinates(simulation):
    from frequensolve.seismic.receivers import ReceiverNode, coordinate_array_metadata

    acquisition = simulation.acquisition
    if acquisition.source_encoding is not None or acquisition.boundary_loadings:
        raise NotImplementedError(
            "Root patches require physical point shots without source encoding or boundary loads"
        )
    if simulation.global_coordinate_system is not None:
        raise NotImplementedError(
            "Patch preparation currently requires global Cartesian acquisition coordinates"
        )
    geometry = acquisition.source_geometry
    if geometry is None or geometry.geometry_type != "Inline":
        raise NotImplementedError(
            "Patch preparation requires a locally available physical source catalog"
        )
    default = simulation.units.defaults.get("length", "km")
    sources = []
    for i, coordinates in enumerate(geometry.coordinate_values(), start=1):
        values, units, system = coordinate_array_metadata(coordinates)
        sources.append(
            _global_coordinates(
                values.reshape(1, -1),
                units or default,
                system,
                simulation.dimension,
                f"Source {i}",
            )[0]
        )
    if not sources:
        raise ValueError("Patch preparation requires at least one physical shot")
    receivers = {}
    for group in acquisition.receiver_groups:
        if type(group.device) is not ReceiverNode:
            raise NotImplementedError(
                f"Receiver group {group.name!r} must use point receivers"
            )
        if group.sampling is not None:
            raise NotImplementedError(
                f"Receiver group {group.name!r} already has sparse sampling; intersect its survey before patch preparation"
            )
        coords = group.coordinates
        receivers[group.name] = _global_coordinates(
            coords.get(),
            getattr(coords, "units", None) or default,
            getattr(coords, "system", None),
            simulation.dimension,
            f"Receiver group {group.name!r}",
        )
    if not receivers:
        raise ValueError("Patch preparation requires receiver groups")
    return np.asarray(sources), receivers


class PatchSet:
    """Reusable source grouping, physical aperture, and propagation-buffer policy.

    A scalar ``max_offset`` selects a radial aperture; a vector selects a box
    with one maximum offset per coordinate direction. Preparation does not certify
    PML construction. Named parent material artifacts must already exist.
    """

    def __init__(self, patches, *, max_offset, padding, depth=None, pml=None):
        self.patches = tuple(patches)
        if not self.patches or not all(
            isinstance(patch, Patch) for patch in self.patches
        ):
            raise ValueError("Supply one or more Patch objects")
        if len({patch.name for patch in self.patches}) != len(self.patches):
            raise ValueError("Patch names must be unique")
        self.shots_per_patch = None
        self._set_policy(max_offset, padding, depth, pml)

    @classmethod
    def around_sources(
        cls, *, shots_per_patch, max_offset, padding, depth=None, pml=None
    ):
        if (
            isinstance(shots_per_patch, bool)
            or not isinstance(shots_per_patch, Integral)
            or shots_per_patch < 1
        ):
            raise ValueError("shots_per_patch must be a positive integer")
        result = cls.__new__(cls)
        result.patches = ()
        result.shots_per_patch = int(shots_per_patch)
        result._set_policy(max_offset, padding, depth, pml)
        return result

    def _set_policy(self, max_offset, padding, depth, pml):
        offset = _metres(max_offset, "max_offset", shape=None)
        if offset.shape not in ((), (2,), (3,)):
            raise ValueError("max_offset must be a scalar or a 2D/3D offset vector")
        self.max_offset = float(offset) if offset.ndim == 0 else tuple(offset)
        self.padding = float(_metres(padding, "padding"))
        if np.any(offset <= 0) or self.padding < 0:
            raise ValueError("max_offset must be positive and padding nonnegative")
        self.depth = None if depth is None else tuple(_metres(depth, "depth", (2,)))
        if self.depth is not None and self.depth[0] >= self.depth[1]:
            raise ValueError("depth must have increasing lower and upper bounds")
        if pml is not None and not isinstance(pml, BoundaryCondition):
            raise TypeError("pml must be a BoundaryCondition")
        self.pml = (
            deepcopy(pml)
            if pml is not None
            else BoundaryCondition(
                conditions=["pml"],
                boundaries=["patch_cut"],
                pml_wavelengths=1,
                pml_reflectivity=1e-4,
            )
        )
        if self.pml.conditions != ["pml"]:
            raise ValueError("The patch PML override must specify the pml condition")
        self.pml.boundaries = ["patch_cut"]

    def _requests(self, simulation, inventory):
        sources, receivers = _acquisition_coordinates(simulation)
        offset = np.asarray(self.max_offset)
        if offset.ndim and offset.shape != (simulation.dimension,):
            raise ValueError("max_offset vector must match the simulation dimension")
        if self.shots_per_patch is None:
            groups = tuple(patch.sources for patch in self.patches)
            names = tuple(patch.name for patch in self.patches)
            assigned = [source for group in groups for source in group]
            if sorted(assigned) != list(range(1, len(sources) + 1)):
                raise ValueError(
                    "Explicit patches must assign every physical shot exactly once"
                )
        else:
            groups = _bisect_sources(sources, self.shots_per_patch)
            names = tuple(f"patch_{i:04d}" for i in range(1, len(groups) + 1))
        parent_lower = np.min(
            [root["sampled_lower"] for root in inventory["roots"]], axis=0
        )
        parent_upper = np.max(
            [root["sampled_upper"] for root in inventory["roots"]], axis=0
        )
        requests, selections = [], []
        for index, (name, shots) in enumerate(zip(names, groups)):
            source_points = sources[np.array(shots) - 1]
            lower, upper = source_points.min(axis=0), source_points.max(axis=0)
            rows = {}
            for group, coordinates in receivers.items():
                by_shot = {}
                for source in shots:
                    delta = coordinates - sources[source - 1]
                    selected = np.flatnonzero(
                        np.all(np.abs(delta) <= offset, axis=1)
                        if offset.ndim
                        else np.linalg.norm(delta, axis=1) <= offset
                    )
                    by_shot[source] = tuple(int(receiver + 1) for receiver in selected)
                    if selected.size:
                        lower = np.minimum(lower, coordinates[selected].min(axis=0))
                        upper = np.maximum(upper, coordinates[selected].max(axis=0))
                rows[group] = by_shot
            if not any(ids for group in rows.values() for ids in group.values()):
                raise ValueError(f"Patch {name!r} has no receivers within max_offset")
            if self.depth is None:
                lower[-1], upper[-1] = parent_lower[-1], parent_upper[-1]
            else:
                if lower[-1] < self.depth[0] or upper[-1] > self.depth[1]:
                    raise ValueError(
                        f"Patch {name!r} depth excludes an assigned source or retained receiver"
                    )
                lower[-1], upper[-1] = self.depth
            points = [
                {
                    "kind": "source",
                    "id": shot,
                    "coordinates": sources[shot - 1].tolist(),
                }
                for shot in shots
            ]
            for group, by_shot in rows.items():
                for receiver in sorted({i for ids in by_shot.values() for i in ids}):
                    points.append(
                        {
                            "kind": "receiver",
                            "group": group,
                            "id": receiver,
                            "coordinates": receivers[group][receiver - 1].tolist(),
                        }
                    )
            request = {"name": name, "padding": self.padding, "points": points}
            if self.shots_per_patch is None:
                roots = self.patches[index].roots
                if roots[-1] > inventory["root_count"]:
                    raise ValueError(
                        f"Patch {name!r} refers to a root absent from the prepared parent"
                    )
                request["roots"] = list(roots)
            else:
                request.update(lower=lower.tolist(), upper=upper.tolist())
            requests.append(request)
            retained = sum(
                len(ids) for group in rows.values() for ids in group.values()
            )
            selections.append(
                {
                    "name": name,
                    "sources": shots,
                    "source_coordinates": sources[np.array(shots) - 1].tolist(),
                    "receivers": rows,
                    "receiver_coordinates": {
                        group: {
                            receiver: receivers[group][receiver - 1].tolist()
                            for receiver in sorted(
                                {
                                    receiver
                                    for ids in by_shot.values()
                                    for receiver in ids
                                }
                            )
                        }
                        for group, by_shot in rows.items()
                    },
                    "retained_pairs": retained,
                    "excluded_pairs": len(shots)
                    * sum(len(coords) for coords in receivers.values())
                    - retained,
                }
            )
        return {"units": "m", "patches": requests}, selections

    def to_dict(self):
        """Serialize the reusable selection policy with explicit metre distances."""
        return {
            "patches": [
                {"name": p.name, "roots": list(p.roots), "sources": list(p.sources)}
                for p in self.patches
            ],
            "shots_per_patch": self.shots_per_patch,
            "max_offset_m": np.asarray(self.max_offset).tolist(),
            "padding_m": self.padding,
            "depth_m": None if self.depth is None else list(self.depth),
            "pml": self.pml.to_fs(),
        }

    @classmethod
    def from_dict(cls, value):
        """Restore a saved patch selection policy through its ordinary validators."""
        if value["shots_per_patch"] is not None and value["patches"]:
            raise ValueError(
                "Saved policy cannot combine explicit patches with automatic grouping"
            )
        options = dict(
            max_offset=np.asarray(value["max_offset_m"]) * ureg.m,
            padding=value["padding_m"] * ureg.m,
            depth=(
                None
                if value["depth_m"] is None
                else np.asarray(value["depth_m"]) * ureg.m
            ),
            pml=BoundaryCondition.from_fs(value["pml"]),
        )
        if value["shots_per_patch"] is not None:
            return cls.around_sources(
                shots_per_patch=value["shots_per_patch"], **options
            )
        return cls([Patch(**item) for item in value["patches"]], **options)

    def prepare(self, simulation, frequencies, *, site):
        """Prepare native curved root geometry and per-shot aperture selections without wave solves."""
        from frequensolve.simulation.jobs._patches import PatchPreparationJob

        # Fail on unsupported acquisition before scheduling geometry work.
        _acquisition_coordinates(simulation)
        inventory_job = PatchPreparationJob(
            "patch_inventory", simulation, frequencies, {"units": "m"}
        )
        site.run(inventory_job, check=True)
        request, selections = self._requests(simulation, inventory_job.geometry_report)
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        job = PatchPreparationJob(
            f"patch_prepare_{key}", simulation, frequencies, request
        )
        site.run(job, check=True)
        geometry = job.geometry_report
        reports = {item["name"]: item for item in geometry["patches"]}
        for patch in request["patches"]:
            report = reports.get(patch["name"], {})
            roots = report.get("point_roots", [])
            if (
                report.get("acquisition_checked") is not True
                or len(roots) != len(patch["points"])
                or not set(roots) <= set(report.get("descriptor", {}).get("roots", []))
            ):
                raise ValueError(
                    f"Patch {patch['name']!r} lacks native acquisition containment; "
                    "use a solver supporting patch point validation"
                )
        return PreparedPatchSet(geometry, selections, (inventory_job, job), self.pml)


class PreparedPatchSet:
    """Native physical-domain preview and original acquisition identities."""

    def __init__(self, geometry, selections, jobs, pml):
        self._geometry = deepcopy(geometry)
        self._selections = deepcopy(selections)
        self.jobs = tuple(jobs)
        self.pml = deepcopy(pml)

    @property
    def geometry(self):
        return deepcopy(self._geometry)

    def freeze_stage(self, control_state, *, name, directory):
        """Pin stage inputs atomically for controlled patch execution and restart."""
        from frequensolve.mesh._stage_snapshot import PatchStageSnapshot

        if not self.jobs or self.jobs[-1].geometry_report != self._geometry:
            raise ValueError(
                "Patch stage requires a current committed preparation result"
            )
        return PatchStageSnapshot.publish(
            directory,
            self.jobs[-1].simulation,
            control_state,
            self,
            name=name,
            frequencies=self.jobs[-1].f_list,
        )

    @property
    def acquisition(self):
        return deepcopy(self._selections)

    def simulations(self, *, name="patch"):
        """Build child simulations with parent catalogs and per-shot sparse apertures."""
        from pathlib import Path

        from frequensolve.mesh._root_patch import RootPatchDescriptor
        from frequensolve.seismic.sparse_survey import SparseSurvey, SparseTrace

        if not self.jobs or self.jobs[-1].geometry_report != self._geometry:
            raise ValueError("Patch execution requires current committed preparation")
        job = self.jobs[-1]
        selections = {item["name"]: item for item in self._selections}
        children = []
        for index, patch in enumerate(self._geometry["patches"]):
            selection = selections[patch["name"]]
            child = job.simulation.copy(f"{name}_{index:04d}")
            child.mesh.mesh = None
            child.mesh.file = str(
                Path(job._result_path) / self._geometry["parent_file"]
            )
            child.mesh.format = "gmp"
            child.mesh.root_patch = RootPatchDescriptor.from_fs(patch["descriptor"])
            if patch["cut_boundary"]:
                child += deepcopy(self.pml)
            child.acquisition.extra["active_sources"] = list(selection["sources"])
            child.acquisition.receiver_groups = [
                group
                for group in child.acquisition.receiver_groups
                if any(selection["receivers"][group.name].values())
            ]
            for group in child.acquisition.receiver_groups:
                survey = SparseSurvey(
                    f"patch_{index:04d}_{group.name}",
                    traces=[
                        SparseTrace(
                            source_id=int(source),
                            receiver_id=int(receiver),
                            receiver_position_id=int(receiver),
                            component_id=component,
                        )
                        for source, receivers in selection["receivers"][
                            group.name
                        ].items()
                        for receiver in receivers
                        for component in range(
                            1, len(list(group.device.output_components())) + 1
                        )
                    ],
                )
                child.acquisition.surveys.append(survey)
                group.sampling = survey.sampling()
            children.append(child)
        return tuple(children)

    @property
    def job(self):
        if len(self.jobs) != 1:
            raise ValueError("Patch preparation used multiple jobs; inspect .jobs")
        return self.jobs[0]

    def plot(self, *, ax=None, patch=None):
        """Draw native sampled curved root edges in metres; PML is not part of this physical preview."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.ticker import MaxNLocator
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        dimension = self._geometry["dimension"]
        if ax is None:
            figure = plt.figure()
            ax = figure.add_subplot(projection="3d" if dimension == 3 else None)
        selected = self._geometry["patches"]
        if patch is not None:
            selected = [item for item in selected if item["name"] == patch]
            if not selected:
                raise KeyError(f"Unknown patch {patch!r}")
        roots = {item["root"]: item for item in self._geometry["roots"]}
        collection_type = LineCollection if dimension == 2 else Line3DCollection

        def edges(ids):
            return [
                edge.T
                for root in ids
                for edge in np.asarray(roots[root]["edge_points"])
                .reshape(dimension, -1, 9)
                .transpose(1, 0, 2)
            ]

        ax.add_collection(collection_type(edges(roots), colors="0.8", linewidths=0.5))
        for i, item in enumerate(selected):
            ax.add_collection(
                collection_type(
                    edges(item["descriptor"]["roots"]),
                    colors=f"C{i % 10}",
                    linewidths=0.6,
                    label=item["name"],
                )
            )
            ax.add_collection(
                collection_type(
                    edges(item["core_roots"]), colors=f"C{i % 10}", linewidths=1.3
                )
            )
            support_added = sorted(
                set(item.get("support_roots", ())) - set(item["core_roots"])
            )
            if support_added:
                ax.add_collection(
                    collection_type(
                        edges(support_added),
                        colors=f"C{i % 10}",
                        linewidths=1.3,
                        linestyles="dotted",
                    )
                )
            if item["added_roots"]:
                ax.add_collection(
                    collection_type(
                        edges(item["added_roots"]),
                        colors=f"C{i % 10}",
                        linewidths=1,
                        linestyles="dashed",
                    )
                )
            acquisition = next(
                (row for row in self._selections if row["name"] == item["name"]), None
            )
            if acquisition is not None:
                sources = np.asarray(acquisition["source_coordinates"])
                ax.scatter(*sources.T, marker="*", color=f"C{i % 10}", s=60, zorder=5)
                coordinates = [
                    xyz
                    for group in acquisition["receiver_coordinates"].values()
                    for xyz in group.values()
                ]
                if coordinates:
                    ax.scatter(
                        *np.asarray(coordinates).T,
                        marker="v",
                        color=f"C{i % 10}",
                        s=12,
                        zorder=4,
                    )
        lower = np.min([root["sampled_lower"] for root in roots.values()], axis=0)
        upper = np.max([root["sampled_upper"] for root in roots.values()], axis=0)
        ax.set_xlim(lower[0], upper[0])
        ax.set_xlabel("x (m)")
        if dimension == 2:
            ax.set_ylim(upper[1], lower[1])
            ax.set_ylabel("z (m)")
            ax.set_aspect("equal")
        else:
            ax.set_ylim(lower[1], upper[1])
            ax.set_zlim(upper[2], lower[2])
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            ax.set_box_aspect(upper - lower)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if dimension == 3:
            ax.zaxis.set_major_locator(MaxNLocator(5))
        ax.set_title("Physical root patches")
        ax.legend()
        return ax
