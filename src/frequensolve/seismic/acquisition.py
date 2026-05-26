"""Python structures defining seismic acquisition geometry"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from frequensolve.geometry.frame import CoordinateValue
from frequensolve.seismic.receivers import (
    ReceiverDevice,
    ReceiverGroup,
    coordinate_array_metadata,
)
from frequensolve.seismic.sources import CompoundSource, PointSource, SourceGroup
from frequensolve.seismic.sparse_survey import ReceiverSampling, SparseSurvey
from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra
from frequensolve.util.named_list import NamedList

__all__ = ["Acquisition"]


@dataclass
class Acquisition(ExtraFieldsMixin):
    """Defines a seismic source and receiver configuration.

    This class reads the input file to retrieve blocks describing sources, receivers, and
    wavelet signatures. It then aggregates them into a single cohesive acquisition definition.

    Attributes:
       source_groups   (NamedList[SourceGroup]): A list of SourceGroup objects describing all shot points.
       receiver_groups (NamedList[ReceiverGroup]): A list of ReceiverGroup objects (stations, geophones, or fibers).
    """

    source_groups: NamedList = field(default_factory=NamedList)
    receiver_groups: NamedList = field(default_factory=NamedList)
    surveys: NamedList = field(default_factory=NamedList)
    extra: Dict = field(default_factory=dict)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    @classmethod
    def from_fs(cls, data: Dict) -> "Acquisition":
        data = copy.deepcopy(data)
        return cls(
            source_groups=NamedList(
                [SourceGroup.from_fs(group) for group in data.pop("source_groups", [])]
            ),
            receiver_groups=NamedList(
                [
                    ReceiverGroup.from_fs(group)
                    for group in data.pop("receiver_groups", [])
                ]
            ),
            surveys=NamedList(
                [SparseSurvey.from_fs(survey) for survey in data.pop("surveys", [])]
            ),
            extra=data,
        )

    def to_fs(self, ctx=None) -> Dict:
        from ..util.printing import print_warn

        # Ensure receiver groups have unique names
        names = {}
        for group in self.receiver_groups:
            name = group.name
            if name in names:
                i = 1
                while f"{name}_{i}" in names:
                    i += 1
                group.name = f"{name}_{i}"
                print_warn(
                    f"Duplicate receiver group names detected. Renaming receiver group {name} to {group.name}"
                )
            names[group.name] = group.name

        survey_component_maps = self._survey_component_maps()

        payload = {
            "source_groups": [group.to_fs(ctx) for group in self.source_groups],
            "receiver_groups": [group.to_fs(ctx) for group in self.receiver_groups],
        }
        if self.surveys:
            payload["surveys"] = [
                (
                    survey.to_fs(
                        ctx, component_map=survey_component_maps.get(survey.name)
                    )
                )
                for survey in self.surveys
            ]
        return merge_extra(payload, self.extra, "Acquisition")

    def add_source_group(
        self,
        kind: str,
        coords: np.ndarray,
        direction: Optional[np.ndarray] = None,
        domain: Optional[int] = None,
    ):
        """Add a group of sources with common kind and direction.

        Args:
           kind (str):              Kind of the source group.
           coords (np.ndarray):     Coordinates of the source group.
           direction (np.ndarray):  Direction of the source group.
           domain (int):            Optional domain in which the source group should be evaluated.
        """

        for row in _source_coordinate_rows(coords):
            isrc = len(self.source_groups)
            source = PointSource(
                kind=kind,
                coordinates=row,
                direction=direction,
                domain=domain,
                name=f"source_{isrc}",
            )
            self.source_groups.append(SourceGroup(source=source))

    def add_compound_source(
        self,
        kind: str,
        coords: np.ndarray,
        weights: np.ndarray,
        direction: Optional[np.ndarray] = None,
        domain: Optional[int] = None,
    ):
        coords = np.asarray(coords, dtype=np.float64)
        weights = np.asarray(weights, dtype=float)
        if direction is not None:
            direction = np.asarray(direction, dtype=float)
        isrc = len(self.source_groups)
        if direction is None:
            direction = np.ones((len(coords), 1))
        elif direction.ndim == 1:
            direction = np.tile(direction, (len(coords), 1))
        elif direction.ndim != 2:
            raise ValueError("direction must be a 1D vector or one row per coordinate")
        if len(direction) != len(coords):
            raise ValueError("direction must have one row per coordinate")
        direction = direction.copy()
        for i, row in enumerate(direction):
            direction[i, :] *= weights[i]
        source = CompoundSource(
            kind=kind,
            coordinates=coords,
            direction=direction,
            domain=domain,
            name=f"source_{isrc}",
        )
        self.source_groups.append(SourceGroup(source=source))

    def add_receiver_group(
        self,
        name: str,
        device: ReceiverDevice,
        coords: np.ndarray,
        domain: Optional[int] = None,
        **kwargs,
    ):
        """Add a group of receivers with common device and coordinates.

        Args:
           name (str):                Name of the receiver group.
           device (ReceiverDevice):   Device defining receiver type and components.
           coordinates (np.ndarray):  Coordinates of the receiver group.
           domain (int):              Optional domain in which the receiver group should be evaluated.
        """
        deprecated_frame_keys = {"frame", "source_frame", "receiver_frame"} & set(
            kwargs
        )
        if deprecated_frame_keys:
            raise TypeError(
                "add_receiver_group frame is no longer supported; receiver coordinates are physical"
            )

        group = ReceiverGroup(
            name=name,
            device=device,
            coordinates=coords,
            domain=domain,
            **kwargs,
        )
        self.receiver_groups.append(group)
        return group

    def add_survey(self, survey: SparseSurvey) -> SparseSurvey:
        """Add or replace a named sparse survey layout."""

        if isinstance(survey, dict):
            survey = SparseSurvey.from_fs(survey)
        try:
            self.surveys[survey.name] = survey
        except ValueError:
            self.surveys.append(survey)
        return survey

    def add_sparse_survey(self, name: str, traces=None, **kwargs) -> SparseSurvey:
        """Create and add a named inline sparse survey."""

        return self.add_survey(SparseSurvey(name=name, traces=traces, **kwargs))

    def add_sparse_receiver_group(
        self,
        name: str,
        device: ReceiverDevice,
        coords: np.ndarray,
        survey: Optional[Union[str, SparseSurvey, Dict]] = None,
        domain: Optional[int] = None,
        **kwargs,
    ) -> ReceiverGroup:
        """Add a receiver group that samples traces from a named sparse survey.

        ``survey`` can be a survey name, a ``SparseSurvey`` object, or a survey
        dictionary loaded from JSON. Survey objects are added to
        ``Acquisition.surveys`` automatically.
        """
        deprecated_frame_keys = {"frame", "source_frame", "receiver_frame"} & set(
            kwargs
        )
        if deprecated_frame_keys:
            raise TypeError(
                "add_sparse_receiver_group frame is no longer supported; receiver coordinates are physical"
            )

        if survey is None:
            raise ValueError(
                "add_sparse_receiver_group requires a survey name or SparseSurvey"
            )
        if isinstance(survey, dict):
            survey = SparseSurvey.from_fs(survey)
        if isinstance(survey, SparseSurvey):
            self.add_survey(survey)
            sampling = survey.sampling()
        else:
            sampling = ReceiverSampling.sparse(str(survey))

        group = ReceiverGroup(
            name=name,
            device=device,
            coordinates=coords,
            domain=domain,
            sampling=sampling,
            **kwargs,
        )
        self.receiver_groups.append(group)
        return group

    def list_fields(self, recv_name: str = "") -> List[str]:
        """List available fields for a specified receiver group or for all groups."""
        field_list = []

        if recv_name:
            group = self.receiver_group(recv_name)
            for field in group.components:
                file = f"{group.name}:{field.name}"
                field_list.append(file)
        else:
            for group in self.receiver_groups:
                for field in group.components:
                    file = f"{group.name}:{field.name}"
                    field_list.append(file)
        return field_list

    def list_sources(self) -> List[int]:
        """List valid source numbers."""
        return list(range(1, len(self.source_groups) + 1))

    def source(self, isrc: int) -> SourceGroup:
        """Retrieve a source by index."""
        try:
            return self.source_groups[isrc - 1]
        except IndexError:
            raise IndexError(f"Source index {isrc} is out of range.")

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path
        for group in self.receiver_groups:
            group._set_path(proj_path, rel_path)
        for group in self.source_groups:
            group._set_path(proj_path, rel_path)
        for survey in self.surveys:
            if hasattr(survey, "_set_path"):
                survey._set_path(proj_path, rel_path)

    def receiver_coords(self, group: Optional[str] = None):
        """Get receiver coordinates."""
        if group is None:
            group_locations = {}
            for group in self.receiver_groups:
                group_locations[group.name] = group.coordinates.get()
            return group_locations
        else:
            return self.receiver_groups[group].coordinates.get()

    def source_coords(self, src: Optional[int] = None):
        """Get source locations for all sources."""
        if src is None:
            return np.array([src.coordinates()[0] for src in self.source_groups])
        else:
            isrc = int(src - 1)
            return self.source_groups[isrc].coordinates()[0]

    def offsets(self, src: int, group: str) -> Dict:
        """Get receiver offsets."""
        diff = self.receiver_coords(group) - self.source_coords(src)
        offsets = np.hypot(diff[:, 0], diff[:, 1])
        return offsets

    def _survey_component_maps(self) -> Dict[str, Dict[str, int]]:
        maps: Dict[str, Dict[str, int]] = {}
        for group in self.receiver_groups:
            survey_name = getattr(group, "survey", None)
            if not survey_name:
                continue
            component_map = maps.setdefault(survey_name, {})
            for index, component in enumerate(group.device.components, start=1):
                component_map.setdefault(str(index), index)
                component_map.setdefault(component.name, index)
                component_map.setdefault(component.name.lower(), index)
                component_map.setdefault(component.field, index)
                component_map.setdefault(component.field.lower(), index)
        return maps

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


def _source_coordinate_rows(coords):
    extra = {}
    if isinstance(coords, CoordinateValue):
        extra = copy.deepcopy(coords.extra)

    values, units, system = coordinate_array_metadata(coords)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("source coordinates must be a 2D array")

    if units is not None or system is not None:
        return [
            CoordinateValue(
                row.tolist(),
                units=units,
                system=system,
                extra=copy.deepcopy(extra),
            )
            for row in values
        ]

    return values
