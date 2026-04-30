"""Python structures defining seismic acquisition geometry"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from frequensolve.seismic.receivers import ReceiverDevice, ReceiverGroup
from frequensolve.seismic.sources import CompoundSource, PointSource, SourceGroup
from frequensolve.seismic.sparse_survey import ReceiverSampling, SparseSurvey
from frequensolve.util.mixins import merge_extra
from frequensolve.util.named_list import NamedList

__all__ = ["Acquisition"]


@dataclass
class Acquisition:
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
    max_batch: Optional[int] = None
    extra: Dict = field(default_factory=dict)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, dict: Dict) -> "Acquisition":
        dict = copy.deepcopy(dict)
        return cls(
            source_groups=NamedList(
                [
                    SourceGroup.from_dict(group)
                    for group in dict.pop("source_groups", [])
                ]
            ),
            receiver_groups=NamedList(
                [
                    ReceiverGroup.from_dict(group)
                    for group in dict.pop("receiver_groups", [])
                ]
            ),
            surveys=NamedList(
                [SparseSurvey.from_dict(survey) for survey in dict.pop("surveys", [])]
            ),
            max_batch=dict.pop("max_batch", None),
            extra=dict,
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
            **({"max_batch": self.max_batch} if self.max_batch is not None else {}),
            "source_groups": [group.to_fs(ctx) for group in self.source_groups],
            "receiver_groups": [group.to_fs(ctx) for group in self.receiver_groups],
        }
        if self.surveys:
            payload["surveys"] = [
                (
                    survey.to_fs(
                        ctx, component_map=survey_component_maps.get(survey.name)
                    )
                    if hasattr(survey, "to_fs")
                    else survey
                )
                for survey in self.surveys
            ]
        return merge_extra(payload, self.extra, "Acquisition")

    def __dict__(self) -> Dict:
        return self.to_fs()

    @property
    def kwargs(self) -> Dict:
        return self.extra

    @kwargs.setter
    def kwargs(self, value: Dict) -> None:
        self.extra = copy.deepcopy(dict(value))

    def add_source_group(
        self,
        kind: str,
        coords: np.ndarray,
        direction: Optional[np.ndarray] = None,
        domain: Optional[int] = None,
        frame: str = "physical",
    ):
        """Add a group of sources with common kind, frame, and direction.

        Args:
           kind (str):              Kind of the source group.
           coords (np.ndarray):     Coordinates of the source group.
           direction (np.ndarray):  Direction of the source group.
           frame (str):             Frame of the source group (e.g., "physical", "global").
           domain (int):            Domain in which the source group should be evaluated
                                    (if a source is defined between multiple domains, responses
                                     will be evaluated in all and averaged by default, setting this
                                     specifies a specific domain to evaluate, neglecting others).
        """

        for row in coords:
            isrc = len(self.source_groups)
            source = PointSource(
                kind=kind,
                frame=frame,
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
        if isinstance(coords, list):
            coords = np.array(coords)
        if isinstance(weights, list):
            weights = np.array(weights)
        if isinstance(direction, list):
            direction = np.array(direction)
        isrc = len(self.source_groups)
        if direction is None:
            direction = np.ones((len(coords), 1))
        for i, row in enumerate(direction):
            direction[i, :] *= weights[i]
        source = CompoundSource(
            kind=kind,
            frame="physical",
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
        frame: str = "physical",
        domain: Optional[int] = None,
        **kwargs,
    ):
        """Add a group of receivers with common device, frame, and coordinates.

        Args:
           name (str):                Name of the receiver group.
           device (ReceiverDevice):   Device defining receiver type and components.
           coordinates (np.ndarray):  Coordinates of the receiver group.
           frame (str):               Frame of the receiver group (e.g., "physical", "global").
           domain (int):              Domain in which the receiver group should be evaluated
                                      (if a receiver is defined between multiple domains, responses
                                       will be evaluated in all and averaged by default, setting this
                                       specifies a specific domain to evaluate, neglecting others).
        """

        self.receiver_groups.append(
            ReceiverGroup(
                name=name,
                device=device,
                frame=frame,
                coordinates=coords,
                domain=domain,
                **kwargs,
            )
        )

    def add_survey(self, survey: SparseSurvey) -> SparseSurvey:
        """Add or replace a named sparse survey layout."""

        if isinstance(survey, dict):
            survey = SparseSurvey.from_dict(survey)
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
        frame: str = "physical",
        domain: Optional[int] = None,
        **kwargs,
    ) -> ReceiverGroup:
        """Add a receiver group that samples traces from a named sparse survey.

        ``survey`` can be a survey name, a ``SparseSurvey`` object, or a survey
        dictionary loaded from JSON. Survey objects are added to
        ``Acquisition.surveys`` automatically.
        """

        if survey is None:
            raise ValueError(
                "add_sparse_receiver_group requires a survey name or SparseSurvey"
            )
        if isinstance(survey, dict):
            survey = SparseSurvey.from_dict(survey)
        if isinstance(survey, SparseSurvey):
            self.add_survey(survey)
            sampling = survey.sampling()
        else:
            sampling = ReceiverSampling.sparse(str(survey))

        group = ReceiverGroup(
            name=name,
            device=device,
            frame=frame,
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
