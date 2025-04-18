"""Python structures defining seismic acquisition geometry"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from frequensolve.seismic.receivers import ReceiverDevice, ReceiverGroup
from frequensolve.seismic.sources import PointSource, SourceGroup
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
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, dict: Dict) -> "Acquisition":
        return cls(
            source_groups=NamedList(
                [SourceGroup.from_dict(group) for group in dict["source_groups"]]
            ),
            receiver_groups=NamedList(
                [ReceiverGroup.from_dict(group) for group in dict["receiver_groups"]]
            ),
        )

    def __dict__(self) -> Dict:
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
                    f"Duplicate reciever group names detected. Renaming receiver group {name} to {group.name}"
                )
            names[group.name] = group.name

        return {
            "source_groups": [group.__dict__() for group in self.source_groups],
            "receiver_groups": [group.__dict__() for group in self.receiver_groups],
        }

    def add_source_group(
        self,
        kind: str,
        coords: np.ndarray,
        direction: Optional[np.ndarray] = None,
        frame: str = "physical",
    ):
        """Add a group of recievers with common kind, frame, and direction.

        Args:
           kind (str): Kind of the receiver group (e.g., "station", "geophone", "fiber").
           coords (np.ndarray): Coordinates of the receiver group.
           direction (np.ndarray): Direction of the receiver group.
           frame (str): Frame of the receiver group (e.g., "physical", "global").
        """

        for row in coords:
            isrc = len(self.source_groups)
            self.source_groups.append(
                SourceGroup(
                    source=PointSource(
                        kind=kind,
                        frame=frame,
                        coordinates=row,
                        direction=direction,
                        name=f"source_{isrc}",
                    )
                )
            )

    def add_receiver_group(
        self,
        name: str,
        device: ReceiverDevice,
        coords: np.ndarray,
        frame: str = "physical",
        domain: Optional[int] = None,
        **kwargs,
    ):
        """Add a group of recievers with common kind, frame, and direction.

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

    def receiver_locations(self) -> Dict:
        """Get receiver locations in physical and reference frames.

        Returns:
           Dict: Dictionary containing physical and reference locations.
        """
        group_locations = {}
        for group in self.receiver_groups:
            group_locations[group.name] = group.coordinates.get()
        return group_locations

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
