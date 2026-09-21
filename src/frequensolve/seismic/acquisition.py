"""Seismic source geometry, source encoding, and receiver acquisition."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from frequensolve.seismic.boundary_loadings import SurfacePressureLoading
from frequensolve.seismic.receivers import (
    CoordsSurfaceCarpet,
    ReceiverDevice,
    ReceiverGroup,
)
from frequensolve.seismic.sources import (
    EncodedSource,
    PointSource,
    SourceEncoding,
    SourceGeometry,
)
from frequensolve.seismic.sparse_survey import ReceiverSampling, SparseSurvey
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
)
from frequensolve.util.named_list import NamedList

__all__ = ["Acquisition"]

_SOURCE_KINDS = {"scalar", "vector", "tensor", "monopole", "dipole"}


def _coerce_source_geometry(value: Any) -> Optional[SourceGeometry]:
    if value is None:
        return None
    if isinstance(value, SourceGeometry):
        return value
    if isinstance(value, Mapping):
        return SourceGeometry.from_fs(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to SourceGeometry")


def _coerce_source_encoding(value: Any) -> Optional[SourceEncoding]:
    if value is None:
        return None
    if isinstance(value, SourceEncoding):
        return value
    if isinstance(value, Mapping):
        return SourceEncoding.from_fs(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to SourceEncoding")


def _encoded_terms(terms: Mapping[Any, Any]) -> Dict[str, Any]:
    encoded: Dict[str, Any] = {}
    for source, coefficient in terms.items():
        if isinstance(source, PointSource):
            if not source.name:
                raise ValueError("PointSource terms require named source points")
            source_name = source.name
        else:
            source_name = str(source)
        encoded[source_name] = coefficient
    return encoded


def _coerce_boundary_loadings(value: Optional[Any]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (SurfacePressureLoading, Mapping)):
        value = [value]
    result = []
    for loading in value:
        if isinstance(loading, SurfacePressureLoading):
            result.append(loading)
        elif isinstance(loading, Mapping):
            result.append(copy.deepcopy(dict(loading)))
        else:
            raise TypeError(
                "Boundary loadings must be SurfacePressureLoading objects or "
                "materialized solver mappings"
            )
    return result


def _loading_source_names(loading: Any) -> List[str]:
    if isinstance(loading, SurfacePressureLoading):
        return loading.source_names()
    fields = loading.get("fields", [])
    return [str(field["source"]) for field in fields if "source" in field]


def _loading_boundary_name(loading: Any) -> Optional[str]:
    if isinstance(loading, SurfacePressureLoading):
        return loading.boundary_condition
    value = loading.get("boundary_condition")
    return None if value is None else str(value)


@dataclass(init=False)
class Acquisition(ExtraFieldsMixin):
    """Physical sources, optional RHS encoding, receivers, and surveys."""

    source_geometry: Optional[SourceGeometry] = None
    source_encoding: Optional[SourceEncoding] = None
    boundary_loadings: List[Any] = field(default_factory=list)
    receiver_groups: NamedList = field(default_factory=NamedList)
    surveys: NamedList = field(default_factory=NamedList)
    max_batch: Optional[int] = None
    write_vtk: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        source_geometry: Optional[Any] = None,
        sources: Optional[Any] = None,
        source_encoding: Optional[Any] = None,
        source_groups: Optional[Any] = None,
        boundary_loadings: Optional[Any] = None,
        receivers: Optional[Any] = None,
        receiver_groups: Optional[Any] = None,
        surveys: Optional[Any] = None,
        max_batch: Optional[int] = None,
        write_vtk: Optional[bool] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if source_geometry is not None and sources is not None:
            raise TypeError("Use either source_geometry or sources, not both")
        if source_groups is not None and (
            source_geometry is not None or sources is not None
        ):
            raise TypeError(
                "Legacy source_groups cannot be combined with source_geometry"
            )
        if receivers is not None and receiver_groups is not None:
            raise TypeError("Use either receivers or receiver_groups, not both")

        extra_fields = copy.deepcopy(dict(extra or {}))
        if "source_groups" in extra_fields:
            raise ValueError(
                "Acquisition.extra cannot contain legacy source_groups; pass "
                "source_groups as the migration argument instead"
            )

        self.source_geometry = _coerce_source_geometry(
            source_geometry if source_geometry is not None else sources
        )
        self.source_encoding = _coerce_source_encoding(source_encoding)
        self.boundary_loadings = _coerce_boundary_loadings(boundary_loadings)
        self.receiver_groups = NamedList(
            receiver_groups if receiver_groups is not None else receivers or []
        )
        self.surveys = NamedList(surveys or [])
        self.max_batch = None if max_batch is None else int(max_batch)
        self.write_vtk = None if write_vtk is None else bool(write_vtk)
        self._init_extra(extra_fields, **kwargs)
        self._coerce_receivers_and_surveys()
        if source_groups is not None:
            from frequensolve.seismic import _legacy_sources

            self.source_geometry, self.source_encoding = (
                _legacy_sources.migrate_source_groups(source_groups)
            )

    @property
    def sources(self) -> Optional[SourceGeometry]:
        """Friendly alias for :attr:`source_geometry`."""

        return self.source_geometry

    @sources.setter
    def sources(self, value: Any) -> None:
        self.source_geometry = _coerce_source_geometry(value)

    @property
    def receivers(self) -> NamedList:
        """Friendly alias for receiver groups."""

        return self.receiver_groups

    @receivers.setter
    def receivers(self, value: Any) -> None:
        self.receiver_groups = NamedList(value or [])
        self._coerce_receivers_and_surveys()

    def _coerce_receivers_and_surveys(self) -> None:
        self.receiver_groups = NamedList(
            [
                (
                    group
                    if isinstance(group, ReceiverGroup)
                    else ReceiverGroup.from_fs(group)
                )
                for group in self.receiver_groups
            ]
        )
        self.surveys = NamedList(
            [
                (
                    survey
                    if isinstance(survey, SparseSurvey)
                    else SparseSurvey.from_fs(survey)
                )
                for survey in self.surveys
            ]
        )
        if self.max_batch is not None and self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")

    def __bool__(self) -> bool:
        """Return whether this acquisition contains authored state."""

        return bool(
            self.source_geometry is not None
            or self.source_encoding is not None
            or self.boundary_loadings
            or self.receiver_groups
            or self.surveys
            or self.max_batch is not None
            or self.write_vtk is not None
            or self.extra
        )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Acquisition":
        """Deserialize current or legacy acquisition JSON."""

        payload = copy.deepcopy(dict(data))
        schema = payload.pop("schema", None)
        if schema not in {None, "fs-acquisition-1", "fs-acquisition-2"}:
            raise ValueError(f"Unsupported acquisition schema {schema!r}")
        source_geometry = payload.pop("source_geometry", payload.pop("sources", None))
        return cls(
            source_geometry=source_geometry,
            source_encoding=payload.pop("source_encoding", None),
            boundary_loadings=payload.pop("boundary_loadings", None),
            source_groups=payload.pop("source_groups", None),
            receiver_groups=NamedList(
                [
                    ReceiverGroup.from_fs(group)
                    for group in payload.pop("receiver_groups", [])
                ]
            ),
            surveys=NamedList(
                [SparseSurvey.from_fs(survey) for survey in payload.pop("surveys", [])]
            ),
            max_batch=payload.pop("max_batch", None),
            write_vtk=payload.pop("write_vtk", None),
            extra=payload,
        )

    def to_fs(
        self,
        ctx=None,
        *,
        boundary_loadings: Optional[Sequence[Any]] = None,
    ) -> Dict:
        """Serialize acquisition geometry for solver input.

        Args:
            ctx: Optional export context used by source, receiver, and survey
                serializers.

        Returns:
            JSON-compatible acquisition block.
        """

        all_boundary_loadings = [
            *self.boundary_loadings,
            *_coerce_boundary_loadings(boundary_loadings),
        ]
        self._validate_export_contract(all_boundary_loadings)

        ctx = ctx or ExportContext()

        survey_component_maps = self._survey_component_maps()

        payload: Dict[str, Any] = {
            "schema": "fs-acquisition-2",
            "receiver_groups": [group.to_fs(ctx) for group in self.receiver_groups],
        }
        if self.max_batch is not None:
            payload["max_batch"] = self.max_batch
        if self.write_vtk is not None:
            payload["write_vtk"] = self.write_vtk
        if self.source_geometry is not None:
            payload["source_geometry"] = self.source_geometry.to_fs(ctx)
        if self.source_encoding is not None:
            payload["source_encoding"] = self.source_encoding.to_fs(ctx)
        if all_boundary_loadings:
            payload["boundary_loadings"] = [
                (
                    loading.to_fs(ctx, loading_index=index)
                    if isinstance(loading, SurfacePressureLoading)
                    else copy.deepcopy(loading)
                )
                for index, loading in enumerate(all_boundary_loadings)
            ]
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

    def _validate_export_contract(
        self, boundary_loadings: Optional[Sequence[Any]] = None
    ) -> None:
        """Reject incomplete or inconsistent acquisition-v2 exports."""

        if "source_groups" in self.extra:
            raise ValueError(
                "Acquisition.extra cannot contain legacy source_groups in a "
                "current acquisition export"
            )
        receiver_names = [group.name for group in self.receiver_groups]
        if len(receiver_names) != len(set(receiver_names)):
            raise ValueError("Receiver group names must be unique")
        survey_names = [survey.name for survey in self.surveys]
        if len(survey_names) != len(set(survey_names)):
            raise ValueError("Survey names must be unique")
        known_surveys = set(survey_names)
        for group in self.receiver_groups:
            survey = group.survey
            if survey is not None and survey not in known_surveys:
                raise ValueError(
                    f"Receiver group {group.name!r} references unknown survey "
                    f"{survey!r}"
                )
        loadings = list(
            self.boundary_loadings if boundary_loadings is None else boundary_loadings
        )
        for loading in loadings:
            if not _loading_boundary_name(loading):
                raise ValueError(
                    "Every surface-pressure loading must name its boundary condition"
                )
            names = _loading_source_names(loading)
            if not names or any(not name.strip() for name in names):
                raise ValueError(
                    "Every surface-pressure loading requires at least one named source"
                )

        geometry = self.source_geometry
        if geometry is None:
            if self.source_encoding is not None:
                raise ValueError("source_encoding requires source_geometry")
            if not loadings:
                raise ValueError(
                    "fs-acquisition-2 requires source_geometry or boundary_loadings"
                )
            return
        if geometry.kind not in _SOURCE_KINDS:
            choices = ", ".join(sorted(_SOURCE_KINDS))
            raise ValueError(
                f"Unsupported source kind {geometry.kind!r}; use one of: {choices}"
            )
        if geometry.geometry_type == "Inline":
            if not geometry.point_count:
                raise ValueError("Inline source_geometry requires source points")
            if not geometry.is_bulk:
                for index, source in enumerate(geometry.sources):
                    if source.kind is not None and (
                        str(source.kind).strip().lower() != geometry.kind
                    ):
                        raise ValueError(
                            "Inline source point kind must match source_geometry.kind: "
                            f"sources[{index}] is {source.kind!r}, geometry is "
                            f"{geometry.kind!r}"
                        )
            geometry.validate_unique_names()

        encoding = self.source_encoding
        if encoding is None or encoding.encoding_type == "HDF5Dense":
            return
        field_names = encoding.field_names()
        if len(field_names) != len(set(field_names)):
            raise ValueError("Source-encoding field names must be unique")
        if encoding.encoding_type == "Named" and geometry.geometry_type == "Inline":
            if geometry.geometry_type == "Inline" and not geometry.has_explicit_names():
                raise ValueError(
                    "Named source encoding requires explicit names for every "
                    "inline physical source point"
                )
            known_names = set(geometry.point_names())
            for field_obj in encoding.fields:
                unknown = sorted(set(field_obj.terms).difference(known_names))
                if unknown:
                    raise ValueError(
                        f"Source encoding references unknown sources: {unknown}"
                    )
        elif encoding.encoding_type in {"JsonDense", "FrequencyDense"}:
            point_count = geometry.point_count
            encoding.validate_dense_shape(point_count)

    def set_sources(self, sources: Any) -> SourceGeometry:
        """Set physical source geometry and return it."""

        geometry = _coerce_source_geometry(sources)
        if geometry is None:
            raise TypeError("sources cannot be None")
        self.source_geometry = geometry
        return geometry

    def set_source_encoding(self, encoding: Optional[Any]) -> Optional[SourceEncoding]:
        """Set or clear explicit source encoding."""

        self.source_encoding = _coerce_source_encoding(encoding)
        return self.source_encoding

    def encode_sources(
        self,
        weights: Optional[Any] = None,
        *,
        coefficients: Optional[Any] = None,
        names: Optional[Sequence[str]] = None,
        reference_coordinates: Optional[Any] = None,
        name: Optional[str] = None,
        conjugate: bool = False,
        frequencies: Optional[Sequence[float]] = None,
    ) -> SourceEncoding:
        """Set a dense complex source encoding on the current geometry.

        Static ``weights`` use encoding-major shape ``(n_encoded, n_source)``.
        Passing ``frequencies`` selects a frequency-dependent tensor with shape
        ``(n_frequency, n_encoded, n_source)``. Set ``conjugate=True`` when the
        values are forward responses that should be time-reversed.
        """

        if weights is not None and coefficients is not None:
            raise TypeError("Use either weights or coefficients, not both")
        if frequencies is None:
            encoding = SourceEncoding.dense(
                weights,
                coefficients=coefficients,
                names=names,
                reference_coordinates=reference_coordinates,
                name=name,
                conjugate=conjugate,
            )
            source_axis = 1
        else:
            if reference_coordinates is not None:
                raise ValueError(
                    "Frequency-dependent source encoding computes reference "
                    "coordinates from physical geometry"
                )
            encoding = SourceEncoding.frequency_dense(
                weights,
                frequencies,
                coefficients=coefficients,
                names=names,
                name=name,
                conjugate=conjugate,
            )
            source_axis = 2
        point_count = self.known_source_point_count()
        assert encoding.weights is not None
        if (
            point_count is not None
            and encoding.weights.shape[source_axis] != point_count
        ):
            raise ValueError(
                "source encoding coefficient count must match the "
                f"{point_count} physical source points"
            )
        self.source_encoding = encoding
        return encoding

    def add_sources(
        self,
        *,
        kind: str,
        coords: Any,
        names: Optional[Sequence[str]] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        domain: Optional[int] = None,
        direction: Optional[Any] = None,
        amplitude: Optional[Any] = None,
        mechanism: Optional[Any] = None,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        """Append physical point sources and return their stable names."""

        existing = self.source_point_count()
        probe = SourceGeometry.points(
            kind=kind,
            coords=coords,
            names=names,
            units=units,
            system=system,
            domain=domain,
            direction=direction,
            amplitude=amplitude,
            mechanism=mechanism,
            defaults=defaults,
        )
        if names is None:
            generated = [
                f"source_{index:06d}"
                for index in range(
                    existing + 1,
                    existing + int(probe.point_count or 0) + 1,
                )
            ]
            probe.set_point_names(generated)
        return self._append_inline_sources(probe)

    add_source_points = add_sources

    def add_source_group(
        self,
        kind: str,
        coords: Any,
        direction: Optional[Any] = None,
        domain: Optional[int] = None,
    ) -> List[str]:
        """Add legacy identity source fields through the compatibility shim."""

        from frequensolve.seismic import _legacy_sources

        return _legacy_sources.add_source_group(
            self,
            kind=kind,
            coords=coords,
            direction=direction,
            domain=domain,
        )

    def add_compound_source(
        self,
        kind: str,
        coords: np.ndarray,
        weights: np.ndarray,
        direction: Optional[np.ndarray] = None,
        domain: Optional[int] = None,
    ) -> EncodedSource:
        """Add a legacy weighted source through the compatibility shim."""

        from frequensolve.seismic import _legacy_sources

        return _legacy_sources.add_compound_source(
            self,
            kind=kind,
            coords=coords,
            weights=weights,
            direction=direction,
            domain=domain,
        )

    def add_encoded_source(
        self,
        name: str,
        terms: Mapping[Any, Any],
    ) -> EncodedSource:
        """Append one sparse named encoded-source field."""

        encoded = _encoded_terms(terms)
        known = set(self.source_point_names())
        unknown = sorted(set(encoded).difference(known))
        if unknown:
            raise ValueError(f"Unknown physical source names: {unknown}")
        field_obj = EncodedSource.named(name, encoded)
        if self.source_encoding is None:
            self.source_encoding = SourceEncoding.named([field_obj])
        elif self.source_encoding.encoding_type != "Named":
            raise ValueError("add_encoded_source can only extend Named source encoding")
        else:
            if name in set(self.source_encoding.field_names()):
                raise ValueError(f"Encoded source {name!r} already exists")
            self.source_encoding.fields.append(field_obj)
        return field_obj

    def add_distributed_source(
        self,
        name: str,
        terms: Mapping[Any, Any],
    ) -> EncodedSource:
        """Deprecated alias for :meth:`add_encoded_source`."""

        warnings.warn(
            "add_distributed_source() is deprecated; use add_encoded_source().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_encoded_source(name, terms)

    def _append_inline_sources(self, geometry: SourceGeometry) -> List[str]:
        if self.source_geometry is None:
            self.source_geometry = geometry
            return geometry.point_names()
        current = self.source_geometry
        if current.geometry_type != "Inline" or geometry.geometry_type != "Inline":
            raise ValueError("Cannot append inline sources to file-backed geometry")
        if current.kind != geometry.kind:
            raise ValueError("All source points in one geometry must share a kind")
        if current.domain != geometry.domain:
            raise ValueError("All source points in one geometry must share a domain")
        if current.defaults != geometry.defaults:
            raise ValueError(
                "All appended source points must share source-geometry defaults"
            )
        known = set(current.point_names())
        new_names = geometry.point_names()
        duplicates = sorted(known.intersection(new_names))
        if duplicates:
            raise ValueError(f"Duplicate source names: {', '.join(duplicates)}")
        current.extend_inline(geometry)
        return new_names

    @property
    def source_groups(self) -> NamedList:
        """Return the deprecated logical-source view from the legacy shim."""

        from frequensolve.seismic import _legacy_sources

        return _legacy_sources.source_groups(self)

    @source_groups.setter
    def source_groups(self, _value: Any) -> None:
        from frequensolve.seismic import _legacy_sources

        _legacy_sources.reject_source_groups_assignment()

    def known_source_point_count(self) -> Optional[int]:
        """Return the physical point count, or ``None`` for external geometry."""

        if self.source_geometry is None:
            return 0
        return self.source_geometry.point_count

    def source_point_count(self) -> int:
        """Return the number of inline physical source points."""

        return int(self.known_source_point_count() or 0)

    def known_source_field_count(self) -> Optional[int]:
        """Return the logical RHS/source-field count when locally known."""

        if self.source_encoding is not None:
            names = self.source_encoding.field_names()
            count = self.source_encoding.field_count
        else:
            names = self.source_point_names()
            count = self.known_source_point_count()
        loading_names = {
            name
            for loading in self.boundary_loadings
            for name in _loading_source_names(loading)
        }
        if loading_names and not names and count != 0:
            # Loading fields may share identities with external sources. Without
            # those identities we cannot determine the size of their union.
            return None
        if names or loading_names:
            return len(set(names) | loading_names)
        return count

    def source_field_count(self) -> int:
        """Return the number of locally addressable RHS/source fields."""

        return int(self.known_source_field_count() or 0)

    def source_field_ids(self) -> List[int]:
        """Return one-based source-field identifiers."""

        return list(range(1, self.source_field_count() + 1))

    def source_field_names(self) -> List[str]:
        """Return source-field names when locally known."""

        if self.source_encoding is not None:
            names = self.source_encoding.field_names()
        else:
            names = self.source_point_names()
        for loading in self.boundary_loadings:
            for name in _loading_source_names(loading):
                if name not in names:
                    names.append(name)
        return names

    def source_point_names(self) -> List[str]:
        """Return physical source-point names when locally known."""

        if self.source_geometry is None:
            return []
        return self.source_geometry.point_names()

    def source_point_coords(self) -> np.ndarray:
        """Return inline physical source-point coordinates."""

        if self.source_geometry is None:
            return np.empty((0, 0), dtype=float)
        return self.source_geometry.coordinates()

    def add_receiver_group(
        self,
        name: str,
        device: ReceiverDevice,
        coords: np.ndarray,
        domain: Optional[int] = None,
        **kwargs,
    ):
        """Add a receiver group with common device and coordinates.

        Args:
            name: Receiver group name.
            device: Device defining receiver type and components.
            coords: Receiver coordinate array or coordinate object.
            domain: Optional domain where the receiver group is evaluated.
            **kwargs: Additional solver-facing receiver group fields.

        Returns:
            Newly added ``ReceiverGroup``.

        Raises:
            TypeError: If deprecated frame arguments are supplied.
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

    def add_receiver_carpet(
        self,
        name: str,
        device: ReceiverDevice,
        *,
        surface: Any,
        x: Any,
        y: Optional[Any] = None,
        units: Optional[Any] = None,
        above: Optional[Any] = None,
        below: Optional[Any] = None,
        domain: Optional[int] = None,
        **kwargs: Any,
    ) -> ReceiverGroup:
        """Add a receiver group on a tensor-product carpet."""

        coords = _carpet_coordinates(
            x=x,
            y=y,
            surface=surface,
            units=units,
            above=above,
            below=below,
        )
        return self.add_receiver_group(
            name=name,
            device=device,
            coords=coords,
            domain=domain,
            **kwargs,
        )

    def add_survey(self, survey: SparseSurvey) -> SparseSurvey:
        """Add or replace a named sparse survey layout.

        Args:
            survey: Sparse survey instance or serialized survey mapping.

        Returns:
            Stored ``SparseSurvey`` instance.
        """

        if isinstance(survey, dict):
            survey = SparseSurvey.from_fs(survey)
        try:
            self.surveys[survey.name] = survey
        except ValueError:
            self.surveys.append(survey)
        return survey

    def add_sparse_survey(self, name: str, traces=None, **kwargs) -> SparseSurvey:
        """Create and add a named inline sparse survey.

        Args:
            name: Survey name.
            traces: Optional initial trace samples.
            **kwargs: Additional ``SparseSurvey`` constructor arguments.

        Returns:
            Newly added ``SparseSurvey`` instance.
        """

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

        Args:
            name: Receiver group name.
            device: Receiver device for the sparse samples.
            coords: Receiver coordinate array or coordinate object.
            survey: Sparse survey name, object, or serialized mapping.
            domain: Optional receiver domain.
            **kwargs: Additional solver-facing receiver group fields.

        Returns:
            Newly added ``ReceiverGroup``.
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
        """List receiver output field selectors.

        Args:
            recv_name: Optional receiver group name. When omitted, all receiver
                groups are included.

        Returns:
            Field selectors of the form ``"<group>:<component>"``.
        """
        field_list = []

        if recv_name:
            group = self.receiver_group(recv_name)
            for field in group.device.output_components():
                file = f"{group.name}:{field.name}"
                field_list.append(file)
        else:
            for group in self.receiver_groups:
                for field in group.device.output_components():
                    file = f"{group.name}:{field.name}"
                    field_list.append(file)
        return field_list

    def list_sources(self) -> List[int]:
        """Return valid one-based source-field numbers."""

        return self.source_field_ids()

    def source_field(self, isrc: int) -> Union[PointSource, EncodedSource]:
        """Return locally available source-field metadata by one-based index.

        Args:
            isrc: One-based source index.

        Returns:
            An inline point source for identity encoding, or an encoded source
            for explicit inline encoding.
        """
        count = self.known_source_field_count()
        if isrc < 1 or (count is not None and isrc > count):
            raise IndexError(f"Source index {isrc} is out of range.") from None
        geometry = self.source_geometry
        encoding = self.source_encoding
        if encoding is None:
            if geometry is None or geometry.geometry_type != "Inline":
                raise ValueError("Source-field metadata is stored externally")
            return geometry.sources[isrc - 1]
        if encoding.encoding_type == "HDF5Dense":
            raise ValueError("Source-field metadata is stored externally")
        return encoding.fields[isrc - 1]

    def source(self, isrc: int) -> Any:
        """Return a legacy source group through the compatibility shim."""

        from frequensolve.seismic import _legacy_sources

        return _legacy_sources.source_group(self, isrc)

    def receiver_group(self, name: str) -> ReceiverGroup:
        """Return a receiver group by name."""

        return self.receiver_groups[name]

    def receiver_coords(self, group: Optional[str] = None):
        """Return receiver coordinates.

        Args:
            group: Optional receiver group name. When omitted, all groups are
                returned as a mapping.
        """
        if group is None:
            group_locations = {}
            for group in self.receiver_groups:
                group_locations[group.name] = group.coordinates.get()
            return group_locations
        else:
            return self.receiver_groups[group].coordinates.get()

    def source_coords(
        self,
        src: Optional[int] = None,
        *,
        preserve_metadata: bool = False,
    ):
        """Return source-field reference coordinates.

        Args:
            src: Optional one-based source index.
            preserve_metadata: Return authored ``CoordinateValue``/quantity
                metadata instead of only numeric values. When all coordinates
                are requested, this returns a list of coordinate values.
        """
        if self.source_geometry is None:
            coords = [] if preserve_metadata else np.empty((0, 0), dtype=float)
        elif self.source_encoding is None:
            coords = (
                self.source_geometry.coordinate_values()
                if preserve_metadata
                else self.source_geometry.coordinates()
            )
        else:
            coords = (
                self.source_encoding.reference_coordinate_values(self.source_geometry)
                if preserve_metadata
                else self.source_encoding.reference_coordinates(self.source_geometry)
            )
        if src is None:
            return coords
        index = int(src)
        if index < 1 or index > len(coords):
            raise IndexError(f"Source index {src} is out of range.")
        return coords[index - 1]

    def offsets(self, src: int, group: str) -> Dict:
        """Return horizontal source-field/receiver offsets.

        Args:
            src: One-based source index.
            group: Receiver group name.
        """
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
            for index, component in enumerate(
                group.device.output_components(), start=1
            ):
                component_map.setdefault(str(index), index)
                component_map.setdefault(component.name, index)
                component_map.setdefault(component.name.lower(), index)
                component_map.setdefault(component.field, index)
                component_map.setdefault(component.field.lower(), index)
        return maps


def _carpet_coordinates(
    *,
    surface: Any,
    x: Any,
    y: Optional[Any],
    units: Optional[Any],
    above: Optional[Any],
    below: Optional[Any],
) -> Any:
    points_grid = getattr(surface, "points_grid", None)
    if not callable(points_grid):
        raise TypeError(
            "surface must provide points_grid(...), such as sim.model_surface(...)"
        )
    compact = CoordsSurfaceCarpet.try_from_surface(
        surface,
        x=x,
        y=y,
        units=units,
        above=above,
        below=below,
    )
    if compact is not None and compact.size > 200:
        return compact
    return points_grid(
        x,
        y,
        units=units,
        above=above,
        below=below,
    )
