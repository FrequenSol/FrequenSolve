"""Utility APIs that do not require optional visualization runtimes."""

from frequensolve.util.data_file import (
    DataArrayFile,
    check_data_exists,
    hash_array_blake3,
    hash_dataarray_blake3,
    hash_file_blake3,
    save_data_if_new,
)
from frequensolve.util.fft import configure_fft, get_fft_backend
from frequensolve.util.fields import canonical_field, canonical_fields
from frequensolve.util.input_parser import InputBlock, InputParser, str_to_array
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    FSSerializableMixin,
    MaterializeMixin,
    PathContextMixin,
    TypeTaggedMixin,
    merge_extra,
)
from frequensolve.util.named_list import NamedList
from frequensolve.util.report_builder import Figure, Report, Section
from frequensolve.util.store import (
    HDF5Reference,
    SimulationStore,
    hash_dataarray_payload,
)

__all__ = [
    "DataArrayFile",
    "ExportContext",
    "ExtraFieldsMixin",
    "FSSerializableMixin",
    "Figure",
    "HDF5Reference",
    "InputBlock",
    "InputParser",
    "MaterializeMixin",
    "NamedList",
    "PathContextMixin",
    "Report",
    "Section",
    "SimulationStore",
    "TypeTaggedMixin",
    "check_data_exists",
    "canonical_field",
    "canonical_fields",
    "configure_fft",
    "get_fft_backend",
    "hash_array_blake3",
    "hash_dataarray_blake3",
    "hash_dataarray_payload",
    "hash_file_blake3",
    "merge_extra",
    "save_data_if_new",
    "str_to_array",
]
