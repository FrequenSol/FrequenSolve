#!/usr/bin/env python3

"""
material_hdf_library.py

An example showing how to store and manage materials in an HDF5-based library.
Materials can be persisted to disk, and extended at runtime by adding new ones.

Requires: h5py
    pip install h5py

Usage:
    python material_hdf_library.py
"""

import abc
import json
import logging
from typing import Dict

import h5py

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------------
# 1) Base Classes and Concrete Material Implementations


class MaterialBase(abc.ABC):
    """Base protocol for materials persisted in an HDF5 material library."""

    @abc.abstractmethod
    def get_property_for_state(self, prop_name: str, state: Dict[str, float]) -> float:
        """Return a material property value for a named state point."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Material name used as the library key."""

    @abc.abstractmethod
    def to_hdf(self, hdf_group: h5py.Group) -> None:
        """Write enough data and metadata to reconstruct the material."""

    @classmethod
    @abc.abstractmethod
    def from_hdf(cls, hdf_group: h5py.Group) -> "MaterialBase":
        """Load a material from an HDF5 group."""


class ConstantMaterial(MaterialBase):
    """Material whose properties do not vary with state."""

    def __init__(self, mat_name: str, properties: Dict[str, float]):
        """Create a material from a fixed property dictionary.

        Args:
            mat_name: Material name.
            properties: Mapping from property name to constant value.
        """
        self._name = mat_name
        self._properties = properties

    @property
    def name(self) -> str:
        """Material name used as the library key."""
        return self._name

    def get_property_for_state(self, prop_name: str, state: Dict[str, float]) -> float:
        """Return a constant property value, ignoring the state argument."""
        if prop_name not in self._properties:
            raise ValueError(f"Property '{prop_name}' not defined in '{self._name}'")
        return self._properties[prop_name]

    def to_hdf(self, hdf_group: h5py.Group) -> None:
        """Store the material type, name, and property dictionary in HDF5."""
        hdf_group.attrs["class_type"] = "ConstantMaterial"
        # Save the name
        hdf_group.attrs["material_name"] = self._name
        # Save properties as JSON (simple approach)
        props_json = json.dumps(self._properties)
        hdf_group.create_dataset("properties_json", data=props_json)

    @classmethod
    def from_hdf(cls, hdf_group: h5py.Group) -> "ConstantMaterial":
        """Recreate a constant material from its HDF5 representation."""
        mat_name = hdf_group.attrs["material_name"]
        props_json = hdf_group["properties_json"][()].decode("utf-8")
        properties = json.loads(props_json)
        return cls(mat_name, properties)
