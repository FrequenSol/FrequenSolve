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
import bisect
import json
import logging
from typing import Any, Dict, List, Tuple, Union

import h5py

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------------------------------------------------
# 1) Base Classes and Concrete Material Implementations


class MaterialBase(abc.ABC):
    """
    Abstract base class for materials. Requires a method to get properties
    for a given state, plus methods to read/write from HDF5 for persistence.
    """

    @abc.abstractmethod
    def get_property_for_state(self, prop_name: str, state: Dict[str, float]) -> float:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def to_hdf(self, hdf_group: h5py.Group) -> None:
        """
        Save all necessary data into hdf_group. Should also save enough metadata
        to reconstruct the object type (class) on load.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_hdf(cls, hdf_group: h5py.Group) -> "MaterialBase":
        """
        Load material data from hdf_group and return the appropriate subclass instance.
        """


class ConstantMaterial(MaterialBase):
    """
    A simple material with constant property values.
    """

    def __init__(self, mat_name: str, properties: Dict[str, float]):
        """
        Attributes:
            mat_name (str): name of the material
            properties (dict): { property_name -> value }
        """
        self._name = mat_name
        self._properties = properties

    @property
    def name(self) -> str:
        return self._name

    def get_property_for_state(self, prop_name: str, state: Dict[str, float]) -> float:
        if prop_name not in self._properties:
            raise ValueError(f"Property '{prop_name}' not defined in '{self._name}'")
        return self._properties[prop_name]

    def to_hdf(self, hdf_group: h5py.Group) -> None:
        """
        Store:
            material class name (so we know how to load it)
            property dictionary as a JSON (or separate datasets).
        """
        hdf_group.attrs["class_type"] = "ConstantMaterial"
        # Save the name
        hdf_group.attrs["material_name"] = self._name
        # Save properties as JSON (simple approach)
        props_json = json.dumps(self._properties)
        hdf_group.create_dataset("properties_json", data=props_json)

    @classmethod
    def from_hdf(cls, hdf_group: h5py.Group) -> "ConstantMaterial":
        """
        Recreate from the stored JSON data.
        """
        mat_name = hdf_group.attrs["material_name"]
        props_json = hdf_group["properties_json"][()].decode("utf-8")
        properties = json.loads(props_json)
        return cls(mat_name, properties)
