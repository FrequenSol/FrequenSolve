"""(De)serialization factory design pattern.

Current support for JSON and YAML.
"""

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from inspect import Signature

import yaml


class Serializer(ABC):
    """Serializer base class.

    Here we define the interface for any serializer implementation.

    Args:
        stream_format: Format of the stream for serialization.
    """

    def __init__(self, format: str) -> None:
        """Initialize serializer callables for a stream format.

        Args:
            format: ``"JSON"`` or ``"YAML"``.
        """

        self.format = format
        self.serialize_func = get_serializer(format)
        self.deserialize_func = get_deserializer(format)

    @abstractmethod
    def serialize(self, payload: dict) -> str:
        """Serialize a payload to the configured stream format.

        Args:
            payload: Mapping to encode.

        Returns:
            Encoded string.
        """

    @abstractmethod
    def deserialize(self, stream: str) -> dict:
        """Deserialize a stream from the configured format.

        Args:
            stream: Encoded JSON/YAML string.

        Returns:
            Decoded mapping.
        """

    @staticmethod
    def validate(data: dict, signature: Signature) -> dict:
        """Validate payload keys against a callable signature.

        Args:
            data: Decoded payload mapping.
            signature: Signature whose parameter names are required.

        Returns:
            Mapping limited to expected keys.

        Raises:
            KeyError: If required keys are missing.
        """
        observed = set(data)
        expected = set(signature.parameters)

        if not expected.issubset(observed):
            raise KeyError(f"Key mismatch: {observed}, expected {expected}")

        if len(observed) != len(expected):
            warnings.warn(f"Ignoring extra key(s): {observed - expected}")
            data = {key: data[key] for key in expected}

        return data


def get_serializer(stream_format: str) -> Callable:
    """Return a serializer function for a stream format.

    Args:
        stream_format: ``"JSON"`` or ``"YAML"``.

    Returns:
        Callable that converts a mapping to a string.

    Raises:
        ValueError: If the format is unsupported.
    """
    stream_format = stream_format.upper()
    if stream_format == "JSON":
        return _serialize_to_json
    elif stream_format == "YAML":
        return _serialize_to_yaml
    else:
        raise ValueError(stream_format)


def get_deserializer(stream_format: str) -> Callable:
    """Return a deserializer function for a stream format.

    Args:
        stream_format: ``"JSON"`` or ``"YAML"``.

    Returns:
        Callable that converts a string to a mapping.

    Raises:
        ValueError: If the format is unsupported.
    """
    stream_format = stream_format.upper()
    if stream_format == "JSON":
        return _deserialize_json
    elif stream_format == "YAML":
        return _deserialize_yaml
    else:
        raise ValueError(stream_format)


def _serialize_to_json(payload: dict) -> str:
    """Convert dictionary to JSON string."""
    return json.dumps(payload)


def _serialize_to_yaml(payload: dict) -> str:
    """Convert dictionary to YAML string."""
    return yaml.dump(payload, sort_keys=False)


def _deserialize_json(stream: str) -> dict:
    """Convert JSON string to dictionary."""
    return json.loads(stream)


def _deserialize_yaml(stream: str) -> dict:
    """Convert YAML string to dictionary."""
    return yaml.safe_load(stream)
