"""Config-file driven site factory."""

from __future__ import annotations

import os
from dataclasses import fields
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping, Optional, Union

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import toml

    class _TomllibCompat:
        @staticmethod
        def loads(text: str) -> dict[str, Any]:
            return toml.loads(text)

    tomllib = _TomllibCompat()

SITE_CONFIG_ENV_VAR = "FREQUENSOLVE_SITE_CONFIG"
DEFAULT_SITE_CONFIG_NAME = "site.toml"

_SITE_TYPES = {
    "aws": "AWSSite",
    "awssite": "AWSSite",
    "cloud": "AWSSite",
    "local": "LocalSite",
    "localsite": "LocalSite",
    "slurm": "SlurmSite",
    "slurmsite": "SlurmSite",
    "stampede3": "Stampede3Site",
    "stampede3site": "Stampede3Site",
    "tacc": "Stampede3Site",
}

_RESERVED_SITE_KEYS = {"type", "backend", "class", "name", "profile"}

__all__ = [
    "DEFAULT_SITE_CONFIG_NAME",
    "SITE_CONFIG_ENV_VAR",
    "Site",
    "load_site_config",
    "site_config_path",
]


def site_config_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return the active FrequenSolve site config path."""

    if path is not None:
        return Path(path).expanduser()
    env_path = os.getenv(SITE_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".frequensolve" / DEFAULT_SITE_CONFIG_NAME


def load_site_config(
    path: Optional[Union[str, Path]] = None,
) -> Mapping[str, Any]:
    """Load the active site config TOML document."""

    config_path = site_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            "No FrequenSolve site config found at "
            f"{config_path}. Create ~/.frequensolve/{DEFAULT_SITE_CONFIG_NAME} "
            f"or set {SITE_CONFIG_ENV_VAR}."
        )
    return tomllib.loads(config_path.read_text())


def Site(
    *,
    config_path: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
    **overrides: Any,
):
    """Create the configured execution site.

    The default config location is ``~/.frequensolve/site.toml``. Tests and
    isolated tools can redirect it with ``FREQUENSOLVE_SITE_CONFIG`` or the
    explicit ``config_path`` argument.
    """

    config = load_site_config(config_path)
    site_config = _site_config_table(config, profile=profile)
    site_type = _site_type(site_config)
    kwargs = _site_kwargs(site_config, site_type)
    kwargs.update(overrides)
    site_class = _resolve_site_class(site_type)
    return site_class(**kwargs)


def _site_config_table(
    config: Mapping[str, Any], *, profile: Optional[str]
) -> Mapping[str, Any]:
    if "site" in config:
        site = config["site"]
        if not isinstance(site, Mapping):
            raise ValueError("FrequenSolve site config [site] must be a table")
        return site

    sites = config.get("sites")
    if isinstance(sites, Mapping):
        selected_profile = (
            profile or config.get("default") or config.get("default_site")
        )
        if not selected_profile:
            raise ValueError(
                "FrequenSolve site config with [sites] must set default or "
                "pass profile=..."
            )
        try:
            site = sites[selected_profile]
        except KeyError as exc:
            raise ValueError(
                f"FrequenSolve site config profile {selected_profile!r} was not found"
            ) from exc
        if not isinstance(site, Mapping):
            raise ValueError(
                f"FrequenSolve site config profile {selected_profile!r} must be a table"
            )
        return site

    if "type" in config or "backend" in config:
        return config

    raise ValueError(
        "FrequenSolve site config must contain a [site] table or [sites.<name>] profiles"
    )


def _site_type(site_config: Mapping[str, Any]) -> str:
    raw_type = site_config.get(
        "type", site_config.get("backend", site_config.get("class"))
    )
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise ValueError("FrequenSolve site config must set site.type")
    return raw_type


def _site_kwargs(site_config: Mapping[str, Any], site_type: str) -> dict[str, Any]:
    kwargs = {
        key: value
        for key, value in site_config.items()
        if key not in _RESERVED_SITE_KEYS
    }
    nested_kwargs = kwargs.pop("kwargs", None)
    if nested_kwargs is not None:
        if not isinstance(nested_kwargs, Mapping):
            raise ValueError("FrequenSolve site config site.kwargs must be a table")
        kwargs.update(nested_kwargs)
    normalized_type = _normalize_site_type(site_type)
    if normalized_type in {"slurm", "slurmsite"}:
        return _slurm_site_kwargs(kwargs)
    if normalized_type in {"stampede3", "stampede3site", "tacc"}:
        return _stampede3_site_kwargs(kwargs)
    return kwargs


def _normalize_site_type(site_type: str) -> str:
    return site_type.replace("_", "").replace("-", "").lower()


def _resolve_site_class(site_type: str):
    normalized = _normalize_site_type(site_type)
    class_name = _SITE_TYPES.get(normalized, site_type)

    from frequensolve.orchestrator import sites

    try:
        return getattr(sites, class_name)
    except AttributeError as exc:
        known = ", ".join(sorted(_SITE_TYPES))
        raise ValueError(
            f"Unknown FrequenSolve site type {site_type!r}. Known types: {known}"
        ) from exc


def _slurm_site_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    hpc = import_module("frequensolve.orchestrator.sites.hpc")
    config_cls = hpc.SlurmSiteConfig
    run_config_cls = hpc.SlurmRunConfig

    source = dict(kwargs)
    site_kwargs = dict(kwargs)
    config_value = site_kwargs.pop("config", None)
    run_config_value = site_kwargs.pop("run_config", None)

    config_values = _matching_values(source, _dataclass_field_names(config_cls))
    run_config_values = _matching_values(
        source, _run_config_field_names(run_config_cls)
    )
    for key in set(config_values) | set(run_config_values):
        site_kwargs.pop(key, None)

    if "default_queue" not in site_kwargs and "queue" in source:
        site_kwargs["default_queue"] = source["queue"]

    site_kwargs["config"] = _coerce_dataclass_config(
        config_cls,
        config_value,
        config_values,
        "SlurmSiteConfig",
    )
    if run_config_value is not None or run_config_values:
        site_kwargs["run_config"] = _coerce_dataclass_config(
            run_config_cls,
            run_config_value,
            run_config_values,
            "SlurmRunConfig",
        )
    return site_kwargs


def _stampede3_site_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    hpc = import_module("frequensolve.orchestrator.sites.hpc")
    run_config_cls = hpc.SlurmRunConfig

    source = dict(kwargs)
    site_kwargs = dict(kwargs)
    run_config_value = site_kwargs.pop("run_config", None)
    config_value = site_kwargs.pop("config", None)
    if config_value is not None:
        raise ValueError("Stampede3 site config does not accept a config table")

    run_config_values = _matching_values(
        source, _run_config_field_names(run_config_cls)
    )
    for key in run_config_values:
        site_kwargs.pop(key, None)

    if "default_queue" not in site_kwargs and "queue" in source:
        site_kwargs["default_queue"] = source["queue"]

    if run_config_value is not None or run_config_values:
        site_kwargs["run_config"] = _coerce_dataclass_config(
            run_config_cls,
            run_config_value,
            run_config_values,
            "SlurmRunConfig",
        )
    return site_kwargs


def _matching_values(source: Mapping[str, Any], names: set[str]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key in names}


def _dataclass_field_names(config_cls: type) -> set[str]:
    return {field.name for field in fields(config_cls)}


def _run_config_field_names(run_config_cls: type) -> set[str]:
    field_names = getattr(run_config_cls, "field_names", None)
    if callable(field_names):
        return set(field_names())
    return _dataclass_field_names(run_config_cls)


def _coerce_dataclass_config(
    config_cls: type,
    value: Any,
    values: Mapping[str, Any],
    name: str,
):
    if value is None:
        config_values = dict(values)
    elif isinstance(value, Mapping):
        config_values = {**values, **value}
    else:
        return value
    try:
        return config_cls(**config_values)
    except TypeError as exc:
        raise ValueError(
            f"Invalid FrequenSolve {name} settings in site config: {exc}"
        ) from exc
