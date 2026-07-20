"""Config-file driven site factory."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import fields
from importlib import import_module
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from frequensolve.orchestrator.sites.base import LocalHostConfig
from frequensolve.storage import frequensolve_home

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
SITE_PRESETS_CONFIG_NAME = "site_presets.toml"
STARTER_SITE_CONFIG = """# FrequenSolve execution site configuration.
# The cloud profile below is active by default and works with app.frequensol.com.
# Replace the placeholders before selecting another profile with `default` or
# fs.Site(profile="..."). Do not store passwords, passphrases, or tokens here.

default = "cloud"

[host]
# Local machine staging for disposable project bundles and transfer tarballs.
# Defaults to Python's platform temp directory when omitted.
# tmp_dir = "/local/tmp/directory"

[sites.cloud]
type = "aws"
domain = "app.frequensol.com"
interactive = true
verbose = true

[sites.local]
type = "local"
solver = "/path/to/local/solver"
shutdown_on_completion = true
verbose = true

[sites.hpc]
type = "slurm"
hostname = "login.example.edu"
username = "your-username"
credential = "example-hpc"
ssh_key = "~/.ssh/id_ed25519"
# HPC launches route every solver phase through this executable.
solver = "/remote/path/to/solver-installation/FS_seismic"
work_dir = "/remote/writable/directory/frequensolve"
# Optional future location for models and other high-I/O data.
scratch_dir = "/remote/scratch/directory/frequensolve"
tmp_dir = "/remote/tmp/directory"
default_partition = "debug"
account = "allocation"
transfer_method = "rsync"
modules = []
verbose = true

# Define one table per partition using limits and node resources supplied by
# the cluster administrator. These `debug` values are illustrative.
[sites.hpc.partitions.debug]
max_duration = "02:00:00"
min_nodes = 1
max_nodes = 4
cores_per_node = 64
sockets_per_node = 2
memory_per_node = 262144 # MiB
gpus_per_node = 0

[sites.hpc.environment]
# Values may reference module-defined variables with simple ${NAME} syntax.
# LD_LIBRARY_PATH = "${PARALLEL_HDF5_LIB}:${LD_LIBRARY_PATH}"

[sites.hpc.run_config]
nodes = 1
duration = "00:30:00"
ranks_per_node = 4
ranks_per_task = 1
poll_interval = 10
scheduler_heartbeat_timeout = 60

# Stampede3 uses built-in host, launcher, partition, and node-shape defaults.
[sites.stampede3]
type = "slurm"
preset = "stampede3"
username = "your-tacc-username"
account = "your-tacc-allocation"
credential = "tacc-stampede3"
ssh_key = "~/.ssh/id_ed25519"
# HPC launches route every solver phase through this executable.
solver = "/remote/path/to/solver-installation/FS_seismic"
# work_dir defaults to $WORK/frequensolve. It may point to any writable remote
# filesystem; TOML does not expand $WORK or $SCRATCH.
# work_dir = "/another/remote/path/frequensolve"
# Optional future location for models and other high-I/O data.
# scratch_dir = "/scratch/your-tacc-username/frequensolve"
# Remote transfer/provision staging. Use a concrete absolute path.
# tmp_dir = "/scratch/your-tacc-username/frequensolve/tmp"
default_partition = "skx-dev"
transfer_method = "rsync"
modules = []
verbose = true

[sites.stampede3.environment]
# Values may reference module-defined variables with simple ${NAME} syntax.
# LD_LIBRARY_PATH = "${PARALLEL_HDF5_LIB}:${LD_LIBRARY_PATH}"

[sites.stampede3.run_config]
nodes = 1
duration = "00:30:00"
"""

_SITE_TYPES = {
    "aws": "AWSSite",
    "awssite": "AWSSite",
    "cloud": "AWSSite",
    "local": "LocalSite",
    "localsite": "LocalSite",
    "slurm": "SlurmSite",
    "slurmsite": "SlurmSite",
    "stampede3": "SlurmSite",
    "stampede3site": "SlurmSite",
    "tacc": "SlurmSite",
}

_RESERVED_SITE_KEYS = {
    "documentation_url",
    "last_verified",
    "preset",
    "type",
}
_UNSUPPORTED_TOP_LEVEL_KEYS = {
    "default_site": "use 'default'",
    "profiles": "use [sites.<profile>] tables",
}
_UNSUPPORTED_SITE_KEYS = {
    "backend": "use 'type'",
    "class": "use 'type'",
    "name": "profile names come from [sites.<profile>] table names",
    "profile": "profile names come from [sites.<profile>] table names",
    "solver_executable": "use 'solver'",
}

__all__ = [
    "DEFAULT_SITE_CONFIG_NAME",
    "SITE_PRESETS_CONFIG_NAME",
    "SITE_CONFIG_ENV_VAR",
    "Site",
    "LocalHostConfig",
    "load_site_config",
    "load_site_presets",
    "site_config_path",
]


def site_config_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return the active FrequenSolve site config path."""

    if path is not None:
        return Path(path).expanduser()
    env_path = os.getenv(SITE_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return frequensolve_home() / DEFAULT_SITE_CONFIG_NAME


def load_site_config(
    path: Optional[Union[str, Path]] = None,
) -> Mapping[str, Any]:
    """Load the active site config TOML document."""

    config_path = site_config_path(path)
    if not config_path.exists():
        if _should_create_starter_config(path):
            _write_starter_site_config(config_path)
            raise FileNotFoundError(
                "Created starter FrequenSolve site config at "
                f"{config_path}. Review the profiles, modify them for your "
                "environment if needed, then rerun. The default profile is "
                "cloud and uses app.frequensol.com."
            )
        raise FileNotFoundError(
            "No FrequenSolve site config found at "
            f"{config_path}. Create "
            f"{frequensolve_home() / DEFAULT_SITE_CONFIG_NAME} "
            f"or set {SITE_CONFIG_ENV_VAR}."
        )
    return tomllib.loads(config_path.read_text())


def load_site_presets() -> Mapping[str, Any]:
    """Load the built-in execution-site preset catalog."""

    resource = files("frequensolve.orchestrator.sites").joinpath(
        SITE_PRESETS_CONFIG_NAME
    )
    return tomllib.loads(resource.read_text())


def _should_create_starter_config(path: Optional[Union[str, Path]]) -> bool:
    return path is None and not os.getenv(SITE_CONFIG_ENV_VAR)


def _write_starter_site_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with config_path.open("x") as file:
            file.write(STARTER_SITE_CONFIG)
    except FileExistsError:  # pragma: no cover - defensive for concurrent first runs
        return


def Site(
    *,
    config_path: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
    queue: Optional[str] = None,
    nodes: Optional[int] = None,
    ranks_per_node: Optional[int] = None,
    duration: Optional[str] = None,
    **overrides: Any,
):
    """Create the configured execution site.

    The default config location is ``~/.frequensolve/site.toml``. Tests and
    isolated tools can redirect it with ``FREQUENSOLVE_SITE_CONFIG`` or the
    explicit ``config_path`` argument. For SLURM profiles, ``queue``, ``nodes``,
    ``ranks_per_node``, and ``duration`` override the profile's default run
    configuration directly.
    """

    config = load_site_config(config_path)
    local_host_config = _local_host_config(config)
    site_config = dict(_site_config_table(config, profile=profile))
    selected_profile = profile or _default_profile(config)
    site_config.update(overrides)
    resource_overrides = {
        name: value
        for name, value in {
            "queue": queue,
            "nodes": nodes,
            "ranks_per_node": ranks_per_node,
            "duration": duration,
        }.items()
        if value is not None
    }
    site_config = _apply_direct_slurm_resources(site_config, resource_overrides)
    site_config = _resolve_site_preset(site_config)
    site_type = _site_type(site_config)
    kwargs = _site_kwargs(site_config, site_type)
    site_class = _resolve_site_class(site_type)
    site = site_class(**kwargs)
    site.local_host_config = local_host_config
    site._site_config_path = site_config_path(config_path)
    site._site_profile = selected_profile
    return site


def _local_host_config(config: Mapping[str, Any]) -> LocalHostConfig:
    """Return local-machine settings from the top-level host table."""

    if "host" in config and "local_host" in config:
        raise ValueError("Use either [host] or [local_host], not both")
    values = config.get("host", config.get("local_host", {}))
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise ValueError("FrequenSolve local host config must be a table")
    unknown = sorted(set(values) - {"tmp_dir"})
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"Unsupported FrequenSolve host config key(s): {names}")
    return LocalHostConfig(tmp_dir=values.get("tmp_dir"))


def _apply_direct_slurm_resources(
    site_config: Mapping[str, Any], resources: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply explicit run resources after all profile-level values."""

    profile = dict(site_config)
    if not resources:
        return profile

    queue = resources.get("queue")
    if queue is not None:
        profile.pop("default_partition", None)
        nested_config = profile.get("config")
        if isinstance(nested_config, Mapping):
            profile["config"] = {**nested_config, "queue": queue}
        elif nested_config is not None and hasattr(nested_config, "queue"):
            nested_config = deepcopy(nested_config)
            nested_config.queue = queue
            profile["config"] = nested_config

    profile.update(resources)
    nested_run_config = profile.get("run_config")
    if isinstance(nested_run_config, Mapping):
        profile["run_config"] = {**nested_run_config, **resources}
    elif nested_run_config is not None:
        merge = getattr(nested_run_config, "merged", None)
        if callable(merge):
            profile["run_config"] = merge(**resources)
        else:
            nested_run_config = deepcopy(nested_run_config)
            for name, value in resources.items():
                if hasattr(nested_run_config, name):
                    setattr(nested_run_config, name, value)
            profile["run_config"] = nested_run_config
    return profile


def _site_config_table(
    config: Mapping[str, Any], *, profile: Optional[str]
) -> Mapping[str, Any]:
    _reject_unsupported_top_level_keys(config)

    if "site" in config:
        raise ValueError(
            "FrequenSolve site config must use top-level default and "
            "[sites.<profile>] tables; [site] is not supported"
        )

    sites = config.get("sites")
    if not isinstance(sites, Mapping):
        raise ValueError(
            "FrequenSolve site config must contain top-level default and "
            "[sites.<profile>] tables"
        )

    default_profile = _default_profile(config)
    if default_profile not in sites:
        raise ValueError(
            f"FrequenSolve site config default profile {default_profile!r} "
            "was not found"
        )
    selected_profile = profile or default_profile

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
    _reject_unsupported_site_keys(site, f"[sites.{selected_profile}]")
    return site


def _default_profile(config: Mapping[str, Any]) -> str:
    default = config.get("default")
    if not isinstance(default, str) or not default.strip():
        raise ValueError("FrequenSolve site config with [sites] must set default")
    return default


def _site_type(site_config: Mapping[str, Any]) -> str:
    raw_type = site_config.get("type")
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise ValueError("FrequenSolve site config must set site.type")
    return raw_type


def _resolve_site_preset(site_config: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay a built-in preset with an individual user profile."""

    profile = dict(site_config)
    raw_type = profile.get("type")
    normalized_type = (
        _normalize_site_type(raw_type) if isinstance(raw_type, str) else None
    )
    slurm_type = normalized_type in {
        "slurm",
        "slurmsite",
        "stampede3",
        "stampede3site",
        "tacc",
    }
    if "queue" in profile and slurm_type:
        queue = profile.pop("queue")
        default_partition = profile.get("default_partition")
        if default_partition is not None and default_partition != queue:
            raise ValueError(
                "FrequenSolve SLURM config cannot set conflicting "
                "default_partition and queue values"
            )
        profile["default_partition"] = queue
    compatibility_alias = normalized_type in {
        "stampede3",
        "stampede3site",
        "tacc",
    }
    preset_name = profile.get("preset")
    if preset_name is None and compatibility_alias:
        preset_name = "stampede3"
    if preset_name is None:
        return profile
    if not isinstance(preset_name, str) or not preset_name.strip():
        raise ValueError("FrequenSolve site config preset must be a non-empty string")

    catalog = load_site_presets().get("presets")
    if not isinstance(catalog, Mapping):  # pragma: no cover - packaged invariant
        raise ValueError("FrequenSolve site preset catalog has no [presets] table")
    preset = catalog.get(preset_name)
    if not isinstance(preset, Mapping):
        known = ", ".join(sorted(str(name) for name in catalog))
        raise ValueError(
            f"Unknown FrequenSolve site preset {preset_name!r}. Known presets: {known}"
        )

    preset_type = preset.get("type")
    if (
        raw_type is not None
        and not compatibility_alias
        and isinstance(preset_type, str)
        and _normalize_site_type(str(raw_type)) != _normalize_site_type(preset_type)
    ):
        raise ValueError(
            f"FrequenSolve site type {raw_type!r} is incompatible with "
            f"preset {preset_name!r} ({preset_type!r})"
        )
    return _deep_merge(preset, profile)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _reject_unsupported_top_level_keys(config: Mapping[str, Any]) -> None:
    for key, replacement in _UNSUPPORTED_TOP_LEVEL_KEYS.items():
        if key in config:
            raise ValueError(
                f"FrequenSolve site config key {key!r} is not supported; {replacement}."
            )


def _reject_unsupported_site_keys(site_config: Mapping[str, Any], context: str) -> None:
    for key, replacement in _UNSUPPORTED_SITE_KEYS.items():
        if key in site_config:
            raise ValueError(
                f"FrequenSolve site config key {key!r} in {context} is not "
                f"supported; {replacement}."
            )


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
    if normalized_type in {
        "slurm",
        "slurmsite",
        "stampede3",
        "stampede3site",
        "tacc",
    }:
        return _slurm_site_kwargs(kwargs)
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
    if "default_partition" in source:
        default_partition = source.pop("default_partition")
        queue = source.get("queue")
        if queue is not None and queue != default_partition:
            raise ValueError(
                "FrequenSolve SLURM config cannot set conflicting "
                "default_partition and queue values"
            )
        source["queue"] = default_partition
    site_kwargs = dict(source)
    config_value = site_kwargs.pop("config", None)
    run_config_value = site_kwargs.pop("run_config", None)

    config_values = _matching_values(source, _dataclass_field_names(config_cls))
    run_config_values = _matching_values(
        source, _run_config_field_names(run_config_cls)
    )
    for key in set(config_values) | set(run_config_values):
        site_kwargs.pop(key, None)

    if "default_partition" not in site_kwargs and "queue" in source:
        site_kwargs["default_partition"] = source["queue"]

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
