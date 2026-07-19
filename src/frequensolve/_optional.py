"""Helpers for optional public dependencies."""

from __future__ import annotations

import importlib
from types import TracebackType
from typing import Any, Iterable, NoReturn, Type, cast


def _dependency_list(dependencies: Iterable[str] | None) -> str:
    values = list(dependencies or [])
    if not values:
        return ""
    return " Required packages: " + ", ".join(values) + "."


def optional_dependency_error(
    symbol: str,
    *,
    extra: str,
    error: BaseException,
    dependencies: Iterable[str] | None = None,
) -> ImportError:
    """Build a clear error for an unavailable optional dependency group.

    Args:
        symbol: Public feature or symbol that requires the optional
            dependency.
        extra: FrequenSolve package extra that installs the dependency group.
        error: Original import error.
        dependencies: Optional package names to mention explicitly.

    Returns:
        ``ImportError`` with an install hint and original error context.
    """

    return ImportError(
        f"{symbol} requires optional FrequenSolve dependencies that are not installed. "
        f"Install them with `pip install frequensolve[{extra}]`."
        f"{_dependency_list(dependencies)} Original import error: {error}"
    )


def _message(
    symbol: str,
    extra: str,
    error: BaseException,
    dependencies: Iterable[str] | None = None,
) -> str:
    return str(
        optional_dependency_error(
            symbol,
            extra=extra,
            error=error,
            dependencies=dependencies,
        )
    )


class _MissingOptionalMeta(type):
    def __getattr__(cls, name: str) -> NoReturn:
        raise cls._import_error()


class _LazyOptionalMeta(type):
    def __getattr__(cls, name: str) -> Any:
        return getattr(cls._load(), name)

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, cls._load())

    def __subclasscheck__(cls, subclass: type[Any]) -> bool:
        return issubclass(subclass, cls._load())


def missing_optional_class(
    symbol: str,
    *,
    extra: str,
    error: BaseException,
    module: str,
    dependencies: Iterable[str] | None = None,
) -> type[Any]:
    """Return a class placeholder that raises an install hint when used.

    Args:
        symbol: Public class name to expose.
        extra: FrequenSolve package extra that installs the dependency group.
        error: Original import error.
        module: Module name assigned to the placeholder class.
        dependencies: Optional package names to mention explicitly.

    Returns:
        Class object whose construction, subclassing, and attribute access
        raise the optional-dependency ``ImportError``.
    """

    class MissingOptionalDependency(metaclass=_MissingOptionalMeta):
        __doc__ = _message(symbol, extra, error, dependencies)
        _symbol = symbol
        _extra = extra
        _error = error
        _dependencies = dependencies

        @classmethod
        def _import_error(cls) -> ImportError:
            return optional_dependency_error(
                cls._symbol,
                extra=cls._extra,
                error=cls._error,
                dependencies=cls._dependencies,
            )

        def __new__(cls, *args: Any, **kwargs: Any) -> NoReturn:
            raise cls._import_error()

        def __init_subclass__(cls, **kwargs: Any) -> NoReturn:
            raise cls._import_error()

        def __enter__(self) -> NoReturn:
            raise self._import_error()

        def __exit__(
            self,
            exc_type: Type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

    MissingOptionalDependency.__name__ = symbol
    MissingOptionalDependency.__qualname__ = symbol
    MissingOptionalDependency.__module__ = module
    return cast(type[Any], MissingOptionalDependency)


def optional_class(
    symbol: str,
    import_path: str,
    *,
    extra: str,
    dependencies: Iterable[str] | None = None,
    module: str,
) -> type[Any]:
    """Return a lightweight proxy that imports an optional class on first use.

    Args:
        symbol: Public class name to expose.
        import_path: Fully qualified target class path.
        extra: FrequenSolve package extra that installs the dependency group.
        dependencies: Optional package names to mention explicitly.
        module: Module name assigned to the proxy class.

    Returns:
        Proxy class that imports and instantiates the target on first use.

    Raises:
        ValueError: If ``import_path`` is not a fully qualified class path.
    """

    module_name, _, attribute = import_path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError("import_path must be a fully qualified class path")

    class OptionalClassProxy(metaclass=_LazyOptionalMeta):
        __doc__ = (
            f"Lazy proxy for `{import_path}`. Requires "
            f"`pip install frequensolve[{extra}]` if dependencies are missing."
        )
        _target: type[Any] | None = None

        @classmethod
        def _load(cls) -> type[Any]:
            if cls._target is None:
                try:
                    imported = importlib.import_module(module_name)
                except ModuleNotFoundError as exc:
                    raise optional_dependency_error(
                        symbol,
                        extra=extra,
                        dependencies=dependencies,
                        error=exc,
                    ) from exc
                cls._target = getattr(imported, attribute)
            return cls._target

        def __new__(cls, *args: Any, **kwargs: Any) -> Any:
            target = cls._load()
            return target(*args, **kwargs)

        def __init_subclass__(cls, **kwargs: Any) -> NoReturn:
            target = cls._load()
            raise TypeError(f"Subclass {target.__name__} directly, not its lazy proxy")

    OptionalClassProxy.__name__ = symbol
    OptionalClassProxy.__qualname__ = symbol
    OptionalClassProxy.__module__ = module
    return OptionalClassProxy
