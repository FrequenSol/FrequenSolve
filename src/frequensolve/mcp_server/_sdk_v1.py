"""Safety adapter for the official MCP Python SDK v1.

All MCP imports and SDK-private compatibility touches live here so the rest of
the simulation assistant can remain independent of the transport library.
"""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable, Mapping

import anyio
import pydantic_core
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ResourceError, ToolError
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import GetPromptResult, ToolAnnotations
from pydantic import AnyUrl, ValidationError

MINIMUM_SDK_VERSION = (1, 28, 1)

READ_ONLY_TOOL_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
CLOUD_READ_ONLY_TOOL_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


class AdapterCompatibilityError(RuntimeError):
    """Raised when the installed SDK no longer matches the isolated adapter."""


class _UnsafeRequest(ValueError):
    """Internal marker for a request that exceeds the protocol budgets."""


class SafeFastMCP(FastMCP):
    """FastMCP with closed tool schemas and non-reflective error handling."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        instructions: str,
        max_request_bytes: int = 262_144,
    ) -> None:
        _require_supported_sdk()
        super().__init__(
            name=name,
            instructions=instructions,
            log_level="ERROR",
        )
        if not hasattr(self, "_mcp_server") or not hasattr(self._mcp_server, "version"):
            raise AdapterCompatibilityError(
                "The installed MCP SDK does not expose the v1 server version hook"
            )
        self._mcp_server.version = version
        self._allowed_tool_names: frozenset[str] = frozenset()
        self._allowed_prompt_names: frozenset[str] = frozenset()
        self._allowed_resource_uris: frozenset[str] = frozenset()
        self._max_request_bytes = max_request_bytes

    def finalize_safety_contract(
        self,
        *,
        tool_names: Iterable[str],
        prompt_names: Iterable[str],
        resource_uris: Iterable[str],
    ) -> None:
        """Close registered schemas and freeze the finite public name sets."""

        if not hasattr(self, "_tool_manager") or not hasattr(
            self._tool_manager, "get_tool"
        ):
            raise AdapterCompatibilityError(
                "The installed MCP SDK does not expose the v1 tool manager hook"
            )

        tools = frozenset(tool_names)
        prompts = frozenset(prompt_names)
        resources = frozenset(resource_uris)
        if not tools or not prompts or not resources:
            raise AdapterCompatibilityError(
                "The MCP safety contract requires non-empty finite name sets"
            )

        for name in tools:
            tool = self._tool_manager.get_tool(name)
            if tool is None:
                raise AdapterCompatibilityError(
                    "A declared MCP tool was not registered"
                )
            metadata = getattr(tool, "fn_metadata", None)
            argument_model = getattr(metadata, "arg_model", None)
            if argument_model is None or not hasattr(argument_model, "model_rebuild"):
                raise AdapterCompatibilityError(
                    "The installed MCP SDK does not expose the v1 argument model"
                )
            argument_model.model_config["extra"] = "forbid"
            argument_model.model_rebuild(force=True)
            tool.parameters = argument_model.model_json_schema(by_alias=True)
            _require_closed_object_schemas(tool.parameters)
            output_schema = getattr(tool, "output_schema", None)
            if not isinstance(output_schema, Mapping):
                raise AdapterCompatibilityError(
                    "A structured MCP tool is missing its output schema"
                )
            _require_closed_object_schemas(output_schema)

        self._allowed_tool_names = tools
        self._allowed_prompt_names = prompts
        self._allowed_resource_uris = resources

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call an allowlisted tool without reflecting invalid input values."""

        if name not in self._allowed_tool_names:
            raise ToolError("Unknown tool; use tools/list.")
        try:
            _validate_request_budget(arguments, maximum_bytes=self._max_request_bytes)
            tool = self._tool_manager.get_tool(name)
            if tool is None:  # pragma: no cover - guarded during finalization
                raise _UnsafeRequest
            tool.fn_metadata.arg_model.model_validate(arguments)
        except (ValidationError, _UnsafeRequest):
            raise ToolError(
                "Invalid tool arguments; use the published input schema."
            ) from None

        try:
            return await super().call_tool(name, arguments)
        except Exception:
            raise ToolError(
                "Tool execution failed safely; retry or inspect diagnostics."
            ) from None

    async def read_resource(
        self,
        uri: Any,
    ) -> Iterable[ReadResourceContents]:
        """Read a fixed resource without echoing or logging unknown URIs."""

        uri_text = str(uri)
        if uri_text not in self._allowed_resource_uris:
            raise ResourceError("Unknown or unavailable resource; use resources/list.")
        try:
            resource = await self._resource_manager.get_resource(
                uri,
                context=self.get_context(),
            )
            if resource is None:  # pragma: no cover - guarded during finalization
                raise ResourceError
            content = await resource.read()
            return [
                ReadResourceContents(
                    content=content,
                    mime_type=resource.mime_type,
                    meta=resource.meta,
                )
            ]
        except Exception:
            raise ResourceError(
                "Unknown or unavailable resource; use resources/list."
            ) from None

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        """Render a fixed zero-argument prompt without reflective logging."""

        if name not in self._allowed_prompt_names or arguments:
            raise ValueError("Unknown or unavailable prompt; use prompts/list.")
        try:
            prompt = self._prompt_manager.get_prompt(name)
            if prompt is None:  # pragma: no cover - guarded during finalization
                raise ValueError
            messages = await prompt.render(None, context=self.get_context())
            return GetPromptResult(
                description=prompt.description,
                messages=pydantic_core.to_jsonable_python(messages),
            )
        except Exception:
            raise ValueError(
                "Unknown or unavailable prompt; use prompts/list."
            ) from None


def run_in_memory_doctor(server: SafeFastMCP) -> dict[str, object]:
    """Verify the v1 server surface through the official in-memory client."""

    return anyio.run(_run_in_memory_doctor, server)


async def _run_in_memory_doctor(server: SafeFastMCP) -> dict[str, object]:
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
        resources = await session.list_resources()
        prompts = await session.list_prompts()
        identity = await session.read_resource(
            AnyUrl("frequensolve://simulation-assistant/identity")
        )
        tool_names = {tool.name for tool in tools.tools}
        resource_uris = {str(resource.uri) for resource in resources.resources}
        prompt_names = {prompt.name for prompt in prompts.prompts}
        if (
            tool_names != server._allowed_tool_names
            or resource_uris != server._allowed_resource_uris
        ):
            raise AdapterCompatibilityError(
                "The MCP doctor found an incomplete server surface"
            )
        if prompt_names != server._allowed_prompt_names or not identity.contents:
            raise AdapterCompatibilityError(
                "The MCP doctor could not read the fixed surface"
            )
        return {
            "ok": True,
            "transport": "in-memory",
            "tools": len(tools.tools),
            "resources": len(resources.resources),
            "prompts": len(prompts.prompts),
        }


def _require_supported_sdk() -> None:
    try:
        raw_version = version("mcp")
    except PackageNotFoundError as exc:  # pragma: no cover - import already guards
        raise AdapterCompatibilityError(
            "Install FrequenSolve with the mcp extra"
        ) from exc
    numeric = raw_version.split("+", 1)[0].split("-", 1)[0]
    try:
        parts = tuple(int(part) for part in numeric.split(".")[:3])
    except ValueError as exc:
        raise AdapterCompatibilityError(
            "The installed MCP SDK version cannot be verified"
        ) from exc
    if len(parts) != 3 or parts[0] != 1 or parts < MINIMUM_SDK_VERSION:
        raise AdapterCompatibilityError(
            "FrequenSolve requires MCP Python SDK >=1.28.1,<2"
        )


def _require_closed_object_schemas(schema: Mapping[str, Any]) -> None:
    stack: list[Any] = [schema]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            if current.get("type") == "object":
                additional = current.get("additionalProperties")
                if additional is not False:
                    raise AdapterCompatibilityError(
                        "MCP object schemas must reject additional properties"
                    )
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _validate_request_budget(value: Any, *, maximum_bytes: int) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    estimated_bytes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > 5_000 or depth > 16:
            raise _UnsafeRequest
        if current is None or isinstance(current, bool):
            estimated_bytes += 16
        elif isinstance(current, int):
            estimated_digits = max(
                1,
                int(current.bit_length() * 0.30103) + 2,
            )
            if estimated_digits > 65_536:
                raise _UnsafeRequest
            estimated_bytes += estimated_digits
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise _UnsafeRequest
            estimated_bytes += 32
        elif isinstance(current, str):
            encoded = current.encode("utf-8", errors="strict")
            if len(encoded) > 65_536:
                raise _UnsafeRequest
            estimated_bytes += len(encoded)
        elif isinstance(current, Mapping):
            if len(current) > 256:
                raise _UnsafeRequest
            for key, item in current.items():
                if not isinstance(key, str):
                    raise _UnsafeRequest
                key_bytes = key.encode("utf-8", errors="strict")
                if len(key_bytes) > 128:
                    raise _UnsafeRequest
                estimated_bytes += len(key_bytes)
                stack.append((item, depth + 1))
        elif isinstance(current, (list, tuple)):
            if len(current) > 1_000:
                raise _UnsafeRequest
            stack.extend((item, depth + 1) for item in current)
        else:
            raise _UnsafeRequest
        if estimated_bytes > maximum_bytes:
            raise _UnsafeRequest
