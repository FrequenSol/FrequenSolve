"""Safety adapter for the official stable MCP Python SDK v2.

The adapter deliberately uses the SDK's public low-level server surface. The
simulation assistant owns its finite schemas and strict validation instead of
depending on host-specific argument coercion or SDK-private manager hooks.
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, TypeVar, get_type_hints

import anyio
from mcp import Client
from mcp.server import Server
from mcp.server.context import ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INVALID_PARAMS,
    CallToolRequestParams,
    CallToolResult,
    GetPromptRequestParams,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    PaginatedRequestParams,
    Prompt,
    PromptMessage,
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
    ToolAnnotations,
)
from pydantic import BaseModel, ValidationError

MINIMUM_SDK_VERSION = (2, 0, 0)

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

_CallableT = TypeVar("_CallableT", bound=Callable[..., Any])


class AdapterCompatibilityError(RuntimeError):
    """Raised when the installed SDK no longer matches the isolated adapter."""


class _UnsafeRequest(ValueError):
    """Internal marker for a request that exceeds the protocol budgets."""


@dataclass(frozen=True)
class _ToolRegistration:
    tool: Tool
    function: Callable[[BaseModel], Awaitable[BaseModel]]
    request_model: type[BaseModel]
    output_model: type[BaseModel]


@dataclass(frozen=True)
class _ResourceRegistration:
    resource: Resource
    function: Callable[[], Awaitable[str]]


@dataclass(frozen=True)
class _PromptRegistration:
    prompt: Prompt
    function: Callable[[], Awaitable[str]]


class SafeMCPServer(Server[dict[str, Any]]):
    """Finite MCP server with strict schemas and non-reflective failures."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        instructions: str,
        max_request_bytes: int = 262_144,
    ) -> None:
        _require_supported_sdk()
        self._tools: dict[str, _ToolRegistration] = {}
        self._resources: dict[str, _ResourceRegistration] = {}
        self._prompts: dict[str, _PromptRegistration] = {}
        self._allowed_tool_names: frozenset[str] = frozenset()
        self._allowed_prompt_names: frozenset[str] = frozenset()
        self._allowed_resource_uris: frozenset[str] = frozenset()
        self._max_request_bytes = max_request_bytes
        self._finalized = False
        super().__init__(
            name,
            version=version,
            instructions=instructions,
            on_list_tools=self._handle_list_tools,
            on_call_tool=self._handle_call_tool,
            on_list_resources=self._handle_list_resources,
            on_read_resource=self._handle_read_resource,
            on_list_prompts=self._handle_list_prompts,
            on_get_prompt=self._handle_get_prompt,
        )

    def tool(
        self,
        *,
        name: str,
        description: str,
        annotations: ToolAnnotations,
        structured_output: bool,
    ) -> Callable[[_CallableT], _CallableT]:
        """Register one strict single-request-model structured tool."""

        if not structured_output:
            raise AdapterCompatibilityError(
                "Every FrequenSolve MCP tool must publish structured output"
            )

        def decorator(function: _CallableT) -> _CallableT:
            self._require_registration_open()
            if name in self._tools:
                raise AdapterCompatibilityError("An MCP tool name was registered twice")
            signature = inspect.signature(function)
            parameters = list(signature.parameters.values())
            hints = get_type_hints(function)
            if (
                len(parameters) != 1
                or parameters[0].name != "request"
                or not inspect.iscoroutinefunction(function)
            ):
                raise AdapterCompatibilityError(
                    "MCP tools must be async functions with one request parameter"
                )
            request_model = hints.get("request")
            output_model = hints.get("return")
            if (
                not isinstance(request_model, type)
                or not issubclass(request_model, BaseModel)
                or not isinstance(output_model, type)
                or not issubclass(output_model, BaseModel)
            ):
                raise AdapterCompatibilityError(
                    "MCP tools require Pydantic request and response models"
                )
            input_schema = _request_schema(request_model)
            output_schema = output_model.model_json_schema(by_alias=True)
            _require_closed_object_schemas(input_schema)
            _require_closed_object_schemas(output_schema)
            self._tools[name] = _ToolRegistration(
                tool=Tool(
                    name=name,
                    description=description,
                    inputSchema=input_schema,
                    outputSchema=output_schema,
                    annotations=annotations,
                ),
                function=function,
                request_model=request_model,
                output_model=output_model,
            )
            return function

        return decorator

    def resource(
        self,
        uri: str,
        *,
        name: str,
        description: str,
        mime_type: str,
    ) -> Callable[[_CallableT], _CallableT]:
        """Register one fixed zero-argument text resource."""

        if not uri or "{" in uri or "}" in uri:
            raise AdapterCompatibilityError(
                "FrequenSolve MCP resources must use fixed URIs"
            )

        def decorator(function: _CallableT) -> _CallableT:
            self._require_registration_open()
            if uri in self._resources:
                raise AdapterCompatibilityError(
                    "An MCP resource URI was registered twice"
                )
            if inspect.signature(
                function
            ).parameters or not inspect.iscoroutinefunction(function):
                raise AdapterCompatibilityError(
                    "MCP resources must be async zero-argument functions"
                )
            self._resources[uri] = _ResourceRegistration(
                resource=Resource(
                    uri=uri,
                    name=name,
                    description=description,
                    mimeType=mime_type,
                ),
                function=function,
            )
            return function

        return decorator

    def prompt(
        self,
        *,
        name: str,
        description: str,
    ) -> Callable[[_CallableT], _CallableT]:
        """Register one fixed zero-argument prompt."""

        def decorator(function: _CallableT) -> _CallableT:
            self._require_registration_open()
            if name in self._prompts:
                raise AdapterCompatibilityError("An MCP prompt was registered twice")
            if inspect.signature(
                function
            ).parameters or not inspect.iscoroutinefunction(function):
                raise AdapterCompatibilityError(
                    "MCP prompts must be async zero-argument functions"
                )
            self._prompts[name] = _PromptRegistration(
                prompt=Prompt(
                    name=name,
                    description=description,
                    arguments=[],
                ),
                function=function,
            )
            return function

        return decorator

    def finalize_safety_contract(
        self,
        *,
        tool_names: Iterable[str],
        prompt_names: Iterable[str],
        resource_uris: Iterable[str],
    ) -> None:
        """Freeze and verify the complete finite public surface."""

        tools = frozenset(tool_names)
        prompts = frozenset(prompt_names)
        resources = frozenset(resource_uris)
        if not tools or not prompts or not resources:
            raise AdapterCompatibilityError(
                "The MCP safety contract requires non-empty finite name sets"
            )
        if tools != self._tools.keys():
            raise AdapterCompatibilityError(
                "The declared MCP tool surface does not match registration"
            )
        if prompts != self._prompts.keys():
            raise AdapterCompatibilityError(
                "The declared MCP prompt surface does not match registration"
            )
        if resources != self._resources.keys():
            raise AdapterCompatibilityError(
                "The declared MCP resource surface does not match registration"
            )
        self._allowed_tool_names = tools
        self._allowed_prompt_names = prompts
        self._allowed_resource_uris = resources
        self._finalized = True

    def run_stdio(self) -> None:
        """Run the official standard STDIO transport."""

        self._require_finalized()
        anyio.run(self._run_stdio_async)

    async def _run_stdio_async(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await super().run(
                read_stream,
                write_stream,
                self.create_initialization_options(),
            )

    async def _handle_list_tools(
        self,
        context: ServerRequestContext[dict[str, Any]],
        params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        del context, params
        self._require_finalized()
        return ListToolsResult(tools=[item.tool for item in self._tools.values()])

    async def _handle_call_tool(
        self,
        context: ServerRequestContext[dict[str, Any]],
        params: CallToolRequestParams,
    ) -> CallToolResult:
        del context
        self._require_finalized()
        registration = self._tools.get(params.name)
        if registration is None:
            return _tool_error("Unknown tool; use tools/list.")
        arguments = params.arguments or {}
        try:
            _validate_request_budget(
                arguments,
                maximum_bytes=self._max_request_bytes,
            )
            if set(arguments) != {"request"}:
                raise _UnsafeRequest
            request = registration.request_model.model_validate(arguments["request"])
        except (ValidationError, _UnsafeRequest):
            return _tool_error(
                "Invalid tool arguments; use the published input schema."
            )
        try:
            raw_result = await registration.function(request)
            result = registration.output_model.model_validate(raw_result)
            structured = result.model_dump(mode="json", by_alias=True)
            text = json.dumps(
                structured,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            return CallToolResult(
                content=[TextContent(type="text", text=text)],
                structuredContent=structured,
            )
        except Exception:
            return _tool_error(
                "Tool execution failed safely; retry or inspect diagnostics."
            )

    async def _handle_list_resources(
        self,
        context: ServerRequestContext[dict[str, Any]],
        params: PaginatedRequestParams | None,
    ) -> ListResourcesResult:
        del context, params
        self._require_finalized()
        return ListResourcesResult(
            resources=[item.resource for item in self._resources.values()]
        )

    async def _handle_read_resource(
        self,
        context: ServerRequestContext[dict[str, Any]],
        params: ReadResourceRequestParams,
    ) -> ReadResourceResult:
        del context
        self._require_finalized()
        registration = self._resources.get(params.uri)
        if registration is None:
            raise MCPError(
                INVALID_PARAMS,
                "Unknown or unavailable resource; use resources/list.",
            )
        try:
            text = await registration.function()
            return ReadResourceResult(
                contents=[
                    TextResourceContents(
                        uri=params.uri,
                        text=text,
                        mimeType=registration.resource.mime_type,
                    )
                ]
            )
        except Exception:
            raise MCPError(
                INVALID_PARAMS,
                "Unknown or unavailable resource; use resources/list.",
            ) from None

    async def _handle_list_prompts(
        self,
        context: ServerRequestContext[dict[str, Any]],
        params: PaginatedRequestParams | None,
    ) -> ListPromptsResult:
        del context, params
        self._require_finalized()
        return ListPromptsResult(
            prompts=[item.prompt for item in self._prompts.values()]
        )

    async def _handle_get_prompt(
        self,
        context: ServerRequestContext[dict[str, Any]],
        params: GetPromptRequestParams,
    ) -> GetPromptResult:
        del context
        self._require_finalized()
        registration = self._prompts.get(params.name)
        if registration is None or params.arguments:
            raise MCPError(
                INVALID_PARAMS,
                "Unknown or unavailable prompt; use prompts/list.",
            )
        try:
            text = await registration.function()
            return GetPromptResult(
                description=registration.prompt.description,
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=text),
                    )
                ],
            )
        except Exception:
            raise MCPError(
                INVALID_PARAMS,
                "Unknown or unavailable prompt; use prompts/list.",
            ) from None

    def _require_registration_open(self) -> None:
        if self._finalized:
            raise AdapterCompatibilityError(
                "The MCP safety contract is already finalized"
            )

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise AdapterCompatibilityError("The MCP safety contract was not finalized")


def run_in_memory_doctor(server: SafeMCPServer) -> dict[str, object]:
    """Verify the modern protocol through the official in-memory client."""

    return anyio.run(_run_in_memory_doctor, server)


async def _run_in_memory_doctor(server: SafeMCPServer) -> dict[str, object]:
    async with Client(server, mode="auto", cache=None) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        identity = await client.read_resource(
            "frequensolve://simulation-assistant/identity"
        )
        tool_names = {tool.name for tool in tools.tools}
        resource_uris = {resource.uri for resource in resources.resources}
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
            "protocol_version": client.protocol_version,
            "tools": len(tools.tools),
            "resources": len(resources.resources),
            "prompts": len(prompts.prompts),
        }


def _request_schema(request_model: type[BaseModel]) -> dict[str, Any]:
    inner = request_model.model_json_schema(by_alias=True)
    nested_definitions = inner.pop("$defs", {})
    definitions = dict(nested_definitions)
    definitions[request_model.__name__] = inner
    return {
        "$defs": definitions,
        "additionalProperties": False,
        "properties": {
            "request": {
                "$ref": f"#/$defs/{request_model.__name__}",
            }
        },
        "required": ["request"],
        "type": "object",
    }


def _tool_error(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


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
    if len(parts) != 3 or parts[0] != 2 or parts < MINIMUM_SDK_VERSION:
        raise AdapterCompatibilityError(
            "FrequenSolve requires MCP Python SDK >=2.0.0,<3"
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
