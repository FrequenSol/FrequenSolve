"""Dedicated opaque workflow OAuth tokens; no SDK or AWS credential imports."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .storage import PrivateStore


class WorkflowError(RuntimeError):
    """Safe diagnostic, never a reflected HTTP body or credential."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise WorkflowError("Workflow redirects are not allowed")


def exchange(
    url: str,
    payload: dict | None = None,
    *,
    token: str | None = None,
    form: bool = False,
) -> dict:
    headers = {"accept": "application/json"}
    if token:
        headers["authorization"] = "Bearer " + token
    data = None
    if payload is not None:
        data = (
            urlencode(payload).encode()
            if form
            else json.dumps(payload, allow_nan=False).encode()
        )
        headers["content-type"] = (
            "application/x-www-form-urlencoded" if form else "application/json"
        )
    try:
        with build_opener(_NoRedirect()).open(
            Request(url, data=data, headers=headers), timeout=25
        ) as response:
            raw = response.read(131073)
        if len(raw) > 131072:
            raise ValueError()
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("error"):
            raise ValueError()
        return value
    except Exception:
        raise WorkflowError(
            "Cloud request could not be confirmed; recover the same intent or reconnect"
        ) from None


@dataclass(frozen=True)
class Connection:
    issuer: str
    resource: str
    client_id: str
    callback: str = "http://127.0.0.1:43881/callback"
    native_approval: bool = False

    def __post_init__(self) -> None:
        issuer, resource = urlsplit(self.issuer), urlsplit(self.resource)
        if (
            issuer.scheme != "https"
            or not issuer.hostname
            or issuer.username
            or issuer.password
            or issuer.path not in ("", "/")
            or issuer.query
            or issuer.fragment
            or resource.scheme != "https"
            or resource.netloc != issuer.netloc
            or resource.path != "/cloud"
            or resource.query
            or resource.fragment
            or self.callback != "http://127.0.0.1:43881/callback"
            or not self.client_id
            or type(self.native_approval) is not bool
        ):
            raise ValueError("Invalid workflow connection")

    @property
    def scopes(self) -> frozenset[str]:
        scopes = frozenset(
            self.resource + suffix
            for suffix in ("/read", "/workflow-prepare", "/workflow-results")
        )
        return scopes | (
            {self.resource + "/workflow-approve"} if self.native_approval else set()
        )

    @property
    def binding(self) -> dict:
        value = asdict(self)
        # Preserve existing explicit connection and journal identities.
        if not self.native_approval:
            value.pop("native_approval")
        return value


class WorkflowAuth:
    def __init__(
        self,
        connection: Connection,
        store: PrivateStore,
        profile: str,
        *,
        request: Callable = exchange,
        now: Callable = time.time,
    ):
        self.connection, self.store, self.profile = connection, store, profile
        self.request, self.now = request, now

    def save(self, response: dict) -> None:
        value = self._tokens(response)
        with self.store.locked(self.profile) as record:
            record.write(value)

    def _tokens(self, response: dict, previous: dict | None = None) -> dict:
        if (
            response.get("token_type") != "Bearer"
            or response.get("resource") != self.connection.resource
            or set(str(response.get("scope", "")).split()) != self.connection.scopes
            or not isinstance(response.get("expires_in"), int)
            or not 1 <= response["expires_in"] <= 3600
            or not all(
                isinstance(response.get(k), str) and response[k]
                for k in (
                    "access_token",
                    "refresh_token",
                    "subject",
                    "grant_id",
                    "account_id",
                )
            )
        ):
            raise WorkflowError(
                "Cloud did not issue the required restricted workflow grant"
            )
        if previous and any(
            response[k] != previous[k] for k in ("subject", "grant_id", "account_id")
        ):
            raise WorkflowError("Workflow grant identity changed; reconnect")
        return {
            **response,
            "binding": self.connection.binding,
            "expires_at": self.now() + response["expires_in"],
        }

    def credentials(self) -> dict:
        # The lock covers read/refresh/replace: server refresh tokens are single-use.
        with self.store.locked(self.profile) as record:
            saved = record.read()
            if not saved or saved.get("binding") != self.connection.binding:
                raise WorkflowError("Run workflow login for this explicit profile")
            if saved["expires_at"] <= self.now() + 60:
                # Persist uncertainty before rotating; never retry a consumed refresh.
                if saved.get("refresh_pending"):
                    raise WorkflowError(
                        "Token refresh outcome is unknown; run workflow login again"
                    )
                saved["refresh_pending"] = True
                record.write(saved)
                result = self.request(
                    self.connection.issuer.rstrip("/") + "/oauth/token",
                    {
                        "client_id": self.connection.client_id,
                        "grant_type": "refresh_token",
                        "refresh_token": saved["refresh_token"],
                        "resource": self.connection.resource,
                    },
                    form=True,
                )
                saved = self._tokens(result, saved)
                record.write(saved)
            return saved

    def login(
        self, open_browser: Callable[[str], object], *, timeout: float = 300
    ) -> None:
        c = self.connection
        metadata = self.request(
            c.issuer.rstrip("/") + "/.well-known/oauth-authorization-server"
        )
        if (
            metadata.get("issuer") != c.issuer
            or metadata.get("authorization_endpoint")
            != c.issuer.rstrip("/") + "/oauth/authorize"
            or metadata.get("token_endpoint") != c.issuer.rstrip("/") + "/oauth/token"
            or not c.scopes.issubset(metadata.get("scopes_supported", []))
        ):
            raise WorkflowError(
                "Workflow authorization metadata does not match the connection"
            )
        verifier = secrets.token_urlsafe(48)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        state = secrets.token_urlsafe(32)
        result: dict = {}

        class Callback(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                url = urlsplit(self.path)
                query = parse_qs(url.query)
                valid = (
                    url.path == "/callback"
                    and query.get("state") == [state]
                    and query.get("iss") == [c.issuer]
                    and len(query.get("code", [])) == 1
                )
                self.send_response(200 if valid else 400)
                self.end_headers()
                self.wfile.write(
                    b"Return to your scientific workflow."
                    if valid
                    else b"Invalid callback."
                )
                if valid:
                    result["code"] = query["code"][0]

            def log_message(self, format: str, *args: Any) -> None:
                pass

        # Bind before opening the authorization page; no password collection.
        with HTTPServer(("127.0.0.1", 43881), Callback) as server:
            server.timeout = 1
            open_browser(
                metadata["authorization_endpoint"]
                + "?"
                + urlencode(
                    {
                        "response_type": "code",
                        "client_id": c.client_id,
                        "redirect_uri": c.callback,
                        "resource": c.resource,
                        "scope": " ".join(sorted(c.scopes)),
                        "state": state,
                        "code_challenge": challenge,
                        "code_challenge_method": "S256",
                    }
                )
            )
            deadline = time.monotonic() + timeout
            while not result and time.monotonic() < deadline:
                server.handle_request()
        if not result:
            raise WorkflowError("Workflow login timed out; start a fresh login")
        self.save(
            self.request(
                metadata["token_endpoint"],
                {
                    "grant_type": "authorization_code",
                    "code": result["code"],
                    "code_verifier": verifier,
                    "client_id": c.client_id,
                    "redirect_uri": c.callback,
                    "resource": c.resource,
                },
                form=True,
            )
        )

    def logout(self) -> None:
        with self.store.locked(self.profile) as record:
            saved = record.read()
            if saved:
                for key in ("access_token", "refresh_token"):
                    self.request(
                        self.connection.issuer.rstrip("/") + "/oauth/revoke",
                        {"client_id": self.connection.client_id, "token": saved[key]},
                        form=True,
                    )
            record.delete()
