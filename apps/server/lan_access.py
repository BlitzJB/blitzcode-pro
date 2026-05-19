"""LAN access for blitzcode-pro.

Two pieces:

  1. `LanAccessStore` — persists whether LAN access is enabled and the
     current shared token. Atomic JSON file, chmod 0600 on the file so
     other users can't read the token from disk.

  2. `make_lan_auth_middleware(...)` — FastAPI middleware that:
       * Allows all loopback requests (the Tauri shell, dev workflows).
       * For non-loopback requests, requires either the
         `X-Blitz-Token` header or `?k=<token>` query param to match
         the stored token. EventSource can't send custom headers, so
         the query-param path is what the streaming endpoint uses.

The middleware uses constant-time compare so token guessing can't be
timing-attacked, but honestly the threat model is "someone on the same
WiFi" — the real defense is keeping the token off shoulders, not
cryptographic hardness.
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


_TOKEN_BYTES = 32  # 256-bit; encoded urlsafe-base64 → ~43 chars


@dataclass
class LanAccessState:
    enabled: bool
    token: Optional[str]


def _empty() -> LanAccessState:
    return LanAccessState(enabled=False, token=None)


class LanAccessStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._state: LanAccessState = _empty()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        enabled = bool(raw.get("enabled", False))
        token = raw.get("token")
        self._state = LanAccessState(
            enabled=enabled,
            token=str(token) if isinstance(token, str) and token else None,
        )

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {
            "enabled": self._state.enabled,
            "token": self._state.token,
        }
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self._path)
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass  # best-effort, e.g. on tmpfs in tests

    @property
    def enabled(self) -> bool:
        return self._state.enabled

    @property
    def token(self) -> Optional[str]:
        return self._state.token

    def public_meta(self) -> dict[str, Any]:
        """What we expose to the client. Token included only when enabled;
        if not enabled, the field is null so leaked snapshots don't
        leak a stale secret."""
        return {
            "enabled": self._state.enabled,
            "token": self._state.token if self._state.enabled else None,
        }

    def enable(self) -> LanAccessState:
        """Enable LAN access. Generates a fresh token (so re-enabling
        rotates it). Returns the new state."""
        token = secrets.token_urlsafe(_TOKEN_BYTES)
        self._state = LanAccessState(enabled=True, token=token)
        self._flush()
        return self._state

    def disable(self) -> LanAccessState:
        """Disable LAN access and drop the token so any leaked QR code
        becomes immediately useless."""
        self._state = LanAccessState(enabled=False, token=None)
        self._flush()
        return self._state


def _is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    return host in ("127.0.0.1", "::1", "localhost")


def _extract_token(scope_headers: list[tuple[bytes, bytes]], query_string: bytes) -> Optional[str]:
    """Accept tokens via (in priority order):
      * `X-Blitz-Token: <token>`
      * `Authorization: Bearer <token>` — what agent-webkit's transport
        already sends; lets us reuse its plumbing unchanged.
      * `?k=<token>` query param — for the URL in the QR code / link
        the user opens on their phone the first time.
    """
    for name, val in scope_headers:
        lname = name.lower()
        if lname == b"x-blitz-token":
            try:
                return val.decode("latin-1")
            except Exception:
                return None
        if lname == b"authorization":
            try:
                raw = val.decode("latin-1")
            except Exception:
                continue
            if raw.lower().startswith("bearer "):
                return raw[7:].strip() or None
    if query_string:
        from urllib.parse import parse_qs
        try:
            qs = parse_qs(query_string.decode("latin-1"))
        except Exception:
            return None
        vals = qs.get("k") or []
        if vals:
            return vals[0]
    return None


# Paths that require auth. Everything else (static export, /_next/*,
# favicon, etc.) is intrinsically public — the bundle contains no
# secrets and the LAN client needs to boot the JS before it can
# present a token via fetch headers.
_PROTECTED_PREFIXES = ("/app/", "/sessions", "/stream", "/genui/")


def _is_protected(path: str) -> bool:
    if not path:
        return False
    # Exact match on /sessions + /stream too (no trailing slash variant).
    if path in ("/sessions", "/stream"):
        return True
    return any(path.startswith(p) for p in _PROTECTED_PREFIXES)


def make_lan_auth_middleware(store: LanAccessStore):
    """ASGI-style middleware factory. Pure-ASGI rather than FastAPI
    `BaseHTTPMiddleware` so it doesn't break SSE streaming (BHTTPM
    buffers the response body).

    Gates only the API + stream paths. Static assets (`/`, `/_next/*`,
    `/favicon.ico`, etc.) are public so the React bundle can boot on a
    LAN client and present the token via Authorization on subsequent
    fetches. The bundle itself contains no secrets — the same JS lives
    in a public GitHub release.
    """

    async def middleware(scope: dict, receive: Callable[[], Awaitable[dict]], send: Callable[[dict], Awaitable[None]], app):  # noqa: E501
        if scope.get("type") != "http":
            return await app(scope, receive, send)
        if not _is_protected(scope.get("path", "")):
            return await app(scope, receive, send)
        client = scope.get("client") or ("", 0)
        client_host = client[0] if client else ""
        if _is_loopback(client_host):
            return await app(scope, receive, send)
        # Non-loopback hitting a protected path: token required, AND the
        # feature must be enabled.
        if not store.enabled or not store.token:
            await _reject(send, 403, "lan_access_disabled", "Enable LAN access in Settings → Network on the host device first.")
            return
        token = _extract_token(scope.get("headers") or [], scope.get("query_string") or b"")
        if not token or not secrets.compare_digest(token, store.token):
            await _reject(send, 401, "bad_token", "Missing or invalid LAN access token.")
            return
        return await app(scope, receive, send)

    return middleware


async def _reject(send: Callable[[dict], Awaitable[None]], status: int, code: str, message: str) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            # Permissive CORS on the error so the browser surfaces the
            # JSON body to fetch() instead of a generic "TypeError: Load
            # failed".
            (b"access-control-allow-origin", b"*"),
        ],
    })
    await send({"type": "http.response.body", "body": body, "more_body": False})
