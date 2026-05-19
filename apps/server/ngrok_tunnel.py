"""ngrok tunnel management for blitzcode-pro.

Two pieces:

  1. `NgrokAuthStore` — persists the user's ngrok authtoken to a
     chmod 0600 JSON file. The token is never echoed back through the
     API; only `{configured: bool}` is exposed.

  2. `NgrokTunnel` — wraps the ngrok Python SDK (`ngrok.forward`,
     `ngrok.disconnect`, `ngrok.kill`). One tunnel per process. State
     machine:

         off ──start()──► starting ──► on
                              │
                              ▼
                            error  ←── retry: stop() then start()
         on  ──stop()───► off

     The SDK is sync and Rust-backed; we run `forward` / `disconnect`
     in a thread executor so we don't block the event loop.

The actual auth gate (which token, what's loopback) still lives in
lan_access.py — the tunnel is just routing. From the middleware's
perspective, ngrok's edge is just another non-loopback IP.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


# ── Authtoken store ─────────────────────────────────────────────────────────


class NgrokAuthStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._token: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(raw, dict):
            tok = raw.get("authtoken")
            if isinstance(tok, str) and tok:
                self._token = tok

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = {"authtoken": self._token} if self._token else {}
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._path)
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass

    def get_authtoken(self) -> Optional[str]:
        return self._token

    async def set(self, token: str) -> None:
        token = (token or "").strip()
        if not token:
            raise ValueError("authtoken cannot be empty")
        self._token = token
        self._flush()

    async def clear(self) -> None:
        self._token = None
        self._flush()

    def public_meta(self) -> dict[str, Any]:
        return {"configured": self._token is not None}


# ── Tunnel state machine ────────────────────────────────────────────────────


@dataclass
class TunnelState:
    state: str = "off"     # "off" | "starting" | "on" | "error"
    url: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "url": self.url, "error": self.error}


class NgrokTunnel:
    """Owns at most one ngrok listener at a time. Methods are safe to
    call from any task; the underlying ngrok SDK is sync + global, so
    we serialize with a single lock.

    The SDK is injected (defaults to the real `ngrok` module) so tests
    can substitute a fake without touching the network."""

    def __init__(self, *, sdk: Any = None) -> None:
        if sdk is None:
            import ngrok as _ngrok  # type: ignore
            sdk = _ngrok
        self._sdk = sdk
        self._state = TunnelState()
        self._lock = asyncio.Lock()
        self._url: Optional[str] = None  # canonical url we asked ngrok for

    def status(self) -> dict[str, Any]:
        return self._state.to_dict()

    async def start(self, port: int, authtoken: str) -> str:
        async with self._lock:
            if self._state.state == "on" and self._state.url:
                # Already up — return existing URL rather than churn.
                return self._state.url
            self._state = TunnelState(state="starting")
            try:
                # `ngrok.forward` is the SDK's blocking entrypoint. Run
                # in a worker thread so we don't stall the event loop.
                listener = await asyncio.to_thread(
                    self._sdk.forward, port, "http", authtoken=authtoken
                )
                url = listener.url() if callable(getattr(listener, "url", None)) else getattr(listener, "url", None)
                if not isinstance(url, str) or not url:
                    raise RuntimeError("ngrok returned no URL")
                self._url = url
                self._state = TunnelState(state="on", url=url)
                logger.info("ngrok tunnel up at %s -> 127.0.0.1:%d", url, port)
                return url
            except Exception as e:
                logger.exception("ngrok tunnel failed to start")
                self._state = TunnelState(state="error", error=str(e))
                # Best-effort cleanup in case forward() partially registered.
                try:
                    await asyncio.to_thread(self._sdk.kill)
                except Exception:
                    pass
                raise

    async def stop(self) -> None:
        async with self._lock:
            if self._state.state == "off":
                return
            url = self._url
            try:
                if url is not None:
                    await asyncio.to_thread(self._sdk.disconnect, url)
                else:
                    await asyncio.to_thread(self._sdk.disconnect)
            except Exception:
                logger.exception("ngrok disconnect failed; will kill")
            try:
                await asyncio.to_thread(self._sdk.kill)
            except Exception:
                pass
            self._url = None
            self._state = TunnelState(state="off")

    async def shutdown(self) -> None:
        """Called by the FastAPI lifespan teardown. Same as stop() but
        never raises — process is dying anyway."""
        try:
            await self.stop()
        except Exception:
            logger.exception("ngrok shutdown swallowed error")
