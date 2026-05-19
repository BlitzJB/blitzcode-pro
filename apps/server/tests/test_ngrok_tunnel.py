"""NgrokAuthStore + NgrokTunnel.

The SDK is mocked so these tests don't touch the network or require
an authtoken. We're pinning the state machine + persistence
contract, nothing more.
"""
import asyncio
import os
from pathlib import Path
from typing import Any, Optional

import pytest

from ngrok_tunnel import NgrokAuthStore, NgrokTunnel, TunnelState


# ── NgrokAuthStore ──────────────────────────────────────────────────────────


def test_authstore_starts_empty(tmp_path: Path):
    s = NgrokAuthStore(tmp_path / "ngrok.json")
    assert s.get_authtoken() is None
    assert s.public_meta() == {"configured": False}


@pytest.mark.asyncio
async def test_authstore_set_persists_and_hides_token(tmp_path: Path):
    p = tmp_path / "ngrok.json"
    s = NgrokAuthStore(p)
    await s.set("abc123-secret-shouldnt-leak")
    assert s.get_authtoken() == "abc123-secret-shouldnt-leak"
    # public_meta NEVER includes the token
    meta = s.public_meta()
    assert meta == {"configured": True}
    assert "authtoken" not in meta
    # Fresh instance reads the same value from disk
    s2 = NgrokAuthStore(p)
    assert s2.get_authtoken() == "abc123-secret-shouldnt-leak"


@pytest.mark.asyncio
async def test_authstore_set_rejects_empty(tmp_path: Path):
    s = NgrokAuthStore(tmp_path / "ngrok.json")
    with pytest.raises(ValueError):
        await s.set("")
    with pytest.raises(ValueError):
        await s.set("   ")


@pytest.mark.asyncio
async def test_authstore_clear(tmp_path: Path):
    s = NgrokAuthStore(tmp_path / "ngrok.json")
    await s.set("xyz")
    await s.clear()
    assert s.get_authtoken() is None
    assert s.public_meta() == {"configured": False}
    # Persists across instances
    s2 = NgrokAuthStore(tmp_path / "ngrok.json")
    assert s2.get_authtoken() is None


def test_authstore_file_is_chmod_0600(tmp_path: Path):
    p = tmp_path / "ngrok.json"
    s = NgrokAuthStore(p)
    asyncio.get_event_loop().run_until_complete(s.set("x"))
    mode = os.stat(p).st_mode & 0o777
    assert mode == 0o600


# ── NgrokTunnel (mocked SDK) ────────────────────────────────────────────────


class _FakeListener:
    def __init__(self, url: str) -> None:
        self._url = url
    def url(self) -> str:
        return self._url


class _FakeSDK:
    """Records every call and lets tests drive forward() success/error."""
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.next_url: Optional[str] = "https://abc-12-34-56-78.ngrok-free.app"
        self.fail_forward: Optional[Exception] = None
        self.fail_disconnect: Optional[Exception] = None
        self.alive = False

    def forward(self, addr, proto=None, **opts):
        self.calls.append(("forward", (addr, proto), opts))
        if self.fail_forward:
            raise self.fail_forward
        self.alive = True
        return _FakeListener(self.next_url or "")

    def disconnect(self, url=None):
        self.calls.append(("disconnect", (url,), {}))
        if self.fail_disconnect:
            raise self.fail_disconnect
        self.alive = False

    def kill(self):
        self.calls.append(("kill", (), {}))
        self.alive = False


@pytest.mark.asyncio
async def test_tunnel_starts_off():
    t = NgrokTunnel(sdk=_FakeSDK())
    s = t.status()
    assert s["state"] == "off" and s["url"] is None and s["error"] is None


@pytest.mark.asyncio
async def test_tunnel_start_returns_url_and_transitions_on():
    sdk = _FakeSDK()
    t = NgrokTunnel(sdk=sdk)
    url = await t.start(51820, "fake-token")
    assert url == sdk.next_url
    assert t.status() == {"state": "on", "url": url, "error": None}
    # Was forward called with the right args + token?
    assert sdk.calls[0][0] == "forward"
    assert sdk.calls[0][1] == (51820, "http")
    assert sdk.calls[0][2] == {"authtoken": "fake-token"}


@pytest.mark.asyncio
async def test_tunnel_start_when_already_on_returns_existing_url():
    sdk = _FakeSDK()
    t = NgrokTunnel(sdk=sdk)
    url1 = await t.start(51820, "fake")
    url2 = await t.start(51820, "fake")
    assert url1 == url2
    # forward called only once
    forwards = [c for c in sdk.calls if c[0] == "forward"]
    assert len(forwards) == 1


@pytest.mark.asyncio
async def test_tunnel_start_error_sets_error_state():
    sdk = _FakeSDK()
    sdk.fail_forward = RuntimeError("auth required")
    t = NgrokTunnel(sdk=sdk)
    with pytest.raises(RuntimeError):
        await t.start(51820, "bad")
    s = t.status()
    assert s["state"] == "error"
    assert "auth required" in (s["error"] or "")
    # Cleanup attempted
    assert any(c[0] == "kill" for c in sdk.calls)


@pytest.mark.asyncio
async def test_tunnel_stop_transitions_off_and_disconnects_url():
    sdk = _FakeSDK()
    t = NgrokTunnel(sdk=sdk)
    await t.start(51820, "fake")
    await t.stop()
    assert t.status() == {"state": "off", "url": None, "error": None}
    disconnects = [c for c in sdk.calls if c[0] == "disconnect"]
    assert len(disconnects) == 1
    # Disconnect was called with the actual URL we got from forward
    assert disconnects[0][1][0] == sdk.next_url


@pytest.mark.asyncio
async def test_tunnel_stop_when_already_off_is_noop():
    sdk = _FakeSDK()
    t = NgrokTunnel(sdk=sdk)
    await t.stop()
    assert sdk.calls == []
    assert t.status()["state"] == "off"


@pytest.mark.asyncio
async def test_tunnel_stop_swallows_disconnect_failures():
    sdk = _FakeSDK()
    sdk.fail_disconnect = RuntimeError("disconnect ate it")
    t = NgrokTunnel(sdk=sdk)
    await t.start(51820, "fake")
    # Should not raise — state must reach "off" regardless
    await t.stop()
    assert t.status()["state"] == "off"


@pytest.mark.asyncio
async def test_tunnel_shutdown_never_raises():
    sdk = _FakeSDK()
    sdk.fail_disconnect = RuntimeError("boom")
    t = NgrokTunnel(sdk=sdk)
    await t.start(51820, "fake")
    # Shutdown is the lifespan-hook entrypoint; must be exception-safe.
    await t.shutdown()
    assert t.status()["state"] == "off"


@pytest.mark.asyncio
async def test_tunnel_restart_after_error():
    """Error state → user provides a working token → start succeeds and
    flips us back to 'on'. Pinning that the error state isn't sticky."""
    sdk = _FakeSDK()
    sdk.fail_forward = RuntimeError("first attempt fails")
    t = NgrokTunnel(sdk=sdk)
    with pytest.raises(RuntimeError):
        await t.start(51820, "bad")
    # Recover: stop (clears error), then start succeeds
    await t.stop()
    sdk.fail_forward = None
    url = await t.start(51820, "good")
    assert url == sdk.next_url
    assert t.status()["state"] == "on"
