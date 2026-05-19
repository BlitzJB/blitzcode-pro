"""LanAccessStore + middleware behavior.

Mirrors the threat model exactly: loopback always passes; non-loopback
needs the feature enabled AND the token (header or `?k=`).
"""
import json
from pathlib import Path

import pytest

from lan_access import LanAccessStore, make_lan_auth_middleware


# ── Store ───────────────────────────────────────────────────────────────────


def test_store_starts_disabled(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    assert s.enabled is False
    assert s.token is None


def test_enable_generates_token(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    state = s.enable()
    assert state.enabled is True
    assert state.token and len(state.token) > 30


def test_re_enable_rotates_token(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    t1 = s.enable().token
    t2 = s.enable().token
    assert t1 != t2


def test_disable_clears_token(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    s.enable()
    s.disable()
    assert s.enabled is False
    assert s.token is None


def test_persists_across_instances(tmp_path: Path):
    p = tmp_path / "lan.json"
    a = LanAccessStore(p)
    a.enable()
    tok = a.token
    b = LanAccessStore(p)
    assert b.enabled is True
    assert b.token == tok


def test_public_meta_hides_token_when_disabled(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    s.enable()
    s.disable()
    meta = s.public_meta()
    assert meta["enabled"] is False
    assert meta["token"] is None


def test_file_is_chmod_0600(tmp_path: Path):
    import os
    p = tmp_path / "lan.json"
    s = LanAccessStore(p)
    s.enable()
    mode = os.stat(p).st_mode & 0o777
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


# ── Middleware ──────────────────────────────────────────────────────────────


class _DummyApp:
    """Records the call so we can assert it was/wasn't reached."""
    def __init__(self):
        self.called = False
    async def __call__(self, scope, receive, send):
        self.called = True
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})


async def _drive(mw, app, *, client_host: str, headers=None, query=b"", path: str = "/app/workspaces"):
    """Run a middleware request through and collect the response.
    Default path is a protected one — earlier tests assumed the
    middleware gated everything, which is no longer the case."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or [])],
        "query_string": query,
        "client": (client_host, 0),
    }
    sent = []
    async def send(msg): sent.append(msg)
    async def recv(): return {"type": "http.disconnect"}
    await mw(scope, recv, send, app)
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body


@pytest.mark.asyncio
async def test_loopback_always_passes_even_when_disabled(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(mw, app, client_host="127.0.0.1")
    assert status == 200
    assert app.called


@pytest.mark.asyncio
async def test_non_loopback_blocked_when_disabled(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, body = await _drive(mw, app, client_host="192.168.1.42")
    assert status == 403
    assert not app.called
    payload = json.loads(body)
    assert payload["error"]["code"] == "lan_access_disabled"


@pytest.mark.asyncio
async def test_non_loopback_blocked_without_token(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    s.enable()
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, body = await _drive(mw, app, client_host="192.168.1.42")
    assert status == 401
    assert not app.called
    assert json.loads(body)["error"]["code"] == "bad_token"


@pytest.mark.asyncio
async def test_non_loopback_passes_with_header_token(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    s.enable()
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(
        mw, app,
        client_host="192.168.1.42",
        headers=[("X-Blitz-Token", s.token)],
    )
    assert status == 200
    assert app.called


@pytest.mark.asyncio
async def test_non_loopback_passes_with_query_token(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    s.enable()
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(
        mw, app,
        client_host="192.168.1.42",
        query=f"k={s.token}".encode(),
    )
    assert status == 200
    assert app.called


@pytest.mark.asyncio
async def test_non_loopback_rejected_with_wrong_token(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    s.enable()
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(
        mw, app,
        client_host="192.168.1.42",
        headers=[("X-Blitz-Token", "definitely-not-the-token")],
    )
    assert status == 401
    assert not app.called


@pytest.mark.asyncio
async def test_non_loopback_passes_with_bearer_token(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    s.enable()
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(
        mw, app,
        client_host="192.168.1.42",
        headers=[("Authorization", f"Bearer {s.token}")],
    )
    assert status == 200
    assert app.called


@pytest.mark.asyncio
async def test_ipv6_loopback_passes(tmp_path: Path):
    s = LanAccessStore(tmp_path / "lan.json")
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(mw, app, client_host="::1")
    assert status == 200


# ── Public vs protected path scoping ───────────────────────────────────────
# Static assets must be reachable from any LAN client without a token so
# the React bundle can boot; only API + stream paths need auth.


PUBLIC_PATHS = [
    "/",
    "/index.html",
    "/_next/static/chunks/main-app.js",
    "/_next/static/css/abc.css",
    "/favicon.ico",
    "/sw.js",
]

PROTECTED_PATHS = [
    "/app/workspaces",
    "/app/settings",
    "/app/lan-access",
    "/sessions",
    "/sessions/abc/history",
    "/sessions/abc/input",
    "/stream",
    "/genui/schema",
]


@pytest.mark.parametrize("path", PUBLIC_PATHS)
@pytest.mark.asyncio
async def test_public_paths_bypass_auth_from_lan(tmp_path: Path, path: str):
    s = LanAccessStore(tmp_path / "lan.json")
    # Even with auth disabled, public paths must pass for any client.
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(mw, app, client_host="192.168.1.42", path=path)
    assert status == 200, f"public path {path!r} should be reachable but got {status}"


@pytest.mark.parametrize("path", PROTECTED_PATHS)
@pytest.mark.asyncio
async def test_protected_paths_block_lan_when_disabled(tmp_path: Path, path: str):
    s = LanAccessStore(tmp_path / "lan.json")
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(mw, app, client_host="192.168.1.42", path=path)
    assert status == 403, f"protected path {path!r} should require auth but got {status}"


@pytest.mark.parametrize("path", PROTECTED_PATHS)
@pytest.mark.asyncio
async def test_protected_paths_pass_loopback_always(tmp_path: Path, path: str):
    s = LanAccessStore(tmp_path / "lan.json")  # disabled
    mw = make_lan_auth_middleware(s)
    app = _DummyApp()
    status, _ = await _drive(mw, app, client_host="127.0.0.1", path=path)
    assert status == 200, f"loopback should bypass auth on {path!r}"
