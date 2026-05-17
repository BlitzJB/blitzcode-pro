"""CredsStore — disk format, validation, perms, no token leak in public_meta."""
import json
import os
import stat

import pytest

from atlassian_creds import AtlassianCreds, CredsStore


@pytest.fixture
def store(tmp_path):
    return CredsStore(tmp_path / "creds.json")


@pytest.mark.asyncio
async def test_starts_empty(store):
    assert not store.has_creds()
    assert store.get() is None
    assert store.public_meta() == {"has_creds": False, "site_url": None, "email": None}


@pytest.mark.asyncio
async def test_set_then_has(store):
    await store.set("https://x.atlassian.net", "a@b.com", "tok")
    assert store.has_creds()
    creds = store.get()
    assert isinstance(creds, AtlassianCreds)
    assert creds.api_token == "tok"
    assert creds.site_url == "https://x.atlassian.net"
    assert creds.email == "a@b.com"


@pytest.mark.asyncio
async def test_set_strips_trailing_slash(store):
    await store.set("https://x.atlassian.net/", "a@b.com", "tok")
    assert store.get().site_url == "https://x.atlassian.net"


@pytest.mark.asyncio
async def test_public_meta_never_includes_token(store):
    await store.set("https://x.atlassian.net", "a@b.com", "shh-secret-token")
    meta = store.public_meta()
    assert meta["has_creds"] is True
    assert meta["site_url"] == "https://x.atlassian.net"
    assert meta["email"] == "a@b.com"
    # Anywhere in the serialized meta — must NOT contain the token.
    assert "shh-secret-token" not in json.dumps(meta)


@pytest.mark.asyncio
async def test_set_requires_all_fields(store):
    with pytest.raises(ValueError):
        await store.set("", "a@b.com", "tok")
    with pytest.raises(ValueError):
        await store.set("https://x", "", "tok")
    with pytest.raises(ValueError):
        await store.set("https://x", "a@b.com", "")


@pytest.mark.asyncio
async def test_set_requires_scheme(store):
    with pytest.raises(ValueError):
        await store.set("no-scheme.atlassian.net", "a@b.com", "tok")


@pytest.mark.asyncio
async def test_file_perms_are_0600_after_write(store, tmp_path):
    await store.set("https://x.atlassian.net", "a@b.com", "tok")
    mode = stat.S_IMODE(os.stat(tmp_path / "creds.json").st_mode)
    assert mode == 0o600


@pytest.mark.asyncio
async def test_persists_across_instances(tmp_path):
    a = CredsStore(tmp_path / "creds.json")
    await a.set("https://x.atlassian.net", "u@u", "abc")
    b = CredsStore(tmp_path / "creds.json")
    assert b.has_creds() and b.get().api_token == "abc"


@pytest.mark.asyncio
async def test_clear(store):
    await store.set("https://x.atlassian.net", "u@u", "abc")
    await store.clear()
    assert not store.has_creds()


@pytest.mark.asyncio
async def test_load_tolerates_corrupt_file(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text("not json")
    s = CredsStore(p)
    assert not s.has_creds()


@pytest.mark.asyncio
async def test_load_tolerates_partial_dict(tmp_path):
    p = tmp_path / "creds.json"
    p.write_text(json.dumps({"site_url": "https://x"}))  # missing email/token
    s = CredsStore(p)
    assert not s.has_creds()
