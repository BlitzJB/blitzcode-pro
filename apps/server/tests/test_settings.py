"""SettingsStore — disk format, merge semantics, null deletes."""
import json

import pytest

from settings import SettingsStore


@pytest.fixture
def store(tmp_path):
    return SettingsStore(tmp_path / "settings.json")


@pytest.mark.asyncio
async def test_starts_empty(store):
    assert store.snapshot() == {}


@pytest.mark.asyncio
async def test_patch_creates_categories(store):
    await store.patch({"appearance": {"theme": "dark"}})
    assert store.snapshot() == {"appearance": {"theme": "dark"}}


@pytest.mark.asyncio
async def test_patch_merges_within_category(store):
    await store.patch({"appearance": {"theme": "dark"}})
    await store.patch({"appearance": {"density": "compact"}})
    assert store.snapshot() == {"appearance": {"theme": "dark", "density": "compact"}}


@pytest.mark.asyncio
async def test_patch_null_deletes_key(store):
    await store.patch({"appearance": {"theme": "dark", "density": "compact"}})
    await store.patch({"appearance": {"theme": None}})
    assert store.snapshot() == {"appearance": {"density": "compact"}}


@pytest.mark.asyncio
async def test_emptying_category_removes_it(store):
    await store.patch({"appearance": {"theme": "dark"}})
    await store.patch({"appearance": {"theme": None}})
    assert store.snapshot() == {}


@pytest.mark.asyncio
async def test_persists_across_instances(tmp_path):
    a = SettingsStore(tmp_path / "settings.json")
    await a.patch({"appearance": {"theme": "system"}})
    b = SettingsStore(tmp_path / "settings.json")
    assert b.snapshot() == {"appearance": {"theme": "system"}}


@pytest.mark.asyncio
async def test_load_tolerates_corrupt_file(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("not json")
    s = SettingsStore(p)
    assert s.snapshot() == {}


@pytest.mark.asyncio
async def test_load_skips_invalid_shape(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"appearance": "not-a-dict", "good": {"a": 1}}))
    s = SettingsStore(p)
    assert s.snapshot() == {"good": {"a": 1}}


@pytest.mark.asyncio
async def test_ignores_invalid_payload_shapes(store):
    # Top-level keys must be strings; values must be dicts. Garbage ignored.
    await store.patch({"appearance": {"theme": "dark"}})
    await store.patch({"appearance": "garbage"})  # type: ignore[arg-type]
    await store.patch({1: {"x": 1}})  # type: ignore[dict-item]
    await store.patch({"good": {1: "v"}})  # type: ignore[dict-item]
    assert store.snapshot() == {"appearance": {"theme": "dark"}}
