"""Initiative store CRUD."""
import pytest

from initiatives import Initiative, InitiativeStore, is_valid_key


class TestKey:
    def test_valid(self):
        for k in ("meowtorq", "internal-ai", "p1", "gear-007"):
            assert is_valid_key(k), k

    def test_invalid(self):
        for k in ("", "Meowtorq", "-foo", "foo_bar", "foo!", "FOO"):
            assert not is_valid_key(k), k


@pytest.fixture
def store(tmp_path):
    return InitiativeStore(tmp_path / "initiatives.json")


@pytest.mark.asyncio
async def test_upsert_then_get(store):
    it = Initiative(key="meowtorq", display_name="Meowtorq", epic_jira_key="LLM-608")
    await store.upsert(it)
    got = store.get("meowtorq")
    assert got and got.display_name == "Meowtorq" and got.epic_jira_key == "LLM-608"


@pytest.mark.asyncio
async def test_upsert_replaces(store):
    await store.upsert(Initiative(key="x", display_name="X"))
    await store.upsert(Initiative(key="x", display_name="X2", epic_jira_key="X-1"))
    got = store.get("x")
    assert got.display_name == "X2" and got.epic_jira_key == "X-1"


@pytest.mark.asyncio
async def test_upsert_rejects_invalid_key(store):
    with pytest.raises(ValueError):
        await store.upsert(Initiative(key="Invalid Key!", display_name="x"))


@pytest.mark.asyncio
async def test_list_returns_clones(store):
    await store.upsert(Initiative(key="a", display_name="A", repo_paths=["/p"]))
    items = store.list()
    items[0].repo_paths.append("/mutated")
    # Mutating the returned list should not affect the store.
    again = store.list()
    assert again[0].repo_paths == ["/p"]


@pytest.mark.asyncio
async def test_patch_partial(store):
    await store.upsert(Initiative(key="a", display_name="A", epic_jira_key="OLD"))
    await store.patch("a", epic_jira_key="NEW")
    got = store.get("a")
    assert got.epic_jira_key == "NEW"
    # display_name preserved.
    assert got.display_name == "A"


@pytest.mark.asyncio
async def test_patch_unknown_returns_none(store):
    assert (await store.patch("nope", display_name="x")) is None


@pytest.mark.asyncio
async def test_associate_repo_dedupes(store):
    await store.upsert(Initiative(key="a", display_name="A"))
    await store.associate_repo("a", "/path")
    await store.associate_repo("a", "/path")
    assert store.get("a").repo_paths == ["/path"]
    await store.associate_repo("a", "/path2")
    assert store.get("a").repo_paths == ["/path", "/path2"]


@pytest.mark.asyncio
async def test_remove(store):
    await store.upsert(Initiative(key="a", display_name="A"))
    assert await store.remove("a") is True
    assert store.get("a") is None
    assert await store.remove("a") is False


@pytest.mark.asyncio
async def test_persistence_across_instances(tmp_path):
    path = tmp_path / "init.json"
    s1 = InitiativeStore(path)
    await s1.upsert(Initiative(key="persist", display_name="P", repo_paths=["/r"]))
    s2 = InitiativeStore(path)
    got = s2.get("persist")
    assert got and got.display_name == "P" and got.repo_paths == ["/r"]
