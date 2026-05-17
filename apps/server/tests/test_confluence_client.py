"""ConfluenceClient — respx-mocked tests."""
import json

import pytest
import respx
from httpx import Response

from atlassian.confluence import ConfluenceClient, ConfluenceError


SITE = "https://x.atlassian.net"


@pytest.fixture
def client():
    return ConfluenceClient(SITE, "u@u", "tok")


def _adf(*content):
    return {"type": "doc", "version": 1, "content": list(content)}


def _adf_response(page_id: str, title: str, version: int, body_adf: dict, **extra) -> dict:
    payload = {
        "id": page_id,
        "title": title,
        "spaceId": "100",
        "parentId": "200",
        "version": {"number": version},
        "body": {"atlas_doc_format": {"representation": "atlas_doc_format", "value": json.dumps(body_adf)}},
        "_links": {"webui": f"/spaces/X/pages/{page_id}/{title}"},
    }
    payload.update(extra)
    return payload


@respx.mock(assert_all_called=True)
async def test_get_page_parses_adf_body(respx_mock, client):
    body_adf = _adf({"type": "paragraph", "content": [{"type": "text", "text": "hi"}]})
    respx_mock.get(f"{SITE}/wiki/api/v2/pages/42").mock(
        return_value=Response(200, json=_adf_response("42", "Hello", 3, body_adf))
    )
    page = await client.get_page("42")
    assert page.id == "42"
    assert page.title == "Hello"
    assert page.version == 3
    assert page.body_adf == body_adf
    assert page.space_id == "100"
    assert page.parent_id == "200"
    assert page.url and page.url.endswith("/wiki/spaces/X/pages/42/Hello")


@respx.mock(assert_all_called=True)
async def test_find_child_by_title_returns_matching_page(respx_mock, client):
    # First page in children listing matches.
    respx_mock.get(f"{SITE}/wiki/api/v2/pages/200/children").mock(
        return_value=Response(200, json={
            "results": [
                {"id": "99", "title": "Other"},
                {"id": "42", "title": "[LLM-1] RFC: Foo"},
            ],
        })
    )
    body_adf = _adf({"type": "paragraph", "content": [{"type": "text", "text": "x"}]})
    respx_mock.get(f"{SITE}/wiki/api/v2/pages/42").mock(
        return_value=Response(200, json=_adf_response("42", "[LLM-1] RFC: Foo", 1, body_adf))
    )
    page = await client.find_child_by_title("200", "[LLM-1] RFC: Foo")
    assert page is not None
    assert page.id == "42"


@respx.mock(assert_all_called=True)
async def test_find_child_by_title_paginates(respx_mock, client):
    # First page no match, second page matches.
    respx_mock.get(f"{SITE}/wiki/api/v2/pages/200/children").mock(side_effect=[
        Response(200, json={
            "results": [{"id": "10", "title": "A"}],
            "_links": {"next": "/wiki/api/v2/pages/200/children?cursor=NEXT123"},
        }),
        Response(200, json={
            "results": [{"id": "42", "title": "wanted"}],
        }),
    ])
    body_adf = _adf({"type": "paragraph", "content": [{"type": "text", "text": "x"}]})
    respx_mock.get(f"{SITE}/wiki/api/v2/pages/42").mock(
        return_value=Response(200, json=_adf_response("42", "wanted", 1, body_adf))
    )
    page = await client.find_child_by_title("200", "wanted")
    assert page is not None
    assert page.id == "42"


@respx.mock(assert_all_called=True)
async def test_find_child_returns_none_when_absent(respx_mock, client):
    respx_mock.get(f"{SITE}/wiki/api/v2/pages/200/children").mock(
        return_value=Response(200, json={"results": []})
    )
    page = await client.find_child_by_title("200", "nope")
    assert page is None


@respx.mock(assert_all_called=True)
async def test_create_page_posts_atlas_doc_format(respx_mock, client):
    body_adf = _adf({"type": "paragraph", "content": [{"type": "text", "text": "hello"}]})
    posted = respx_mock.post(f"{SITE}/wiki/api/v2/pages").mock(
        return_value=Response(200, json=_adf_response("42", "T", 1, body_adf))
    )
    page = await client.create_page(space_id="100", parent_id="200", title="T", body_adf=body_adf)
    assert page.id == "42"
    body = json.loads(posted.calls.last.request.read().decode())
    assert body["spaceId"] == "100"
    assert body["parentId"] == "200"
    assert body["title"] == "T"
    assert body["body"]["representation"] == "atlas_doc_format"
    # The body.value must be a STRING (Confluence v2 quirk).
    assert isinstance(body["body"]["value"], str)
    # And that string must JSON-decode back to our ADF.
    assert json.loads(body["body"]["value"]) == body_adf


@respx.mock(assert_all_called=True)
async def test_update_page_bumps_version(respx_mock, client):
    body_adf = _adf({"type": "paragraph", "content": [{"type": "text", "text": "v2"}]})
    posted = respx_mock.put(f"{SITE}/wiki/api/v2/pages/42").mock(
        return_value=Response(200, json=_adf_response("42", "T", 4, body_adf))
    )
    page = await client.update_page(page_id="42", title="T", body_adf=body_adf, current_version=3)
    assert page.version == 4
    body = json.loads(posted.calls.last.request.read().decode())
    assert body["version"]["number"] == 4  # current + 1


@respx.mock(assert_all_called=True)
async def test_update_conflict_raises(respx_mock, client):
    body_adf = _adf({"type": "paragraph", "content": [{"type": "text", "text": "x"}]})
    respx_mock.put(f"{SITE}/wiki/api/v2/pages/42").mock(
        return_value=Response(409, text='{"errors":[{"detail":"Version conflict"}]}')
    )
    with pytest.raises(ConfluenceError) as exc:
        await client.update_page(page_id="42", title="T", body_adf=body_adf, current_version=2)
    assert exc.value.status == 409
    assert "Version conflict" in (exc.value.body or "")


@respx.mock(assert_all_called=True)
async def test_search_pages_returns_normalized_hits(respx_mock, client):
    route = respx_mock.get(f"{SITE}/wiki/rest/api/search").mock(
        return_value=Response(200, json={
            "results": [
                {"content": {"id": "11", "title": "Meowtorq root", "_links": {"webui": "/spaces/M/pages/11"}}},
                {"content": {"id": "12", "title": "Meowtorq RFCs", "_links": {"webui": "/spaces/M/pages/12"}}},
            ]
        })
    )
    out = await client.search_pages("Meowtorq")
    assert [(r["id"], r["title"]) for r in out] == [("11", "Meowtorq root"), ("12", "Meowtorq RFCs")]
    # CQL got injected with prefix-match
    call = route.calls.last
    assert b'cql=' in call.request.url.query
    assert b'title' in call.request.url.query


async def test_search_pages_empty_query_no_call(client):
    out = await client.search_pages("")
    assert out == []


@respx.mock(assert_all_called=True)
async def test_get_page_tolerates_corrupt_body_value(respx_mock, client):
    respx_mock.get(f"{SITE}/wiki/api/v2/pages/42").mock(
        return_value=Response(200, json={
            "id": "42", "title": "x", "spaceId": "100", "parentId": "200",
            "version": {"number": 1},
            "body": {"atlas_doc_format": {"value": "not valid json"}},
        })
    )
    page = await client.get_page("42")
    assert page.body_adf == {"type": "doc", "version": 1, "content": []}
