"""JiraClient — respx-mocked HTTP. Covers JQL shape, status transitions,
flag toggling, link-action-item, error mapping."""
import pytest
import respx
from httpx import Response

from atlassian.jira import JiraClient, JiraError, _typeahead_jql


SITE = "https://x.atlassian.net"


@pytest.fixture
def client():
    return JiraClient(SITE, "u@u", "tok")


# ────────────────────────────────────────────────────────────────────────────
# typeahead JQL builder — pure unit
# ────────────────────────────────────────────────────────────────────────────


class TestTypeaheadJQL:
    def test_empty(self):
        # Not tested directly; typeahead() short-circuits on empty.
        pass

    def test_bare_alpha_is_free_text(self):
        # "LLM" is ambiguous (project name vs. word in a title). We treat
        # it as free-text and let the user add "-" to opt into key mode.
        jql = _typeahead_jql("LLM")
        assert "summary ~" in jql
        assert "project =" not in jql

    def test_key_dash_no_digits(self):
        # User typed "LLM-" — project filter, no key match yet
        assert _typeahead_jql("LLM-") == 'project = "LLM" ORDER BY updated DESC'

    def test_key_full(self):
        jql = _typeahead_jql("LLM-12")
        assert 'key = "LLM-12"' in jql
        assert 'project = "LLM"' in jql
        assert 'summary ~ "LLM-12*"' in jql

    def test_bare_alpha_long_word_is_free_text(self):
        # Critical regression test — "checkout" must NOT be misread as
        # project="CHECKOUT". This used to return zero results.
        jql = _typeahead_jql("checkout")
        assert 'summary ~ "checkout*"' in jql
        assert "project =" not in jql

    def test_free_text_long(self):
        jql = _typeahead_jql("checkout flow")
        assert "summary ~" in jql and "text ~" in jql
        assert "checkout flow*" in jql

    def test_free_text_short_skips_text_index(self):
        # text ~ requires >= 3 chars in some indexes; we fall back to summary
        jql = _typeahead_jql("ab")
        assert "text ~" not in jql
        assert "summary ~" in jql

    def test_quotes_escaped(self):
        jql = _typeahead_jql('say "hi"')
        assert 'say \\"hi\\"' in jql


# ────────────────────────────────────────────────────────────────────────────
# HTTP calls — respx mocks
# ────────────────────────────────────────────────────────────────────────────


@respx.mock(assert_all_called=True)
async def test_search_jql_posts_correct_body(respx_mock, client):
    route = respx_mock.post(f"{SITE}/rest/api/3/search/jql").mock(
        return_value=Response(200, json={"issues": [], "isLast": True})
    )
    out = await client.search_jql("project = LLM", max_results=5)
    assert out == {"issues": [], "isLast": True}
    call = route.calls.last
    body = call.request.read()
    assert b'"jql":"project = LLM"' in body
    assert b'"maxResults":5' in body
    assert b'"summary"' in body  # default fields included


@respx.mock(assert_all_called=True)
async def test_search_jql_passes_pagination_cursor(respx_mock, client):
    route = respx_mock.post(f"{SITE}/rest/api/3/search/jql").mock(
        return_value=Response(200, json={"issues": []})
    )
    await client.search_jql("anything", next_page_token="abc123")
    body = route.calls.last.request.read()
    assert b'"nextPageToken":"abc123"' in body


@respx.mock(assert_all_called=True)
async def test_typeahead_returns_summaries(respx_mock, client):
    respx_mock.post(f"{SITE}/rest/api/3/search/jql").mock(
        return_value=Response(200, json={
            "issues": [
                {"key": "LLM-1", "fields": {"summary": "First", "status": {"name": "Open"}, "issuetype": {"name": "Task"}}},
                {"key": "LLM-2", "fields": {"summary": "Second", "status": {"name": "In Progress"}, "issuetype": {"name": "Story"}}},
            ]
        })
    )
    out = await client.typeahead("LLM")
    assert [t.key for t in out] == ["LLM-1", "LLM-2"]
    assert out[1].status == "In Progress"
    assert out[1].issuetype == "Story"


async def test_typeahead_empty_returns_no_call(client):
    out = await client.typeahead("")
    assert out == []


@respx.mock(assert_all_called=True)
async def test_get_issue(respx_mock, client):
    respx_mock.get(f"{SITE}/rest/api/3/issue/LLM-1").mock(
        return_value=Response(200, json={"key": "LLM-1", "fields": {"summary": "X"}})
    )
    out = await client.get_issue("LLM-1")
    assert out["key"] == "LLM-1"


@respx.mock(assert_all_called=True)
async def test_transitions_list(respx_mock, client):
    respx_mock.get(f"{SITE}/rest/api/3/issue/LLM-1/transitions").mock(
        return_value=Response(200, json={"transitions": [
            {"id": "21", "name": "Start", "to": {"name": "In Progress"}},
            {"id": "31", "name": "Done", "to": {"name": "Done"}},
        ]})
    )
    ts = await client.transitions("LLM-1")
    assert ts[0]["id"] == "21"


@respx.mock(assert_all_called=True)
async def test_set_status_finds_transition(respx_mock, client):
    respx_mock.get(f"{SITE}/rest/api/3/issue/LLM-1/transitions").mock(
        return_value=Response(200, json={"transitions": [
            {"id": "21", "name": "Start", "to": {"name": "In Progress"}},
            {"id": "31", "name": "Done", "to": {"name": "Code Review"}},
        ]})
    )
    posted = respx_mock.post(f"{SITE}/rest/api/3/issue/LLM-1/transitions").mock(
        return_value=Response(204)
    )
    tid = await client.set_status("LLM-1", "Code Review")
    assert tid == "31"
    body = posted.calls.last.request.read()
    assert b'"id":"31"' in body


@respx.mock(assert_all_called=True)
async def test_set_status_raises_when_no_transition(respx_mock, client):
    respx_mock.get(f"{SITE}/rest/api/3/issue/LLM-1/transitions").mock(
        return_value=Response(200, json={"transitions": [
            {"id": "21", "name": "Start", "to": {"name": "In Progress"}},
        ]})
    )
    with pytest.raises(JiraError):
        await client.set_status("LLM-1", "Code Review")


@respx.mock(assert_all_called=True)
async def test_add_comment_sends_adf_body(respx_mock, client):
    posted = respx_mock.post(f"{SITE}/rest/api/3/issue/LLM-1/comment").mock(
        return_value=Response(201, json={"id": "10001"})
    )
    adf = {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": "hi"}]}]}
    out = await client.add_comment("LLM-1", adf)
    assert out["id"] == "10001"
    body = posted.calls.last.request.read()
    assert b'"body":' in body
    assert b'"type":"doc"' in body


@respx.mock(assert_all_called=True)
async def test_set_flag(respx_mock, client):
    posted = respx_mock.post(f"{SITE}/rest/greenhopper/1.0/xboard/issue/flag/flag.json").mock(
        return_value=Response(200, json={})
    )
    await client.set_flag("LLM-1", flagged=True, comment="why")
    body = posted.calls.last.request.read()
    assert b'"issueKeys":["LLM-1"]' in body
    assert b'"flag":true' in body
    assert b'"comment":"why"' in body


@respx.mock(assert_all_called=True)
async def test_link_action_item(respx_mock, client):
    posted = respx_mock.post(f"{SITE}/rest/api/3/issueLink").mock(
        return_value=Response(201, json={})
    )
    await client.link_action_item("LLM-1", "LLM-2")
    body = posted.calls.last.request.read()
    assert b'"name":"Action item"' in body
    assert b'"LLM-1"' in body and b'"LLM-2"' in body


@respx.mock(assert_all_called=True)
async def test_error_mapping(respx_mock, client):
    respx_mock.get(f"{SITE}/rest/api/3/issue/NOPE").mock(
        return_value=Response(404, text='{"errorMessages":["Issue Does Not Exist"]}')
    )
    with pytest.raises(JiraError) as exc:
        await client.get_issue("NOPE")
    assert exc.value.status == 404
    assert "Issue Does Not Exist" in (exc.value.body or "")


def test_constructor_validates_site():
    with pytest.raises(ValueError):
        JiraClient("", "u", "tok")


def test_strips_trailing_slash():
    c = JiraClient("https://x.atlassian.net/", "u", "tok")
    assert c._site == "https://x.atlassian.net"
