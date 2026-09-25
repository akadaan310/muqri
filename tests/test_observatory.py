"""The Observatory is read-only, token-gated, and serves only round 1-5 recordings."""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")


@pytest.fixture()
def client(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    import observatory.server as srv
    monkeypatch.setattr(srv, "TOKEN_FILE", tmp_path / "token")
    (tmp_path / "token").write_text("t0ken-for-tests")
    return TestClient(srv.create_app(), base_url="https://testserver")


def test_no_token_no_access(client) -> None:  # type: ignore[no-untyped-def]
    assert client.get("/muqri-observatory").status_code == 401


def test_writes_are_refused_even_with_the_token(client) -> None:  # type: ignore[no-untyped-def]
    client.cookies.set("obs", "t0ken-for-tests")
    for path in ("/sessions/submit", "/review/label", "/analyze", "/muqri-observatory"):
        assert client.post(path).status_code == 405


def test_token_link_sets_a_cookie_then_pages_serve(client) -> None:  # type: ignore[no-untyped-def]
    r = client.get("/muqri-observatory?token=t0ken-for-tests", follow_redirects=False)
    assert r.status_code == 303 and "token" not in r.headers["location"]
    client.cookies.set("obs", "t0ken-for-tests")
    assert client.get("/api/observatory/manifest").status_code == 200


def test_only_rounds_one_to_five_and_no_path_escape(client) -> None:  # type: ignore[no-untyped-def]
    client.cookies.set("obs", "t0ken-for-tests")
    assert client.get("/sessions/round/6").status_code == 404
    assert client.get("/observatory/audio/r6e1/A/x.m4a").status_code == 404
    assert client.get("/observatory/audio/r5e1/A/..%2F..%2F..%2Fsessions_results.jsonl").status_code == 404
    assert client.get("/observatory/audio/r5e1/C/x.m4a").status_code == 404
