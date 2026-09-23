"""End-to-end tests for the read-only per-source gap scan."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        yield c, db_path


def create_session(client, sources=("a", "b"), lateness=0.0, session_id="s"):
    r = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "source_ids": list(sources),
            "allowed_lateness_seconds": lateness,
        },
    )
    assert r.status_code == 201, r.text
    return session_id


def report(client, sid, readings):
    return client.post(f"/sessions/{sid}/readings", json={"readings": readings})


def seal(client, sid, source, final_observed_at):
    return client.post(
        f"/sessions/{sid}/sources/{source}/seal",
        json={"final_observed_at": final_observed_at},
    )


def gaps(client, sid, **params):
    return client.get(f"/sessions/{sid}/source-gaps", params=params)


def setup_two_source_stream(client, sid="s"):
    """Publish eight readings across two sources (lateness 0):

    a: 10:00:00, 10:00:10, 10:01:20, 10:01:30   (one 70s gap)
    b: 10:00:00, 10:00:05, 10:02:00, 10:02:10   (one 115s gap)

    The clock-advancing reports a#100/b#100@10:10:00 stay buffered.
    """
    create_session(client, session_id=sid)
    r = report(client, sid, [
        {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": "a1"},
        {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": "b1"},
        {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:00:10Z", "value": "a2"},
        {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:00:05Z", "value": "b2"},
        {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T10:01:20Z", "value": "a3"},
        {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T10:02:00Z", "value": "b3"},
        {"source_id": "a", "seq": 4, "observed_at": "2026-01-01T10:01:30Z", "value": "a4"},
        {"source_id": "b", "seq": 4, "observed_at": "2026-01-01T10:02:10Z", "value": "b4"},
        {"source_id": "a", "seq": 100, "observed_at": "2026-01-01T10:10:00Z", "value": "ac"},
        {"source_id": "b", "seq": 100, "observed_at": "2026-01-01T10:10:00Z", "value": "bc"},
    ])
    assert r.json()["published_total"] == 8, r.text
    return sid


class TestGapDetection:
    def test_gaps_detected_per_source(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        r = gaps(c, sid, max_gap_seconds="60")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == sid
        assert body["source_id"] is None
        assert body["from"] is None
        assert body["to"] is None
        assert body["max_gap_seconds"] == 60.0
        assert body["limit"] == 100
        assert body["matched"] == 2
        assert body["returned"] == 2
        assert body["has_more"] is False
        assert body["next_cursor"] is None

        # Deterministic order: (source_id, start key) -> a's gap first.
        a_gap, b_gap = body["gaps"]
        assert a_gap["source_id"] == "a"
        assert a_gap["start_key"] == {
            "observed_at": "2026-01-01T10:00:10Z", "source_id": "a", "seq": 2}
        assert a_gap["end_key"] == {
            "observed_at": "2026-01-01T10:01:20Z", "source_id": "a", "seq": 3}
        assert a_gap["gap_seconds"] == 70.0
        assert b_gap["source_id"] == "b"
        assert b_gap["start_key"] == {
            "observed_at": "2026-01-01T10:00:05Z", "source_id": "b", "seq": 2}
        assert b_gap["end_key"] == {
            "observed_at": "2026-01-01T10:02:00Z", "source_id": "b", "seq": 3}
        assert b_gap["gap_seconds"] == 115.0

    def test_strictly_greater_threshold(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        # a's gap is exactly 70s: not reported at threshold 70.
        body = gaps(c, sid, max_gap_seconds="70").json()
        assert [g["source_id"] for g in body["gaps"]] == ["b"]
        # b's gap is exactly 115s: not reported at threshold 115.
        body = gaps(c, sid, max_gap_seconds="115").json()
        assert body["matched"] == 0
        assert body["gaps"] == []

    def test_microsecond_boundary(self, client):
        c, _ = client
        create_session(c)
        r = report(c, "s", [
            # a: gap of exactly 60.000001 seconds -> strictly greater than 60.
            {"source_id": "a", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T10:01:00.000001Z", "value": 1},
            # b: gap of exactly 60 seconds -> not a gap.
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T10:01:00Z", "value": 1},
            {"source_id": "a", "seq": 100, "observed_at": "2026-01-01T10:10:00Z", "value": 1},
            {"source_id": "b", "seq": 100, "observed_at": "2026-01-01T10:10:00Z", "value": 1},
        ])
        assert r.json()["published_total"] == 4, r.text
        body = gaps(c, "s", max_gap_seconds="60").json()
        assert body["matched"] == 1
        assert body["gaps"][0]["source_id"] == "a"
        assert body["gaps"][0]["gap_seconds"] == pytest.approx(60.000001)

    def test_only_published_readings_participate(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        # A late reading lands in the quarantine; buffered clock readings stay
        # buffered.  Neither may appear in (or extend) the gap scan.
        r = report(c, sid, [
            {"source_id": "a", "seq": 50, "observed_at": "2026-01-01T10:00:01Z", "value": "late"},
        ])
        assert r.json()["quarantined"] == 1, r.text
        body = gaps(c, sid, max_gap_seconds="60").json()
        assert body["matched"] == 2
        keys = [(g["start_key"]["seq"], g["end_key"]["seq"]) for g in body["gaps"]]
        assert keys == [(2, 3), (2, 3)]
        # The buffered a#100@10:10:00 must not extend a's last gap.
        assert all(g["end_key"]["seq"] != 100 for g in body["gaps"])

    def test_empty_result_shape(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        body = gaps(c, sid, max_gap_seconds="100000").json()
        assert body["matched"] == 0
        assert body["returned"] == 0
        assert body["has_more"] is False
        assert body["next_cursor"] is None
        assert body["gaps"] == []

    def test_same_event_time_orders_by_seq(self, client):
        c, _ = client
        create_session(c)
        r = report(c, "s", [
            {"source_id": "a", "seq": 5, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "a", "seq": 7, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T10:02:00Z", "value": 1},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T10:00:00Z", "value": 1},
            {"source_id": "a", "seq": 100, "observed_at": "2026-01-01T10:10:00Z", "value": 1},
            {"source_id": "b", "seq": 100, "observed_at": "2026-01-01T10:10:00Z", "value": 1},
        ])
        assert r.json()["published_total"] == 4, r.text
        body = gaps(c, "s", max_gap_seconds="60").json()
        assert body["matched"] == 1
        gap = body["gaps"][0]
        # The predecessor in (observed_at, seq) order is seq=7, not seq=5.
        assert gap["start_key"]["seq"] == 7
        assert gap["end_key"]["seq"] == 9
        assert gap["gap_seconds"] == 120.0

    def test_closed_session_still_scannable(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        assert seal(c, sid, "a", "2026-01-01T10:10:00Z").status_code == 200
        r = seal(c, sid, "b", "2026-01-01T10:10:00Z")
        assert r.json()["closed"] is True, r.text
        # The tail flush published a#100/b#100@10:10:00, so each source now
        # shows its original gap plus the gap to the tail-published reading.
        body = gaps(c, sid, max_gap_seconds="60").json()
        assert body["matched"] == 4
        assert [g["source_id"] for g in body["gaps"]] == ["a", "a", "b", "b"]


class TestFilters:
    def test_source_filter(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        body = gaps(c, sid, max_gap_seconds="60", source_id="b").json()
        assert body["source_id"] == "b"
        assert body["matched"] == 1
        assert body["gaps"][0]["source_id"] == "b"
        assert body["gaps"][0]["gap_seconds"] == 115.0

    def test_unknown_source_rejected(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        r = gaps(c, sid, max_gap_seconds="60", source_id="zzz")
        assert r.status_code == 400, r.text
        assert "unknown source_id" in r.json()["detail"]

    def test_blank_source_rejected(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        assert gaps(c, sid, max_gap_seconds="60", source_id="").status_code == 400
        assert gaps(c, sid, max_gap_seconds="60", source_id="  ").status_code == 400

    def test_from_to_closed_interval(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        # Both boundary readings participate (inclusive): a's 70s gap between
        # exactly `from` and exactly `to` is reported.
        body = gaps(
            c, sid, max_gap_seconds="60",
            **{"from": "2026-01-01T10:00:10Z", "to": "2026-01-01T10:01:20Z"},
        ).json()
        assert body["from"] == "2026-01-01T10:00:10Z"
        assert body["to"] == "2026-01-01T10:01:20Z"
        assert body["matched"] == 1
        assert body["gaps"][0]["source_id"] == "a"
        assert body["gaps"][0]["gap_seconds"] == 70.0

    def test_interval_redefines_participation(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        # Excluding b#2@10:00:05 removes b's 115s gap (its remaining readings
        # are 10s apart); a's gap is cut off by the same lower bound.
        body = gaps(
            c, sid, max_gap_seconds="60", **{"from": "2026-01-01T10:00:11Z"}
        ).json()
        assert body["matched"] == 0
        # Cutting b#3@10:02:00 off removes b's gap while a's 70s gap survives.
        body = gaps(
            c, sid, max_gap_seconds="60", **{"to": "2026-01-01T10:01:59Z"}
        ).json()
        assert [g["source_id"] for g in body["gaps"]] == ["a"]

    def test_from_after_to_rejected(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        r = gaps(
            c, sid, max_gap_seconds="60",
            **{"from": "2026-01-02T00:00:00Z", "to": "2026-01-01T00:00:00Z"},
        )
        assert r.status_code == 400, r.text
        assert "must not be later than" in r.json()["detail"]

    def test_invalid_time_bounds_rejected(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        assert gaps(c, sid, max_gap_seconds="60", **{"from": "junk"}).status_code == 400
        assert gaps(c, sid, max_gap_seconds="60", to="junk").status_code == 400
        assert gaps(c, sid, max_gap_seconds="60", **{"from": ""}).status_code == 400


class TestParameterValidation:
    def test_max_gap_seconds_required(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        r = gaps(c, sid)
        assert r.status_code == 400, r.text
        assert "max_gap_seconds" in r.json()["detail"]

    @pytest.mark.parametrize("bad", ["abc", "0", "-5", "nan", "inf", "-inf", "", "  ", "1.5.2"])
    def test_max_gap_seconds_invalid(self, client, bad):
        c, _ = client
        sid = setup_two_source_stream(c)
        r = gaps(c, sid, max_gap_seconds=bad)
        assert r.status_code == 400, (bad, r.text)

    @pytest.mark.parametrize("good", ["60", "60.5", "1e2", "0.000001"])
    def test_max_gap_seconds_valid_forms(self, client, good):
        c, _ = client
        sid = setup_two_source_stream(c)
        r = gaps(c, sid, max_gap_seconds=good)
        assert r.status_code == 200, (good, r.text)

    @pytest.mark.parametrize("bad", ["0", "-1", "1001", "abc", "1.5", ""])
    def test_limit_invalid(self, client, bad):
        c, _ = client
        sid = setup_two_source_stream(c)
        r = gaps(c, sid, max_gap_seconds="60", limit=bad)
        assert r.status_code == 400, (bad, r.text)

    def test_limit_boundaries(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        assert gaps(c, sid, max_gap_seconds="60", limit="1").status_code == 200
        assert gaps(c, sid, max_gap_seconds="60", limit="1000").status_code == 200

    def test_unknown_session_404(self, client):
        c, _ = client
        r = gaps(c, "nope", max_gap_seconds="60")
        assert r.status_code == 404, r.text

    def test_malformed_cursor_rejected(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        assert gaps(c, sid, max_gap_seconds="60", cursor="!!!").status_code == 400
        # Valid base64/JSON but wrong shape or part types.
        from app.cursor import encode_cursor

        for bad in [("a",), ("a", "ts"), ("a", "ts", "notint"), (1, 2, 3),
                    ("a", "ts", True), ("a", "ts", 1, "extra")]:
            r = gaps(c, sid, max_gap_seconds="60", cursor=encode_cursor(bad))
            assert r.status_code == 400, (bad, r.text)


class TestPagination:
    def test_keyset_pagination_concatenates(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        page1 = gaps(c, sid, max_gap_seconds="60", limit="1").json()
        assert page1["matched"] == 2
        assert page1["returned"] == 1
        assert page1["has_more"] is True
        assert page1["next_cursor"] is not None
        assert [g["source_id"] for g in page1["gaps"]] == ["a"]

        page2 = gaps(
            c, sid, max_gap_seconds="60", limit="1", cursor=page1["next_cursor"]
        ).json()
        # `matched` counts the filters, not the remaining page window.
        assert page2["matched"] == 2
        assert page2["returned"] == 1
        assert page2["has_more"] is False
        assert page2["next_cursor"] is None
        assert [g["source_id"] for g in page2["gaps"]] == ["b"]

        unpaginated = gaps(c, sid, max_gap_seconds="60").json()
        assert page1["gaps"] + page2["gaps"] == unpaginated["gaps"]

    def test_pagination_with_filters_stays_deterministic(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        params = {"max_gap_seconds": "100", "limit": "1",
                  "from": "2026-01-01T10:00:00Z", "to": "2026-01-01T10:05:00Z"}
        page1 = gaps(c, sid, **params).json()
        assert page1["matched"] == 1  # only b's 115s gap lies in the window
        assert page1["has_more"] is False

        # A threshold that yields both gaps, paged one at a time with a
        # source-agnostic filter, must produce the same order on every page.
        params = {"max_gap_seconds": "60", "limit": "1"}
        seen = []
        cursor = None
        while True:
            q = dict(params)
            if cursor:
                q["cursor"] = cursor
            body = gaps(c, sid, **q).json()
            seen.extend(g["source_id"] for g in body["gaps"])
            if not body["has_more"]:
                break
            cursor = body["next_cursor"]
        assert seen == ["a", "b"]

    def test_cursor_beyond_last_gap_returns_empty_page(self, client):
        c, _ = client
        sid = setup_two_source_stream(c)
        page1 = gaps(c, sid, max_gap_seconds="60", limit="1").json()
        page2 = gaps(
            c, sid, max_gap_seconds="60", limit="1", cursor=page1["next_cursor"]
        ).json()
        assert page2["has_more"] is False
        # Reusing the last cursor is a stable empty page, not an error.
        page3 = gaps(
            c, sid, max_gap_seconds="60", limit="1",
            cursor=page1["next_cursor"],
        ).json()
        assert page3["gaps"] == page2["gaps"]


class TestReadOnly:
    def test_calls_do_not_mutate_anything(self, client):
        c, db_path = client
        sid = setup_two_source_stream(c)
        # Give the session alert state, a correlated rule, a checkpoint and a
        # quarantined reading, then snapshot every read surface.
        r = c.post(f"/sessions/{sid}/alert-rules", json={
            "rule_id": "a-high", "source_id": "a", "direction": "above",
            "enter_threshold": 5.0, "recover_threshold": 1.0,
            "enter_consecutive": 1, "recover_consecutive": 1,
        })
        assert r.status_code == 201, r.text
        r = c.post(f"/sessions/{sid}/checkpoints", json={"name": "consumer-1"})
        assert r.status_code == 201, r.text
        report(c, sid, [
            {"source_id": "a", "seq": 50, "observed_at": "2026-01-01T10:00:01Z", "value": "late"},
        ])

        surfaces = [
            f"/sessions/{sid}",
            f"/sessions/{sid}/stream?limit=1000",
            f"/sessions/{sid}/quarantine?limit=1000",
            f"/sessions/{sid}/alert-rules",
            f"/sessions/{sid}/alerts/history?limit=1000",
            f"/sessions/{sid}/checkpoints",
        ]
        before = {url: c.get(url).json() for url in surfaces}

        for _ in range(3):
            r = gaps(c, sid, max_gap_seconds="60")
            assert r.status_code == 200, r.text
            r = gaps(c, sid, max_gap_seconds="1", source_id="a", limit="1")
            assert r.status_code == 200, r.text

        after = {url: c.get(url).json() for url in surfaces}
        assert before == after

    def test_no_writes_reach_the_database(self, client):
        c, db_path = client
        sid = setup_two_source_stream(c)
        r = gaps(c, sid, max_gap_seconds="60")
        assert r.status_code == 200, r.text
        # A read-only endpoint leaves the schema (and every table) untouched.
        conn = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            conn.close()
        assert "source_gaps" not in tables
