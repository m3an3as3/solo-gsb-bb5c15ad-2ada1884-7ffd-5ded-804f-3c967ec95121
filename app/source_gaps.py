"""Read-only per-source gap scan over the published timeline.

``GET /sessions/{session_id}/source-gaps`` answers, inside one SQLite read
snapshot: "for each declared source (or one selected source), where do two
consecutively published readings of that source sit **strictly more** than
``max_gap_seconds`` apart in event time?"  Nothing is mutated: the scan only
reads, so watermarks, alert state machines and checkpoints are untouched and
repeated calls are idempotent.

Semantics:

* Only ``published`` readings participate; buffered and quarantined rows
  never form gap boundaries.
* Each source is scanned independently in the deterministic per-source order
  ``(observed_at, seq)`` (``seq`` is unique per source, so the order is
  total).  A *gap* is a pair of adjacent readings in that order whose
  event-time distance is strictly greater than ``max_gap_seconds`` -- a pair
  exactly on the threshold is not a gap.
* ``from`` / ``to`` are inclusive event-time bounds restricting which
  published readings participate (``from <= observed_at <= to``); adjacency
  is then computed among the remaining readings of each source.
* The result order ``(source_id, start observed_at, start seq)`` is total:
  per source a reading starts at most one gap, so the start-reading keyset
  uniquely identifies a gap.  Keyset pagination over that order is stable
  across pages, and filters do not change it.

The strict comparison is computed with exact ``timedelta`` arithmetic in
Python (microsecond resolution), never with floating-point julian-day SQL,
so a gap of exactly ``max_gap_seconds`` plus one microsecond is reported
while exactly ``max_gap_seconds`` is not.
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import timedelta
from typing import Any, Optional

from .cursor import decode_cursor as _decode_cursor
from .cursor import encode_cursor as _encode_cursor
from .database import read_transaction
from .errors import BadRequestError
from .service import _load_session
from .time_utils import format_event_time, parse_event_time, store_event_time

__all__ = ["list_source_gaps", "DEFAULT_LIMIT", "MAX_LIMIT"]

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000


# ---------------------------------------------------------------------------
# request validation (every semantic error must surface as HTTP 400)
# ---------------------------------------------------------------------------

def _validate_max_gap_seconds(value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise BadRequestError(
            "max_gap_seconds is required and must be a positive finite number"
        )
    if isinstance(value, str) and not value.strip():
        raise BadRequestError(
            "max_gap_seconds is required and must be a positive finite number"
        )
    try:
        gap = float(value)
    except (TypeError, ValueError) as exc:
        raise BadRequestError(
            "max_gap_seconds must be a positive finite number"
        ) from exc
    if not math.isfinite(gap) or gap <= 0:
        raise BadRequestError("max_gap_seconds must be a positive finite number")
    return gap


def _validate_limit(value: Any) -> int:
    if value is None:
        return DEFAULT_LIMIT
    # ``bool`` is a subclass of ``int``; never silently accept it as 0/1.
    if isinstance(value, bool):
        raise BadRequestError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    if isinstance(value, int):
        limit_value = value
    elif isinstance(value, str):
        try:
            limit_value = int(value.strip())
        except ValueError as exc:
            raise BadRequestError(
                f"limit must be an integer between 1 and {MAX_LIMIT}"
            ) from exc
    else:
        raise BadRequestError(f"limit must be an integer between 1 and {MAX_LIMIT}")
    if not 1 <= limit_value <= MAX_LIMIT:
        raise BadRequestError(f"limit must be between 1 and {MAX_LIMIT}")
    return limit_value


def _optional_time(value: Any, field: str) -> Optional[str]:
    """Validate/normalise an optional inclusive event-time bound."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BadRequestError(f"{field} must be a non-empty ISO 8601 string")
    # TimestampError is mapped to HTTP 400 by the application handler.
    return store_event_time(parse_event_time(value))


def _decode_gap_cursor(cursor: str) -> tuple[str, str, int]:
    """Decode a pagination cursor and check the keyset part types.

    The parts are compared in Python (not bound into SQL), so a cursor whose
    parts have the wrong types must be rejected here instead of blowing up
    the tuple comparison with a TypeError.
    """
    parts = _decode_cursor(cursor, 3)
    source_id, observed_at, seq = parts
    if (
        not isinstance(source_id, str)
        or not isinstance(observed_at, str)
        or isinstance(seq, bool)
        or not isinstance(seq, int)
    ):
        raise BadRequestError("malformed cursor")
    return source_id, observed_at, seq


# ---------------------------------------------------------------------------
# serialisation helpers
# ---------------------------------------------------------------------------

def _key_out(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "observed_at": format_event_time(parse_event_time(row["observed_at"])),
        "source_id": row["source_id"],
        "seq": row["seq"],
    }


# ---------------------------------------------------------------------------
# gap scan
# ---------------------------------------------------------------------------

def list_source_gaps(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    max_gap_seconds: Any = None,
    source_id: Any = None,
    from_: Any = None,
    to: Any = None,
    limit: Any = None,
    cursor: Any = None,
) -> dict[str, Any]:
    """Find per-source event-time gaps in the published timeline.

    Read-only: every row comes from one deferred read snapshot and no table
    is ever written.  Malformed parameters raise
    :class:`~app.errors.BadRequestError` / :class:`TimestampError` (400); an
    unknown session raises 404; an undeclared ``source_id`` raises 400.
    """
    max_gap = _validate_max_gap_seconds(max_gap_seconds)
    limit_value = _validate_limit(limit)
    from_store = _optional_time(from_, "from")
    to_store = _optional_time(to, "to")
    if from_store is not None and to_store is not None and from_store > to_store:
        raise BadRequestError(
            f"'from' ({format_event_time(parse_event_time(from_store))}) must not"
            f" be later than 'to' ({format_event_time(parse_event_time(to_store))})"
        )
    if source_id is not None and (
        not isinstance(source_id, str) or not source_id.strip()
    ):
        raise BadRequestError("source_id must be a non-empty string")
    if cursor is not None and not isinstance(cursor, str):
        raise BadRequestError("malformed cursor")
    cursor_key = _decode_gap_cursor(cursor) if cursor else None

    where = "session_id = ? AND status = 'published'"
    params: list[Any] = [session_id]
    if source_id is not None:
        where += " AND source_id = ?"
        params.append(source_id)
    if from_store is not None:
        where += " AND observed_at >= ?"
        params.append(from_store)
    if to_store is not None:
        where += " AND observed_at <= ?"
        params.append(to_store)

    with read_transaction(conn):
        session = _load_session(conn, session_id)
        if source_id is not None:
            allowed_sources = set(json.loads(session["source_ids"]))
            if source_id not in allowed_sources:
                raise BadRequestError(
                    f"unknown source_id {source_id!r} for this session;"
                    f" declared sources: {sorted(allowed_sources)}"
                )
        # One ordered scan of the snapshot: consecutive rows of the same
        # source are exactly the adjacent published pairs of that source.
        rows = conn.execute(
            f"SELECT source_id, observed_at, seq FROM readings WHERE {where}"
            " ORDER BY source_id, observed_at, seq",
            params,
        ).fetchall()

    # Exact timedelta comparison: strictly greater than the threshold.
    threshold = timedelta(seconds=max_gap)
    gaps: list[tuple[sqlite3.Row, sqlite3.Row, float]] = []
    for prev, cur in zip(rows, rows[1:]):
        if cur["source_id"] != prev["source_id"]:
            continue
        delta = parse_event_time(cur["observed_at"]) - parse_event_time(
            prev["observed_at"]
        )
        if delta > threshold:
            gaps.append((prev, cur, delta.total_seconds()))

    # Total gaps matching the filters (pre-pagination): stable across pages.
    matched = len(gaps)

    # The scan order (source_id, observed_at, seq) already orders gaps by
    # their start-reading keyset, which is unique per gap: keyset pagination
    # is a deterministic filter on that order.
    if cursor_key is not None:
        gaps = [
            g
            for g in gaps
            if (g[0]["source_id"], g[0]["observed_at"], g[0]["seq"]) > cursor_key
        ]

    page = gaps[:limit_value]
    has_more = len(gaps) > limit_value
    next_cursor = None
    if has_more and page:
        last = page[-1][0]
        next_cursor = _encode_cursor(
            (last["source_id"], last["observed_at"], last["seq"])
        )

    return {
        "session_id": session_id,
        "source_id": source_id,
        "from": (
            format_event_time(parse_event_time(from_store))
            if from_store is not None
            else None
        ),
        "to": (
            format_event_time(parse_event_time(to_store))
            if to_store is not None
            else None
        ),
        "max_gap_seconds": max_gap,
        "limit": limit_value,
        "matched": matched,
        "returned": len(page),
        "has_more": has_more,
        "next_cursor": next_cursor,
        "gaps": [
            {
                "source_id": prev["source_id"],
                "start_key": _key_out(prev),
                "end_key": _key_out(cur),
                "gap_seconds": gap_seconds,
            }
            for prev, cur, gap_seconds in page
        ],
    }
