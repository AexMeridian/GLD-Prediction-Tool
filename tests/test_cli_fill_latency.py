"""Tests for cli._load_fill_latency_samples -- specifically that a signal
which expired with no user action at all (no fill, no explicit skip) is
still counted as a miss, not silently dropped. See delay_profile.py's
FillLatencySample.delay_s docstring for why that case has no real delay."""

import sqlite3
from datetime import UTC, datetime, timedelta

from gold_edge.cli import _load_fill_latency_samples
from gold_edge.recorder import Recorder

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _insert_signal(conn, signal_id, reason, status, created_at):
    expires_at = created_at + timedelta(seconds=6)
    conn.execute(
        "INSERT INTO signals (id, window_ticker, action, side, limit_price, size, fair, "
        "market_price, edge_after_costs, reason, created_at, expires_at, status) "
        "VALUES (?, 'W1', 'BUY', 'YES', '0.50', '1', 0.55, '0.50', 0.03, ?, ?, ?, ?)",
        (signal_id, reason, created_at.isoformat(), expires_at.isoformat(), status),
    )


def _insert_fill(conn, signal_id, logged_at):
    conn.execute(
        "INSERT INTO fills (signal_id, window_ticker, side, action, price, size, "
        "logged_at, is_skip) VALUES (?, 'W1', 'YES', 'BUY', '0.50', '1', ?, 0)",
        (signal_id, logged_at.isoformat()),
    )


def test_filled_signal_has_a_real_delay(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()
    conn = sqlite3.connect(sqlite_path)
    _insert_signal(conn, "s1", "enter_edge", "FILLED", T0)
    _insert_fill(conn, "s1", T0 + timedelta(seconds=1.5))
    conn.commit()
    conn.close()

    samples = _load_fill_latency_samples(sqlite_path)
    assert len(samples) == 1
    assert samples[0].was_missed is False
    assert samples[0].delay_s == 1.5


def test_expired_signal_with_no_fill_row_counts_as_a_miss(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()
    conn = sqlite3.connect(sqlite_path)
    _insert_signal(conn, "s2", "enter_edge", "EXPIRED", T0)
    conn.commit()
    conn.close()

    samples = _load_fill_latency_samples(sqlite_path)
    assert len(samples) == 1
    assert samples[0].signal_id == "s2"
    assert samples[0].was_missed is True
    assert samples[0].delay_s is None


def test_mix_of_filled_and_silently_expired_signals(tmp_path):
    sqlite_path = tmp_path / "test.sqlite"
    Recorder(sqlite_path).close()
    conn = sqlite3.connect(sqlite_path)
    _insert_signal(conn, "s1", "enter_edge", "FILLED", T0)
    _insert_fill(conn, "s1", T0 + timedelta(seconds=1.0))
    _insert_signal(conn, "s2", "enter_edge", "EXPIRED", T0 + timedelta(seconds=10))
    conn.commit()
    conn.close()

    samples = _load_fill_latency_samples(sqlite_path)
    assert {s.signal_id for s in samples} == {"s1", "s2"}
    missed = {s.signal_id: s for s in samples}["s2"]
    assert missed.was_missed is True
    assert missed.delay_s is None
