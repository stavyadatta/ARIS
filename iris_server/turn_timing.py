"""Per-turn latency measurement for the Iris request path.

One turn is one ProcessAudioImg RPC. Spans nest, are kept in call order so the
block reads as a timeline, and are emitted both as human-readable lines and as
one JSON object per turn, so a session can be aggregated without parsing print
output.

This lives at the package root rather than under utils/ because every layer
imports it -- core_api, utils and media_manager alike -- while utils/__init__
constructs the Neo4j driver on import. Instrumentation must not drag that into
modules that do not otherwise need it.

Measurement must never break a turn: with no active turn every entry point is a
no-op, and emission failures are reported rather than raised.
"""

import json
import os
import threading
from contextlib import contextmanager
from time import perf_counter

DEFAULT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "turn_timing.jsonl"
)
TIMING_LOG_PATH = os.environ.get("IRIS_TIMING_LOG", DEFAULT_LOG_PATH)

MS_PER_SECOND = 1000
NAME_COLUMN_WIDTH = 38

# gRPC serves each RPC on its own worker thread and drives that request's
# response generator on the same thread, so the active turn is thread state.
_active = threading.local()
_log_file_lock = threading.Lock()


def _current_turn():
    return getattr(_active, "turn", None)


def _elapsed_ms_since(started_at):
    return (perf_counter() - started_at) * MS_PER_SECOND


class _Turn:
    """Ordered spans and facts for one request."""

    def __init__(self, label):
        self.label = label
        self.started_at = perf_counter()
        self.total_ms = 0.0
        self.spans = []
        self.facts = {}
        self._open_span_count = 0

    def open_span(self, name):
        """Reserve this span's place so the report reads parent-before-child."""
        depth = self._open_span_count
        self._open_span_count += 1
        self.spans.append({"name": name, "depth": depth, "ms": 0.0})
        return depth, len(self.spans) - 1

    def close_span(self, depth, position, elapsed_ms):
        self._open_span_count = depth
        self.spans[position]["ms"] = round(elapsed_ms, 1)

    def record_fact(self, name, value):
        self.facts[name] = value

    def has_fact(self, name):
        return name in self.facts

    def elapsed_ms(self):
        return round(_elapsed_ms_since(self.started_at), 1)

    def finish(self):
        self.total_ms = self.elapsed_ms()


@contextmanager
def turn(label):
    """Measure one request, emitting the breakdown when it ends.

    A nested call keeps the outermost turn so a helper that measures itself
    cannot split a turn in two.
    """
    if _current_turn() is not None:
        yield
        return

    record = _Turn(label)
    _active.turn = record
    try:
        yield
    finally:
        _active.turn = None
        record.finish()
        _emit(record)


@contextmanager
def span(name):
    """Measure one section of the active turn."""
    record = _current_turn()
    if record is None:
        yield
        return

    depth, position = record.open_span(name)
    started_at = perf_counter()
    try:
        yield
    finally:
        record.close_span(depth, position, _elapsed_ms_since(started_at))


def mark(name, value):
    """Attach a non-timing fact, such as the route the turn took."""
    record = _current_turn()
    if record is not None:
        record.record_fact(name, value)


def record_first_moment(name):
    """Attach how far into the turn something first happened.

    Later calls are ignored, so a per-chunk call site records time-to-first
    chunk rather than time-to-last.
    """
    record = _current_turn()
    if record is not None and not record.has_fact(name):
        record.record_fact(name, record.elapsed_ms())


def _emit(record):
    try:
        _print_breakdown(record)
        _append_json_line(record)
    except Exception as e:
        print(f"[timing] could not emit breakdown: {e}")


def _print_breakdown(record):
    print(f"[timing] {record.label} total={record.total_ms}ms")
    for entry in record.spans:
        print(_span_line(entry, record.total_ms))
    if record.facts:
        print(f"[timing] facts {_format_facts(record.facts)}")


def _span_line(entry, total_ms):
    label = "  " * (entry["depth"] + 1) + entry["name"]
    share = entry["ms"] / total_ms * 100 if total_ms else 0
    return f"[timing] {label:<{NAME_COLUMN_WIDTH}} {entry['ms']:>8.1f}ms {share:>5.1f}%"


def _format_facts(facts):
    return " ".join(f"{name}={value}" for name, value in facts.items())


def _append_json_line(record):
    line = json.dumps({
        "label": record.label,
        "total_ms": record.total_ms,
        "spans": record.spans,
        "facts": record.facts,
    })
    os.makedirs(os.path.dirname(TIMING_LOG_PATH), exist_ok=True)
    with _log_file_lock:
        with open(TIMING_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")
