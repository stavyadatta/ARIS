"""ANSI color helpers for BRAID logs.

speaker_server.py installs a stream handler (colors on) and a file handler
that strips ANSI via ``_PlainFormatter``, so we can inject escape codes
freely into log messages — the log file stays clean.

Usage:
    from .log_style import C
    logger.info(f"{C.tick}[tick]{C.r} started")

Colors are intentionally module-scoped so scanning stdout reveals which
phase produced a line at a glance.
"""
from __future__ import annotations


class C:
    # reset / styles
    r      = "\033[0m"
    bold   = "\033[1m"
    dim    = "\033[2m"
    # phase / module prefixes
    tick       = "\033[1;36m"   # bold cyan
    perception = "\033[95m"     # magenta
    assoc      = "\033[94m"     # blue
    posterior  = "\033[93m"     # yellow
    decision   = "\033[92m"     # green
    action     = "\033[1;33m"   # bold yellow
    gallery    = "\033[96m"     # cyan
    temporal   = "\033[35m"     # purple
    grpc       = "\033[90m"     # gray
    client     = "\033[1;35m"   # bold magenta
    # severities / states
    ok     = "\033[92m"
    warn   = "\033[93m"
    err    = "\033[91m"
    # decision state palette
    RECOGNISE = "\033[1;92m"   # bold green
    CONFIRM   = "\033[1;93m"   # bold yellow
    ENROL     = "\033[1;96m"   # bold cyan
    EXPLORE   = "\033[1;95m"   # bold magenta
    UNKNOWN   = "\033[1;90m"   # bold gray


def state_color(state_name: str) -> str:
    return getattr(C, state_name, C.r)
