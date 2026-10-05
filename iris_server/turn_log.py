"""What the server prints for a turn, and how loudly.

A turn used to scatter thirty-odd lines across the log -- a raw model response
object, three state names, the same reply printed by two layers -- so the one
thing worth finding, what the person said, was buried. Now a turn prints the
words heard, one line per decision, the words Iris replies with, and a one-line
timing summary. Everything else is `debug`, shown only when IRIS_LOG_VERBOSE=1.

Colour is on by default because the log is read through `docker logs` in a
terminal; set IRIS_LOG_COLOR=0 when piping it to a file.
"""

import os

VERBOSE_ENV = "IRIS_LOG_VERBOSE"
COLOR_ENV = "IRIS_LOG_COLOR"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLACK_ON_YELLOW = "\033[1;30;43m"
BLACK_ON_CYAN = "\033[1;30;46m"

HEARD_TAG = " HEARD "
IRIS_TAG = " IRIS  "


def _enabled(name: str, default: str) -> bool:
    return os.environ.get(name, default) not in ("0", "", "false", "no")


def _paint(style: str, text: str) -> str:
    return f"{style}{text}{RESET}" if _enabled(COLOR_ENV, "1") else text


def is_verbose() -> bool:
    return _enabled(VERBOSE_ENV, "0")


def heard(transcription: str) -> None:
    """The words the person said, as the speech recogniser wrote them."""
    print(f"\n{_paint(BLACK_ON_YELLOW, HEARD_TAG)} "
          f"{_paint(BOLD, repr(transcription.strip()))}", flush=True)


def said(reply: str, action: str) -> None:
    """The words Iris replies with, and the gesture that goes with them."""
    gesture = "" if action in ("", "none") else f"  [{action}]"
    words = reply.strip() or "(silence)"
    print(f"{_paint(BLACK_ON_CYAN, IRIS_TAG)} {_paint(BOLD, repr(words))}{gesture}",
          flush=True)


def step(tag: str, text: str) -> None:
    """One decision the pipeline made, kept short and dim."""
    print(_paint(DIM, f"  [{tag}] {text}"), flush=True)


def debug(text: str) -> None:
    """Detail for diagnosing a problem; hidden unless IRIS_LOG_VERBOSE=1."""
    if is_verbose():
        print(_paint(DIM, f"  (debug) {text}"), flush=True)
