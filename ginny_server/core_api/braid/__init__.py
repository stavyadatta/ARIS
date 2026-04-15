"""BRAID — Bayesian Robotic Audio-Visual Identity Decision Network.

See /workspace/blueprint/BRAID_Blueprint_v1.md and
/workspace/.omc/specs/deep-interview-braid.md for the authoritative spec.

This package is activated via the --braid flag on ginny_server.main. It is
self-contained and does not mutate the existing speaker_recognition modules;
it only *imports* them.
"""

from .config import BraidConfig, load_config  # noqa: F401
