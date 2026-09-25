"""Shared harness for the iris_server smoke tests.

These run as plain scripts because the server image ships no pytest. Both
suites need the same two things: iris_server on the import path, and the heavy
model layer kept out of the way — core_api/__init__.py instantiates Whisper on
a CUDA device at import time, so importing anything under it pulls in GPU
weights the tests do not need.
"""

import os
import sys
import types

# Defaults to where the Dockerfile puts the package; override to run the
# suites against a checkout somewhere else.
IRIS_SERVER_PATH = os.environ.get("IRIS_SERVER_PATH", "/workspace/iris_server")

# Names core_api exports that the code under test may reference.
_MODEL_SINGLETONS = (
    "Llama", "ChatGPT", "Grok", "ClipClassification", "PersonDetectionCropper",
    "YOLODetector", "Claude", "RelationshipChecker", "AttributeFinder", "OCSort",
)


class _InertModel:
    """Absorbs any call an API makes on a model these tests do not exercise.

    Several APIs finish by handing their turn to a model singleton — the
    relationship checker, for one. Those calls have to succeed without doing
    anything, or the assertion after them never runs.
    """

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def add_iris_server_to_path():
    if IRIS_SERVER_PATH not in sys.path:
        sys.path.insert(0, IRIS_SERVER_PATH)


def stub_core_api_models(face_recognition=None, transcribe=None):
    """Replace core_api wholesale with stand-ins for the model singletons.

    Use for tests of the pipeline, which cares what the models return rather
    than how they work.
    """
    core_api = types.ModuleType("core_api")
    for name in _MODEL_SINGLETONS:
        setattr(core_api, name, _InertModel())
    core_api.FaceRecognition = face_recognition
    core_api.WhisperSpeech2Text = transcribe
    sys.modules["core_api"] = core_api
    return core_api


def stub_core_api_package():
    """Make core_api importable without running its __init__.

    Use for tests of the real submodules: relative imports still resolve, but
    no model is instantiated.
    """
    package = types.ModuleType("core_api")
    package.__path__ = [os.path.join(IRIS_SERVER_PATH, "core_api")]
    sys.modules["core_api"] = package
    return package


class Checks:
    """Collects pass/fail results so one run reports every failure."""

    def __init__(self):
        self.failures = []

    def equal(self, label, got, want):
        if got == want:
            print(f"ok   {label}")
            return True
        self.failures.append(f"{label}\n    got:  {got!r}\n    want: {want!r}")
        print(f"FAIL {label}")
        return False

    def section(self, title):
        print(f"\n--- {title} ---")

    def report(self, summary):
        print()
        if self.failures:
            print(f"{len(self.failures)} FAILURE(S):")
            for failure in self.failures:
                print(" -", failure)
            sys.exit(1)
        print(summary)
