"""Make the server's planner importable here without models or a database.

Importing the server's own packages normally loads GPU speech models and
connects to Neo4j (a graph database). Neither is wanted for an evaluation, so
the same stand-ins as live_action_planner_check.py are installed first.
"""

import importlib
import os

from harness import (
    IRIS_SERVER_PATH,
    add_iris_server_to_path,
    stub_core_api_models,
    stub_neo4j_database,
)

CHATGPT_MODULE = "core_api.chatgpt.chatgpt"


def prepare_server_imports() -> None:
    add_iris_server_to_path()
    stub_neo4j_database()
    core_api = stub_core_api_models()
    # Lets the real OpenAI handler module load while the models stay stubbed.
    core_api.__path__ = [os.path.join(IRIS_SERVER_PATH, "core_api")]


def gesture_table() -> dict:
    """The server's allow list: reasoner state -> gesture description."""
    return importlib.import_module("apis.g1_gesture").G1_GESTURES


def allowed_actions() -> set:
    return {gesture["action"] for gesture in gesture_table().values()}


def action_by_state() -> dict:
    return {state: gesture["action"] for state, gesture in gesture_table().items()}


def real_chatgpt_handler():
    """The server's own OpenAI handler; it reads the key from the environment."""
    return importlib.import_module(CHATGPT_MODULE)._OpenAIHandler()


def planner_model_name() -> str:
    """The model the handler will ask: a name only, never a credential."""
    return importlib.import_module(CHATGPT_MODULE).DEFAULT_PLANNER_MODEL
