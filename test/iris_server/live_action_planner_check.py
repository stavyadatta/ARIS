"""OPT-IN live check of the gesture-queue planner against the real model.

NOT part of run_tests.sh and never run automatically: it makes real OpenAI
requests, which cost credits and need OPENAI_API_KEY in the environment. It
sends five sample sentences through the same code the server uses and prints
what came back next to what was expected, so a person can judge the model.

    OPENAI_API_KEY=... IRIS_PLANNER_MODEL=gpt-4o-mini \\
        test/iris_server/live_action_planner_check.py

(inside the iris-server image, with the same mounts and PYTHONPATH as
run_tests.sh). It needs no database. IRIS_PLANNER_MODEL is optional and
defaults to OPENAI_MODEL, then gpt-4o-mini. The key is never printed.
"""

import importlib
import os

from harness import IRIS_SERVER_PATH, add_iris_server_to_path, stub_core_api_models, stub_neo4j_database

# (what the person says, what a good planner returns)
SAMPLE_REQUESTS = (
    ("Can you dance, wave and then shake my hand",
     "queue: waist_drum_dance, wave, handshake"),
    ("Can you go over there, pick up the towel, clean the table, then sit",
     "unsupported: go over there / pick up the towel / clean the table / sit"),
    ("Please wave and then dance and then wave again",
     "queue: wave, waist_drum_dance, wave"),
    ("Can you dance like a DJ",
     "queue: waist_drum_dance (one dance, not two gestures)"),
    ("Can you tell me a joke and then tell me the weather",
     "could not plan: no physical step"),
)


def describe(outcome) -> str:
    from reasoner.action_planner import PlannedQueue, UnsupportedRequest

    if isinstance(outcome, PlannedQueue):
        return "queue: " + ", ".join(outcome.states)
    if isinstance(outcome, UnsupportedRequest):
        return "unsupported: " + " / ".join(outcome.steps)
    return f"could not plan: {outcome.reason}"


def real_handler():
    """The server's own OpenAI handler, loaded without starting any model."""
    add_iris_server_to_path()
    stub_neo4j_database()
    core_api = stub_core_api_models()
    core_api.__path__ = [os.path.join(IRIS_SERVER_PATH, "core_api")]
    return importlib.import_module("core_api.chatgpt.chatgpt")._OpenAIHandler()


def main():
    handler = real_handler()
    from reasoner.action_planner import model_call_through, plan_robot_steps

    ask_model = model_call_through(handler)
    model = importlib.import_module("core_api.chatgpt.chatgpt").DEFAULT_PLANNER_MODEL
    print(f"planner model: {model}\n")
    for sentence, expected in SAMPLE_REQUESTS:
        print(f"said:     {sentence!r}")
        print(f"expected: {expected}")
        print(f"got:      {describe(plan_robot_steps(sentence, ask_model))}\n")


if __name__ == "__main__":
    main()
