"""Connect the evaluation to the planner: a way to plan, and a way to read outcomes.

`plan(sentence)` is anything that returns the planner's outcome. The real one
calls the model; the fake answers from the labels, so the runner and the
metrics can be checked offline.
"""

import importlib

import planner_environment
from error_scrubbing import scrubbed
from eval_cases import KIND_CHAT, KIND_COULD_NOT_PLAN, KIND_QUEUE, KIND_UNSUPPORTED

# The planner's own reason when a sentence asks for nothing physical.
NO_PHYSICAL_STEP_REASON = "no physical step requested"

FAKE_UNSUPPORTED_STEP = "something the robot cannot do"


def _planner_module():
    return importlib.import_module("reasoner.action_planner")


def label_of(outcome, action_by_state: dict) -> dict:
    """The planner's typed outcome as {"kind", "actions"} (plus "reason" if lost)."""
    planner = _planner_module()
    if isinstance(outcome, planner.PlannedQueue):
        actions = [action_by_state[state] for state in outcome.states]
        return {"kind": KIND_QUEUE, "actions": actions}
    if isinstance(outcome, planner.UnsupportedRequest):
        return {"kind": KIND_UNSUPPORTED, "actions": []}
    if outcome.reason == NO_PHYSICAL_STEP_REASON:
        return {"kind": KIND_CHAT, "actions": []}
    return {"kind": KIND_COULD_NOT_PLAN, "actions": [], "reason": outcome.reason}


def real_planner():
    """A plan function that asks the live model; costs real requests.

    Model errors are scrubbed (see error_scrubbing.py) because the planner puts
    the error text into its reason, which the report prints.
    """
    planner = _planner_module()
    ask_model = planner.model_call_through(planner_environment.real_chatgpt_handler())
    return lambda sentence: planner.plan_robot_steps(sentence, scrubbed(ask_model))


def fake_planner(cases: list, action_by_state: dict):
    """A plan function that answers every case's sentence with its expected label."""
    state_by_action = {action: state for state, action in action_by_state.items()}
    expected_by_sentence = {case["sentence"]: case["expected"] for case in cases}

    def plan(sentence):
        return _outcome_for_expected(expected_by_sentence[sentence], state_by_action)

    return plan


def _outcome_for_expected(expected: dict, state_by_action: dict):
    planner = _planner_module()
    if expected["kind"] == KIND_QUEUE:
        states = tuple(state_by_action[action] for action in expected["actions"])
        return planner.PlannedQueue(states)
    if expected["kind"] == KIND_UNSUPPORTED:
        return planner.UnsupportedRequest((FAKE_UNSUPPORTED_STEP,))
    return planner.CouldNotPlan(NO_PHYSICAL_STEP_REASON)
