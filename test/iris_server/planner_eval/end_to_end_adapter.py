"""Connect the end-to-end evaluation to the server's own reasoner.

The evaluation does not copy the routing: it builds the production `_Reasoner`
with the model calls swapped in, and asks it to `decide` each sentence, so the
classifier prompt, the physical-state rule and the planner routing under test
are the ones the robot runs.
"""

import importlib

import planner_environment
from error_scrubbing import scrubbed
from eval_cases import KIND_CHAT, KIND_QUEUE, KIND_UNSUPPORTED


def _module(name: str):
    return importlib.import_module(name)


def reasoner_with(ask_classifier_model, ask_planner_model):
    """The production reasoner, asking these two model calls."""
    return _module("reasoner.reasoner")._Reasoner(
        ask_planner_model=ask_planner_model,
        ask_classifier_model=ask_classifier_model,
    )


def real_reasoner():
    """A reasoner that asks the live model for both stages; costs real requests.

    The classifier has no Grok fallback here, so a failure is measured rather
    than hidden. Errors are scrubbed (see error_scrubbing.py).
    """
    handler = planner_environment.real_chatgpt_handler()
    classifier_call = _module("reasoner.classifier_call")
    planner = _module("reasoner.action_planner")
    return reasoner_with(
        scrubbed(classifier_call.classifier_call_through(handler, classifier_call.CLASSIFY_SPAN)),
        scrubbed(planner.model_call_through(handler)),
    )


def label_of_decision(decision, action_by_state: dict) -> dict:
    """What the robot would do for `decision`, as {"kind", "actions"}.

    A confirmation question ("did you mean wave?") moves nothing, so it is
    chat here; the report counts those separately.
    """
    utils = _module("utils")
    if decision.state == utils.G1_SEQUENCE_STATE:
        return _queue([action_by_state[state] for state in decision.gestures])
    if decision.state in action_by_state:
        return _queue([action_by_state[decision.state]])
    if decision.state == utils.UNSUPPORTED_ACTION_STATE:
        return {"kind": KIND_UNSUPPORTED, "actions": []}
    return {"kind": KIND_CHAT, "actions": []}


def _queue(actions: list) -> dict:
    return {"kind": KIND_QUEUE, "actions": actions}


def stage_facts(decision) -> dict:
    """What the classifier and planner each did for one turn, for the report."""
    turn_decision = _module("reasoner.turn_decision")
    answer = decision.classifier_answer
    return {
        "classifier_answer": answer,
        "classifier_flagged_physical": turn_decision.is_physical_state(answer),
        "asked_to_confirm": answer.startswith(turn_decision.CONFIRM_STATE_PREFIX),
        "planner_asked": decision.planner_was_asked,
        "planner_vetoed": decision.vetoed_by_planner,
        "planner_failed": decision.planner_failed,
        "route": decision.state,
    }


NO_STAGE_FACTS = {
    "classifier_answer": None,
    "classifier_flagged_physical": False,
    "asked_to_confirm": False,
    "planner_asked": False,
    "planner_vetoed": False,
    "planner_failed": False,
    "route": None,
}
