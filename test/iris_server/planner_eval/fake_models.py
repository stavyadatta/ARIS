"""Offline stand-ins for the classifier and planner models, answering from the labels.

They let the end-to-end runner and its metrics be checked without a network. A
perfect pair must score 100%. A flawed pair makes a few known mistakes so the
metrics that count those mistakes can be tested too.
"""

import importlib
import json
from dataclasses import dataclass

from eval_cases import KIND_CHAT, KIND_QUEUE, KIND_UNSUPPORTED

CLASSIFIER_MISSES_IN_QUEUE_CASES = 2
CLASSIFIER_MISSES_IN_UNSUPPORTED_CASES = 1
CLASSIFIER_OVERFLAGS_IN_CHAT_CASES = 3
# Of the over-flagged chat cases, how many also fool the planner into acting.
PLANNER_FOOLED_IN_CHAT_CASES = 1

NO_CHANGE_ANSWER = "no change"
OVERFLAG_ANSWER = "g1 wave"
OVERFLAG_ACTION = "wave"
FAKE_UNSUPPORTED_STEP = "something the robot cannot do"
REQUEST_OPENING = 'Spoken request: """'
REQUEST_CLOSING = '"""'


@dataclass(frozen=True)
class Flaws:
    """Sentences on which the flawed fakes go wrong."""

    missed_by_classifier: frozenset
    overflagged_by_classifier: frozenset
    fooled_planner: frozenset


def choose_flaws(cases: list) -> Flaws:
    """The first few cases of each kind, so the flaws are the same every run."""
    def first(kind, count):
        return [case["sentence"] for case in cases if case["expected"]["kind"] == kind][:count]

    overflagged = first(KIND_CHAT, CLASSIFIER_OVERFLAGS_IN_CHAT_CASES)
    return Flaws(
        missed_by_classifier=frozenset(
            first(KIND_QUEUE, CLASSIFIER_MISSES_IN_QUEUE_CASES)
            + first(KIND_UNSUPPORTED, CLASSIFIER_MISSES_IN_UNSUPPORTED_CASES)),
        overflagged_by_classifier=frozenset(overflagged),
        fooled_planner=frozenset(overflagged[:PLANNER_FOOLED_IN_CHAT_CASES]),
    )


def _unsupported_state() -> str:
    return importlib.import_module("utils").UNSUPPORTED_ACTION_STATE


def _perfect_classifier_answer(expected: dict, state_by_action: dict) -> str:
    """What a perfect classifier says: the first step in time order, or no change."""
    if expected["kind"] == KIND_QUEUE:
        return state_by_action[expected["actions"][0]]
    if expected["kind"] == KIND_UNSUPPORTED:
        return _unsupported_state()
    return NO_CHANGE_ANSWER


def fake_classifier(cases: list, state_by_action: dict, flaws: Flaws = None):
    """A classifier call answering from the labels; with `flaws`, wrong on those sentences."""
    answers = {case["sentence"]: _perfect_classifier_answer(case["expected"], state_by_action)
               for case in cases}
    if flaws:
        answers.update({sentence: NO_CHANGE_ANSWER for sentence in flaws.missed_by_classifier})
        answers.update({sentence: OVERFLAG_ANSWER for sentence in flaws.overflagged_by_classifier})

    def ask(messages: list) -> str:
        return answers[messages[-1]["content"]]

    return ask


def _steps_for(expected: dict) -> list:
    if expected["kind"] == KIND_QUEUE:
        return [{"action": action} for action in expected["actions"]]
    if expected["kind"] == KIND_UNSUPPORTED:
        return [{"unsupported": FAKE_UNSUPPORTED_STEP}]
    return []


def _sentence_in(messages: list) -> str:
    content = messages[-1]["content"]
    return content.split(REQUEST_OPENING, 1)[1].rsplit(REQUEST_CLOSING, 1)[0]


def fake_planner_model(cases: list, flaws: Flaws = None):
    """A planner model call answering from the labels; with `flaws`, fooled on those sentences."""
    steps_by_sentence = {case["sentence"]: _steps_for(case["expected"]) for case in cases}
    if flaws:
        steps_by_sentence.update(
            {sentence: [{"action": OVERFLAG_ACTION}] for sentence in flaws.fooled_planner})

    def ask(messages: list, schema: dict) -> str:
        return json.dumps({"steps": steps_by_sentence[_sentence_in(messages)]})

    return ask
