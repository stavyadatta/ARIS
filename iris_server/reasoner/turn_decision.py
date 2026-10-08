"""What a classified turn becomes: the classifier decides, the planner builds.

The per-turn classifier (see prompt.py) already reads every sentence, so it is
the only thing that decides whether a sentence is a request for the robot's
body. When it says so, the planner (action_planner.py) reads the whole sentence
and builds the queue, because the classifier names one state and cannot order
several steps or notice a step the robot cannot do. No word list takes part in
that decision.

`decide_turn` is a pure function of the classifier's answer, the sentence and
the planner's model call, so the reasoner and the evaluation run the same code.
"""

import random
from dataclasses import dataclass
from typing import Optional

import turn_log
from turn_timing import span
from utils import G1_SEQUENCE_STATE, UNSUPPORTED_ACTION_STATE
from .action_planner import (
    CouldNotPlan,
    ModelCall,
    NoStepsRequested,
    PlannedQueue,
    UnsupportedRequest,
    plan_robot_steps,
)

STATE_SPEAK = "speak"
G1_STATE_PREFIX = "g1 "
CONFIRM_STATE_PREFIX = "g1 confirm "

# The classifier names a *family* of gestures ("g1 greeting", "g1 farewell")
# rather than one specific action, so the family can be resolved to one of its
# concrete states by weighted probability instead of always picking the same
# gesture. Equal weights today; skew them (or add more entries) without
# touching the resolution logic.
_GREETING_GESTURE_WEIGHTS = {"g1 wave": 0.5, "g1 handshake": 0.5}
_FAREWELL_GESTURE_WEIGHTS = {"g1 blow kiss left": 0.5, "g1 blow kiss right": 0.5}
_GESTURE_FAMILY_WEIGHTS = {
    "g1 greeting": _GREETING_GESTURE_WEIGHTS,
    "g1 farewell": _FAREWELL_GESTURE_WEIGHTS,
}

PLANNER_SPAN = "reasoner.plan_steps"


@dataclass(frozen=True)
class TurnDecision:
    """The state to route a turn to, and everything that state needs.

    `gestures` is the queue when `state` is the sequence state, and
    `unsupported_steps` the person's own words when it is the unsupported-action
    state; both are empty otherwise. `planner_outcome` is None when the planner
    was not asked.
    """

    state: str
    classifier_answer: str
    gestures: tuple = ()
    unsupported_steps: tuple = ()
    planner_outcome: Optional[object] = None

    @property
    def planner_was_asked(self) -> bool:
        return self.planner_outcome is not None

    @property
    def vetoed_by_planner(self) -> bool:
        """The classifier flagged a request and the planner found none in it."""
        return isinstance(self.planner_outcome, NoStepsRequested)

    @property
    def planner_failed(self) -> bool:
        """The planner's model call failed or answered badly (not a deliberate "no steps")."""
        return (isinstance(self.planner_outcome, CouldNotPlan)
                and not self.vetoed_by_planner)


def is_physical_state(state: str) -> bool:
    """Whether the classifier flagged a request for the robot's body.

    Greeting and farewell are social: they are answered with a gesture picked
    from their family and never reach the planner, so a plain "hello" costs no
    extra model call. A confirmation is a question about a gesture, not a
    request to perform it.
    """
    return (
        state.startswith(G1_STATE_PREFIX)
        and state not in _GESTURE_FAMILY_WEIGHTS
        and not state.startswith(CONFIRM_STATE_PREFIX)
    )


def _weighted_choice(options: dict) -> str:
    """Pick one key from options, weighted by its probability value."""
    return random.choices(list(options.keys()), weights=list(options.values()), k=1)[0]


def resolve_gesture_family(classifier_answer: str) -> str:
    gesture_family = _GESTURE_FAMILY_WEIGHTS.get(classifier_answer)
    if gesture_family is None:
        return classifier_answer
    resolved_gesture = _weighted_choice(gesture_family)
    print(f"[g1_action] llm_category={classifier_answer!r} resolved={resolved_gesture}")
    return resolved_gesture


def _decision_after_planning(classifier_answer: str, state: str, outcome) -> TurnDecision:
    """The decision for a plan, or for no plan at all, given the classifier's state."""
    if isinstance(outcome, PlannedQueue):
        turn_log.step("plan", f"{list(outcome.states)}")
        return _queue_decision(classifier_answer, outcome)
    if isinstance(outcome, UnsupportedRequest):
        turn_log.step("plan", f"unsupported steps {list(outcome.steps)}; performing nothing")
        return TurnDecision(UNSUPPORTED_ACTION_STATE, classifier_answer,
                            unsupported_steps=outcome.steps, planner_outcome=outcome)
    if isinstance(outcome, NoStepsRequested):
        turn_log.step("plan", "classifier flagged a request but the planner found no physical "
                              "step; treating it as conversation")
        return TurnDecision(STATE_SPEAK, classifier_answer, planner_outcome=outcome)
    # The model failed. Before the planner existed the classifier's own state
    # was what the robot did, so that is the safest thing to fall back to.
    turn_log.step("plan", f"could not plan ({outcome.reason}); keeping the classifier's {state!r}")
    return TurnDecision(state, classifier_answer, planner_outcome=outcome)


def _queue_decision(classifier_answer: str, queue: PlannedQueue) -> TurnDecision:
    if len(queue.states) == 1:
        return TurnDecision(queue.states[0], classifier_answer, planner_outcome=queue)
    return TurnDecision(G1_SEQUENCE_STATE, classifier_answer, gestures=queue.states,
                        planner_outcome=queue)


def _ask_planner(transcription: str, ask_planner_model: ModelCall):
    with span(PLANNER_SPAN):
        return plan_robot_steps(transcription, ask_planner_model)


def decide_turn(classifier_answer: str, transcription: str,
                ask_planner_model: ModelCall) -> TurnDecision:
    """Route the classifier's answer; ask the planner only for a physical one."""
    state = resolve_gesture_family(classifier_answer)
    if not is_physical_state(classifier_answer):
        return TurnDecision(state, classifier_answer)
    outcome = _ask_planner(transcription, ask_planner_model)
    return _decision_after_planning(classifier_answer, state, outcome)
