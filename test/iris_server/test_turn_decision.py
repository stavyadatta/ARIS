"""Test that the classifier decides whether a turn is physical and the planner builds the queue.

Every model call is a fake, so nothing here reaches a network, and the Neo4j
client is stubbed, so no database is needed.
"""

import json
from types import SimpleNamespace

from harness import Checks, add_iris_server_to_path, stub_core_api_models, stub_neo4j_database

add_iris_server_to_path()
stub_neo4j_database()
stub_core_api_models()

import reasoner.reasoner as reasoner_module
from apis.g1_gesture import G1_CONFIRMATIONS, G1_GESTURES
from reasoner.action_planner import NoStepsRequested, PlannedQueue
from reasoner.classifier_call import (
    classifier_call_through,
    default_classifier_call,
    with_fallback,
)
from reasoner.prompt import action_reasoner_prompt
from reasoner.reasoner import _Reasoner
from reasoner.turn_decision import (
    STATE_SPEAK,
    decide_turn,
    is_physical_state,
)
from utils import (
    G1_SEQUENCE_ATTRIBUTE,
    G1_SEQUENCE_STATE,
    PersonDetails,
    UNSUPPORTED_ACTION_STATE,
    UNSUPPORTED_STEPS_ATTRIBUTE,
)

WAVE, CLAP, DANCE, HANDSHAKE = "g1 wave", "g1 clap", "g1 waist drum dance", "g1 handshake"
NO_CHANGE = "no change"
GREETING_STATES = {"g1 wave", "g1 handshake"}
FAREWELL_STATES = {"g1 blow kiss left", "g1 blow kiss right"}
REMOVED_GATE_NAMES = (
    "_GESTURE_REQUEST_MARKERS", "_G1_GESTURE_PHRASES", "_G1_GESTURE_PATTERNS",
    "_gestures_mentioned", "_has_request_marker", "_physical_request",
    "_planned_or_single", "_route_requested_gesture", "_uncertain_g1_gesture",
    "_route_uncertain_gesture", "_sounds_like_wave", "_WAVE_MISHEARINGS",
)

check = Checks()


class FakePlannerModel:
    """Answers with the steps it is told, and counts how often it was asked."""

    def __init__(self):
        self.calls = []
        self.reply = json.dumps({"steps": []})
        self.error = None

    def answers(self, *actions):
        self.reply = json.dumps({"steps": [{"action": action} for action in actions]})
        self.error = None

    def answers_unsupported(self, *steps):
        self.reply = json.dumps({"steps": [{"unsupported": step} for step in steps]})
        self.error = None

    def answers_raw(self, text):
        self.reply, self.error = text, None

    def fails_with(self, error):
        self.error = error

    def __call__(self, messages, schema):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply


class FakeClassifierModel:
    """Answers with what it is told, and keeps the messages it was sent."""

    def __init__(self):
        self.answer = NO_CHANGE
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.answer


planner = FakePlannerModel()
classifier = FakeClassifierModel()
reasoner = _Reasoner(ask_planner_model=planner, ask_classifier_model=classifier)


def decide(classifier_says, sentence="any sentence"):
    planner.calls.clear()
    return decide_turn(classifier_says, sentence, planner)


def turn(sentence, classifier_says, state="speak"):
    """Run a whole turn through the reasoner and return the routed person."""
    person = PersonDetails({"state": state, "face_id": "f1"})
    reasoner_module.Neo4j.get_person_details = lambda face_id: person
    classifier.answer = classifier_says
    classifier.calls.clear()
    planner.calls.clear()
    return reasoner(sentence, "f1")


check.section("the physical-state rule")
PHYSICAL_STATES = [*G1_GESTURES, UNSUPPORTED_ACTION_STATE, G1_SEQUENCE_STATE]
NOT_PHYSICAL_STATES = [
    "g1 greeting", "g1 farewell", *G1_CONFIRMATIONS,
    "no change", "speak", "silent", "vision", "bad input", "thanks", "no face", "",
]
for state in PHYSICAL_STATES:
    check.equal(f"{state!r} is physical", is_physical_state(state), True)
for state in NOT_PHYSICAL_STATES:
    check.equal(f"{state!r} is not physical", is_physical_state(state), False)

check.section("a physical answer: the planner is asked once and its queue is routed")
planner.answers("wave")
single = decide(WAVE, "can you wave at me")
check.equal("one planner call", len(planner.calls), 1)
check.equal("one gesture is routed as that gesture",
            (single.state, single.gestures), (WAVE, ()))
planner.answers("clamp")
check.equal("the planner's gesture wins over the classifier's",
            decide(WAVE).state, CLAP)
planner.answers("waist_drum_dance", "wave", "handshake")
several = decide(WAVE, "dance, wave and then shake my hand")
check.equal("several gestures are routed as the sequence, in order",
            (several.state, several.gestures), (G1_SEQUENCE_STATE, (DANCE, WAVE, HANDSHAKE)))
check.equal("the planner read the whole sentence",
            "dance, wave and then shake my hand" in planner.calls[0][1]["content"], True)

check.section("an unsupported plan: nothing is performed")
planner.answers_unsupported("pick up the towel", "sit")
refused = decide(WAVE)
check.equal("routed to the unsupported action",
            (refused.state, refused.gestures, refused.unsupported_steps),
            (UNSUPPORTED_ACTION_STATE, (), ("pick up the towel", "sit")))

check.section("no steps from the planner: the classifier was wrong, so it is conversation")
planner.answers()
vetoed = decide(WAVE)
check.equal("routed to plain conversation", (vetoed.state, vetoed.gestures), (STATE_SPEAK, ()))
check.equal("the veto is recorded", vetoed.vetoed_by_planner, True)
check.equal("the planner outcome is the typed one", type(vetoed.planner_outcome), NoStepsRequested)

check.section("the planner failed: the classifier's own state stands")
for label, arrange in [
    ("model raises", lambda: planner.fails_with(TimeoutError("slow"))),
    ("not json", lambda: planner.answers_raw("sure, I will dance")),
    ("cut off mid-reply", lambda: planner.answers_raw('{"steps": [{"action": "wa')),
    ("unknown action", lambda: planner.answers("moonwalk")),
]:
    arrange()
    for classifier_state in (CLAP, UNSUPPORTED_ACTION_STATE):
        kept = decide(classifier_state)
        check.equal(f"{label}: keeps {classifier_state!r}",
                    (kept.state, kept.gestures, kept.unsupported_steps, kept.vetoed_by_planner),
                    (classifier_state, (), (), False))

check.section("a non-physical answer: the planner is never asked")
planner.answers("wave")
SENTENCES = ["Cats and dogs", "Can you tell me about cats and dogs"]
for classifier_state in ("no change", "speak", "silent", "vision", "bad input"):
    for sentence in SENTENCES:
        outcome = decide(classifier_state, sentence)
        check.equal(f"{classifier_state!r} for {sentence!r}: planner not called",
                    (planner.calls, outcome.state, outcome.planner_was_asked),
                    ([], classifier_state, False))

check.section("greeting and farewell: no planner call, a gesture from the family")
for family, members in (("g1 greeting", GREETING_STATES), ("g1 farewell", FAREWELL_STATES)):
    resolved = {decide(family, "hello there").state for _ in range(60)}
    check.equal(f"{family}: planner not called", planner.calls, [])
    check.equal(f"{family}: only family members, and more than one", (resolved <= members, len(resolved) > 1),
                (True, True))

check.section("a confirmation: no planner call, the question is asked")
for confirmation in G1_CONFIRMATIONS:
    outcome = decide(confirmation, "can you waive")
    check.equal(f"{confirmation!r}: stays a confirmation",
                (outcome.state, planner.calls), (confirmation, []))

check.section("a whole turn through the reasoner")
planner.answers("waist_drum_dance", "wave")
routed = turn("dance and then wave", WAVE)
check.equal("the classifier is asked once with the prompt then the words",
            [(m["role"]) for m in classifier.calls[0]], ["system", "user"])
check.equal("the words are sent as said", classifier.calls[0][1]["content"], "dance and then wave")
check.equal("the planner is asked once", len(planner.calls), 1)
check.equal("sequence state", routed.get_attribute("state"), G1_SEQUENCE_STATE)
check.equal("sequence in order", routed.get_attribute(G1_SEQUENCE_ATTRIBUTE), [DANCE, WAVE])

planner.answers("handshake")
check.equal("one planned gesture is that gesture",
            turn("shake hands", HANDSHAKE).get_attribute("state"), HANDSHAKE)

planner.answers_unsupported("sit")
refused = turn("please sit", WAVE)
check.equal("unsupported: state", refused.get_attribute("state"), UNSUPPORTED_ACTION_STATE)
check.equal("unsupported: the steps are kept for the log",
            refused.get_attribute(UNSUPPORTED_STEPS_ATTRIBUTE), ["sit"])

planner.answers()
check.equal("vetoed: plain conversation",
            turn("can you tell me how she waved", WAVE, state="vision").get_attribute("state"),
            STATE_SPEAK)

planner.fails_with(TimeoutError("slow"))
check.equal("planner down: the classifier's gesture",
            turn("clap for me", CLAP).get_attribute("state"), CLAP)

planner.answers("wave")
check.equal("no change keeps the current state and asks no planner",
            (turn("cats and dogs", NO_CHANGE, state="vision").get_attribute("state"),
             len(planner.calls)), ("vision", 0))
check.equal("a confirmation question is routed as asked, with no planner",
            (turn("can you waive at me", "g1 confirm wave").get_attribute("state"),
             len(planner.calls)), ("g1 confirm wave", 0))
check.equal("a greeting is resolved, with no planner",
            (turn("hello iris", "g1 greeting").get_attribute("state") in GREETING_STATES,
             len(planner.calls)), (True, 0))

check.section("turns that need no model at all")
confirmed = turn("Yes, please!", "no change", state="g1 confirm wave")
check.equal("yes to an open question performs it, with no model",
            (confirmed.get_attribute("state"), classifier.calls, planner.calls), (WAVE, [], []))
declined = turn("no thanks", NO_CHANGE, state="g1 confirm wave")
check.equal("anything else drops the question and is classified afresh",
            (declined.get_attribute("state"), len(classifier.calls)), (STATE_SPEAK, 1))
thanked = turn("Thank you, Iris!", WAVE)
check.equal("a pure thank-you needs no model",
            (thanked.get_attribute("state"), classifier.calls, planner.calls), ("thanks", [], []))

check.section("a failing classifier stops the turn, as before")


def classifier_is_down(messages):
    raise ConnectionError("no network in tests")


failing = _Reasoner(ask_planner_model=planner, ask_classifier_model=classifier_is_down)
try:
    failing("hello", "f1")
    raised = False
except Exception:
    raised = True
check.equal("the error reaches the caller", raised, True)

check.section("the classifier's model call")
FAKE_REPLY = SimpleNamespace(model="m", choices=[SimpleNamespace(message=SimpleNamespace(content="g1 wave"))])


class FakeHandler:
    def __init__(self, reply=FAKE_REPLY, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def send_text(self, messages, stream):
        self.calls.append((messages, stream))
        if self.error:
            raise self.error
        return self.reply


handler = FakeHandler()
messages = [{"role": "user", "content": "hi"}]
check.equal("returns the answer text",
            classifier_call_through(handler, "span")(messages), "g1 wave")
check.equal("sends the messages, not streamed", handler.calls, [(messages, False)])
broken, backup = FakeHandler(error=RuntimeError("down")), FakeHandler()
check.equal("the fallback answers when the first model raises",
            default_classifier_call(broken, backup)(messages), "g1 wave")
check.equal("both were asked", (len(broken.calls), len(backup.calls)), (1, 1))
check.equal("the fallback is not asked when the first answers",
            (default_classifier_call(handler, backup)(messages), len(backup.calls)), ("g1 wave", 1))
try:
    with_fallback(classifier_is_down, classifier_is_down)(messages)
    both_down_raises = False
except ConnectionError:
    both_down_raises = True
check.equal("both failing raises", both_down_raises, True)

check.section("the old keyword gates are gone")
for name in REMOVED_GATE_NAMES:
    check.equal(f"{name} no longer exists",
                (hasattr(reasoner_module, name), hasattr(_Reasoner, name)), (False, False))

check.section("the classifier prompt")
for confirmation in G1_CONFIRMATIONS:
    check.equal(f"offers {confirmation!r}", confirmation in action_reasoner_prompt, True)
for phrase in ("FIRST step", "data to classify", "g1 unsupported action", "misheard"):
    check.equal(f"says {phrase!r}", phrase in action_reasoner_prompt, True)

check.report("TURN DECISION OK")
