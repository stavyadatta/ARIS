"""Smoke test for the G1 action queue: one request, several gestures, in order.

Covers the queue planner (the model call is a fake), how the reasoner routes
what the classifier and planner decide, the optional `actions` array in the
reply contract, the gesture API that emits it, and that the voice step does not
drop it on the way to the client. Needs no database: the Neo4j client is
stubbed. When the classifier is asked at all is covered by test_turn_decision.
"""

import json
import re
from collections import deque

from harness import (
    Checks,
    add_iris_server_to_path,
    stub_core_api_models,
    stub_neo4j_database,
)

add_iris_server_to_path()
stub_neo4j_database()
core_api = stub_core_api_models()


class RecordingChatGPT:
    """Keeps the prompt the gesture API sends, so the test can read what it asked for."""

    def __init__(self):
        self.prompts = []

    def send_text(self, prompt, stream, max_tokens):
        self.prompts.append(prompt)
        raise RuntimeError("no model in tests")


chat = RecordingChatGPT()
core_api.ChatGPT = chat

from apis import api_call
from apis.g1_gesture import G1_GESTURES
from executor.executor import find_best_match
from media_manager.grpc_handle import MediaManager
from apis.unsupported_action import OFFERED_GESTURES, UNSUPPORTED_ACTION_REPLY
from reasoner.action_planner import (
    CouldNotPlan,
    NoStepsRequested,
    PlannedQueue,
    SCHEMA_NAME,
    UnsupportedRequest,
    plan_robot_steps,
    response_format,
)
from reasoner.reasoner import _Reasoner
from utils import (
    G1_SEQUENCE_ATTRIBUTE,
    G1_SEQUENCE_STATE,
    PersonDetails,
    UNSUPPORTED_ACTION_STATE,
    UNSUPPORTED_STEPS_ATTRIBUTE,
    g1_action_payload,
)

IMAGE_QUEUE_CAPACITY = 50
LONG_QUEUE_LENGTH = 50

check = Checks()


class FakePlannerModel:
    """Stands in for the network call: answers with what it is told, and counts calls."""

    def __init__(self):
        self.calls = []
        self.reply = None
        self.error = None

    def answers(self, *actions):
        self.answers_steps(*({"action": action} for action in actions))

    def answers_steps(self, *steps):
        self.answers_raw(json.dumps({"steps": list(steps)}))

    def answers_raw(self, text):
        self.reply, self.error = text, None

    def fails_with(self, error):
        self.error = error

    def __call__(self, messages, schema):
        self.calls.append((messages, schema))
        if self.error:
            raise self.error
        return self.reply


class FakeClassifierModel:
    """Stands in for the per-turn classifier: always answers with what it is told."""

    def __init__(self):
        self.answer = "no change"

    def __call__(self, messages):
        return self.answer


planner_model = FakePlannerModel()
classifier_model = FakeClassifierModel()
reasoner = _Reasoner(ask_planner_model=planner_model, ask_classifier_model=classifier_model)
HANDSHAKE, DANCE, WAVE = "g1 handshake", "g1 waist drum dance", "g1 wave"
SPIN_DISCS, HIGH_FIVE = "g1 spin discs", "g1 high five"


def gestures(text):
    """The gesture states the planner queues for `text`; [] when it queues none."""
    outcome = plan_robot_steps(text, planner_model)
    return list(outcome.states) if isinstance(outcome, PlannedQueue) else []


def routed_person(sentence, classifier_says):
    """The person record the reasoner routes `sentence` to, given the classifier's answer."""
    classifier_model.answer = classifier_says
    person = PersonDetails({"state": "speak", "face_id": "f1"})
    prompt = [{"role": "user", "content": sentence}]
    return reasoner._route_decision(person, reasoner.decide(sentence), sentence, prompt)


check.section("several steps: the planner orders them")
owner_sentence = "Can you dance, wave and then shake my hand"
planner_model.answers("waist_drum_dance", "wave", "handshake")
planner_model.calls.clear()
check.equal("the owner's sentence, in the order spoken",
            gestures(owner_sentence), [DANCE, WAVE, HANDSHAKE])
check.equal("one planner call", len(planner_model.calls), 1)
sent_messages, sent_schema = planner_model.calls[0]
check.equal("the model is sent the words as they were said",
            owner_sentence in sent_messages[1]["content"], True)
check.equal("it is sent the strict schema", sent_schema, response_format())

planner_model.answers("wave", "waist_drum_dance", "wave")
check.equal("repeats are kept",
            gestures("please wave and then dance and then wave again"), [WAVE, DANCE, WAVE])
planner_model.answers("waist_drum_dance")
check.equal("dance like a DJ planned as one dance",
            gestures("please dance like a DJ"), [DANCE])
planner_model.answers("wave")
check.equal("narration dropped by the planner",
            gestures("please dance, then she waved"), [WAVE])
planner_model.answers("hug", "clamp")
check.equal("any allow-listed gesture may be planned, not only keyword ones",
            gestures("can you dance and wave, then hug me and clap"), ["g1 hug", "g1 clap"])

check.section("no cap: a long queue is planned whole, in order")
LONG_REQUEST = "please wave, dance and shake hands over and over, and then some more"
long_actions = ["wave", "waist_drum_dance", "handshake"] * (LONG_QUEUE_LENGTH // 3) + ["hug"] * 2
long_states = [WAVE, DANCE, HANDSHAKE] * (LONG_QUEUE_LENGTH // 3) + ["g1 hug"] * 2
planner_model.answers(*long_actions)
check.equal("fifty actions are accepted in order",
            gestures(LONG_REQUEST), long_states)
check.equal("fifty is not a special number", len(long_states), LONG_QUEUE_LENGTH)

check.section("a step the robot cannot do: nothing is performed")
owner_chores = "Can you wave, go over there, pick up the towel, clean the table, then sit"
planner_model.answers_steps({"unsupported": "go over there"}, {"unsupported": "pick up the towel"},
                            {"unsupported": "clean the table"}, {"unsupported": "sit"})
refusal = plan_robot_steps(owner_chores, planner_model)
check.equal("the owner's example is refused whole",
            (type(refusal), refusal.steps),
            (UnsupportedRequest, ("go over there", "pick up the towel", "clean the table", "sit")))
planner_model.answers_steps({"action": "wave"}, {"unsupported": "pick up the towel"},
                            {"action": "handshake"})
mixed = plan_robot_steps("please wave, pick up the towel, then shake my hand", planner_model)
check.equal("one unsupported step among supported ones refuses everything",
            (type(mixed), mixed.steps), (UnsupportedRequest, ("pick up the towel",)))
check.equal("so no gesture is returned for it",
            gestures("please wave, pick up the towel, then shake my hand"), [])
planner_model.answers_steps({"unsupported": "  clean the table  "})
check.equal("the person's words are trimmed",
            plan_robot_steps("can you wave and clean the table", planner_model).steps,
            ("clean the table",))

check.section("an untrustworthy plan is a CouldNotPlan, never a queue")
for label, arrange in [
    ("unknown action", lambda: planner_model.answers("wave", "moonwalk")),
    ("model raises", lambda: planner_model.fails_with(TimeoutError("slow"))),
    ("not json", lambda: planner_model.answers_raw("sure, I will dance")),
    ("cut off mid-reply", lambda: planner_model.answers_raw('{"steps": [{"action": "wa')),
    ("no steps key", lambda: planner_model.answers_raw('{"queue": ["wave"]}')),
    ("steps is not a list", lambda: planner_model.answers_raw('{"steps": "wave"}')),
    ("a bare string step", lambda: planner_model.answers_raw('{"steps": ["wave"]}')),
    ("a non-string action", lambda: planner_model.answers_steps({"action": 7})),
    ("a step with both keys", lambda: planner_model.answers_steps({"action": "wave", "unsupported": "x"})),
    ("a step with neither key", lambda: planner_model.answers_steps({"move": "wave"})),
    ("a blank unsupported step", lambda: planner_model.answers_steps({"unsupported": "  "})),
    ("a nested action", lambda: planner_model.answers_steps({"action": ["wave"]})),
    ("reply is None", lambda: planner_model.answers_raw(None)),
    ("a state name instead of an action", lambda: planner_model.answers("g1 wave")),
]:
    arrange()
    failure = plan_robot_steps("can you dance and wave", planner_model)
    check.equal(f"{label}: could not plan, and not 'no steps'",
                (isinstance(failure, CouldNotPlan), isinstance(failure, NoStepsRequested)),
                (True, False))

check.section("no steps from the planner: the gesture word was narration")
NARRATION = "can you tell me how she waved"
planner_model.answers()
check.equal("an empty list is not a request: no queue",
            gestures(NARRATION), [])
planner_model.calls.clear()
check.equal("also when several gestures are named",
            gestures("can you tell me how she waved and danced"), [])
check.equal("each was one planner call", len(planner_model.calls), 1)
check.equal("the empty list is reported as its own outcome",
            type(plan_robot_steps(NARRATION, planner_model)), NoStepsRequested)
check.equal("which is still a CouldNotPlan for code that only knows that",
            isinstance(plan_robot_steps(NARRATION, planner_model), CouldNotPlan), True)
planner_model.answers("wave")
check.equal("the same words with a queue from the planner are performed",
            gestures("can you wave"), [WAVE])

check.section("planner: typed result")
planner_model.answers("handshake", "waist_drum_dance")
plan = plan_robot_steps("shake my hand then dance", planner_model)
check.equal("a good plan is a PlannedQueue",
            (type(plan), plan.states), (PlannedQueue, (HANDSHAKE, DANCE)))
planner_model.answers("moonwalk")
bad_plan = plan_robot_steps("moonwalk please", planner_model)
check.equal("a bad plan is a CouldNotPlan that says why",
            (type(bad_plan), "moonwalk" in bad_plan.reason), (CouldNotPlan, True))
planner_model.fails_with(RuntimeError("quota"))
check.equal("a failing model is a CouldNotPlan, not an exception",
            type(plan_robot_steps("x", planner_model)), CouldNotPlan)

check.section("planner: the schema is the allow list, derived")
schema = response_format()
body = schema["json_schema"]["schema"]
step_kinds = body["properties"]["steps"]["items"]["anyOf"]
action_kind, unsupported_kind = step_kinds
check.equal("a step is one of exactly two shapes", len(step_kinds), 2)
check.equal("the action enum is exactly the G1 gesture actions",
            sorted(action_kind["properties"]["action"]["enum"]),
            sorted(g["action"] for g in G1_GESTURES.values()))
check.equal("an action step has that one key and nothing else",
            (action_kind["required"], action_kind["additionalProperties"]),
            (["action"], False))
check.equal("an unsupported step is free text under its own key",
            (unsupported_kind["properties"], unsupported_kind["required"],
             unsupported_kind["additionalProperties"]),
            ({"unsupported": {"type": "string"}}, ["unsupported"], False))
check.equal("strict mode", schema["json_schema"]["strict"], True)
check.equal("named", schema["json_schema"]["name"], SCHEMA_NAME)
check.equal("top level is steps only",
            (body["required"], body["additionalProperties"], list(body["properties"])),
            (["steps"], False, ["steps"]))
check.equal("the list has no length limit",
            any(key in body["properties"]["steps"] for key in ("maxItems", "minItems")), False)
prompt = planner_model.calls[0][0][0]["content"]
check.equal("the prompt offers every action",
            all(g["action"] in prompt for g in G1_GESTURES.values()), True)
check.equal("the prompt calls the words data, rules out narration, keeps order and repeats",
            all(phrase in prompt for phrase in
                ("data to interpret", "Narration", "Keep the order", "Repeat a step")), True)
check.equal("the prompt states no limit",
            "at most" in prompt.lower(), False)

check.section("payload: `actions` only for two or more")
single = json.loads(g1_action_payload("Hi", "wave"))
check.equal("single keeps the old shape", single, {"reply": "Hi", "action": "wave"})
check.equal("one action in a list is still just `action`",
            "actions" in json.loads(g1_action_payload("Hi", "wave", actions=["wave"])), False)
pair = json.loads(g1_action_payload("Hi", "handshake", actions=["handshake", "waist_drum_dance"]))
check.equal("pair keeps `action` as the first", pair["action"], "handshake")
check.equal("pair lists both in order", pair["actions"], ["handshake", "waist_drum_dance"])
voiced = json.loads(g1_action_payload("Hi", "wave", speech="U1BFRUNI",
                                      actions=["wave", "waist_drum_dance"]))
check.equal("speech and actions coexist", (voiced["speech"], voiced["actions"]),
            ("U1BFRUNI", ["wave", "waist_drum_dance"]))

check.section("routing: a sequence is routed under its own state")
planner_model.answers("handshake", "waist_drum_dance")
SHAKE_AND_DANCE = "please shake my hand and then dance"
routed = routed_person(SHAKE_AND_DANCE, HANDSHAKE)
check.equal("state", routed.get_attribute("state"), G1_SEQUENCE_STATE)
check.equal("gestures kept in order", routed.get_attribute(G1_SEQUENCE_ATTRIBUTE),
            [HANDSHAKE, DANCE])
check.equal("executor picks the gesture api exactly",
            find_best_match(G1_SEQUENCE_STATE, api_call.keys()), G1_SEQUENCE_STATE)
planner_model.answers("wave")
check.equal("a plan of one is routed as that single gesture",
            routed_person("please wave and dance", WAVE).get_attribute("state"), WAVE)
check.equal("single gesture keeps its own state",
            routed_person("please wave", WAVE).get_attribute("state"), WAVE)

check.section("gesture api: one reply, the whole sequence")
sequence_chunks = list(api_call[G1_SEQUENCE_STATE](routed))
check.equal("one chunk", len(sequence_chunks), 1)
sequence_payload = json.loads(sequence_chunks[0].textchunk)
check.equal("action is the first", sequence_payload["action"], "handshake")
check.equal("actions in order", sequence_payload["actions"], ["handshake", "waist_drum_dance"])
check.equal("reply is one string", isinstance(sequence_payload["reply"], str), True)
check.equal("speaks the G1 contract", sequence_chunks[0].mode, "g1_action")
check.equal("state returns to speak", routed.get_attribute("state"), "speak")

wave_chunks = list(api_call[WAVE](PersonDetails({"state": WAVE, "face_id": "f1"})))
check.equal("single gesture payload has no `actions`",
            "actions" in json.loads(wave_chunks[0].textchunk), False)

system_prompt = chat.prompts[0][0]["content"]
check.equal("the one reply is asked to cover both gestures",
            "shake their hand and then dancing" in system_prompt
            and "one after the other" in system_prompt, True)
single_prompt = chat.prompts[1][0]["content"]
check.equal("a single gesture is described as before",
            "You are waving hello to them right now" in single_prompt, True)

check.section("no cap: a long queue is routed, serialised and kept in order")
planner_model.answers(*long_actions)
long_person = routed_person(LONG_REQUEST, WAVE)
check.equal("routed as a sequence", long_person.get_attribute("state"), G1_SEQUENCE_STATE)
check.equal("all fifty states carried, in order",
            long_person.get_attribute(G1_SEQUENCE_ATTRIBUTE), long_states)
long_chunks = list(api_call[G1_SEQUENCE_STATE](long_person))
long_payload = json.loads(long_chunks[0].textchunk)
check.equal("all fifty actions sent, in order", long_payload["actions"], long_actions)
check.equal("action is the first", long_payload["action"], long_actions[0])
long_prompt = chat.prompts[-1][0]["content"]
check.equal("the reply prompt stays short however long the queue",
            len(long_prompt) < 1500, True)
check.equal("but says there is more", "the rest of what they asked" in long_prompt, True)

check.section("an unsupported request: routed to the decline, performing nothing")
planner_model.answers_steps({"unsupported": "go over there"}, {"unsupported": "pick up the towel"},
                            {"unsupported": "clean the table"}, {"unsupported": "sit"})
chores = routed_person(owner_chores, WAVE)
check.equal("routed to the unsupported-action state",
            chores.get_attribute("state"), UNSUPPORTED_ACTION_STATE)
check.equal("the executor reaches the unsupported-action api",
            api_call[find_best_match(UNSUPPORTED_ACTION_STATE, api_call.keys())] is
            api_call[UNSUPPORTED_ACTION_STATE], True)
check.equal("it carries the person's steps",
            chores.get_attribute(UNSUPPORTED_STEPS_ATTRIBUTE),
            ["go over there", "pick up the towel", "clean the table", "sit"])
declined_chunks = list(api_call[UNSUPPORTED_ACTION_STATE](chores))
declined = json.loads(declined_chunks[0].textchunk)
check.equal("no body action at all", (declined["action"], "actions" in declined), ("none", False))
check.equal("speaks the G1 contract", declined_chunks[0].mode, "g1_action")
check.equal("the reply is the one fixed wording", declined["reply"], UNSUPPORTED_ACTION_REPLY)

check.section("the refusal wording")
check.equal("it says it is a safety matter and not configured yet",
            "Due to safety, those actions have not been configured for me yet." in UNSUPPORTED_ACTION_REPLY,
            True)
check.equal("it offers the list in a natural spoken form",
            UNSUPPORTED_ACTION_REPLY.endswith(
                "What I can do now is wave, shake hands, give a high five, dance, DJ or throw money."),
            True)
spoken_words = set(re.findall(r"[a-z']+", declined["reply"].lower()))
persons_words = {word for step in chores.get_attribute(UNSUPPORTED_STEPS_ATTRIBUTE)
                 for word in re.findall(r"[a-z']+", step.lower())}
check.equal("none of the person's words are spoken", spoken_words & persons_words, set())
check.equal("not the sentence the person said either",
            owner_chores.lower() in declined["reply"].lower(), False)
generic = json.loads(list(api_call[UNSUPPORTED_ACTION_STATE](
    PersonDetails({"state": "custom movement"})))[0].textchunk)
check.equal("the planner path and the classifier path say identical words",
            generic["reply"], declined["reply"])
other_steps = PersonDetails({"state": UNSUPPORTED_ACTION_STATE,
                             UNSUPPORTED_STEPS_ATTRIBUTE: ["sit"]})
check.equal("whatever the steps are, the words do not change",
            json.loads(list(api_call[UNSUPPORTED_ACTION_STATE](other_steps))[0].textchunk)["reply"],
            declined["reply"])

check.section("the offered gestures are real, allow-listed ones")
allow_listed_actions = {gesture["action"] for gesture in G1_GESTURES.values()}
check.equal("every offered action is a G1 gesture action",
            all(action in allow_listed_actions for action, _ in OFFERED_GESTURES), True)
check.equal("every offered phrase is spoken in the reply",
            all(phrase in UNSUPPORTED_ACTION_REPLY for _, phrase in OFFERED_GESTURES), True)
check.equal("nothing is offered twice",
            len({action for action, _ in OFFERED_GESTURES}), len(OFFERED_GESTURES))
never_offered = {G1_GESTURES[state]["action"] for state in
                 ("g1 hug", "g1 hand on heart", "g1 blow kiss left", "g1 blow kiss right")}
check.equal("a hug, hand on heart and the blow kisses are never offered",
            never_offered & {action for action, _ in OFFERED_GESTURES}, set())

check.section("gesture api: an unsound sequence is refused, never trimmed")
for label, states in [("not a list", "g1 wave"),
                      ("missing", []),
                      ("one entry", [WAVE]),
                      ("unknown gesture", [WAVE, "g1 moonwalk"])]:
    refused = list(api_call[G1_SEQUENCE_STATE](
        PersonDetails({"state": G1_SEQUENCE_STATE, G1_SEQUENCE_ATTRIBUTE: states})))
    check.equal(f"{label}: error mode", refused[0].mode, "g1_action_error")
    check.equal(f"{label}: nothing to perform",
                json.loads(refused[0].textchunk), {"reply": "", "action": ""})

check.section("voice step keeps the sequence")
manager = MediaManager(image_queue=deque(maxlen=IMAGE_QUEUE_CAPACITY))
with_voice = json.loads(manager._text_chunk(
    g1_action_payload("Sure.", "handshake", actions=["handshake", "waist_drum_dance"]),
    "g1_action").text)
check.equal("actions survive", with_voice["actions"], ["handshake", "waist_drum_dance"])
check.equal("action survives", with_voice["action"], "handshake")
check.equal("speech attached", "speech" in with_voice, True)
check.equal("single action still has no `actions`",
            "actions" in json.loads(manager._text_chunk(
                g1_action_payload("Sure.", "wave"), "g1_action").text), False)

check.report("ACTION QUEUE OK")
