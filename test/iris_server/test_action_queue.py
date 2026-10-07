"""Smoke test for the G1 action queue: one request, several gestures, in order.

Covers the gesture gate and queue planner in the reasoner (the model call is a fake), the optional `actions` array in the
reply contract, the gesture API that emits it, and that the voice step does not
drop it on the way to the client. Needs no database: the Neo4j client is stubbed.
"""

import json
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
from apis.unsupported_action import UNSUPPORTED_ACTION_REPLIES, spoken_refusal_naming
from reasoner.action_planner import (
    CouldNotPlan,
    PlannedQueue,
    SCHEMA_NAME,
    UnsupportedRequest,
    plan_robot_steps,
    response_format,
)
from reasoner.reasoner import _G1_GESTURE_PHRASES, _Reasoner
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


planner_model = FakePlannerModel()
reasoner = _Reasoner(ask_planner_model=planner_model)
HANDSHAKE, DANCE, WAVE = "g1 handshake", "g1 waist drum dance", "g1 wave"
SPIN_DISCS, HIGH_FIVE = "g1 spin discs", "g1 high five"


def gestures(text):
    """The gesture states the reasoner would perform for `text`; [] when none."""
    request = reasoner._physical_request(text)
    return list(request.states) if isinstance(request, PlannedQueue) else []


check.section("one gesture word and no sequencing word: deterministic, no model call")
planner_model.calls.clear()
check.equal("wave", gestures("can you wave at me"), [WAVE])
check.equal("handshake", gestures("please shake hands"), [HANDSHAKE])
check.equal("high five", gestures("can you give a high five"), [HIGH_FIVE])
check.equal("dance", gestures("could you dance"), [DANCE])
check.equal("throw money", gestures("please make it rain"), ["g1 throw money"])
check.equal("the planner was never called", planner_model.calls, [])

check.section("no request marker: never the planner")
check.equal("gestures without a marker", gestures("shake my hand and then dance"), [])
check.equal("narration", gestures("and then she waved goodbye and danced"), [])
check.equal("a sequencing word alone", gestures("and then we left"), [])
check.equal("not a gesture", gestures("please tell me a joke"), [])
check.equal("dj inside a word", gestures("please tell me about djibouti"), [])
check.equal("the planner was never called either", planner_model.calls, [])

check.section("when the planner is asked")
planner_model.answers("wave")
for label, sentence in [
    ("two gesture words", "please dance like a DJ"),
    ("one gesture word and a sequencing word", "can you wave and say hello"),
    ("a sequencing word and no gesture word", "can you go over there and then sit"),
]:
    planner_model.calls.clear()
    gestures(sentence)
    check.equal(f"{label}: asked once", len(planner_model.calls), 1)
for word in ("and", "then", "and then", "after that", "afterwards", "next", "followed by"):
    planner_model.calls.clear()
    gestures(f"please sit {word} stand")
    check.equal(f"sequencing word {word!r} triggers the planner", len(planner_model.calls), 1)
planner_model.calls.clear()
gestures("could you wave")
gestures("could you wait a moment")
check.equal("'and' inside another word does not trigger",
            gestures("could you wave at the band"), [WAVE])
check.equal("... and neither did those", planner_model.calls, [])

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
long_actions = ["wave", "waist_drum_dance", "handshake"] * (LONG_QUEUE_LENGTH // 3) + ["hug"] * 2
long_states = [WAVE, DANCE, HANDSHAKE] * (LONG_QUEUE_LENGTH // 3) + ["g1 hug"] * 2
planner_model.answers(*long_actions)
check.equal("fifty actions are accepted in order",
            gestures("please do all of these and then some more"), long_states)
check.equal("fifty is not a special number", len(long_states), LONG_QUEUE_LENGTH)

check.section("a step the robot cannot do: nothing is performed")
owner_chores = "Can you go over there, pick up the towel, clean the table, then sit"
planner_model.answers_steps({"unsupported": "go over there"}, {"unsupported": "pick up the towel"},
                            {"unsupported": "clean the table"}, {"unsupported": "sit"})
refusal = reasoner._physical_request(owner_chores)
check.equal("the owner's example is refused whole",
            (type(refusal), refusal.steps),
            (UnsupportedRequest, ("go over there", "pick up the towel", "clean the table", "sit")))
planner_model.answers_steps({"action": "wave"}, {"unsupported": "pick up the towel"},
                            {"action": "handshake"})
mixed = reasoner._physical_request("please wave, pick up the towel, then shake my hand")
check.equal("one unsupported step among supported ones refuses everything",
            (type(mixed), mixed.steps), (UnsupportedRequest, ("pick up the towel",)))
check.equal("so no gesture is returned for it",
            gestures("please wave, pick up the towel, then shake my hand"), [])
planner_model.answers_steps({"unsupported": "  clean the table  "})
check.equal("the person's words are trimmed",
            reasoner._physical_request("can you clean the table and then sit").steps,
            ("clean the table",))

check.section("an untrustworthy plan falls back to the old single pick")
# "dance" and "wave" are both named; the old priority order puts wave first.
fallback_sentence = "can you dance and wave"
for label, arrange in [
    ("unknown action", lambda: planner_model.answers("wave", "moonwalk")),
    ("empty list", lambda: planner_model.answers()),
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
    check.equal(f"{label}: one gesture, by priority", gestures(fallback_sentence), [WAVE])
planner_model.fails_with(TimeoutError("slow"))
check.equal("no gesture word and a failed plan: not a gesture request at all",
            reasoner._physical_request("can you go over there and sit"), None)

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
person = PersonDetails({"state": "speak", "face_id": "f1"})
user_prompt = [{"role": "user", "content": "please shake my hand and then dance"}]
routed = reasoner._routed_by_gates(person, "please shake my hand and then dance", user_prompt)
check.equal("state", routed.get_attribute("state"), G1_SEQUENCE_STATE)
check.equal("gestures kept in order", routed.get_attribute(G1_SEQUENCE_ATTRIBUTE),
            [HANDSHAKE, DANCE])
check.equal("executor picks the gesture api exactly",
            find_best_match(G1_SEQUENCE_STATE, api_call.keys()), G1_SEQUENCE_STATE)
planner_model.answers("wave")
check.equal("a plan of one is routed as that single gesture",
            reasoner._routed_by_gates(PersonDetails({"state": "speak", "face_id": "f1"}),
                                      "please wave and dance", user_prompt).get_attribute("state"),
            WAVE)
check.equal("single gesture keeps its own state",
            reasoner._routed_by_gates(PersonDetails({"state": "speak", "face_id": "f1"}),
                                      "please wave", user_prompt).get_attribute("state"),
            WAVE)

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
long_person = reasoner._routed_by_gates(PersonDetails({"state": "speak", "face_id": "f1"}),
                                        "please do all of these and then some more", user_prompt)
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
chores = reasoner._routed_by_gates(PersonDetails({"state": "speak", "face_id": "f1"}),
                                   owner_chores, user_prompt)
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
check.equal("it names what it cannot do",
            "go over there, pick up the towel, clean the table and more" in declined["reply"], True)
check.equal("it says nothing was started", "will not start any of it" in declined["reply"], True)
check.equal("it offers something it can do",
            any(offer in declined["reply"] for offer in
                ("wave", "dance", "shake hands", "throw money")), True)
check.equal("speaks the G1 contract", declined_chunks[0].mode, "g1_action")
check.equal("a single unsupported step reads naturally",
            spoken_refusal_naming(["sit"]).startswith("I cannot sit yet,"), True)
check.equal("two read naturally",
            spoken_refusal_naming(["sit", "stand"]).startswith("I cannot sit and stand yet,"), True)
generic = json.loads(list(api_call[UNSUPPORTED_ACTION_STATE](
    PersonDetails({"state": "custom movement"})))[0].textchunk)
check.equal("without recorded steps it declines generally, as before",
            generic["reply"] in UNSUPPORTED_ACTION_REPLIES, True)

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

check.section("every queued gesture is allow-listed")
check.equal("all gate states have gesture entries",
            all(state in G1_GESTURES for state, _ in _G1_GESTURE_PHRASES), True)

check.report("ACTION QUEUE OK")
