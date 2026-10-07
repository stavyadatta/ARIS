"""Smoke test for the G1 action queue: one request, several gestures, in order.

Covers the ordering gate in the reasoner, the optional `actions` array in the
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
from reasoner.reasoner import _G1_GESTURE_PHRASES, _Reasoner
from utils import (
    G1_SEQUENCE_ATTRIBUTE,
    G1_SEQUENCE_STATE,
    MAX_ACTIONS_PER_REQUEST,
    PersonDetails,
    g1_action_payload,
)

IMAGE_QUEUE_CAPACITY = 50

check = Checks()
reasoner = _Reasoner()
HANDSHAKE, DANCE, WAVE = "g1 handshake", "g1 waist drum dance", "g1 wave"

check.section("ordering: gestures come back in the order they were spoken")
gestures = reasoner._requested_g1_gestures
check.equal("shake then dance",
            gestures("please shake my hand and then dance"), [HANDSHAKE, DANCE])
check.equal("dance then shake: spoken order, not table order",
            gestures("can you dance and then shake my hand"), [DANCE, HANDSHAKE])
check.equal("dance then wave",
            gestures("can you dance and then wave"), [DANCE, WAVE])
check.equal("wave then dance",
            gestures("can you wave and then dance"), [WAVE, DANCE])
check.equal("DJ is a word of its own",
            gestures("please be a DJ and then give me a high five"),
            ["g1 spin discs", "g1 high five"])
check.equal("case is ignored",
            gestures("Please WAVE and then DANCE"), [WAVE, DANCE])

check.section("two gesture words are not a sequence without a sequencing word")
check.equal("dance like a DJ is one dance",
            gestures("please dance like a DJ"), [DANCE])
check.equal("the wave dance is one wave",
            gestures("please do the wave dance"), [WAVE])
check.equal("'do the wave dance' alone has no request marker, as before",
            gestures("do the wave dance"), [])
check.equal("a comma alone does not chain",
            gestures("please wave, dance"), [WAVE])
check.equal("a bare 'and' does",
            gestures("please wave and dance"), [WAVE, DANCE])
check.equal("the same gesture twice is one",
            gestures("please dance and then dance"), [DANCE])
check.equal("each sequencing word chains",
            [gestures(f"please wave {word} dance") for word in
             ("then", "and then", "after that", "afterwards", "next", "followed by")],
            [[WAVE, DANCE]] * 6)
check.equal("'and' inside another word does not chain",
            gestures("please wave handsomely dance"), [WAVE])
check.equal("the chain stops where the sequencing word stops",
            gestures("please wave and dance like a DJ"), [WAVE, DANCE])
check.equal("an unchained pair falls back to the old priority order",
            gestures("please dance, then shake my hand like a DJ"), [DANCE, HANDSHAKE])
check.equal("an unchained first pair is the old priority pick",
            gestures("can you dance, wave and then shake my hand"), [HANDSHAKE])

check.section("single gesture is unchanged")
check.equal("wave", gestures("can you wave at me"), [WAVE])
check.equal("handshake", gestures("please shake hands"), [HANDSHAKE])
check.equal("high five", gestures("can you give a high five"), ["g1 high five"])
check.equal("dance", gestures("could you dance"), [DANCE])
check.equal("throw money", gestures("please make it rain"), ["g1 throw money"])

check.section("nothing without a request")
check.equal("no request marker", gestures("shake my hand and then dance"), [])
check.equal("narration", gestures("then she waved goodbye and danced"), [])
check.equal("not a gesture", gestures("please tell me a joke"), [])
check.equal("dj inside a word", gestures("please tell me about djibouti"), [])

check.section("duplicates and the cap")
check.equal("same gesture twice is one",
            gestures("please wave and then wave again"), [WAVE])
check.equal("two phrasings of one gesture are one",
            gestures("please shake hands, then give me a handshake"), [HANDSHAKE])
five = "please wave, then dance, then shake my hand, then high five, then make it rain"
check.equal("capped at the first three spoken",
            gestures(five), [WAVE, DANCE, HANDSHAKE])
check.equal("cap is three", MAX_ACTIONS_PER_REQUEST, 3)

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
person = PersonDetails({"state": "speak", "face_id": "f1"})
user_prompt = [{"role": "user", "content": "please shake my hand and then dance"}]
routed = reasoner._routed_by_gates(person, "please shake my hand and then dance", user_prompt)
check.equal("state", routed.get_attribute("state"), G1_SEQUENCE_STATE)
check.equal("gestures kept in order", routed.get_attribute(G1_SEQUENCE_ATTRIBUTE),
            [HANDSHAKE, DANCE])
check.equal("executor picks the gesture api exactly",
            find_best_match(G1_SEQUENCE_STATE, api_call.keys()), G1_SEQUENCE_STATE)
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

check.section("gesture api: an unsound sequence is refused, never trimmed")
for label, states in [("not a list", "g1 wave"),
                      ("missing", []),
                      ("one entry", [WAVE]),
                      ("over the cap", [WAVE, DANCE, HANDSHAKE, "g1 high five"]),
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
