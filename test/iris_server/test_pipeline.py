"""Smoke test for the Iris conversation pipeline.

Covers the path a spoken turn actually takes — transcribe, resolve a face,
reason, then fold the executor's output into the G1 reply/action contract —
plus the gRPC endpoints that feed it.
"""

import importlib.util
import json
import queue
import sys

import numpy as np

from harness import Checks, add_iris_server_to_path, stub_core_api_models

add_iris_server_to_path()

IMAGE_QUEUE_CAPACITY = 50


class FakeFaceRecognition:
    """Stands in for the insightface-backed recogniser."""

    def __init__(self):
        self.face_img_queue = queue.Queue()
        self.face_id_queue = []
        self.min_area = 4500
        self.votes = None
        self.relaxed = None
        self.bbox = None

    def get_most_frequent_face_id(self):
        return self.votes

    def recognize_face_relaxed(self, image):
        return self.relaxed

    def add2face_img_queue(self, image):
        self.face_img_queue.put(image)

    def get_face_box(self, image):
        return self.bbox


def fake_transcription(audio_img_item):
    return audio_img_item["fake_transcription"]


face_recognition = FakeFaceRecognition()
stub_core_api_models(face_recognition=face_recognition, transcribe=fake_transcription)

from collections import deque

from apis import api_call
from executor.executor import find_best_match
from media_manager.grpc_handle import MediaManager
from reasoner.reasoner import _Reasoner
from utils import PersonDetails, ApiObject, g1_action_payload
import utils

# The APIs under test only need the write to succeed, not to reach Neo4j.
utils.Neo4j.add_message_to_person = lambda person_details: None

check = Checks()


def png_bytes(width=8, height=8):
    import cv2
    _, buffer = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    return buffer.tobytes()


class FakeStreamRequest:
    def __init__(self, image_data, face_min_area=0):
        self.image_data = image_data
        self.face_min_area = face_min_area


check.section("executor routing")
for state in ["no face", "bad input", "speak", "g1 wave", "g1 confirm wave", "vision"]:
    check.equal(f"route {state!r}", find_best_match(state, api_call.keys()), state)

check.section("reasoner: no face short-circuit")
reasoner = _Reasoner()
check.equal("face_id None -> no face",
            reasoner(transcription="hello iris", face_id=None).get_attribute("state"),
            "no face")

check.section("reasoner: explicit gesture requests")
check.equal("wave", reasoner._requested_g1_gesture("can you wave at me"), "g1 wave")
check.equal("handshake", reasoner._requested_g1_gesture("please shake hands"), "g1 handshake")
check.equal("high five", reasoner._requested_g1_gesture("can you give a high five"), "g1 high five")
check.equal("narration ignored", reasoner._requested_g1_gesture("she waved goodbye"), None)
check.equal("no request marker ignored", reasoner._requested_g1_gesture("wave"), None)

check.section("reasoner: mis-heard gesture requests ask before acting")
for heard in ["wait", "waive", "weave", "wade"]:
    check.equal(f"{heard!r} -> confirm wave",
                reasoner._uncertain_g1_gesture(f"can you {heard}"), "g1 wave")
# These scored 0.75 against "wave" under the old similarity threshold and
# wrongly asked to wave. An ordinary sentence must stay ordinary.
for innocent in ["have a look", "gave it to me", "save that", "cave", "wake me"]:
    check.equal(f"{innocent!r} is not a wave request",
                reasoner._uncertain_g1_gesture(f"can you {innocent}"), None)
check.equal("shake", reasoner._uncertain_g1_gesture("please shake"), "g1 handshake")
check.equal("unrelated request", reasoner._uncertain_g1_gesture("please tell me a joke"), None)

check.section("reasoner: confirmation vocabulary")
check.equal("yes", reasoner._confirmed_g1_gesture("yes"), True)
check.equal("yes please", reasoner._confirmed_g1_gesture("Yes, please!"), True)
check.equal("no thanks", reasoner._confirmed_g1_gesture("no thanks"), False)

check.section("no face api")
chunks = list(api_call["no face"](PersonDetails({"state": "no face"})))
check.equal("one chunk", len(chunks), 1)
check.equal("mode", chunks[0].mode, "g1_action")
payload = json.loads(chunks[0].textchunk)
check.equal("no action", payload["action"], "none")
check.equal("asks to be seen", "see" in payload["reply"].lower(), True)

check.section("g1 gesture api")
gesture_chunks = list(api_call["g1 wave"](PersonDetails({"state": "g1 wave", "face_id": "f1"})))
check.equal("one chunk", len(gesture_chunks), 1)
# The spoken words are generated per turn from what the person said, so only
# the action is fixed. An unreachable model yields a silent gesture, never a
# canned line, so reply is a string either way.
gesture_payload = json.loads(gesture_chunks[0].textchunk)
check.equal("action", gesture_payload["action"], "wave")
check.equal("reply is text", isinstance(gesture_payload["reply"], str), True)
check.equal("mode", gesture_chunks[0].mode, "g1_action")

unknown = list(api_call["g1 wave"](PersonDetails({"state": "g1 moonwalk"})))
check.equal("unknown state refuses to guess", json.loads(unknown[0].textchunk),
            {"reply": "", "action": ""})
check.equal("unknown state mode", unknown[0].mode, "g1_action_error")

check.section("response folding")
manager = MediaManager(image_queue=deque(maxlen=IMAGE_QUEUE_CAPACITY))


def fold(chunks):
    return list(manager._g1_conversation_chunks(iter(chunks)))


speech = [ApiObject("Hello ", mode="default"), ApiObject("there", mode="default")]
check.equal("speech only", fold(speech),
            [(g1_action_payload("Hello there", "none"), "g1_action")])
check.equal("speech then structured",
            fold(speech + [ApiObject("{}", mode="custom_movement")]),
            [(g1_action_payload("Hello there", "none"), "g1_action"),
             ("{}", "custom_movement")])
check.equal("structured only", fold([ApiObject("{}", mode="g1_action")]),
            [("{}", "g1_action")])

blank = fold([ApiObject("", mode="default")])
check.equal("blank speech falls back", len(blank), 1)
check.equal("blank speech scratches head", json.loads(blank[0][0])["action"], "scratch_head")

empty = fold([])
check.equal("no chunks falls back", len(empty), 1)
check.equal("no chunks scratches head", json.loads(empty[0][0])["action"], "scratch_head")

check.section("face id resolution")
face_recognition.votes, face_recognition.relaxed = "face_7", None
check.equal("votes win", manager._resolve_face_id(None, skip_face_validation=False), "face_7")
face_recognition.votes, face_recognition.relaxed = None, "face_9"
check.equal("retries this request's frame",
            manager._resolve_face_id(None, skip_face_validation=False), "face_9")
face_recognition.votes, face_recognition.relaxed = None, None
check.equal("both empty", manager._resolve_face_id(None, skip_face_validation=False), None)
face_recognition.votes, face_recognition.relaxed = "face_7", "face_2"
check.equal("relaxed skips voting",
            manager._resolve_face_id(None, skip_face_validation=True), "face_2")

check.section("transcription")
check.equal("silence becomes a placeholder",
            manager._transcribe({"fake_transcription": "."}), "You")
check.equal("speech is kept",
            manager._transcribe({"fake_transcription": "hello iris"}), "hello iris")

check.section("StreamImages fills both queues")
stream_manager = MediaManager(image_queue=deque(maxlen=IMAGE_QUEUE_CAPACITY))
face_recognition.face_img_queue = queue.Queue()
stream_manager.StreamImages(iter([FakeStreamRequest(png_bytes(), face_min_area=900)]), None)
check.equal("bbox queue got the frame", len(stream_manager.image_queue), 1)
check.equal("recognition queue got the frame", face_recognition.face_img_queue.qsize(), 1)
check.equal("face_min_area applied", face_recognition.min_area, 900)

check.section("GetBbox answers once frames exist")
bbox_manager = MediaManager(image_queue=deque(maxlen=IMAGE_QUEUE_CAPACITY))
face_recognition.bbox = (11, 22, 33, 44)
check.equal("empty queue answers zero", bbox_manager.GetBbox(None, None).x2, 0)
for _ in range(IMAGE_QUEUE_CAPACITY):
    bbox_manager.image_queue.append(np.zeros((8, 8, 3), dtype=np.uint8))
found = bbox_manager.GetBbox(None, None)
check.equal("full queue answers the detected box",
            (found.x1, found.y1, found.x2, found.y2), (11, 22, 33, 44))
face_recognition.bbox = None
check.equal("no face answers zero", bbox_manager.GetBbox(None, None).x2, 0)

check.section("ClearQueue drains what exists")
clear_manager = MediaManager(image_queue=deque(maxlen=IMAGE_QUEUE_CAPACITY))
face_recognition.face_img_queue = queue.Queue()
for _ in range(3):
    face_recognition.face_img_queue.put(object())
    clear_manager.image_queue.append(object())
removal = clear_manager.ClearQueue(None, None)
check.equal("reports success", removal.removed, True)
check.equal("recognition queue emptied", face_recognition.face_img_queue.qsize(), 0)
check.equal("bbox queue emptied", len(clear_manager.image_queue), 0)
check.equal("clearing an empty queue still succeeds",
            clear_manager.ClearQueue(None, None).removed, True)

check.section("a server fault is apologised for aloud, not reported as 'error'")
# Failures used to travel as mode="error" carrying a stack trace, which the
# client could not voice -- the robot said the word "error" at whoever was
# standing in front of it. A fault is not the listener's fault and not
# something they can act on, so it now speaks in the ordinary reply contract.
error_chunks = list(manager.ProcessAudioImg(object(), None))
check.equal("one chunk", len(error_chunks), 1)
check.equal("speaks the G1 contract", error_chunks[0].mode, "g1_action")
error_payload = json.loads(error_chunks[0].text)
check.equal("no body action", error_payload["action"], "none")
# Check every reply, not the one this run happened to draw. The code picks at
# random, so asserting on a single sample is a test that passes most of the
# time -- which is worse than one that fails.
from media_manager.grpc_handle import G1_ERROR_REPLIES

check.equal("several to rotate between", len(G1_ERROR_REPLIES) > 1, True)
for reply in G1_ERROR_REPLIES:
    short = reply[:32] + "..."
    check.equal(f"apologises: {short}",
                "sorry" in reply.lower() or "apolog" in reply.lower(), True)
    # An exception is not fixed by the listener repeating themselves. These
    # must send them to a person, not into a retry loop -- which is what
    # separates them from G1_LISTENING_FALLBACKS.
    check.equal(f"points at a human: {short}",
                any(w in reply.lower() for w in
                    ("team", "engineer", "someone", "technical")), True)
    check.equal(f"never says 'error': {short}", "error" in reply.lower(), False)

# And the distinction is real, not accidental: the listening fallbacks ask for
# a repeat and must not send anyone looking for an engineer.
from media_manager.grpc_handle import G1_LISTENING_FALLBACKS

for reply in G1_LISTENING_FALLBACKS:
    check.equal(f"listening fallback asks for a repeat: {reply[:32]}...",
                any(w in reply.lower() for w in ("again", "repeat", "once more")), True)
check.equal("the drawn reply is one of them",
            error_payload["reply"] in G1_ERROR_REPLIES, True)

check.section("silence is deliberate, not a failure to answer")
# "be quiet" used to reach the listening fallback, which made Iris say
# "could you repeat that?" and scratch its head — the opposite of the
# instruction. A silent turn must stay silent.
silent_person = PersonDetails({"state": "silent", "face_id": "f1"})
silent_person.set_latest_usr_message({"role": "user", "content": "be quiet"})
silent_chunks = list(manager._g1_conversation_chunks(api_call["silent"](silent_person)))
check.equal("one chunk", len(silent_chunks), 1)
check.equal("speaks the G1 contract", silent_chunks[0][1], "g1_action")
silent_payload = json.loads(silent_chunks[0][0])
check.equal("says nothing", silent_payload["reply"], "")
check.equal("does nothing", silent_payload["action"], "none")
check.equal("records the turn for later context",
            silent_person.get_latest_llm_message()["content"], "Silence noted")

# A genuinely empty reply still asks for a repeat: bad input depends on it.
bad_input_chunks = list(manager._g1_conversation_chunks(
    api_call["bad input"](PersonDetails({"state": "bad input"}))))
check.equal("bad input still scratches its head",
            json.loads(bad_input_chunks[0][0])["action"], "scratch_head")

check.section("no route can emit another robot's joint angles")
# "Can you wipe your hands?" used to reach Pepper's movement API and answer
# with NAO joint names the G1 does not have. Every motion state must now
# decline in speech instead.
from apis.unsupported_action import _UnsupportedAction

for pepper_state in ["custom movement", "standard movement", "g1 unsupported action"]:
    check.equal(f"{pepper_state!r} declines",
                isinstance(api_call[pepper_state], _UnsupportedAction), True)

declined = list(api_call["g1 unsupported action"](PersonDetails({"state": "custom movement"})))
check.equal("one chunk", len(declined), 1)
check.equal("speaks the G1 contract", declined[0].mode, "g1_action")
declined_payload = json.loads(declined[0].textchunk)
check.equal("no body action", declined_payload["action"], "none")
check.equal("offers what it can do",
            "high five" in declined_payload["reply"], True)

check.equal("Pepper movement package is gone",
            importlib.util.find_spec("apis.movement"), None)
check.equal("Pepper auto package is gone",
            importlib.util.find_spec("apis.pepper_auto"), None)
for state, api in api_call.items():
    check.equal(f"{state!r} never emits joint angles",
                type(api).__name__ in {"_Speaking", "_Silent", "_PersonAttribute",
                                       "_BadInput", "_NoFace", "_UnsupportedAction",
                                       "_SecondaryChannel", "_G1Gesture"}, True)

check.report("PIPELINE OK")
