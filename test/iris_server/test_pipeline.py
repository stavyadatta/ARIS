"""Smoke test for the Iris conversation pipeline.

Covers the path a spoken turn actually takes — transcribe, resolve a face,
reason, then fold the executor's output into the G1 reply/action contract —
plus the gRPC endpoints that feed it.
"""

import base64
import importlib.util
import io
import contextlib
import os
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
        self.relaxed_calls = 0
        self.bbox = None

    def get_most_frequent_face_id(self):
        return self.votes

    def recognize_face_relaxed(self, image):
        self.relaxed_calls += 1
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
from media_manager.grpc_handle import MediaManager, ResolvedFace
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
for state in ["no face", "bad input", "speak", "g1 wave", "g1 sequence", "g1 confirm wave", "vision"]:
    check.equal(f"route {state!r}", find_best_match(state, api_call.keys()), state)

check.section("reasoner: no face short-circuit")
reasoner = _Reasoner()
check.equal("face_id None -> no face",
            reasoner(transcription="hello iris", face_id=None).get_attribute("state"),
            "no face")

check.section("reasoner: explicit gesture requests")
def requested_states(sentence):
    request = reasoner._physical_request(sentence)
    return list(request.states) if request else []


check.equal("wave", requested_states("can you wave at me"), ["g1 wave"])
check.equal("handshake", requested_states("please shake hands"), ["g1 handshake"])
check.equal("high five", requested_states("can you give a high five"), ["g1 high five"])
check.equal("narration ignored", requested_states("she waved goodbye"), [])
check.equal("no request marker ignored", requested_states("wave"), [])

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

class FakePersonDetector:
    """Stands in for YOLO: reports a body, no body, or a fault."""

    def __init__(self, outcome):
        self.outcome = outcome

    def detect_and_crop_person(self, image):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def no_face_reply(detector, image):
    no_face_module = sys.modules["apis.no_face"]
    no_face_module.PersonDetectionCropper = detector
    details = PersonDetails({"state": "no face"})
    details.set_image(image)
    chunks = list(api_call["no face"](details))
    return json.loads(chunks[0].textchunk)["reply"]


check.section("no face api: body in shot")
frame = np.zeros((480, 640, 3), dtype=np.uint8)
body_replies = sys.modules["apis.no_face"].BODY_ONLY_REPLIES
check.equal("body without face says so",
            no_face_reply(FakePersonDetector(frame), frame) in body_replies, True)
check.equal("body-only reply never claims who it is",
            all("who you are" in r or "face" in r for r in body_replies), True)
no_face_replies = sys.modules["apis.no_face"].NO_FACE_REPLIES
check.equal("nobody in shot asks to be seen",
            no_face_reply(FakePersonDetector(None), frame) in no_face_replies, True)
check.equal("detector fault still answers",
            no_face_reply(FakePersonDetector(RuntimeError("cuda")), frame) in no_face_replies, True)
check.equal("missing frame still answers",
            no_face_reply(FakePersonDetector(frame), None) in no_face_replies, True)

check.section("thanks is a courtesy, not bad input")
from reasoner.reasoner import _is_pure_thanks
for thanks in ("Thank you.", "thanks", "Thanks, Iris!", "Thank you very much", "THANK YOU SO MUCH.", "cheers"):
    check.equal(f"{thanks!r} is pure thanks", _is_pure_thanks(thanks), True)
for other in ("Thank you for the dance", "thanks, now wave", "no thank you", "You", "", "thank"):
    check.equal(f"{other!r} is not pure thanks", _is_pure_thanks(other), False)

thanks_person = PersonDetails({"state": "speak", "face_id": "f1"})
routed = reasoner._route_thanks(thanks_person, "Thank you.", [{"role": "user", "content": "Thank you."}])
check.equal("gate routes thanks", routed.get_attribute("state"), "thanks")
check.equal("gate ignores a real request",
            reasoner._route_thanks(PersonDetails({"state": "speak"}), "thanks, now wave", [{}]), None)
check.equal("gate runs before the classifier",
            reasoner._routed_by_gates(PersonDetails({"state": "speak", "face_id": "f1"}), "Thank you.",
                                      [{"role": "user", "content": "Thank you."}]).get_attribute("state"),
            "thanks")
check.equal("route thanks", find_best_match("thanks", api_call.keys()), "thanks")

thanks_replies = sys.modules["apis.thanks"].THANKS_REPLIES
seen = set()
for _ in range(60):
    chunk = list(api_call["thanks"](thanks_person))
    reply_payload = json.loads(chunk[0].textchunk)
    seen.add(reply_payload["reply"])
    check_action = reply_payload["action"]
check.equal("every reply is from the rotation", seen <= set(thanks_replies), True)
check.equal("rotates", len(seen) > 1, True)
check.equal("never asks for a repeat", any("catch" in r.lower() or "repeat" in r.lower() for r in thanks_replies), False)
check.equal("does nothing physical", check_action, "none")

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
resolve = lambda relaxed_flag: manager._resolve_face_id(None, skip_face_validation=relaxed_flag)
face_recognition.votes, face_recognition.relaxed = "face_7", None
check.equal("votes win", resolve(False), ("face_7", True))
face_recognition.votes, face_recognition.relaxed = None, "face_9"
check.equal("request frame is a guess, so unverified", resolve(False), ("face_9", False))
face_recognition.votes, face_recognition.relaxed = None, None
check.equal("both empty", resolve(False), (None, False))
face_recognition.votes, face_recognition.relaxed = "face_7", "face_2"
check.equal("client-requested relaxed skips voting and is trusted",
            resolve(True), ("face_2", True))

check.section("unconfirmed face is never assumed")
unconfirmed = manager._reason_about("hello iris", ResolvedFace("face_9", False), None)
check.equal("routes to face unconfirmed", unconfirmed.get_attribute("state"), "face unconfirmed")
check.equal("carries no identity", bool(unconfirmed.get_attribute("face_id")), False)
unconfirmed_chunks = list(api_call["face unconfirmed"](unconfirmed))
unconfirmed_payload = json.loads(unconfirmed_chunks[0].textchunk)
check.equal("no body action", unconfirmed_payload["action"], "none")
unconfirmed_replies = sys.modules["apis.face_unconfirmed"].UNCONFIRMED_FACE_REPLIES
check.equal("speaks one of its replies", unconfirmed_payload["reply"] in unconfirmed_replies, True)
check.equal("every reply asks the person to lean in",
            all("lean" in r.lower() or "closer" in r.lower() for r in unconfirmed_replies), True)
check.equal("route face unconfirmed",
            find_best_match("face unconfirmed", api_call.keys()), "face unconfirmed")
check.equal("route no face still exact",
            find_best_match("no face", api_call.keys()), "no face")

check.section("server speaks: voice attached to g1_action chunks")
voice = sys.modules["core_api"].KokoroTts
spoken = json.loads(manager._text_chunk(g1_action_payload("Hello there.", "wave"), "g1_action").text)
check.equal("reply kept", spoken["reply"], "Hello there.")
check.equal("action kept", spoken["action"], "wave")
check.equal("speech attached", spoken.get("speech"), voice.speech)
check.equal("synthesised the reply text", voice.spoken[-1], "Hello there.")

apology = json.loads(manager._text_chunk(*manager._apology_chunk()).text)
check.equal("apology is voiced too", "speech" in apology, True)

voice.spoken.clear()
silent = json.loads(manager._text_chunk(g1_action_payload("", "none"), "g1_action").text)
check.equal("empty reply carries no speech", "speech" in silent, False)

voice.speech = None
unvoiced = json.loads(manager._text_chunk(g1_action_payload("Hello there.", "wave"), "g1_action").text)
check.equal("failed synthesis drops speech, keeps the turn",
            (unvoiced["reply"], unvoiced["action"], "speech" in unvoiced),
            ("Hello there.", "wave", False))
voice.speech = "U1BFRUNI"

check.equal("non-g1 modes pass through untouched",
            manager._text_chunk("raw streamed words", "default").text, "raw streamed words")
check.equal("unreadable g1 payload passes through",
            manager._text_chunk("not json", "g1_action").text, "not json")

check.section("voice report")
import io, contextlib
voice.speech = base64.b64encode(b"\x00\x01" * 16000).decode()   # 1.0 s of samples
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    manager._text_chunk(g1_action_payload("Hello there.", "wave"), "g1_action")
line = captured.getvalue()
check.equal("logs an attached voice", "[voice] attached" in line, True)
check.equal("reports audio seconds", "voice_audio_s=1.0" in line, True)
check.equal("reports wire size and reply length",
            "voice_wire_kb=" in line and "reply_chars=12" in line, True)

voice.speech = None
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    manager._text_chunk(g1_action_payload("Hello there.", "wave"), "g1_action")
check.equal("names a failed synthesis", "[voice] synthesis_failed" in captured.getvalue(), True)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    manager._text_chunk(g1_action_payload("", "none"), "g1_action")
check.equal("names an empty reply", "[voice] skipped_empty_reply" in captured.getvalue(), True)
voice.speech = "U1BFRUNI"

check.section("filler-only turns stay quiet")
for filler in ("Um...", "Oh.", "Uh, hmm", "  Hmm?  ", "mhm"):
    check.equal(f"{filler!r} is filler", manager._is_filler_only(filler), True)
for real in ("Um, hello", "Oh no", "Iris", "I love chess", "You", "", "..."):
    check.equal(f"{real!r} is not filler", manager._is_filler_only(real), False)

quiet = manager._quiet_chunk()
check.equal("speaks the G1 contract", quiet[1], "g1_action")
quiet_payload = json.loads(quiet[0])
check.equal("says nothing", quiet_payload["reply"], "")
check.equal("does nothing", quiet_payload["action"], "none")

filler_turn = list(manager._getting_response({"fake_transcription": "Um...", "image_data": np.zeros((4, 4, 3), dtype=np.uint8)}))
check.equal("filler turn yields one quiet chunk", len(filler_turn), 1)
check.equal("it is silent, not the listening fallback",
            json.loads(filler_turn[0][0]), {"reply": "", "action": "none"})
relaxed_calls_before = face_recognition.relaxed_calls
list(manager._getting_response({"fake_transcription": "Um...", "image_data": np.zeros((4, 4, 3), dtype=np.uint8)}))
check.equal("it never reached the face stage", face_recognition.relaxed_calls, relaxed_calls_before)

check.section("turn log: heard and said stand out, the rest stays quiet")
import turn_log
import turn_timing

def printed(call, *args, **env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            call(*args)
    finally:
        for key, value in old.items():
            os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
    return buffer.getvalue()

heard_line = printed(turn_log.heard, "Hello Iris", IRIS_LOG_COLOR="0")
check.equal("heard names the words", "HEARD" in heard_line and "'Hello Iris'" in heard_line, True)
check.equal("no colour codes when colour is off", "\033[" in heard_line, False)
check.equal("colour on by default", "\033[" in printed(turn_log.heard, "Hello Iris"), True)

said_line = printed(turn_log.said, "Hi there!", "wave", IRIS_LOG_COLOR="0")
check.equal("said shows reply and gesture", "'Hi there!'" in said_line and "[wave]" in said_line, True)
check.equal("no gesture tag for none", "[" in printed(turn_log.said, "Hi", "none", IRIS_LOG_COLOR="0"), False)
check.equal("silence is spelled out", "(silence)" in printed(turn_log.said, "", "none", IRIS_LOG_COLOR="0"), True)

check.equal("debug hidden by default", printed(turn_log.debug, "x"), "")
check.equal("debug shown when verbose", "x" in printed(turn_log.debug, "x", IRIS_LOG_VERBOSE="1"), True)

record = turn_timing._Turn("process_audio_img")
record.total_ms = 4019.2
record.spans = [
    {"name": "transcribe", "depth": 0, "ms": 185.5},
    {"name": "whisper.transcribe", "depth": 1, "ms": 145.9},
    {"name": "resolve_face_id", "depth": 0, "ms": 0.0},
    {"name": "reason", "depth": 0, "ms": 748.6},
    {"name": "api_response", "depth": 0, "ms": 3080.2},
]
record.facts = {"route": "g1 handshake", "first_chunk_at_ms": 1938.1}
summary = printed(turn_timing._print_breakdown, record, IRIS_LOG_COLOR="0")
check.equal("one line per turn", len(summary.strip().splitlines()), 1)
check.equal("names the stages in plain words",
            all(word in summary for word in ("heard 0.19s", "classify 0.75s", "reply 3.08s")), True)
check.equal("shows total and first reply", "4.02s total" in summary and "first reply at 1.94s" in summary, True)
check.equal("drops sub-spans and zero-cost stages", "whisper" not in summary and "face" not in summary, True)
verbose = printed(turn_timing._print_breakdown, record, IRIS_LOG_VERBOSE="1")
check.equal("verbose keeps the full table", "whisper.transcribe" in verbose and "facts route=" in verbose, True)

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
                                       "_BadInput", "_NoFace", "_FaceUnconfirmed", "_Thanks",
                                       "_UnsupportedAction",
                                       "_SecondaryChannel", "_G1Gesture"}, True)

check.report("PIPELINE OK")
