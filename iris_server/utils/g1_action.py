"""The single response contract spoken between Iris and the G1 client.

Every reply the robot produces travels as one JSON object with a `reply` to
speak and an `action` to perform. The G1 client validates `action` against its
own allow-list, so a new action name here is inert until that client learns it.
"""

import json

# TextChunk has no dedicated action field, so the mode marks which chunks
# carry this contract rather than raw streamed speech.
G1_ACTION_MODE = "g1_action"
G1_ACTION_ERROR_MODE = "g1_action_error"

ACTION_NONE = "none"
ACTION_SCRATCH_HEAD = "scratch_head"


def g1_action_payload(reply: str, action: str) -> str:
    """Serialise one reply/action pair for the G1 client.

    Speech is deliberately not carried here. Iris's voice is synthesised by the
    Kokoro sidecar the client talks to directly (g1_client_cpp/robot/
    kokoro_client), which keeps the audio off this wire and lets the sidecar run
    the Python 3.10 that kokoro-onnx needs -- the robot's system Python is 3.8.
    """
    return json.dumps({"reply": reply, "action": action}, ensure_ascii=False)

