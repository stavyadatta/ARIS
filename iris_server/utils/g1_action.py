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


def g1_action_payload(reply: str, action: str, speech: str = None) -> str:
    """Serialise one reply/action pair for the G1 client.

    `speech` is base64 16 kHz mono 16-bit PCM of `reply`, generated on the
    server so the robot's Jetson does not have to synthesise it. It is omitted
    when generation is unavailable, and the client then falls back to its own
    voice -- so a failure here costs speed, never the turn.
    """
    payload = {"reply": reply, "action": action}
    if speech:
        payload["speech"] = speech
    return json.dumps(payload, ensure_ascii=False)
