"""The single response contract spoken between Iris and the G1 client.

Every reply the robot produces travels as one JSON object with a `reply` to
speak and an `action` to perform. The G1 client validates `action` against its
own allow-list, so a new action name here is inert until that client learns it.

A request that names several gestures ("shake my hand and then dance") adds an
optional `actions` array, in the order to perform them. `action` still carries
the first one, so a client that predates `actions` performs that one and
ignores the rest, and a client that reads `actions` still works against a
server that only sends `action`.
"""

import json
from typing import Sequence

# TextChunk has no dedicated action field, so the mode marks which chunks
# carry this contract rather than raw streamed speech.
G1_ACTION_MODE = "g1_action"
G1_ACTION_ERROR_MODE = "g1_action_error"

ACTION_NONE = "none"
ACTION_SCRATCH_HEAD = "scratch_head"

# `actions` is only sent from this length up; one action is just `action`.
# There is no upper limit on a sequence: the only bound is the transport's
# message size.
MIN_ACTIONS_FOR_SEQUENCE = 2

# The state a multi-gesture request is routed under. The gestures themselves
# travel beside it in this attribute because the executor fuzzy-matches the
# state string to an API, and a state spelling out several gestures could match
# the wrong one -- or none -- and move the robot differently than asked.
G1_SEQUENCE_STATE = "g1 sequence"
G1_SEQUENCE_ATTRIBUTE = "g1_sequence"

# A request with a step the robot cannot do is routed here instead, so nothing at
# all is performed and the reply says what is not possible. The model's wording
# of each such step is only logged, never spoken or stored, because it is
# unsanitised model text.
UNSUPPORTED_ACTION_STATE = "g1 unsupported action"


def g1_action_payload(reply: str, action: str, speech: str = None,
                      actions: Sequence[str] = ()) -> str:
    """Serialise one reply and its action, or action sequence, for the G1 client.

    `actions` is the full ordered sequence and is sent only when it holds at
    least MIN_ACTIONS_FOR_SEQUENCE entries; `action` is then its first entry.

    `speech` is base64 16 kHz mono 16-bit PCM of `reply`, generated on the
    server so the robot's Jetson does not have to synthesise it. It is omitted
    when generation is unavailable, and the client then falls back to its own
    voice -- so a failure here costs speed, never the turn.
    """
    payload = {"reply": reply, "action": action}
    if speech:
        payload["speech"] = speech
    if len(actions) >= MIN_ACTIONS_FOR_SEQUENCE:
        payload["actions"] = list(actions)
    return json.dumps(payload, ensure_ascii=False)
