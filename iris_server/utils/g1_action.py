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

    `speech` is base64 16 kHz mono 16-bit PCM of `reply`, generated here rather
    than on the robot so Iris does not have to use the G1's own text-to-speech.
    It is omitted when speech generation is unavailable, and the client then
    falls back to the robot's voice -- so this degrades quality, never the turn.
    """
    payload = {"reply": reply, "action": action}
    if speech:
        payload["speech"] = speech
    return json.dumps(payload, ensure_ascii=False)


def g1_spoken_payload(reply: str, action: str) -> str:
    """The same payload, with Iris's own voice attached.

    Use this wherever the robot actually says something. Silent and error
    replies should keep using `g1_action_payload`, which costs nothing.

    Speech is generated here rather than on the robot because the G1's built-in
    text-to-speech is Chinese-first and sounds it, and because the workstation
    has the graphics cards. If generation fails the key is simply absent and the
    client falls back to the robot's own voice, so this degrades the voice and
    never the turn.
    """
    if not reply or not reply.strip():
        return g1_action_payload(reply, action)

    # Imported here, not at module scope: utils is imported by core_api, so a
    # top-level import would be circular.
    try:
        from core_api.kokoro_tts import KokoroTts
        speech = KokoroTts.speech_base64(reply)
    except Exception as e:
        print(f"[kokoro] unavailable, using the robot's voice: {e}")
        speech = None

    return g1_action_payload(reply, action, speech=speech)
