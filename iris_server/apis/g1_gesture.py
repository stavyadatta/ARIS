"""Safe, structured gesture intents for the Unitree G1 client.

This API intentionally does not know anything about robot joint angles or
Unitree action IDs.  It emits a small allow-listed intent which the G1 client
must validate and map to its own approved action implementation.
"""

from utils import (
    ACTION_NONE,
    ApiObject,
    G1_ACTION_ERROR_MODE,
    G1_ACTION_MODE,
    Neo4j,
    PersonDetails,
    g1_action_payload,
    message_format,
)
from turn_timing import span
from .api_base import ApiBase


# The reasoner may select only these states.  Keep replies short because the
# robot client can speak them before it performs the corresponding gesture.
G1_GESTURES = {
    "g1 wave": {
        "action": "wave",
        "reply": "Hello! It is nice to meet you.",
    },
    "g1 handshake": {
        "action": "handshake",
        "reply": "Nice to meet you too.",
    },
    "g1 high five": {
        "action": "high_five",
        "reply": "High five!",
    },
    "g1 blow kiss left": {
        "action": "blow_kiss_with_left_hand",
        "reply": "Goodbye! It was lovely talking with you.",
    },
    "g1 blow kiss right": {
        "action": "blow_kiss_with_right_hand",
        "reply": "See you next time! Take care.",
    },
    "g1 clap": {
        "action": "clamp",
        "reply": "Bravo!",
    },
    "g1 hug": {
        "action": "hug",
        "reply": "Come here, let me give you a hug.",
    },
    "g1 hand on heart": {
        "action": "right_hand_on_heart",
        "reply": "That means a lot to me, thank you.",
    },
}

G1_CONFIRMATIONS = {
    "g1 confirm wave": "Did you ask me to wave? Please say yes to confirm.",
    "g1 confirm handshake": "Did you ask me to shake hands? Please say yes to confirm.",
    "g1 confirm high five": "Did you ask me for a high five? Please say yes to confirm.",
}

STATE_SPEAK = "speak"


class _G1Gesture(ApiBase):
    """Emit one G1 gesture contract instead of Pepper joint-angle JSON."""

    def __call__(self, person_details: PersonDetails):
        state = str(person_details.get_attribute("state"))

        if state in G1_CONFIRMATIONS:
            yield self._ask_for_confirmation(person_details, state)
        elif state in G1_GESTURES:
            yield self._perform_gesture(person_details, state)
        else:
            yield self._reject_unknown_state(state)

    def _ask_for_confirmation(self, person_details: PersonDetails,
                              state: str) -> ApiObject:
        question = G1_CONFIRMATIONS[state]
        self._remember_reply(person_details, question)
        # Keep the confirmation state in Neo4j until the next utterance.
        with span("g1_gesture.persist"):
            Neo4j.add_message_to_person(person_details)
        print(f"[g1_action] state={state} action={ACTION_NONE} (awaiting confirmation)")
        return ApiObject(
            g1_action_payload(question, ACTION_NONE), mode=G1_ACTION_MODE
        )

    def _perform_gesture(self, person_details: PersonDetails,
                         state: str) -> ApiObject:
        gesture = G1_GESTURES[state]
        self._remember_reply(person_details, gesture["reply"])
        person_details.set_attribute("state", STATE_SPEAK)
        with span("g1_gesture.persist"):
            Neo4j.add_message_to_person(person_details)
        print(f"[g1_action] state={state} action={gesture['action']}")
        return ApiObject(
            g1_action_payload(gesture["reply"], gesture["action"]),
            mode=G1_ACTION_MODE,
        )

    def _reject_unknown_state(self, state: str) -> ApiObject:
        """Never substitute a guessed action for an unrecognised state.

        Unreachable while the reasoner prompt is obeyed, but a wrong gesture
        is a physical event, so an unknown state must fail loudly instead.
        """
        print(f"[g1_action] rejected unexpected state={state!r}")
        return ApiObject(g1_action_payload("", ""), mode=G1_ACTION_ERROR_MODE)

    def _remember_reply(self, person_details: PersonDetails, reply: str):
        reply_message = message_format("assistant", reply)
        person_details.set_latest_llm_message(reply_message)
        person_details.set_relevant_messages([reply_message])
