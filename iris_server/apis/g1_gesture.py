"""Safe, structured gesture intents for the Unitree G1 client.

This API intentionally does not know anything about robot joint angles or
Unitree action IDs.  It emits a small allow-listed intent which the G1 client
must validate and map to its own approved action implementation.
"""

from core_api import ChatGPT
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

# One short spoken line. The reply travels as a single g1_action JSON object,
# so there is nothing to stream and a blocking call costs no more than a
# streamed one.
GESTURE_REPLY_MAX_TOKENS = 40


# The reasoner may select only these states.  Keep replies short because the
# robot client can speak them before it performs the corresponding gesture.
G1_GESTURES = {
    "g1 wave": {
        "action": "wave",
        "doing": "waving hello to them",
        "reply": "Hello! It is nice to meet you.",
    },
    "g1 handshake": {
        "action": "handshake",
        "doing": "reaching out to shake their hand",
        "reply": "Nice to meet you too.",
    },
    "g1 high five": {
        "action": "high_five",
        "doing": "giving them a high five",
        "reply": "High five!",
    },
    "g1 blow kiss left": {
        "action": "blow_kiss_with_left_hand",
        "doing": "blowing them a goodbye kiss",
        "reply": "Goodbye! It was lovely talking with you.",
    },
    "g1 blow kiss right": {
        "action": "blow_kiss_with_right_hand",
        "doing": "blowing them a goodbye kiss",
        "reply": "See you next time! Take care.",
    },
    "g1 clap": {
        "action": "clamp",
        "doing": "applauding them",
        "reply": "Bravo!",
    },
    "g1 hug": {
        "action": "hug",
        "doing": "opening your arms for a hug",
        "reply": "Come here, let me give you a hug.",
    },
    "g1 hand on heart": {
        "action": "right_hand_on_heart",
        "doing": "placing a hand on your heart",
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
        """Speak first, then record the turn.

        Every reply here is a constant string that is already known, so making
        the person wait on two embedding round trips and a Cypher write before
        hearing it buys nothing. The consumer drains this generator inside the
        same RPC, so the write still completes before the turn ends -- the
        pending-confirmation state is read back on a later RPC and cannot race
        it. silent.py:31-33 already orders it this way.
        """
        state = str(person_details.get_attribute("state"))

        if state in G1_CONFIRMATIONS:
            yield self._ask_for_confirmation(person_details, state)
        elif state in G1_GESTURES:
            yield self._perform_gesture(person_details, state)
        else:
            yield self._reject_unknown_state(state)
            return

        self._record_turn(person_details)

    def _record_turn(self, person_details: PersonDetails):
        with span("g1_gesture.persist"):
            Neo4j.add_message_to_person(person_details)

    def _ask_for_confirmation(self, person_details: PersonDetails,
                              state: str) -> ApiObject:
        question = G1_CONFIRMATIONS[state]
        # Keep the confirmation state in Neo4j until the next utterance.
        self._remember_reply(person_details, question)
        print(f"[g1_action] state={state} action={ACTION_NONE} (awaiting confirmation)")
        return ApiObject(
            g1_action_payload(question, ACTION_NONE), mode=G1_ACTION_MODE
        )

    def _perform_gesture(self, person_details: PersonDetails,
                         state: str) -> ApiObject:
        gesture = G1_GESTURES[state]
        reply = self._spoken_reply_for(person_details, gesture)
        self._remember_reply(person_details, reply)
        person_details.set_attribute("state", STATE_SPEAK)
        print(f"[g1_action] state={state} action={gesture['action']} reply={reply!r}")
        return ApiObject(
            g1_action_payload(reply, gesture["action"]),
            mode=G1_ACTION_MODE,
        )

    def _spoken_reply_for(self, person_details: PersonDetails,
                          gesture: dict) -> str:
        """Answer what the person actually said, rather than a written line.

        Falls back to that written line on any failure. The arm is already
        committed by the time this runs, so a text model being slow or
        unavailable must never stop the robot speaking at all.
        """
        try:
            with span("g1_gesture.reply_llm"):
                response = ChatGPT.send_text(
                    self._reply_prompt(person_details, gesture),
                    stream=False,
                    max_tokens=GESTURE_REPLY_MAX_TOKENS,
                )
            reply = response.choices[0].message.content.strip()
            return reply or gesture["reply"]
        except Exception as e:
            print(f"[g1_action] reply generation failed, using written line: {e}")
            return gesture["reply"]

    def _reply_prompt(self, person_details: PersonDetails, gesture: dict) -> list:
        """Build a deliberately small prompt.

        No Neo4j retrieval and no conversation history: the reasoner already
        fetched this person's record, and a gesture turn is the fastest path
        the robot has. Everything added here is paid before the arm moves.
        """
        system_prompt = f"""
            You are Iris, a humanoid robot talking with {person_details.get_attribute("name")}.
            You are {gesture["doing"]} right now, because they asked you to.

            Reply with ONE short spoken sentence, under fifteen words. React to
            what they actually said -- if they mentioned news, a feeling or a
            reason, respond to that, not just to the gesture. Do not narrate the
            gesture; they can see it.

            You are speaking out loud: no lists, no emoji, no stage directions.
            Warm and natural, the way a person would say it.
        """
        return [
            message_format("system", system_prompt),
            person_details.get_latest_user_message(),
        ]

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
