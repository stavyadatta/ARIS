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


# The reasoner may select only these states. `doing` describes the physical
# act so the reply can be written to fit it; the words themselves are
# generated per turn from what the person actually said.
G1_GESTURES = {
    "g1 wave": {
        "action": "wave",
        "doing": "waving hello to them",
    },
    "g1 handshake": {
        "action": "handshake",
        "doing": "reaching out to shake their hand",
    },
    "g1 high five": {
        "action": "high_five",
        "doing": "giving them a high five",
    },
    "g1 blow kiss left": {
        "action": "blow_kiss_with_left_hand",
        "doing": "blowing them a goodbye kiss",
    },
    "g1 blow kiss right": {
        "action": "blow_kiss_with_right_hand",
        "doing": "blowing them a goodbye kiss",
    },
    "g1 clap": {
        "action": "clamp",
        "doing": "applauding them",
    },
    "g1 hug": {
        "action": "hug",
        "doing": "opening your arms for a hug",
    },
    "g1 hand on heart": {
        "action": "right_hand_on_heart",
        "doing": "placing a hand on your heart",
    },
    # Whole-body routines the G1 client runs by name rather than by numeric id
    # (ExecuteAction(custom_name)). These move far more of the robot than the
    # gestures above, so the client still gates them behind its own allow list.
    "g1 waist drum dance": {
        "action": "waist_drum_dance",
        "doing": "dancing a waist drum dance for them",
    },
    "g1 spin discs": {
        "action": "spin_discs",
        "doing": "spinning discs like a DJ for them",
    },
    "g1 throw money": {
        "action": "throw_money",
        "doing": "throwing money in the air for them",
    },
}

G1_CONFIRMATIONS = {
    "g1 confirm wave": "wave at them",
    "g1 confirm handshake": "shake their hand",
    "g1 confirm high five": "give them a high five",
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
        # Keep the confirmation state in Neo4j until the next utterance.
        question = self._generated_line(
            person_details,
            f"""
            You think {self._who(person_details)} may have asked you
            to {G1_CONFIRMATIONS[state]}, but you are not sure you heard right.

            Ask them ONE short yes/no question to check, and make it obvious
            that "yes" is the answer that confirms it. Do not perform anything
            yet. Speak it out loud: no lists, no emoji, no stage directions.
            """,
        )
        self._remember_reply(person_details, question)
        print(f"[g1_action] state={state} action={ACTION_NONE} "
              f"(awaiting confirmation) question={question!r}")
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
        return self._generated_line(
            person_details,
            f"""
            You are {gesture["doing"]} right now, because they asked you to.

            React to what they actually said -- if they mentioned news, a
            feeling or a reason, respond to that, not just to the gesture. Do
            not narrate the gesture; they can see it.
            """,
        )

    def _generated_line(self, person_details: PersonDetails,
                        situation: str) -> str:
        """One short spoken line for this turn, or nothing.

        On failure the robot performs the gesture in silence rather than
        speaking a canned line. The arm is already committed by the time this
        runs, and a stock greeting answering an unrelated sentence reads worse
        than saying nothing at all.
        """
        try:
            with span("g1_gesture.reply_llm"):
                response = ChatGPT.send_text(
                    self._prompt(person_details, situation),
                    stream=False,
                    max_tokens=GESTURE_REPLY_MAX_TOKENS,
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[g1_action] reply generation failed, staying silent: {e}")
            return ""

    def _who(self, person_details: PersonDetails) -> str:
        """Name the person, or say plainly that Iris does not know them.

        A guest has no name, and interpolating the empty value left the prompt
        reading "talking with []", which invites the model to fill the blank.
        """
        name = person_details.get_attribute("name")
        return str(name) if name else "someone whose name you do not know"

    def _prompt(self, person_details: PersonDetails, situation: str) -> list:
        """Build a deliberately small prompt.

        No Neo4j retrieval and no conversation history: the reasoner has
        already fetched this person's record, and a gesture turn is the
        fastest path the robot has. Everything added here is paid before the
        arm moves.
        """
        system_prompt = f"""
            You are Iris, a humanoid robot talking with {self._who(person_details)}.
            {situation}
            Reply with ONE short spoken sentence, under fifteen words. You are
            speaking out loud: no lists, no emoji, no stage directions. Warm
            and natural, the way a person would say it.
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
