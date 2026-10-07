"""Decline a physical request the G1 has no approved action for.

Iris inherited Pepper's movement APIs, which answer any motion request by
having an LLM invent NAO joint angles ("RShoulderPitch", "RWristYaw"). The G1
has neither those joints nor that contract, so a request like "can you wipe
your hands?" produced joint angles for the wrong robot.

Saying so is the honest answer, and naming what Iris *can* do turns a refusal
into an offer. The reply is the same sentence whichever path got here (the
planner found a step it cannot do, or the classifier routed a movement), and it
never repeats the person's words: those reached us through a language model and
this reply is read aloud, so quoting them would speak unchecked model text.
"""

from utils import (
    ACTION_NONE,
    ApiObject,
    G1_ACTION_MODE,
    PersonDetails,
    g1_action_payload,
)
from .api_base import ApiBase
from .g1_gesture import G1_GESTURES


def _action_named(state: str) -> str:
    return G1_GESTURES[state]["action"]


# What the refusal offers instead, as (allow-listed action name, spoken
# phrase), in the order they are spoken. Action names come from G1_GESTURES so
# a rename there cannot leave a stale offer behind.
#
# Deliberately never offered: a hug, which reads oddly volunteered; hand on
# heart, which is a response to sentiment rather than something to propose;
# and the blow-kiss pair, which only makes sense in a farewell.
OFFERED_GESTURES = (
    (_action_named("g1 wave"), "wave"),
    (_action_named("g1 handshake"), "shake hands"),
    (_action_named("g1 high five"), "give a high five"),
    (_action_named("g1 waist drum dance"), "dance"),
    (_action_named("g1 spin discs"), "DJ"),
    (_action_named("g1 throw money"), "throw money"),
)


def _spoken_list(phrases: list) -> str:
    """"a", "a or b", or "a, b or c"."""
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " or " + phrases[-1]


UNSUPPORTED_ACTION_REPLY = (
    "Due to safety, those actions have not been configured for me yet. "
    f"What I can do now is {_spoken_list([phrase for _, phrase in OFFERED_GESTURES])}."
)


class _UnsupportedAction(ApiBase):
    """Answer with speech only, never with a guessed body movement.

    Deliberately no Neo4j write: the exchange carries no new information about
    the person, and the reasoner has already returned them to conversation.
    """

    def __call__(self, person_details: PersonDetails):
        print(f"[g1_action] unsupported action requested; action={ACTION_NONE}")
        yield ApiObject(
            g1_action_payload(UNSUPPORTED_ACTION_REPLY, ACTION_NONE),
            mode=G1_ACTION_MODE,
        )
