"""Decline a physical request the G1 has no approved action for.

Iris inherited Pepper's movement APIs, which answer any motion request by
having an LLM invent NAO joint angles ("RShoulderPitch", "RWristYaw"). The G1
has neither those joints nor that contract, so a request like "can you wipe
your hands?" produced joint angles for the wrong robot.

Saying so is the honest answer, and naming what Iris *can* do turns a refusal
into an offer.
"""

import random

from utils import (
    ACTION_NONE,
    ApiObject,
    G1_ACTION_MODE,
    PersonDetails,
    UNSUPPORTED_STEPS_ATTRIBUTE,
    g1_action_payload,
)
from .api_base import ApiBase


# Each refusal names a different couple of things Iris can do instead of
# reciting the whole repertoire. There are eleven gestures; listing them all
# would be a twenty-second menu, and this reply is spoken aloud -- every
# character costs speaking time and then a matching stretch of microphone
# blanking on the client before it can hear an answer. Rotating keeps each
# refusal short while letting somebody discover more across a few attempts.
#
# Deliberately never offered: a hug, which reads oddly volunteered; hand on
# heart, which is a response to sentiment rather than something to propose;
# and the blow-kiss pair, which only makes sense in a farewell.
_REFUSALS_WITH_OFFERS = (
    ("I have not learned that movement yet.", "I can wave or give you a high five."),
    ("Sorry, that one is beyond me for now.", "Ask me to dance, though."),
    ("I cannot do that one yet, I am afraid.", "I can shake hands, or DJ for you."),
    ("That is not something I know how to do.", "Ask me to throw money, I am good at that."),
)
UNSUPPORTED_ACTION_REPLIES = tuple(
    f"{refusal} {offer}" for refusal, offer in _REFUSALS_WITH_OFFERS
)
_OFFERS = tuple(offer for _, offer in _REFUSALS_WITH_OFFERS)

# A request can hold any number of steps the robot cannot do, but the reply is
# spoken, so it names only the first few and says there are more. Every step
# is still refused: this only shortens the sentence.
STEPS_NAMED_ALOUD = 3


def _named_aloud(steps: list) -> str:
    """"a", "a and b", "a, b and c", or "a, b, c and more"."""
    named = steps[:STEPS_NAMED_ALOUD]
    if len(steps) > STEPS_NAMED_ALOUD:
        return ", ".join(named) + " and more"
    if len(named) == 1:
        return named[0]
    return ", ".join(named[:-1]) + " and " + named[-1]


def spoken_refusal_naming(steps: list) -> str:
    """Say what cannot be done, that none of it was started, and what can be."""
    return (
        f"I cannot {_named_aloud(steps)} yet, so I will not start any of it. "
        f"{random.choice(_OFFERS)}"
    )


class _UnsupportedAction(ApiBase):
    """Answer with speech only, never with a guessed body movement.

    Deliberately no Neo4j write: the exchange carries no new information about
    the person, and the reasoner has already returned them to conversation.
    """

    def __call__(self, person_details: PersonDetails):
        reply = self._reply_for(person_details)
        print(f"[g1_action] unsupported action requested; action={ACTION_NONE}")
        yield ApiObject(g1_action_payload(reply, ACTION_NONE), mode=G1_ACTION_MODE)

    def _reply_for(self, person_details: PersonDetails) -> str:
        """Name the refused steps when the reasoner recorded them, else decline generally."""
        steps = person_details.get_attribute(UNSUPPORTED_STEPS_ATTRIBUTE)
        if steps:
            return spoken_refusal_naming(steps)
        return random.choice(UNSUPPORTED_ACTION_REPLIES)
