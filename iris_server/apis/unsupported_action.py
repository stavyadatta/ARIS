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
UNSUPPORTED_ACTION_REPLIES = (
    "I have not learned that movement yet. I can wave or give you a high five.",
    "Sorry, that one is beyond me for now. Ask me to dance, though.",
    "I cannot do that one yet, I am afraid. I can shake hands, or DJ for you.",
    "That is not something I know how to do. Ask me to throw money, I am good at that.",
)


class _UnsupportedAction(ApiBase):
    """Answer with speech only, never with a guessed body movement.

    Deliberately no Neo4j write: the exchange carries no new information about
    the person, and the reasoner has already returned them to conversation.
    """

    def __call__(self, person_details: PersonDetails):
        reply = random.choice(UNSUPPORTED_ACTION_REPLIES)
        print(f"[g1_action] unsupported action requested; action={ACTION_NONE}")
        yield ApiObject(g1_action_payload(reply, ACTION_NONE), mode=G1_ACTION_MODE)
