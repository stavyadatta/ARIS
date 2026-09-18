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


# Keep in step with the gestures G1_GESTURES exposes in g1_gesture.py. Only
# the ones worth offering unprompted: the blow-kiss pair reads oddly outside
# a farewell.
OFFERABLE_GESTURES = "wave, shake hands, give you a high five, or dance for you"

UNSUPPORTED_ACTION_REPLIES = (
    f"I have not learned that movement yet. I can {OFFERABLE_GESTURES}.",
    f"Sorry, that one is beyond me for now. I can {OFFERABLE_GESTURES}.",
    f"I cannot do that movement yet, I am afraid. I can {OFFERABLE_GESTURES}.",
    f"That is not something I know how to do yet. I can {OFFERABLE_GESTURES}.",
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
