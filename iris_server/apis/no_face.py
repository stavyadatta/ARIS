"""Ask the person to step into view when the camera cannot find them.

A missing face id used to collapse into "bad input", which the G1 client
spoke as an audio retry request ("I did not catch that").  That sent people
repeating themselves louder at a robot whose ears were fine and whose eyes
were the problem.  This API keeps the two failures distinct.
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


# Phrased as a request to be seen, never as a complaint about hearing.
NO_FACE_REPLIES = (
    "Sorry, I cannot see you at the moment. Could you let me see you, please?",
    "I am not able to see your face right now. Would you mind standing in front of me?",
    "I seem to have lost sight of you. Could you come where I can see you, please?",
    "I cannot quite see you. Could you please face me so I know who I am talking to?",
    "My camera is not finding you just now. Please step in front of me so I can see you.",
)


class _NoFace(ApiBase):
    """Emit a speech-only request for visibility.

    Deliberately no Neo4j write and no body action: with no face id there is
    no person record to attach the exchange to, and the robot must not move
    towards someone it cannot currently locate.
    """

    def __call__(self, person_details: PersonDetails):
        reply = random.choice(NO_FACE_REPLIES)
        print(f"[iris_action] state=no face action={ACTION_NONE} reply={reply!r}")
        yield ApiObject(g1_action_payload(reply, ACTION_NONE), mode=G1_ACTION_MODE)
