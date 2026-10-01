"""Ask the person to step into view when the camera cannot find them.

A missing face id used to collapse into "bad input", which the G1 client
spoke as an audio retry request ("I did not catch that").  That sent people
repeating themselves louder at a robot whose ears were fine and whose eyes
were the problem.  This API keeps the two failures distinct.
"""

import random

from core_api import PersonDetectionCropper
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

# The body is in shot but the face is not -- the usual case, because the
# camera looks down at the chest. Moving back would make it worse, so ask
# the person to bring their face closer to the lens.
BODY_ONLY_REPLIES = (
    "I can see you, but not your face. Could you lean in a little so I know who you are?",
    "I can tell someone is here, but I cannot see your face yet. Would you mind bringing it closer to my camera?",
    "I can see you standing there, but my camera is not catching your face. Could you lean in toward me, please?",
)


class _NoFace(ApiBase):
    """Emit a speech-only request for visibility.

    When a person is in shot without a visible face, say so; it is a different
    request from "I cannot see you at all". The face stays unrecognised either
    way, so this never guesses who the person is.

    Deliberately no Neo4j write and no body action: with no face id there is
    no person record to attach the exchange to, and the robot must not move
    towards someone it cannot currently locate.
    """

    def __call__(self, person_details: PersonDetails):
        replies = BODY_ONLY_REPLIES if self._body_in_view(person_details.image) else NO_FACE_REPLIES
        reply = random.choice(replies)
        print(f"[iris_action] state=no face action={ACTION_NONE} reply={reply!r}")
        yield ApiObject(g1_action_payload(reply, ACTION_NONE), mode=G1_ACTION_MODE)

    @staticmethod
    def _body_in_view(image) -> bool:
        """A detector fault must not silence the robot, so it counts as no body."""
        if image is None:
            return False
        try:
            return PersonDetectionCropper.detect_and_crop_person(image) is not None
        except Exception as e:
            print(f"[iris_action] person detector failed: {e}")
            return False
