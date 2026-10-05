"""Ask who is speaking when the face match is only a guess.

A face matched through the relaxed fallback skips the size and side-face
checks, so a clipped or turned face can land on the wrong stored person --
which is how Iris came to greet people by someone else's name. Rather than
trust the guess, ask to see the face properly. The reply names no one, writes
nothing to Neo4j and moves nothing: the record belongs to whoever the face
really is, and we do not yet know.
"""

import random

import turn_log
from utils import (
    ACTION_NONE,
    ApiObject,
    G1_ACTION_MODE,
    PersonDetails,
    g1_action_payload,
)
from .api_base import ApiBase


UNCONFIRMED_FACE_REPLIES = (
    "I am not sure who I am talking to yet. Could you lean in so I can see your face properly?",
    "I would rather not guess who you are. Could you bring your face a little closer to my camera?",
    "Sorry, I cannot tell who you are from here. Would you mind leaning in so I can check?",
    "I only caught part of your face, so I am not certain who you are. Could you lean in for me?",
)


class _FaceUnconfirmed(ApiBase):
    """Emit a speech-only request to see the face, without naming anyone."""

    def __call__(self, person_details: PersonDetails):
        reply = random.choice(UNCONFIRMED_FACE_REPLIES)
        turn_log.step("unconfirmed", "face match is a guess; asking before assuming who")
        yield ApiObject(g1_action_payload(reply, ACTION_NONE), mode=G1_ACTION_MODE)
