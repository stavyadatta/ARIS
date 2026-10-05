"""Answer a plain "thank you" with a short "you're welcome".

The classifier prompt used to file a bare "Thank you" under "bad input", so a
person being polite got "I did not catch that" and a head scratch. The
reasoner now routes a pure thanks here before any model call. Nothing is
written to Neo4j and nothing moves: it is a courtesy, not a topic, and skipping
the embeddings keeps the reply fast.
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


THANKS_REPLIES = (
    "You are very welcome!",
    "Anytime, happy to help.",
    "My pleasure!",
    "No problem at all.",
    "You're welcome, glad I could help.",
)


class _Thanks(ApiBase):
    """Emit a speech-only acknowledgement of thanks."""

    def __call__(self, person_details: PersonDetails):
        reply = random.choice(THANKS_REPLIES)
        turn_log.step("thanks", "polite thanks; replying with a courtesy")
        yield ApiObject(g1_action_payload(reply, ACTION_NONE), mode=G1_ACTION_MODE)
