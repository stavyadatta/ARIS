"""Answer an instruction to be quiet with a deliberate, explicit silence.

Yielding empty speech chunks is not enough. The G1 conversation folding reads
a blank reply as the executor having failed and substitutes a "could you
repeat that?" line with a head scratch, so being told to be silent used to
make Iris speak. Saying nothing has to be said explicitly.
"""

from core_api import RelationshipChecker
from utils import (
    ACTION_NONE,
    ApiObject,
    G1_ACTION_MODE,
    Neo4j,
    PersonDetails,
    g1_action_payload,
    message_format,
)

from .api_base import ApiBase

# What the transcript records for a turn the robot deliberately sat out, so
# later context shows it was asked to stay quiet rather than that it failed.
SILENCE_TRANSCRIPT_NOTE = "Silence noted"


class _Silent(ApiBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, person_details: PersonDetails):
        yield ApiObject(g1_action_payload("", ACTION_NONE), mode=G1_ACTION_MODE)
        self._record_silent_turn(person_details)

    def _record_silent_turn(self, person_details: PersonDetails):
        latest_usr_message = person_details.get_latest_user_message()
        silence_note = message_format("assistant", SILENCE_TRANSCRIPT_NOTE)
        person_details.set_latest_llm_message(silence_note)
        person_details.set_relevant_messages([latest_usr_message, silence_note])

        Neo4j.add_message_to_person(person_details)
        RelationshipChecker.adding_text2relationship_checker(person_details)
