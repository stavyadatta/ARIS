import random

from core_api import RelationshipChecker
from utils import Neo4j, ApiObject, message_format

from .api_base import ApiBase

hmm_variations = [
    "Hmm", "hmm", "hmmmmm", "hhmmmm", "Hmmmm", "hmmm...", "hmm-hmm", "hmm.", "hmm?", "hmm?!",
    "hmm~", "HMMM", "hmm..", "hmm...!", "hmm?!!", "hmm!!", "hmm~?", "hmm-hmm-hmm", "hmmmm", "hmmz",
    "hmmph", "hmmff", "hmmf", "hmm-hm", "hmm, hmm", "hmm-mm", "Hmm~", "hmmm....", "Hmm!", "Hmm?",
    "Hmm...​", "Hmmmph", "Hmmmff", "Hmmf~", "hmmmmmm", "hmmmm?!", "HMMMM", "hmm. Hmm.", "hmm-hm-hm",
    "hmmmmmm....", "hmmmmmmmmmmmm", "hmm.. hmm?", "hmm-hm-hmmm", "hmm-hmm...", "hmmmmm~", "hmmmmmph",
    "hmmmmmff", "hmmmff..", "hmmf?!", "hmm-HMMM", "HMMMPH!", "Hmm, hmm.", "HMM?!", "hmmmmmmmm-hmmm",
    "hmm~hmm", "Hmm-hmm-hmm.", "Hmm...hmm?", "Hmmmmm?", "Hmmf!", "Hmmm-hmmm", "hmmz-hmm", "Hmm...",
    "hmmHmm", "Hmm, hmmm.", "Hmmmmmmm~", "HMMMMMMMMMMMM", "hmmz-hmmz", "Hmm~hmm~", "HMMMmmmmm",
    "Hmmhmm...", "hmmmhm", "hmmmhmm", "hmmmhmhmm", "Hmm...~", "hmmm?!", "hmmmmmf", "Hmm-hm-hm",
    "HmmMmMm", "Hmm...Hm!", "HmmmffHmmm", "Hmm...hmph", "hmmmmmph...", "Hmmf-hmm~", "hmmHmmHmm",
    "Hmm..~", "hmmHmmHmm~", "Hmm..Hmm..", "HmmmmmHmm", "Hmm~Hmm~Hmm", "HmmHmmHmm..", "Hmmph",
    "Hmmhm", "HmmMmmMm", "Hmmhmph", "HmmhmmMm"
]

# Function to randomly choose one "Hmm"
def random_hmm():
    return random.choice(hmm_variations)


class _Silent(ApiBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, person_details):
        latest_msg = person_details.get_latest_user_message()
        messages = [latest_msg]
        llm_pseudo_response = "Silence noted"

        for empty in llm_pseudo_response.split():
            yield ApiObject("")

        llm_dict = message_format("assistant", llm_pseudo_response)
        person_details.set_latest_llm_message(llm_dict)
        person_details.set_relevant_messages(messages + [llm_dict])

        Neo4j.add_message_to_person(person_details)
        RelationshipChecker.adding_text2relationship_checker(person_details)
