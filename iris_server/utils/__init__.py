from fuzzywuzzy import fuzz

from .api_object import ApiObject
from .person_details import PersonDetails
from .neo4j_db import _Neo4j
from .secondary_details import SecondaryDetails
from .g1_action import (
    ACTION_NONE,
    ACTION_SCRATCH_HEAD,
    G1_ACTION_ERROR_MODE,
    G1_ACTION_MODE,
    g1_action_payload,
)

Neo4j = _Neo4j()

def message_format(role: str, content: str):
    return {"role": role, "content": content}

def name_similarity(name_1, name_2):
    """ 
        Fuzzy search to find similarity between two names
    """
    return fuzz.ratio(name_1, name_2)


__all__ = [
    "ACTION_NONE",
    "ACTION_SCRATCH_HEAD",
    "ApiObject",
    "G1_ACTION_ERROR_MODE",
    "G1_ACTION_MODE",
    "Neo4j",
    "PersonDetails",
    "SecondaryDetails",
    "g1_action_payload",
    "message_format",
]
