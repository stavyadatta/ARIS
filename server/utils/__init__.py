from fuzzywuzzy import fuzz

from .api_object import ApiObject
from .person_details import PersonDetails
from .neo4j_db import _Neo4j

Neo4j = _Neo4j()

def message_format(role: str, content: str):
    return {"role": role, "content": content}

def name_similarity(name_1, name_2):
    """ 
        Fuzzy search to find similarity between two names
    """
    return fuzz.ratio(name_1, name_2)


__all__ = ["Neo4j", "PersonDetails", "message_format", "ApiObject"]
