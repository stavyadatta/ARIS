from typing import Iterator

from apis import api_call 
from utils import PersonDetails, ApiObject

from difflib import SequenceMatcher

def find_best_match(input_string, dictionary_keys):
    """
    Finds the dictionary key that best matches the input string.

    Parameters:
        input_string (str): The string to match.
        dictionary_keys (list): A list of dictionary keys.

    Returns:
        str: The key that matches the input string the most.
    """
    best_match = None
    highest_similarity = 0
    
    for key in dictionary_keys:
        # Compute similarity using SequenceMatcher
        similarity = SequenceMatcher(None, input_string, key).ratio()
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = key

    return best_match

class _Executor():
    def __init__(self):
        pass

    def __call__(self, person_details: PersonDetails) -> Iterator[ApiObject]:
        state = str(person_details.get_attribute("state"))

        best_key = find_best_match(state, api_call.keys())

        response = api_call[best_key](person_details)
        return response
