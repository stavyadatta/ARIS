from typing import Any

from utils import PersonDetails

class ApiBase():
    def __init__(self) -> None:
        pass

    # TODO check whether the person_details object is required or can we make 
    # do with something lighter
    def __call__(self, person_details: PersonDetails) -> Any:
        pass
