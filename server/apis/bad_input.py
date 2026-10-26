from utils import ApiObject
from .api_base import ApiBase

class _BadInput(ApiBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, person_details):
        response = ""

        yield ApiObject(response)

