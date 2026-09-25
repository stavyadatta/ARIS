from secondary_apis import secondary_apis_kit
from utils import SecondaryDetails, ApiObject

class _SecondaryChannel():
    def __init__(self) -> None:
        pass

    def _process_api_details(self, api_details, img, audio):
        return SecondaryDetails(audio, img, api_details)

    def __call__(self, img=None, audio=None, api_task: dict={}) -> ApiObject:
        print("The api task is \n", api_task, "\n")
        api_name = str(api_task.get("api_name"))
        api_details = api_task.get("api_details")
        secondary_details = self._process_api_details(api_details, img, audio)
        return secondary_apis_kit[api_name](secondary_details)


SecondaryChannel = _SecondaryChannel()

__all__ = ["SecondaryChannel"]
