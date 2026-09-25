from utils import ApiObject, Neo4j, PersonDetails, message_format
from core_api import PersonDetectionCropper, ChatGPT, Grok, RelationshipChecker

from .api_base import ApiBase

class _PersonAttribute(ApiBase):
    def __init__(self) -> None:
        super().__init__()

    def _describe_person(self, cropped_person, person_details):
        try:
            return ChatGPT.process_image_and_text(cropped_person, person_details)
        except Exception as e:
            print("chatgpt failed ", e)
            return Grok.process_image_and_text(cropped_person, person_details)

    def __call__(self, person_details: PersonDetails):
        try:
            image = person_details.image
            if image is None:
                raise Exception("Image is None when vision was asked")

            cropped_person = PersonDetectionCropper.detect_and_crop_person(image)
            assert cropped_person is not None

            response = self._describe_person(cropped_person, person_details)

            llm_response = ""
            for chunk in response:
                llm_response += chunk
                yield ApiObject(chunk)
        
            person_details.set_latest_llm_message(
                message_format("assistant", llm_response))
            person_details.set_attribute("state", "speak")
            Neo4j.add_message_to_person(person_details)
            RelationshipChecker.adding_text2relationship_checker(person_details)

        except Exception as e:
            raise Exception(f"Exception in the _PersonAttribute {e}")

