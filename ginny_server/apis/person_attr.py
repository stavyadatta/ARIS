from utils import ApiObject
from .api_base import ApiBase

from utils import PersonDetails, Neo4j, message_format
from core_api import PersonDetectionCropper, ChatGPT, Grok, RelationshipChecker, AttributeFinder

class _PersonAttribute(ApiBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, person_details: PersonDetails):
        try:
            image = person_details.image
            if image is None:
                raise Exception("Image is None when vision was asked")

            cropped_person = PersonDetectionCropper.detect_and_crop_person(image)
            assert cropped_person is not None

            try:
                response = ChatGPT.process_image_and_text(cropped_person, person_details)
            except Exception as e:
                print("chatgpt failed ", e)
                response = Grok.process_image_and_text(cropped_person, person_details)

            llm_response = ""
            for chunk in response:
                llm_response += chunk
                yield ApiObject(chunk)
        
            llm_dict = message_format("assistant", llm_response)
            person_details.set_latest_llm_message(llm_dict)
            person_details.set_attribute("state", "speak")
            Neo4j.add_message_to_person(person_details)
            RelationshipChecker.adding_text2relationship_checker(person_details)

        except Exception as e:
            raise Exception(f"Exception in the _PersonAttribute {e}")

