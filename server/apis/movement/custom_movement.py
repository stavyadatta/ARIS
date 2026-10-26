from typing import Any

from core_api import ChatGPT
from utils import PersonDetails, Neo4j, message_format, ApiObject

from .custom_prompt import movement_prompt
from ..api_base import ApiBase

class _CustomMovement(ApiBase):
    def __init__(self) -> None:
        super().__init__()

    def _developing_system_prompt(self):
        system_prompt = movement_prompt
        system_dict = message_format("system", system_prompt)
        return [system_dict]
        
    def __call__(self, person_details: PersonDetails) -> Any:
        latest_usr_message = person_details.get_latest_user_message()
        system_dict = self._developing_system_prompt()
        total_prompt = system_dict + [latest_usr_message]
        
        print("The total prompt is ", total_prompt)
        response = ChatGPT.send_text_get_json(total_prompt, stream=False, max_tokens=2000)

        print(response.choices[0].message.content)
        yield ApiObject(response.choices[0].message.content, mode='custom_movement') 

        llm_dict = message_format("assistant", "The movement has been performed")
        person_details.set_latest_llm_message(llm_dict)
        person_details.set_attribute("state", "speak")
        Neo4j.add_message_to_person(person_details)
