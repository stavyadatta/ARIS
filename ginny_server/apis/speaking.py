from typing import Any

from core_api import ChatGPT, Grok, RelationshipChecker, AttributeFinder
from utils import PersonDetails, Neo4j, message_format, ApiObject
from .api_base import ApiBase

class _Speaking(ApiBase):
    def __init__(self) -> None:
        super().__init__()

    def _developing_system_prompt(self, 
                                  person_name, 
                                  person_attributes, 
                                  person_relationships
        ):
        system_prompt = f"""
            Your are playing the role of Ginny robot which is a humanoid, as part of this 
            role you are a supposed to have friendly human conversations similar to 
            how people on facebook messenger chat.

            Take care of following before responding 
            1) Is my answer relevant to what the user asked 
            2) Is my answer short, I hope I am not speaking a lot 
            3) Am I polite
            4) Never admit visual errors; visual input is handled by another assistant discreetly.
            6) You may receive dancing context, do not admit you cannot dance, being a 
            robot you can certainly dance. 
            7) Replace your output "as large language model" with "as a robot"
            8) Donot use the following characters: '*', 'bullet points', 'numbered list'

            If you are asked questions like do you remember me, if you have context you should reply 
            with yes along with their names and there shared experiences with you 

            for example, treat texts in <> as conditional prompts
            ```

            input: Hey do you remember me
            response: <If name in context> yes I remmeber you, your name is <name> and you like <examples from context>

            input: Hey how are you 
            output: I am good, great to see you <name> how are you doing

            input: What did you say before 
            output: <Use latest conversation messages to answer this question>

            input: What do you know about my friendship <or any other relationship>
            output: I know you are friends <or any other relationship> with <people name if details have been provided>
            ```

            Here are some more details about the person 

            name: {person_name}
            person_attributes: {person_attributes}

            Here are the relationships this person has with people: {person_relationships}
        """

        system_dict = message_format("system", system_prompt)
        return [system_dict]
        
    def __call__(self, person_details: PersonDetails) -> Any:
        face_id = person_details.get_attribute("face_id")
        latest_msg = person_details.get_latest_user_message()
        messages = Neo4j.get_person_messages(latest_msg, face_id)

        # Developing system prompt 
        person_attributes = person_details.get_attribute("attributes")
        person_name = person_details.get_attribute("name")
        person_relationships = Neo4j.describe_relationships_by_face_id(face_id)
        system_dict = self._developing_system_prompt(
            person_name, 
            person_attributes, 
            person_relationships
        )

        total_prompt = system_dict + messages 
        
        # response = Llama.send_to_model(total_prompt, stream=True)
        try:
            response = Grok.send_text(total_prompt, stream=True, grok_model="grok-3")
        except Exception as e:
            print("grok failed ", e)
            response = ChatGPT.send_text(total_prompt, stream=True, model='gpt-4-turbo') 
        # response = Claude.process_text(messages, system_dict, stream=True)

        llm_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                llm_response += content
                yield ApiObject(content)
        
        llm_dict = message_format("assistant", llm_response)
        person_details.set_latest_llm_message(llm_dict)
        person_details.set_relevant_messages(messages + [llm_dict])

        Neo4j.add_message_to_person(person_details)
        RelationshipChecker.adding_text2relationship_checker(person_details)
