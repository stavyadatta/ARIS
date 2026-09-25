import json
from queue import Queue
from threading import Thread

from utils import PersonDetails, Neo4j, message_format, name_similarity
from .prompt import relationship_check_prompt

class _RelationshipChecker:
    def __init__(self) -> None:
        self.relationship_queue: Queue[PersonDetails] = Queue()
        self.relationship_prompt = message_format("system", relationship_check_prompt)

        relationship_thread = Thread(
                target=self.relationship_checker,
                daemon=True
        )
        relationship_thread.start()

    def adding_text2relationship_checker(self, person_details: PersonDetails):
        self.relationship_queue.put(person_details)

    def compare_name2db_names(self, name, threshold=55):
        db_names = Neo4j.get_db_people_names()
        highest_ratio = -1
        closest_name = None
        for db_name in db_names:
            ratio = name_similarity(name, db_name)
            if ratio > highest_ratio:
                highest_ratio = ratio
                closest_name = db_name

        if highest_ratio > threshold:
            print("The clostest name is ", closest_name)
            return closest_name
        else:
            print("No name is found, returning name is ", name)
            return name

    def find_similar_name(self, name_list):
        new_names = []
        for name in name_list:
            closest_name = self.compare_name2db_names(name, threshold=70)
            new_names.append(closest_name)
        return new_names
        
    def _develop_query_param(self, person_details: PersonDetails, name_list: list):
        query_param = {"face_id": person_details.get_attribute("face_id")}

        if name_list == []:
            raise Exception("The name list in the relation checker is empty")

        new_name_list = self.find_similar_name(name_list)

        for name, new_name in zip(name_list, new_name_list):
            query_param[name] = new_name

        return query_param

    def relationship_checker(self):
        from core_api import ChatGPT, AttributeFinder
        while True:
            person_details = self.relationship_queue.get()
            text_message = person_details.get_latest_user_message()

            total_messages = [self.relationship_prompt, text_message]

            try:
                response = ChatGPT.send_text_get_json(total_messages, stream=False)
            except Exception:
                print("Chatgpt failed to get relationships")
                try:
                    AttributeFinder.attr_checker(person_details)
                except Exception:
                    continue
                continue

            output = response.choices[0].message.content
            output_json = json.loads(output)

            if not output_json.get("is_relationship"):
                try:
                    AttributeFinder.attr_checker(person_details)
                except Exception:
                    continue
                continue

            name_list = output_json.get("names")
            query = output_json.get("cypher_query")

            try:
                query_param = self._develop_query_param(person_details, name_list)
            except Exception:
                # No name found
                try:
                    AttributeFinder.attr_checker(person_details)
                except Exception:
                    continue
                continue

            print("The relationship query parameter is going to execute ", query_param, query)

            Neo4j.write_query(query, **query_param)
            Neo4j.update_db_name_list()

            # going into the attribute checker 
            try:
                AttributeFinder.attr_checker(person_details)
            except Exception as e:
                print(f"Coming from relationship check into attr {e}" )
                continue
