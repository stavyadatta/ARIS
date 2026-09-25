import json
from logging import exception
from threading import Thread
from queue import Queue

from utils import PersonDetails, Neo4j, message_format, person_details
from .prompt import get_attr_prompt, check_friend_name_prompt

class _AttributeFinder():
    def __init__(self) -> None:
        self.possible_attr_queue: Queue[PersonDetails] = Queue()
        
        attribute_thread = Thread(
            target=self.attr_checker_parallel,
            daemon=True
        )

        attribute_thread.start()

    def adding_text2attr_finder(self, person_details: PersonDetails):
        self.possible_attr_queue.put(person_details)

    def check_friend_name(self, messages):
        check_friend_prompt = check_friend_name_prompt()
        check_friend_system_message = message_format("system", check_friend_prompt)
        total_messages = [check_friend_system_message] + messages
        from core_api import ChatGPT

        try:
            response = ChatGPT.send_text_get_json(total_messages, stream=False)
        except Exception as e:
            print("Chatgpt failed at friend attributes part")
            raise e

        friend_name_content = response.choices[0].message.content
        friend_name_dict = json.loads(friend_name_content)
        return friend_name_dict

    def third_person_update(self, person_details: PersonDetails, output_attribute):
        from core_api import RelationshipChecker

        all_relevant_messages = person_details.get_relevant_messages()
        try:
            friend_name_dict = self.check_friend_name(all_relevant_messages)
        except Exception as e:
            raise e

        friend_name = friend_name_dict.get("name")
        if not bool(friend_name):
            return

        closest_name = RelationshipChecker.compare_name2db_names(friend_name)

        assert closest_name != person_details.get_attribute("name")

        friend_current_attributes_query = """ 
            MATCH  (p:Person {name: $name})
            RETURN elementId(p) AS pid, p.attributes AS attributes
            LIMIT 1
        """
        friend_result = Neo4j.read_query(
            friend_current_attributes_query,
            name=closest_name
        )


        if friend_result:
            friend_attr_list = friend_result[0].get("attributes", []) or []
            friend_pid = friend_result[0].get("pid")
            friend_attr_list.append(output_attribute)

            Neo4j.update_name_or_attribute(
                name=closest_name, 
                attributes=friend_attr_list, 
                pid=friend_pid
            )

    def have_I_heard_about_you(self, name):
        from core_api import RelationshipChecker
        closest_name = RelationshipChecker.compare_name2db_names(name, threshold=70)

        # Figure out if the person is without face_id in the db
        try:
            has_face_id = Neo4j.get_people_without_face_id(closest_name)
        except Exception as e:
            raise e

        # If face_id does not exist that means this person was talked about 
        # by other person therefore has heard about them
        if not has_face_id:
            return True
        else:
            return False

    def merging_nodes(self,
                      p1_person_details: PersonDetails,
                      p2_person_name: str) -> None:
        face_id = p1_person_details.get_attribute("face_id")

        # 1. fetch elementIds and attribute lists
        records = Neo4j.read_query(
            """
            MATCH (p1:Person {face_id:$face_id}),
                  (p2:Person {name:$name})
            WHERE elementId(p1) <> elementId(p2)
            RETURN elementId(p1) AS p1_eid,
                   elementId(p2) AS p2_eid,
                   p1.attributes AS attrs1,
                   p2.attributes AS attrs2
            """,
            face_id=face_id,
            name=p2_person_name
        )
        if not records:
            return
        rec     = records[0]
        p2_eid  = rec["p2_eid"]
        attrs1  = rec.get("attrs1") or []
        attrs2  = rec.get("attrs2") or []

        # 2. merge + dedupe attributes
        merged_attrs = list(dict.fromkeys(attrs1 + attrs2))

        # 3. update p1’s attributes & name
        # Neo4j.write_query(
        #     """
        #     MATCH (p1:Person {face_id:$face_id})
        #     SET p1.attributes = $merged_attrs,
        #         p1.name       = $new_name
        #     """,
        #     face_id=face_id,
        #     merged_attrs=merged_attrs,
        #     new_name=p2_person_name
        # )
        #
        # 4a. reattach incoming rels from p2 to p1
        incoming = Neo4j.read_query(
            """
            MATCH (p2)<-[r]-(n)
            WHERE elementId(p2) = $p2_eid
            RETURN elementId(n) AS nid, type(r) AS rtype, properties(r) AS props
            """,
            p2_eid=p2_eid
        )
        for rel in incoming:
            nid, rtype, props = rel["nid"], rel["rtype"], rel["props"]
            Neo4j.write_query(
                """
                MATCH (p1:Person {face_id:$face_id}), (n)
                WHERE elementId(n) = $nid
                CREATE (n)-[newR:`%s`]->(p1)
                SET newR = $props
                """ % rtype,
                face_id=face_id,
                nid=nid,
                props=props
            )

        # 4b. reattach outgoing rels from p2 to p1
        outgoing = Neo4j.read_query(
            """
            MATCH (p2)-[r]->(n)
            WHERE elementId(p2) = $p2_eid
            RETURN elementId(n) AS nid, type(r) AS rtype, properties(r) AS props
            """,
            p2_eid=p2_eid
        )
        for rel in outgoing:
            nid, rtype, props = rel["nid"], rel["rtype"], rel["props"]
            Neo4j.write_query(
                """
                MATCH (p1:Person {face_id:$face_id}), (n)
                WHERE elementId(n) = $nid
                CREATE (p1)-[newR:`%s`]->(n)
                SET newR = $props
                """ % rtype,
                face_id=face_id,
                nid=nid,
                props=props
            )

        # 5. delete p2 by elementId
        Neo4j.write_query(
            """
            MATCH (p2)
            WHERE elementId(p2) = $p2_eid
            DETACH DELETE p2
            """,
            p2_eid=p2_eid
        )

        print(f"The face id {face_id} name is {p2_person_name} attributes {merged_attrs}")
        Neo4j.update_name_or_attribute(
            face_id=face_id,
            name=p2_person_name,
            attributes=merged_attrs
        )

    def attr_checker(self, person_details: PersonDetails):
        from core_api import RelationshipChecker
        from core_api import ChatGPT
        text_message = person_details.get_latest_user_message()

        person_attributes = person_details.get_attribute("attributes")
        if person_attributes == None:
            person_attributes = []


        system_text = get_attr_prompt(person_attributes)
        system_message = message_format("system", system_text)
        total_messages = [system_message, text_message]

        try:
            response = ChatGPT.send_text_get_json(total_messages, stream=False)
        except Exception as e:
            print("Chatgpt failed to get attributes")
            raise e

        output = response.choices[0].message.content
        output_json = json.loads(output)

        output_attribute = output_json.get("attribute")
        input_name = output_json.get("name")
        check_friend = output_json.get("check_friend")

        attribute_bool = bool(output_attribute)
        name_bool = bool(input_name)
        face_id = person_details.get_attribute("face_id")

        # Adding the attribute to the existing attributes list
        person_attributes.append(output_attribute)

        print("These are the attributes going in ", person_attributes, attribute_bool, name_bool)

        # For checking if the attributes are being described for a third person 
        # who is friend, th expectation is of using pronounds like She or He
        if check_friend:
            try:
                self.third_person_update(person_details, output_attribute)
            except Exception as e:
                raise e

        if attribute_bool or name_bool:
            # If the person has been talked aboout before by someone else 
            # then use this
            try:
                have_I_heard_about_you = self.have_I_heard_about_you(input_name)
            except ValueError:
                have_I_heard_about_you = False
            except Exception as e:
                print(f"Some other error in the have_I_heard_about_you {e}")
                raise e

            if name_bool and have_I_heard_about_you:
                closest_name = RelationshipChecker.compare_name2db_names(input_name)
                self.merging_nodes(person_details, closest_name)
            if not check_friend and not have_I_heard_about_you:
                Neo4j.update_name_or_attribute(face_id=face_id, 
                                            name=input_name, 
                                            attributes=person_attributes
                                        )

    def attr_checker_parallel(self):
        while True:
            person_details = self.possible_attr_queue.get()
            try:
                self.attr_checker(person_details)
            except Exception as e:
                print(e)
                continue
