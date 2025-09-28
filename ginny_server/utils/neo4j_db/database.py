import os
import json
import uuid
import traceback
from queue import Queue
from time import perf_counter
from neo4j import GraphDatabase


from utils import PersonDetails

class _Neo4j:
    def __init__(self, neo4j_url="bolt://172.27.72.27:7687"):
        neo4j_passwd = os.environ["NEO4J_PASSWORD"]
        neo4j_user = "neo4j"
        self.driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_passwd))
        print("Connected to the database")

        self.relationship_queue = Queue()
        self.update_db_name_list()

    def close(self):
        self.driver.close()

    def read_query(self, query, **params):
        with self.driver.session() as session:
            return session.run(query, **params).data()

    def write_query(self, query, **params):
        with self.driver.session() as session:
            session.run(query, **params)

    def update_name_or_attribute(self, face_id=None, name=None, attributes=None, pid=None):
        if face_id:
            query = """
                MERGE (p:Person {face_id: $face_id})
                ON MATCH SET 
                    p.name = CASE 
                                WHEN $name IS NULL OR trim($name) = '' THEN p.name 
                                ELSE $name 
                            END,
                    p.attributes = COALESCE($attributes, p.attributes)
                """
            self.write_query(query, face_id=face_id, name=name, attributes=attributes)
        else:
            if name:
                query = """ 
                    MATCH (p:Person {name: $name})
                    WHERE elementId(p) = $pid
                    SET p.attributes = COALESCE($attributes, p.attributes) 
                """

                self.write_query(query, name=name, attributes=attributes, pid=pid)


    def create_or_update_person(self, face_id=None, name=None, state='speak'):
        from core_api import ChatGPT
        query = """
                MERGE (p:Person {face_id: $face_id})
                ON CREATE SET p.messages = '[]', p.state = $state
                ON MATCH SET 
                    p.messages = COALESCE(p.messages, '[]'),
                    p.state = COALESCE($state, p.state),
                    p.attributes = COALESCE(p.attributes, '[]')
                SET p.name = COALESCE($name, p.name)

                WITH p
                MERGE (latestMessage:Message {message_id: $assistant_message_id})
                ON CREATE SET 
                    latestMessage.message_number = 0,
                    latestMessage.role = "assistant",
                    latestMessage.text = $assistant_text,
                    latestMessage.embedding = $assistant_embedding,
                    latestMessage.face_id = $face_id

                MERGE (p)-[:MESSAGE]->(latestMessage)
        """
        assistant_text = "Hello"
        assistant_embedding = ChatGPT.get_openai_embedding(assistant_text)
        assistant_message_id = str(uuid.uuid4())

        with self.driver.session() as session:
            session.run(
                query, face_id=face_id, name=name, state=state,
                assistant_text=assistant_text, assistant_embedding=assistant_embedding,
                assistant_message_id=assistant_message_id
            )
        self.update_db_name_list()
        print("Created a new person")

    def get_person_details(self, face_id) -> PersonDetails:
        """
        Retrieve the details of a person 
        :param face_id: Unique identifier for the person.
        :return: PersonDetails Object
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {face_id: $face_id})
                RETURN p.face_id AS face_id, p.name AS name, p.messages AS messages, p.state AS state, p.attributes as attributes
                """,
                face_id=face_id
            )
            record = result.single()
            if record:
                return PersonDetails({
                    "face_id": record.get("face_id"),
                    "name": record.get("name"),
                    "messages": json.loads(record["messages"]) if record.get("messages") else [],
                    "state": record.get("state"),
                    "attributes": record.get("attributes")
                })
            else:
                return PersonDetails() 

    def describe_relationships_by_face_id(self, face_id):
        query = """
        MATCH (p:Person {face_id: $face_id})
        OPTIONAL MATCH (p)-[r]->(other:Person)
        OPTIONAL MATCH (other2:Person)-[r2]->(p)
        RETURN p.name AS selfName, 
            collect({
                type: type(r), direction: 'out', 
                target: other.name, attributes: other.attributes
            }) +
            collect({
                type: type(r2), direction: 'in', 
                source: other2.name, attributes: other2.attributes
            }) AS relations
        """

        result = self.read_query(query, face_id=face_id)
        if not result:
            return "No person found with that face_id."

        self_name = result[0]["selfName"] or "This person"
        relations = result[0]["relations"]
        sentences = []

        for rel in relations:
            attr_list = rel.get("attributes")
            attr_text = ""
            if attr_list and isinstance(attr_list, list) and len(attr_list) > 0:
                attr_text = f" Their attributes are: {', '.join(attr_list)}."

            if rel["direction"] == "out" and rel.get("target"):
                sentences.append(f"{self_name} is {rel['type'].lower()} of {rel['target']}.{attr_text}")
            elif rel["direction"] == "in" and rel.get("source"):
                sentences.append(f"{rel['source']} is {rel['type'].lower()} of {self_name}.{attr_text}")

        return " ".join(sentences) if sentences else f"{self_name} has no relationships."

    def bkp_describe_relationships_by_face_id(self, face_id):
        query = """
        MATCH (p:Person {face_id: $face_id})
        OPTIONAL MATCH (p)-[r]->(other:Person)
        OPTIONAL MATCH (other2:Person)-[r2]->(p)
        RETURN p.name AS selfName, 
            collect({type: type(r), direction: 'out', target: other.name}) +
            collect({type: type(r2), direction: 'in', source: other2.name}) AS relations
        """

        result = self.read_query(query, face_id=face_id)
        if not result:
            return "No person found with that face_id."

        self_name = result[0]["selfName"] or "This person"
        relations = result[0]["relations"]
        sentences = []

        for rel in relations:
            if rel["direction"] == "out" and rel["target"]:
                sentences.append(f"{self_name} is {rel['type'].lower()} of {rel['target']}.")
            elif rel["direction"] == "in" and rel["source"]:
                sentences.append(f"{rel['source']} is {rel['type'].lower()} of {self_name}.")

        return " ".join(sentences) if sentences else f"{self_name} has no relationships."


    def update_db_name_list(self):
        """ 
            Gets the latest names in the db and stores in the database name variable
        """
        query = """
            MATCH (p:Person)
            WHERE p.name IS NOT NULL AND p.name <> ""
            RETURN p.name AS name
        """
        name_result = self.read_query(query)
        self.people_names = [result["name"] for result in name_result]

    def get_people_without_face_id(self, name):
        query = """ 
            MATCH (p:Person {name: $name})
            RETURN 
            CASE
                WHEN p.face_id IS NOT NULL THEN true
                ELSE false
            END as hasFaceID
        """
        result = self.read_query(query, name=name)

        if result:
            return result[0]['hasFaceID']
        else:
            raise ValueError(f"The person {name} does not exist in the database")
        
    def get_db_people_names(self):
        return self.people_names

    def get_cos_msgs(self, text, face_id, top_k=20):
        """ 
            Gets all the cosine distance messages from the query of the face_id
        """
        cosine_query = """ 
        CALL db.index.vector.queryNodes(
            'message_embeddings', 
            $top_k, 
            $query_embedding
        ) YIELD node AS message, score
        WHERE message.face_id = $face_id
        MATCH (p:Person {face_id: $face_id})
        WITH message, score, p
        MATCH window = (m0:Message)-[:NEXT*0..1]->(message)-[:NEXT*0..1]->(m1:Message)
        RETURN message.message_id AS id, 
               message.text AS text, 
               message.role AS role, 
               p.name AS name,
               p.attributes AS attributes,
               score, 
               message.message_number as message_number,
               nodes(window) as chain
        """

        from utils import message_format
        # Getting query embedding 
        from core_api import ChatGPT
        query_embedding = ChatGPT.get_openai_embedding(text)


        results = self.read_query(cosine_query, query_embedding=query_embedding, 
                                  top_k=top_k, face_id=face_id)
        messages = []
        message_set = set()
        message_num_list = []

        for idx, row in enumerate(results):
            for msg in row["chain"]:
                if msg['message_number'] not in message_set:
                    llm_dict = message_format(msg['role'], msg['text'])
                    message_set.add(msg['message_number'])
                    message_num_list.append(msg['message_number'])
                    messages.append(llm_dict)

        return messages, message_num_list

    def get_last_k_msgs(self, face_id, k=20, all_msgs=False):
        """ 
            Get messages of the face_id:
            - If all_msgs=True → entire chain
            - Else → only last k messages
        """
        limit = ".." if all_msgs else f"..{k}"

        query = f""" 
            MATCH (p:Person {{face_id: $face_id}})
            WITH p
            MATCH (p)-[:MESSAGE]->(m:Message)
            WITH m
            MATCH window = (m0:Message)-[:NEXT*0{limit}]->(m)
            RETURN nodes(window) AS chain
        """

        from utils import message_format

        messages = []
        message_set = set()
        message_num_list = []

        t0 = perf_counter()
        results = self.read_query(query, face_id=face_id, k=k)
        t1 = perf_counter()
        time_counter = (t1 - t0) * 1000
        for result in results:
            row = result["chain"]
            for msg in row:
                if msg['message_number'] not in message_set:
                    llm_dict = message_format(msg['role'], msg['text'])
                    message_set.add(msg['message_number'])
                    message_num_list.append(msg['message_number'])
                    messages.append(llm_dict)

        return messages, message_num_list

    def get_person_messages(self, latest_message: dict, face_id: str, is_rag: bool=True, message_save: bool=False):
        """ 
            Takes the query, does the cosine distance on the messages of the person 
            and then returns them in ascending order, also reduces the returns the 
            past 20 messages along with the latest message at the end
        """
        # Message is in format {"<user>": "<message>"}
        latest_text = latest_message["content"]

        if is_rag:
            cos_msgs, cos_msg_num_list = self.get_cos_msgs(latest_text, face_id)

            # Getting last 5 messages
            last_20_msgs, last_20_msg_num_list = self.get_last_k_msgs(face_id)

            # Merge message_num_list and remove duplicates while maintaining order
            merged_message_num_list = sorted(set(cos_msg_num_list) | set(last_20_msg_num_list))
            # Create a mapping of message numbers to messages
            message_mapping = {num: msg for num, msg in zip(cos_msg_num_list, cos_msgs)}
            message_mapping.update({num: msg for num, msg in zip(last_20_msg_num_list, last_20_msgs)})  # Update with latest
            # Reconstruct merged messages list based on sorted message numbers
            merged_messages = [message_mapping[num] for num in merged_message_num_list]
        else:
            t0 = perf_counter()
            read_person_query = """ 
                MATCH (p:Person {face_id: $face_id})
                RETURN p.messages as messages
            """
            read_query_param = {
                "face_id": face_id
            }
            response = self.read_query(read_person_query, **read_query_param)
            merged_messages = json.loads(response[0].get("messages"))
            t1 = perf_counter()

            non_rag_time = (t1 - t0) * 1000.0
            print("The non rag time for Retrieve is ", non_rag_time)

        if message_save and not is_rag:
            query = """
                MATCH (p:Person {face_id: $face_id})
                SET p.messages = $merged_messages
            """
            query_params = {
                "face_id": face_id,
                "merged_messages": json.dumps(merged_messages)
            }
            self.write_query(query, **query_params)

        # Adding the latest_message to list 
        merged_messages.append(latest_message)
        return merged_messages

    def add_message_to_person(self, person_details: PersonDetails):
        from core_api import ChatGPT

        add_llm_msg_query = """ 
            MATCH (p:Person {face_id:$face_id})-[:MESSAGE]->(latestMessage:Message)
            WITH p, latestMessage
            CREATE (userMessage:Message {
                message_id: $user_message_id,
                message_number: latestMessage.message_number + 1,
                role: "user",
                text: $user_text,
                embedding: $user_embedding,
                face_id: $face_id
            })
            CREATE (llmMessage:Message {
                message_id: $llm_message_id,
                message_number: latestMessage.message_number + 2,
                role: "assistant",
                text: $llm_text,
                embedding: $llm_embedding,
                face_id: $face_id
            })
            MERGE (latestMessage)-[:NEXT]->(userMessage)
            MERGE (userMessage)-[:NEXT]->(llmMessage)
            MERGE (p)-[:MESSAGE]->(llmMessage)
            WITH p, latestMessage
            MATCH (p)-[oldRel:MESSAGE]->(latestMessage)
            SET p.state = COALESCE($state, p.state)
            DELETE oldRel
        """

        add_only_usr_msg = """ 
            MATCH (p:Person {face_id:$face_id})-[:MESSAGE]->(latestMessage:Message)
            WITH p, latestMessage
            CREATE (userMessage:Message {
                message_id: $user_message_id,
                message_number: latestMessage.message_number + 1,
                role: "user",
                text: $user_text,
                embedding: $user_embedding,
                face_id: $face_id
            })
            MERGE (latestMessage)-[:NEXT]->(userMessage)
            MERGE (p)-[:MESSAGE]->(userMessage)
            WITH p, userMessage
            MATCH (p)-[oldRel:MESSAGE]->(latestMessage)
            SET p.state = COALESCE($state, p.state)
            DELETE oldRel
        """

        face_id = person_details.get_attribute("face_id")
        state = person_details.get_attribute("state")
        print("The State inside Neo4j Add message function is ", state)

        usr_dict = person_details.get_latest_user_message()
        usr_txt = usr_dict["content"]
        usr_embedding = ChatGPT.get_openai_embedding(usr_txt)
        usr_message_id = str(uuid.uuid4())

        llm_dict = person_details.get_latest_llm_message()
        llm_txt = llm_dict.get("content")

        if llm_txt is not None:
            llm_embedding = ChatGPT.get_openai_embedding(llm_txt)
        else:
            llm_embedding = None
        llm_message_id = str(uuid.uuid4())

        query_params = {
            "face_id": face_id,
            "state": state,
            "user_text": usr_txt,
            "llm_text": llm_txt,
            "user_embedding": usr_embedding,
            "llm_embedding": llm_embedding,
            "user_message_id": usr_message_id,
            "llm_message_id": llm_message_id
        }

        try:
            if llm_dict == {}:
                self.write_query(add_only_usr_msg, **query_params)
            else:
                self.write_query(add_llm_msg_query, **query_params)
        except Exception as e:
            print(f"Error in add_message_to_person: {e}")
            traceback.print_exc()

    def adding_text2relationship_checker(self, person_details: PersonDetails):
        latest_usr_msg = person_details.get_latest_user_message()
        self.relationship_queue.put(latest_usr_msg)
