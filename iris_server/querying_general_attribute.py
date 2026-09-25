import enum
import json
from core_api import ChatGPT
from utils import Neo4j, message_format

def main(face_id):
    query = """
    What is my name, reply in json format which please with single word. Also please write 
    down attributes about the person that you found relevant, this can be 
    their likings, their hobbies, their jobs, what they do in there part time 
    if they are male or female, if they are introvert or extrovert

    Here is an example of output 
    ```
    {
        "name_reasoning": <reasoning>,
        "name": <name>
        "attribute_reasoning": <reasoning>,
        "attributes": [
            <attribute_1>,
            <attribute_2>,
            <attribute_3>
        ]
    }

    ```
    If you think you donot know the name then put it out like this 
    ```
    {
        "reasoning": <reasoning>,
        "name": "",
        "attribute_reasoning": <reasoning>,
        "attributes": [
            <attribute_1>,
            <attribute_2>,
            <attribute_3>
        ]
    }
    ```
    Attributes can be something like this 
    {
        "reasoning": <reasoning>,
        "name": <name>,
        "attribute_reasoning": <reasoning>,
        "attributes": [
            "likes baseball",
            "Maybe like academics or robotics",
            "Can be a guy not sure"
        ]
    }

    Be as detailed as possible, can be more than 3 points
    """

    getting_all_messages = """ 
    MATCH (p:Person {face_id:$face_id})
    WITH p
    MATCH (p)-[:MESSAGE]->(lastMessage:Message)
    WITH lastMessage
    MATCH window = (m0:Message)-[:NEXT*]->(lastMessage:Message)
    RETURN [n IN nodes(window) | {message_id: n.message_id, text: n.text, role: n.role, message_number: n.message_number}] AS chain
    """ 
    results = Neo4j.read_query(getting_all_messages, face_id=face_id)
    messages = []
    message_set = set()
    message_num_list = []

    for idx, result in enumerate(results):
        row = result["chain"]
        for msg in row:
            if msg['message_number'] not in message_set:
                print("The text is ", msg['text'])
                llm_dict = message_format(msg['role'], msg['text'])
                message_set.add(msg['message_number'])
                message_num_list.append(msg['message_number'])
                messages.append(llm_dict)

    print("The message_num_list is ", message_num_list)
    if message_num_list != []:
        sorted_pairs = sorted(zip(message_num_list, messages), key=lambda x: x[0])
        message_num_list, messages= zip(*sorted_pairs)
        messages = list(messages)
        messages.append(message_format("user", query))

        response = ChatGPT.send_text_get_json(messages[:], stream=False)
        print("The response is ", response)
        print("The statement is this ", response.choices[0].message.content)

        personality_attr = json.loads(response.choices[0].message.content)
        name = personality_attr.get("name")
        attributes = personality_attr.get("attributes")

        create_attr = """ 
        MERGE (p:Person {face_id:$face_id})
        ON MATCH SET 
            p.name = COALESCE($name, p.name),
            p.attributes = COALESCE($attributes, p.attributes)
        """
        Neo4j.write_query(create_attr, face_id=face_id, name=name, attributes=attributes)

if __name__ == "__main__":
    get_name_query = """
        MATCH (p:Person)
        WHERE p.face_id = "face_65"
        RETURN p.face_id as face_id
    """
    results = Neo4j.read_query(get_name_query)
    for idx, row in enumerate(results):
        print("All tried face ids ", row["face_id"])
        face_id = row["face_id"]
        main(face_id)
