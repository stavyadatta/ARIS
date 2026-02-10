import json
from openai import OpenAI
from utils import Neo4j, message_format
from core_api.grok import _GrokHandler

Grok = _GrokHandler()
face_id = "face_1"

def developing_system_prompt(name, attributes):
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
        ```

        For this person here are the details that maybe you should use in your response
        The name of the person is: {name} 

        The personal attributes are: {attributes}
    """

    system_dict = message_format("system", system_prompt)
    return [system_dict]

client = OpenAI()

def get_openai_embedding(text):
    """Generates OpenAI embedding for a given text."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def find_top_similar_messages_with_chains_optimized(query_text, top_k=20):
    """Finds the top-K similar messages and retrieves their conversation chains in a single query."""
    
    # Generate embedding for the query text
    query_embedding = get_openai_embedding(query_text)
    
    # Single optimized query combining vector search and chain retrieval
    query = """
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
               message.message_number AS number,
               nodes(window) AS chain
    """
    
    results = Neo4j.read_query(query, query_embedding=query_embedding, top_k=top_k, face_id=face_id)
    messages = []
    message_set = set()
    message_num_list = []
    system_prompt = ""

    num_cos_msgs = 0

    # Print results
    print("\nTop", top_k, "similar messages for:", query_text)
    for idx, row in enumerate(results):
        print(f"{idx+1}. ({row['score']:.4f}), msg_num: [{row['number']}] {row['text']}")
        num_cos_msgs += 1
        name = row["name"]
        attributes = row["attributes"]
        system_prompt = developing_system_prompt(name, attributes)
        print("  Conversation Chain:")
        for msg in row["chain"]:
            if msg['message_number'] not in message_set:
                llm_dict = message_format(msg['role'], msg['text'])
                message_set.add(msg['message_number'])
                message_num_list.append(msg['message_number'])
                messages.append(llm_dict)

    print("The number of cos msgs are ", num_cos_msgs, len(message_set), message_set)
    print("\n")
    for cos_msg in messages:
        print(cos_msg)

    print("\n")

    # developing latest messages
    latest_query = """ 
        MATCH (p:Person {face_id: $face_id})
        WITH p
        MATCH (p)-[:MESSAGE]->(m:Message)
        WITH m
        MATCH window = (m0:Message)-[NEXT*0..20]->(m)
        RETURN nodes(window) as chain
    """
    results = Neo4j.read_query(latest_query, face_id=face_id)

    num_lst_messages = 0

    for idx, result in enumerate(results):
        row = result["chain"]
        for msg in row:
            if msg['message_number'] not in message_set:
                num_lst_messages += 1
                llm_dict = message_format(msg['role'], msg['text'])
                message_set.add(msg['message_number'])
                message_num_list.append(msg['message_number'])
                messages.append(llm_dict)

    print("\nNumber of last messages are ", num_lst_messages)

    sorted_pairs = sorted(zip(message_num_list, messages), key=lambda x: x[0])
    message_num_list, messages= zip(*sorted_pairs)
    messages = list(messages)


    print("\nFinal Messages list\n")
    for num, tot_msgs in zip(message_num_list, messages):
        print("Message num is ", num, " total msg ", tot_msgs)

    print("\n")


    latest_input = message_format("user", query_text)
    messages.append(latest_input)
    total_prompt = system_prompt + messages

    # for p in total_prompt:
    #     print(p)
    #     print("\n")

    response = Grok.send_text(total_prompt, stream=False, grok_model="grok-3")
    print("\n\nResponse is: ", response.choices[0].message.content)

# Run the function for a sample query
find_top_similar_messages_with_chains_optimized("Okay, what is my mom's name?", top_k=20)
