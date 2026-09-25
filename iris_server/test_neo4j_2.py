import json
import uuid
from openai import OpenAI
from utils import Neo4j

client = OpenAI()
# Create a vector index in Neo4j if it does not exist
Neo4j.write_query("""
  CREATE VECTOR INDEX message_embeddings IF NOT EXISTS
  FOR (m:Message) ON (m.embedding) 
  OPTIONS { indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }}
""")

def main(face_id):
# Fetch messages from Neo4j
    cypher_query = """
        MATCH (p:Person {face_id: $face_id})
        RETURN p.messages AS messages
    """
    data = Neo4j.read_query(cypher_query, face_id=face_id)
    messages = json.loads(data[0]["messages"]) if data and data[0].get("messages") else []

# Fetch the last inserted message node for the face_id
    last_message_query = """
        MATCH (m:Message {face_id: $face_id})
        RETURN m.message_id AS message_id ORDER BY m.message_id DESC LIMIT 1
    """
    last_message_data = Neo4j.read_query(last_message_query, face_id=face_id)
    previous_message_id = last_message_data[0]["message_id"] if last_message_data else None

    message_number = 0
# Loop through messages and insert into Neo4j with embeddings
    for message in messages:
        print(message)
        role = message["role"]
        text = message["content"]
        
        # Generate a unique message ID
        message_id = str(uuid.uuid4())

        # Generate OpenAI embedding
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding_vector = response.data[0].embedding

        # Insert new message node with message_id
        Neo4j.write_query("""
            CREATE (m:Message {message_id: $message_id, face_id: $face_id, role: $role, text: $text, embedding: $embedding, message_number: $message_number})
        """, message_id=message_id, face_id=face_id, role=role, text=text, embedding=embedding_vector, message_number=message_number)

        # Link the new message to the previous one using `BEFORE`
        if previous_message_id:
            Neo4j.write_query("""
                MATCH (new:Message {message_id: $new_id}), (old:Message {message_id: $old_id})
                CREATE (old)-[:NEXT]->(new)
            """, new_id=message_id, old_id=previous_message_id)

        # Update previous message reference
        previous_message_id = message_id
        message_number += 1

    joining_person_message = """ 
        MATCH (p:Person {face_id: $face_id}), (m:Message {message_id: $previous_message_id})
        MERGE (p)-[:MESSAGE]->(m)
    """
    Neo4j.write_query(joining_person_message, face_id=face_id, previous_message_id=previous_message_id)

    print("Messages stored with embeddings and linked in Neo4j.")

if __name__ == "__main__":
    get_name_query = """
        MATCH (p:Person)
        WHERE p.face_id = "face_1"
        RETURN p.face_id as face_id
    """
    results = Neo4j.read_query(get_name_query)
    for idx, row in enumerate(results):
        print("All tried face ids ", row["face_id"])
        face_id = row["face_id"]
        main(face_id)
