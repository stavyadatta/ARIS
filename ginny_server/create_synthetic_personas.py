import os
import json
import uuid
import time
from tqdm import tqdm
from utils import Neo4j, message_format
from core_api.qwen.qwen import Qwen

# Configuration
START_FACE_ID = 8000
BATCH_SIZE = 500  # For Neo4j writes

# Personas and Message Counts
MESSAGE_COUNTS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]

PERSONAS = [
    {
        "name": "Narendra Modi",
        "description": "Prime Minister of India. Known for his oratory skills, focus on development, technology, and Indian culture. Speaks formally but connects with masses.",
        "context": "You are Narendra Modi. Discuss policies, yoga, technology, and the future of India. Be charismatic and authoritative."
    },
    {
        "name": "Shah Rukh Khan",
        "description": "Bollywood Actor, 'King Khan'. Witty, charming, philosophical, and humble. Known for romance and interviews.",
        "context": "You are Shah Rukh Khan. Talk about movies, love, your journey, and struggle. Be witty, charming, and use self-deprecating humor."
    },
    {
        "name": "Mukesh Ambani",
        "description": "Business Magnate, Chairman of Reliance Industries. Focus on business expansion, digital revolution (Jio), and family values.",
        "context": "You are Mukesh Ambani. Discuss business strategies, the digital economy, reliance, and future technologies. Be professional and visionary."
    },
    {
        "name": "Virat Kohli",
        "description": "Indian Cricketer. Aggressive, passionate, fitness enthusiast, and focused on excellence.",
        "context": "You are Virat Kohli. Talk about cricket, fitness, discipline, and winning mentality. Be energetic and passionate."
    },
    {
        "name": "A.R. Rahman",
        "description": "Music Composer. Spiritual, soft-spoken, genius, focuses on harmony and innovation.",
        "context": "You are A.R. Rahman. Discuss music, spirituality, harmony, and technology in art. Be soft-spoken, humble, and profound."
    },
    {
        "name": "Priyanka Chopra",
        "description": "Global Actor and Producer. Confident, articulate, focuses on women empowerment and global cinema.",
        "context": "You are Priyanka Chopra. Talk about Hollywood, Bollywood, breaking glass ceilings, and life between India and USA. Be confident and sassy."
    },
    {
        "name": "APJ Abdul Kalam",
        "description": "Former President & Scientist. 'Missile Man'. Inspirational, loves students, science, and vision for 2020.",
        "context": "You are APJ Abdul Kalam. Talk about science, dreams, youth, and education. Be inspirational, teacher-like, and visionary."
    },
    {
        "name": "Ratan Tata",
        "description": "Industrialist & Philanthropist. Dignified, ethical, cares for animals and the nation.",
        "context": "You are Ratan Tata. Discuss ethics in business, philanthropy, cars, and nation-building. Be dignified, humble, and thoughtful."
    },
    {
        "name": "Sachin Tendulkar",
        "description": "Cricket Legend. 'God of Cricket'. Humble, dedicated, loves the game above all.",
        "context": "You are Sachin Tendulkar. Talk about cricket memories, respect for the game, and batting technique. Be humble and simple."
    },
    {
        "name": "Amitabh Bachchan",
        "description": "Bollywood Legend. Deep voice, articulate, poetic (uses Hindi/English), respectful.",
        "context": "You are Amitabh Bachchan. Talk about cinema history, poetry (Harivansh Rai Bachchan), and discipline. Be baritone, respectful, and articulate."
    },
    {
        "name": "Lata Mangeshkar",
        "description": "Legendary Singer. 'Nightingale of India'. Devotional, pure, respectful, loves music.",
        "context": "You are Lata Mangeshkar. Talk about music riyaz, old bollywood, and devotion. Be very polite, traditional, and sweet."
    }
]

# Ginny System Prompt (From speaking.py)
def get_ginny_system_prompt(person_name, person_attributes="[]", person_relationships=""):
    return f"""
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
            7) Replace your output \"as large language model\" with \"as a robot\"
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

def simulate_conversation(persona, msg_target):
    print(f"Starting simulation for {persona['name']} (Target: {msg_target} messages)")
    
    # Initialize Contexts
    ginny_system = get_ginny_system_prompt(persona['name'])
    persona_system = f"""{persona['context']}    
    You are talking to GINNY, a friendly robot assistant.
    You will engage in a long conversation sharing your likes, dislikes, life experiences, and opinions.
    Keep your responses natural, sometimes short, sometimes detailed.
    Do not repeat yourself too much.
    """

    messages = [] # List of dicts {role, content, message_number}
    
    # Initial Exchange
    # 1. User (Persona) starts
    first_msg = "Hey Ginny"
    messages.append({"role": "user", "content": first_msg})
    
    current_count = 1
    pbar = tqdm(total=msg_target)
    pbar.update(1)

    while current_count < msg_target:
        # 2. Assistant (Ginny) Responds
        # Construct prompt for Ginny: System + Recent History
        ginny_input_msgs = [{"role": "system", "content": ginny_system}] + \
                           [{"role": m["role"], "content": m["content"]} for m in messages[-20:]] # Context window 20
        
        try:
            response = Qwen.send_text(ginny_input_msgs, stream=False, qwen_model="qwen")
            ginny_content = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating Ginny response: {e}")
            ginny_content = "I am listening."

        messages.append({"role": "assistant", "content": ginny_content})
        current_count += 1
        pbar.update(1)
        if current_count >= msg_target: break

        # 3. User (Persona) Responds
        # Construct prompt for Persona: System + Recent History
        # We need to invert roles for the Persona model (Assistant -> User, User -> Assistant) 
        # because the model plays the 'assistant' role in generation but represents the 'user' in our chat data.
        # However, simpler is to just feed it the history and tell it "You are the user".
        # Actually, standard Chat API expects 'user' and 'assistant'. 
        # If we want the model to generate the "User" response, we treat the "User" as the "Assistant" in the API call context.
        
        persona_input_msgs = [{"role": "system", "content": persona_system}]
        for m in messages[-20:]:
            # Invert roles: If it was 'user' (Persona previously), it's now 'assistant' from Persona's own POV?
            # No, standard practice: 
            # We want model to output "User" content.
            # So previous "assistant" (Ginny) msg is "user" input to this model.
            # Previous "user" (Persona) msg is "assistant" (Self) output.
            role = "user" if m["role"] == "assistant" else "assistant"
            persona_input_msgs.append({"role": role, "content": m["content"]})

        try:
            response = Qwen.send_text(persona_input_msgs, stream=False, qwen_model="qwen")
            persona_content = response.choices[0].message.content
        except Exception as e:
            print(f"Error generating Persona response: {e}")
            persona_content = "Tell me more."

        messages.append({"role": "user", "content": persona_content})
        current_count += 1
        pbar.update(1)

    pbar.close()
    return messages

def save_to_neo4j(face_id, persona, messages):
    print(f"Saving {len(messages)} messages for {persona['name']} ({face_id}) to Neo4j...")
    
    # 1. Create Person Node
    # Store all messages as JSON in the person node as requested
    messages_json_data = [{"role": m["role"], "content": m["content"]} for m in messages]
    messages_json_str = json.dumps(messages_json_data)
    
    person_query = """
    MERGE (p:Person {face_id: $face_id})
    SET p.name = $name,
        p.messages = $messages_json,
        p.attributes = '[]',
        p.state = 'speak'
    """
    Neo4j.write_query(person_query, face_id=face_id, name=persona['name'], messages_json=messages_json_str)

    # 2. Create Message Chain
    # We will batch this
    
    # Create the first message
    if not messages:
        return

    # Assign IDs and Numbers
    linked_messages = []
    for i, m in enumerate(messages):
        linked_messages.append({
            "message_id": str(uuid.uuid4()),
            "message_number": i, # 0-indexed
            "role": m["role"],
            "text": m["content"],
            "face_id": face_id,
            "embedding": [] # Empty embedding to save time/space or we can generate if needed (skipping for speed)
        })

    # Cypher for batch creation
    # We'll create nodes in batches and link them
    # To link efficiently, we can CREATE the nodes and then LINK them based on ID or index
    
    # Batch strategy:
    # 1. UNWIND list to CREATE Messages
    # 2. UNWIND list to MATCH and LINK (NEXT)
    # 3. LINK Person to Last Message
    
    # Note: Creating 22k nodes might be heavy. Let's do batches of 1000.
    
    print("Creating message nodes...")
    total_msgs = len(linked_messages)
    for start_idx in range(0, total_msgs, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_msgs)
        batch = linked_messages[start_idx:end_idx]
        
        create_nodes_query = """
        UNWIND $batch AS msg
        CREATE (m:Message {
            message_id: msg.message_id,
            message_number: msg.message_number,
            role: msg.role,
            text: msg.text,
            face_id: msg.face_id
        })
        """
        Neo4j.write_query(create_nodes_query, batch=batch)
        print(f"  Created nodes {start_idx} to {end_idx}")

    print("Linking message nodes...")
    # Linking requires matching the nodes we just created.
    # Since they are sequential, we can link m[i] to m[i+1]
    
    # We can do this by matching the whole set (might be slow) or by range.
    # A better way for large chains:
    # MATCH (m:Message {face_id: $face_id}) WITH m ORDER BY m.message_number
    # WITH collect(m) as msgs
    # FOREACH (i in range(0, size(msgs)-2) | 
    #    FOREACH (a in [msgs[i]] | FOREACH (b in [msgs[i+1]] | MERGE (a)-[:NEXT]->(b))))
    
    # But for 22k, `collect(m)` might blow up memory.
    # Let's link in batches using message_number.
    
    link_query = """
    UNWIND range($start_num, $end_num - 1) AS i
    MATCH (a:Message {face_id: $face_id, message_number: i})
    MATCH (b:Message {face_id: $face_id, message_number: i+1})
    MERGE (a)-[:NEXT]->(b)
    """
    
    for start_idx in range(0, total_msgs - 1, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_msgs - 1)
        Neo4j.write_query(link_query, face_id=face_id, start_num=start_idx, end_num=end_idx)
        print(f"  Linked nodes {start_idx} to {end_idx}")

    # 3. Link Person to Last Message
    print("Linking Person to Last Message...")
    last_msg_num = total_msgs - 1
    person_link_query = """
    MATCH (p:Person {face_id: $face_id})
    MATCH (last:Message {face_id: $face_id, message_number: $last_num})
    MERGE (p)-[:MESSAGE]->(last)
    """
    Neo4j.write_query(person_link_query, face_id=face_id, last_num=last_msg_num)
    print("Done saving.")


def main():
    if len(PERSONAS) != len(MESSAGE_COUNTS):
        print("Error: Number of Personas and Message Counts must match.")
        return

    # Check database connection
    try:
        Neo4j.read_query("RETURN 1")
    except Exception as e:
        print("Failed to connect to Neo4j. Check environment variables.")
        print(e)
        return

    for i, persona in enumerate(PERSONAS):
        face_id = f"face_{START_FACE_ID + i}"
        msg_count = MESSAGE_COUNTS[i]
        
        print(f"\n--- Processing {persona['name']} ({face_id}) ---")
        
        # 1. Simulate
        messages = simulate_conversation(persona, msg_count)
        
        # 2. Save
        save_to_neo4j(face_id, persona, messages)

if __name__ == "__main__":
    main()
