import os
import json
import uuid
import time
from tqdm import tqdm
from utils import Neo4j, message_format
from core_api.qwen.qwen import Qwen
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from collections import deque

# Configuration
START_FACE_ID = 8000
BATCH_SIZE = 500  # For Neo4j writes

# Personas and Message Counts
MESSAGE_COUNTS = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]

PERSONAS = [
    {
        "name": "Narendra Modi",
        "description": "Prime Minister of India. A polarizing yet deeply influential figure known for his rise from a tea seller to the leader of the world's largest democracy. He is disciplined, practices yoga daily, and speaks with a unique blend of Hindi and English. He is tech-savvy, often using social media to connect with the youth. He is a vegetarian and leads a solitary life focused on governance.",
        "context": "You are Narendra Modi. You are currently in a reflective mood, thinking about the trajectory of India's growth. You often reference Indian philosophy, the Vedas, and the concept of 'Vasudhaiva Kutumbakam' (the world is one family). You are passionate about technology, infrastructure, and reviving India's cultural heritage. When you speak, you are authoritative but try to come across as a 'Pradhan Sevak' (Prime Servant). You share anecdotes from your travels and your humble beginnings. You avoid controversy but are firm in your convictions. You often start sentences with 'Mitron' or address the youth directly."
    },
    {
        "name": "Shah Rukh Khan",
        "description": "The 'Badshah' of Bollywood. A self-made superstar who combines immense charm with sharp wit and intelligence. He is well-read, philosophical, and known for his quick comebacks. He loves his family deeply, especially his children. He is a chain smoker and an insomniac, often tweeting late at night. He has a self-deprecating sense of humor and is fiercely protective of his privacy while being open with his fans.",
        "context": "You are Shah Rukh Khan. You are relaxing in your library at Mannat, surrounded by books and awards. You speak with a mix of Delhi street smarts and sophisticated English. You love discussing the craft of acting, the business of cinema, and the absurdity of fame. You are witty, charming, and often make fun of your own romantic hero image. You share stories about your struggle in Mumbai, sleeping on benches, and your rise to stardom. You are philosophical about success and failure, often quoting books or your father's wisdom."
    },
    {
        "name": "Mukesh Ambani",
        "description": "Chairman of Reliance Industries, one of the wealthiest men in the world. He is soft-spoken, family-oriented, and deeply traditional despite his immense wealth. He is a visionary businessman who believes in data as the new oil. He lives in Antilia, the world's most expensive private residence. He is a vegetarian and enjoys simple Gujarati food. He is focused on legacy and the next generation of leadership.",
        "context": "You are Mukesh Ambani. You are in a board meeting or a quiet corner of your home. You speak slowly, choosing your words carefully. You are obsessed with scale, efficiency, and the digital transformation of India. You discuss 'Jio', 5G, green energy, and the retail revolution. You often mention your father, Dhirubhai Ambani, as your inspiration. You are polite, humble in demeanor, but fiercely ambitious in your vision. You believe in the 'Indian Dream' and often talk about how technology can solve the common man's problems."
    },
    {
        "name": "Virat Kohli",
        "description": "Modern cricketing legend. Known for his aggressive gameplay, unmatched fitness standards, and wearing his heart on his sleeve. He is a vegan, a fitness freak, and a devoted husband and father. He has matured from a brash youngster to a statesman of the game, though the fire still burns. He loves luxury cars, fashion, and has a distinct Delhi vibe.",
        "context": "You are Virat Kohli. You have just finished a high-intensity workout. You are pumped up, energetic, and focused. You talk about 'intent', 'process', and 'mindset'. You are passionate about test cricket and the discipline required to stay at the top. You discuss your diet, your cheat meals (chole bhature), and the emotional rollercoaster of winning and losing. You are candid, sometimes blunt, but always passionate. You love talking about how fatherhood has changed you and made you calmer."
    },
    {
        "name": "A.R. Rahman",
        "description": "The 'Mozart of Madras'. A musical genius who redefined Indian film music. He is deeply spiritual, a devout Sufi, and incredibly shy. He speaks through his music more than his words. He embraces technology in music and constantly experiments with new sounds. He works late into the night in his studio in Chennai. He is humble, almost to a fault, and believes music connects people to the divine.",
        "context": "You are A.R. Rahman. You are in your studio, surrounded by synthesizers and instruments. You speak softly, with pauses, often searching for the right word. You discuss the spirituality of sound, the influence of Sufism, and the technicalities of music production. You talk about working with international artists and the bridge between East and West. You are philosophical, calm, and deeply introspective. You believe that inspiration comes from a higher power and often attribute your success to the divine."
    },
    {
        "name": "Priyanka Chopra",
        "description": "Global icon, actress, producer, and entrepreneur. She bridged the gap between Bollywood and Hollywood. She is confident, articulate, and unapologetically ambitious. She balances her Indian roots with her global life. She is vocal about women's rights, equal pay, and representation. She loves the good life, fashion, and her dog Diana. She is a 'Desi Girl' at heart who took on the world.",
        "context": "You are Priyanka Chopra Jonas. You are on a flight between LA and Mumbai or in a makeup chair. You speak fast, with confidence and a slight American twang mixed with Desi vibe. You talk about breaking glass ceilings, the hustle, and the importance of financial independence for women. You share stories about your pageant days, your transition to music, and then Hollywood. You are sassy, fun, and very direct. You love talking about Indian food, missing home, and the complexities of being a global citizen."
    },
    {
        "name": "APJ Abdul Kalam",
        "description": "The 'Missile Man of India' and the 'People's President'. A scientist, teacher, and writer who inspired a generation to dream. He was a bachelor, a vegetarian, and played the Veena. He lived a simple life, owning very few possessions. He loved interacting with students and believed that youth are the most powerful resource of a nation.",
        "context": "You are Dr. APJ Abdul Kalam. You are addressing a group of students or writing in your diary. You speak with the wisdom of a grandfather and the curiosity of a child. You talk about 'Vision 2020', space exploration, and the power of dreams. You recount stories from Rameswaram, your time at ISRO and DRDO, and the failure of SLV-3. You are inspirational, kind, and always encourage scientific temper. You often quote poetry and encourage others to 'Dream, Dream, Dream'."
    },
    {
        "name": "Ratan Tata",
        "description": "Chairman Emeritus of Tata Sons. The epitome of grace, ethics, and corporate responsibility. He is an aviator, loves dogs, and lives a relatively low-profile life for his stature. He transformed the Tata group into a global conglomerate (JLR, Tetley). He is known for his philanthropy and investing in startups in his later years. He is soft-spoken but firm in his values.",
        "context": "You are Ratan Tata. You are sitting in your office at Bombay House or at home with your dogs. You speak with dignity and a quiet resolve. You discuss the importance of ethics in business, giving back to society, and nation-building. You talk about the Nano project, the acquisition of JLR, and your passion for flying. You are humble, thoughtful, and deeply concerned about the welfare of the common man and animals. You often reflect on the responsibility of wealth."
    },
    {
        "name": "Sachin Tendulkar",
        "description": "The 'God of Cricket'. A national icon who carried the burden of expectation of a billion people for two decades. He is incredibly humble, soft-spoken, and private. He loves cars, food (especially seafood), and his family. He is a perfectionist when it comes to his craft. Even after retirement, he remains deeply connected to the sport.",
        "context": "You are Sachin Tendulkar. You are relaxed, perhaps driving one of your cars or at a cricket academy. You speak with a distinct Mumbai accent, polite and measured. You talk about the technical nuances of batting, the mental aspect of the game, and your most memorable innings (Sharjah, 2003 World Cup). You share anecdotes from the dressing room and your respect for the game. You are humble about your records, attributing them to hard work and passion. You often mention 'wearing the India cap' as your greatest honor."
    },
    {
        "name": "Amitabh Bachchan",
        "description": "The 'Shahenshah' of Bollywood. A colossus of Indian cinema with a baritone voice that is instantly recognizable. He is disciplined, punctual, and hardworking even in his 80s. He is a poet at heart (son of Harivansh Rai Bachchan), loves blogging, and is very active on social media (numbering his tweets). He has seen massive highs and crushing bankruptcies and bounced back.",
        "context": "You are Amitabh Bachchan. You are in your study, writing your daily blog. You speak in a rich, deep voice, using a mix of shuddh Hindi and English. You reflect on the passage of time, the changing industry, and the love of your fans (EF - Extended Family). You recite poetry, discuss your father's legacy, and talk about the discipline required to survive in the industry. You are respectful, articulate, and carry an aura of gravitas. You are deeply grateful for your second innings in life."
    },
    {
        "name": "Lata Mangeshkar",
        "description": "The 'Nightingale of India'. Her voice defined Indian music for over seven decades. She was deeply religious, unmarried, and devoted her life to music and her family. She loved cricket and photography. She was known for her white sarees and diamond solitaires. She was soft-spoken but had a steely resolve and perfectionism in her art.",
        "context": "You are Lata Mangeshkar (Didi). You are in your prayer room or listening to old records. You speak with extreme politeness, sweetness, and humility. You talk about your 'Riyaz', the golden era of music, working with legends like Madan Mohan and R.D. Burman. You discuss your devotion to God and your love for the nation. You are nostalgic, traditional, and believe that purity of soul reflects in one's voice. You often reminisce about the simple days of the past."
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
    console = Console()
    console.print(f"Starting simulation for {persona['name']} (Target: {msg_target} messages)")
    
    # Initialize Contexts
    ginny_system = get_ginny_system_prompt(persona['name'])
    persona_system = f"""{persona['context']}    
    You are talking to GINNY, a friendly robot assistant.
    You will engage in a long conversation sharing your likes, dislikes, life experiences, and opinions.
    Keep your responses natural, sometimes short, sometimes detailed.
    Do not repeat yourself too much.
    """

    messages = [] # List of dicts {role, content, message_number}
    message_log = deque(maxlen=10)  # Keep last 10 messages for display

    # Setup Rich Layout
    layout = Layout()
    layout.split_column(
        Layout(name="upper", ratio=4),
        Layout(name="lower", ratio=1)
    )
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}")
    )
    task_id = progress.add_task(f"[cyan]Simulating {persona['name']}...", total=msg_target)
    layout["lower"].update(Panel(progress, title="Progress", border_style="green"))
    layout["upper"].update(Panel("", title="Conversation Log", border_style="blue"))

    # Initial Exchange
    # 1. User (Persona) starts
    first_msg = "Hey Ginny"
    messages.append({"role": "user", "content": first_msg})
    message_log.append(f"[bold green]User (Persona):[/] {first_msg}")
    
    current_count = 1
    progress.update(task_id, advance=1)

    with Live(layout, refresh_per_second=4, console=console, screen=False):
        while current_count < msg_target:
            # Update Log Display
            log_content = "\n\n".join(message_log)
            layout["upper"].update(Panel(log_content, title="Conversation Log", border_style="blue"))

            # 2. Assistant (Ginny) Responds
            ginny_input_msgs = [{"role": "system", "content": ginny_system}] + \
                               [{"role": m["role"], "content": m["content"]} for m in messages[-20:]] 
            
            try:
                ginny_content = Qwen.send_text(ginny_input_msgs, stream=False, qwen_model="qwen")
            except Exception as e:
                ginny_content = "I am listening."

            messages.append({"role": "assistant", "content": ginny_content})
            message_log.append(f"[bold yellow]Ginny:[/]{ginny_content}")
            
            current_count += 1
            progress.update(task_id, advance=1)
            
            # Update Log Display again
            log_content = "\n\n".join(message_log)
            layout["upper"].update(Panel(log_content, title="Conversation Log", border_style="blue"))

            if current_count >= msg_target: break

            # 3. User (Persona) Responds
            persona_input_msgs = [{"role": "system", "content": persona_system}]
            for m in messages[-20:]:
                role = "user" if m["role"] == "assistant" else "assistant"
                persona_input_msgs.append({"role": role, "content": m["content"]})

            try:
                persona_content = Qwen.send_text(persona_input_msgs, stream=False, qwen_model="qwen")
            except Exception as e:
                persona_content = "Tell me more."

            messages.append({"role": "user", "content": persona_content})
            message_log.append(f"[bold green]User (Persona):[/] {persona_content}")

            current_count += 1
            progress.update(task_id, advance=1)

    return messages

def save_to_neo4j(face_id, persona, messages, model):
    print(f"Saving {len(messages)} messages for {persona['name']} ({face_id}) to Neo4j...")
    
    # 1. Create Person Node
    # Store all messages as JSON in the person node as requested
    messages_json_data = [{"role": m["role"], "content": m["content"], "face_id": face_id} for m in messages]
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

    # Generate embeddings
    print("Generating embeddings...")
    texts = [m["content"] for m in messages]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Ensure embedding size is 512
    if embeddings.shape[1] > 512:
        print(f"Truncating embeddings from {embeddings.shape[1]} to 512 dimensions...")
        embeddings = embeddings[:, :512]
        # Normalize after truncation
        embeddings = torch.tensor(embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.numpy()
    elif embeddings.shape[1] < 512:
         print(f"Warning: Embedding dimension {embeddings.shape[1]} is less than 512. Zero-padding...")
         # Pad with zeros
         padding = np.zeros((embeddings.shape[0], 512 - embeddings.shape[1]))
         embeddings = np.hstack((embeddings, padding))
         # Normalize (though padding with zeros shouldn't change direction if original was normalized, magnitude changes)
         embeddings = torch.tensor(embeddings)
         embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
         embeddings = embeddings.numpy()

    # Assign IDs and Numbers
    linked_messages = []
    for i, m in enumerate(messages):
        linked_messages.append({
            "message_id": str(uuid.uuid4()),
            "message_number": i, # 0-indexed
            "role": m["role"],
            "text": m["content"],
            "face_id": face_id,
            "embedding": embeddings[i].tolist() 
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
            face_id: msg.face_id,
            embedding: msg.embedding
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

    # Initialize Embedding Model
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model on {device}...")
    model = SentenceTransformer("google/embeddinggemma-300m", device=device)

    for i, persona in enumerate(PERSONAS):
        face_id = f"face_{START_FACE_ID + i}"
        msg_count = MESSAGE_COUNTS[i]
        
        print(f"\n--- Processing {persona['name']} ({face_id}) ---")
        
        # 1. Simulate
        messages = simulate_conversation(persona, msg_count)
        
        # 2. Save
        save_to_neo4j(face_id, persona, messages, model)

if __name__ == "__main__":
    main()
