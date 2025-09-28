import json
from utils import Neo4j  # adjust import to your project structure

# configure range
FACE_START, FACE_END = 7000, 7012
REPEATS = 25

# queries
FETCH_PERSON = """
MATCH (p:Person {face_id: $face_id})
OPTIONAL MATCH (p)-[:MESSAGE]->(last:Message)
OPTIONAL MATCH (m:Message)-[:NEXT*0..]->(last)
WITH p, collect(DISTINCT m) AS ms
RETURN p.messages AS messages_json, size(ms) AS msg_count
"""

UPDATE_PERSON = """
MATCH (p:Person {face_id: $face_id})
SET p.messages = $messages_json
"""

def main():
    for i in range(FACE_START, FACE_END + 1):
        face_id = f"face_{i}"
        print(f"Processing {face_id} ...")

        rows = list(Neo4j.read_query(FETCH_PERSON, face_id=face_id))
        if not rows:
            print(f"⚠️ No person found for {face_id}, skipping")
            continue

        row = rows[0]
        messages_json = row["messages_json"]
        msg_count = row["msg_count"]

        if not messages_json:
            print(f"⚠️ No messages property for {face_id}, skipping")
            continue

        # parse JSON string -> list
        try:
            messages_list = json.loads(messages_json)
        except Exception as e:
            print(f"❌ Failed to parse messages for {face_id}: {e}")
            continue

        # repeat 100x
        repeated_list = messages_list * REPEATS

        # safety check
        if len(repeated_list) != msg_count:
            print(f"❌ Length mismatch for {face_id}: "
                  f"repeated={len(repeated_list)} vs nodes={msg_count}")
            continue

        # update back into DB
        new_json = json.dumps(repeated_list)
        Neo4j.write_query(UPDATE_PERSON, face_id=face_id, messages_json=new_json)

        print(f"✅ Updated {face_id} with {len(repeated_list)} messages")

if __name__ == "__main__":
    main()

