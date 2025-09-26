# Assumptions:
# - You have a Neo4j helper with:
#     Neo4j.read_query(cypher, **params)  -> iterable of dict rows
#     Neo4j.write_query(cypher, **params) -> iterable/summary (not used here)
# - A Person node has label :Person and property face_id.
# - Messages are a single linear chain linked via [:NEXT].
# - The Person points to the LAST message via [:MESSAGE].
# - You want to keep all other properties unchanged.
# - If a Message lacks 'face_id', we simply don't add it.

from typing import List, Dict, Any
from utils import Neo4j

FETCH_CHAIN_CYPHER = """
MATCH (p:Person {face_id: $face_id})-[:MESSAGE]->(last:Message)
MATCH path = (first:Message)-[:NEXT*0..]->(last)
WHERE NOT (first)<-[:NEXT]-(:Message)   // ensure we start at the head of the chain
RETURN properties(p) AS person,
       [m IN nodes(path) | properties(m)] AS messages
LIMIT 1
"""

CREATE_COPY_CYPHER = """
WITH $personProps AS personProps, $messages AS messages
// 1) Create new Person
CREATE (np:Person)
SET np += personProps

// 2) Create copies of all messages (works even if $messages is empty)
WITH np, messages
UNWIND messages AS m
CREATE (nm:Message)
SET nm += m

// 3) Gather in order and re-link NEXT
WITH np, collect(nm) AS mlist
UNWIND range(0, size(mlist)-2) AS i
WITH np, mlist, mlist[i] AS a, mlist[i+1] AS b
CREATE (a)-[:NEXT]->(b)

// 4) Collapse to one row, then link Person -> last message once
WITH DISTINCT np, mlist
WITH np,
     CASE WHEN size(mlist) > 0 THEN mlist[size(mlist)-1] ELSE NULL END AS lastMsg
FOREACH (_ IN CASE WHEN lastMsg IS NULL THEN [] ELSE [1] END |
  CREATE (np)-[:MESSAGE]->(lastMsg)
)

RETURN np
"""

def _fetch_person_and_chain(db, face_id: str):
    rows = list(db.read_query(FETCH_CHAIN_CYPHER, face_id=face_id))
    if not rows:
        raise ValueError(f"No person or chain found for face_id={face_id}")
    person_props = rows[0]["person"] or {}
    messages_props = rows[0]["messages"] or []
    return person_props, messages_props

def _with_updated_face_id(props: Dict[str, Any], new_face_id: str) -> Dict[str, Any]:
    out = dict(props)  # shallow copy
    out["face_id"] = new_face_id
    return out

def copy_people_and_chains(db, source_face_ids: List[str], start_from: int = 7000, prefix: str = "face_") -> List[str]:
    """
    Copies each Person and their message chain.
    Returns the list of newly created face_ids in the same order as inputs.

    Note: If your original graph enforces uniqueness on Person.face_id,
    make sure 'face_{start_from + i}' does not collide.
    """
    new_face_ids = []

    for i, src_fid in enumerate(source_face_ids):
        new_fid = f"{prefix}{start_from + i}"

        # 1) Read original person + ordered chain of messages
        person_props, messages_props = _fetch_person_and_chain(db, src_fid)

        # 2) Update face_id on new Person
        new_person_props = _with_updated_face_id(person_props, new_fid)

        # 3) Update face_id on every Message (only if present originally)
        new_messages = []
        for msg in messages_props:
            if "face_id" in msg:
                new_messages.append(_with_updated_face_id(msg, new_fid))
            else:
                new_messages.append(dict(msg))  # unchanged

        # 4) Create the copy in one write
        db.write_query(
            CREATE_COPY_CYPHER,
            personProps=new_person_props,
            messages=new_messages
        )

        new_face_ids.append(new_fid)

    return new_face_ids

# -----------------------------
# Example usage:
# -----------------------------
sources = ["face_318"]
created = copy_people_and_chains(Neo4j, sources, start_from=7000)
print("Created new people:", created)

