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

REPEAT_TIMES = 100  # make this configurable if you want

CREATE_COPY_CYPHER = """
WITH $personProps AS personProps, $messages AS messages, $repeats AS repeats
// 1) Create new Person
CREATE (np:Person)
SET np += personProps

// 2) Load original chain into an ordered list mlist
WITH np, personProps, [m IN messages | m] AS mlist, repeats

// 3) Create copies for each repeat and each position, preserving global order
UNWIND range(0, repeats-1) AS r
UNWIND range(0, size(mlist)-1) AS i
WITH np, personProps, r, i, mlist[i] AS m
CREATE (nm:Message)
SET nm += m
SET nm.face_id = personProps.face_id
WITH np, r, i, nm
ORDER BY r, i
WITH np, collect(nm) AS allMsgs

// 4) Re-link NEXT across the whole repeated sequence
UNWIND range(0, size(allMsgs)-2) AS k
WITH np, allMsgs, allMsgs[k] AS a, allMsgs[k+1] AS b
CREATE (a)-[:NEXT]->(b)

// 5) Collapse to a single row and link Person -> last message once
WITH DISTINCT np, allMsgs
WITH np,
     CASE WHEN size(allMsgs) > 0 THEN allMsgs[size(allMsgs)-1] ELSE NULL END AS lastMsg
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

def copy_people_and_chains(db, source_face_ids, start_from=7000, prefix="face_", repeats=REPEAT_TIMES):
    new_face_ids = []

    for i, src_fid in enumerate(source_face_ids):
        new_fid = f"{prefix}{start_from + i}"

        # Read original person + ordered chain
        person_props, messages_props = _fetch_person_and_chain(db, src_fid)

        # Person copy: overwrite face_id
        new_person_props = dict(person_props)
        new_person_props["face_id"] = new_fid

        # Message templates: keep properties as-is (face_id will be overwritten in Cypher)
        new_messages = [dict(m) for m in messages_props]

        db.write_query(
            CREATE_COPY_CYPHER,
            personProps=new_person_props,
            messages=new_messages,
            repeats=repeats
        )

        new_face_ids.append(new_fid)

    return new_face_ids

#
# -----------------------------
# Example usage:
# -----------------------------
sources = ["face_318"]
created = copy_people_and_chains(Neo4j, sources, start_from=7000)
print("Created new people:", created)

