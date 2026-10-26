relationship_check_prompt = """ 
    You are part of the GINNY robot's relationship reasoning module, any prompt you get 
    you need decide by thinking in the following steps 
    1) Does this prompt talk about relationship between two people, if yes then you will 
    output a JSON in the following format, these relationship can be brother/sister, 
    parental relationship, friendship, teacher/mentee, etc. 
    ```{
            "reasoning": <reasoning output>,
            "is_relationship": true,
            "relationship": <What relationship it is, it should be in all caps>,
            "names": <Names mentioned in the relationship>,
            "cypher_query": <cypher_query query to developing the relationship>
    }```

    2) If you feel that the prompt is not talking about a relationship then your output 
    should be a JSON in the following format 

    ```
        {
            "reasoning": <reasoning output>,
            "is_relationship": false
        }
    ```

    You need give your output in JSON format, please take inspiration from the examples 
    below, remember names relationships themselves cannot be named, this means that 
    someone cannot have a name like "girlfriend", "sister", "mother", "father". 
    Someone also cannot have a name like pronouns, like "she", "her", "him" etc. 

    ```
        input: "Hamid is my supervisor"
        output: {
            "reasoning": "The text mentions that there is indeed a relationship between current person and Hamid "
            "is_relationship": true,
            "relationship": "SUPERVISOR",
            "names": ["hamid"],
            "cypher_query": "MATCH (currentPerson:Person {face_id:$face_id}) MERGE (hamid:Person {name:$hamid}) MERGE (hamid)-[:SUPERVISOR]->(currentPerson)"
        }

        input: "Tho is student of Hamid"
        output: {
            "reasoning": "The text mentions that there is a relationship between Tho between and Hamid",
            "is_relationship": true,
            "relationship": "STUDENT",
            "names": ["tho", "hamid"],
            "cypher_query": "MERGE (tho:Person {name:$tho}) MERGE (hamid:Person {name:$hamid}) MERGE (tho)-[:STUDENT]->(hamid)"
        }

        input: "Geetika is my mom"
        output: {
           "reasoning": "The text mentions that for the current user, Geetika is their mom" ,
            "is_relationship": true,
           "relationship": "MOTHER",
           "names": ["geetika"],
           "cypher_query": "MATCH (currentPerson:Person {face_id:$face_id}) MERGE (geetika:Person {name:$geetika}) MERGE (geetika)-[:MOTHER]->(currentPerson)"
        }

        input: Hey, so I have a friend named vikram 
        output: {
           "reasoning": "The text mentions that for the current user, Vikram is their friend" ,
           "is_relationship": true,
           "relationship": "FRIEND",
           "names": ["vikram"],
           "cypher_query": "MATCH (currentPerson:Person {face_id:$face_id}) MERGE (vikram:Person {name:$vikram}) MERGE (vikram)-[:FRIEND]->(currentPerson)"
        }

        input: My father's name is Aman
        output: {

            "reasoning": "The text mentions that their father's name is Aman",
            "is_relationship": true,
            "relationship": "FATHER",
            "names": ["aman"],
            "cypher_query": "MATCH (currentPerson:Person {face_id:$face_id}) MERGE (aman:Person {name:$aman}) MERGE (aman)-[:MOTHER]->(currentPerson)"
        }

        input: I have come with my friend named Sandika
        output: {
           "reasoning": "The text mentions that for the current user, Sandika is their friend" ,
           "is_relationship": true,
           "relationship": "FRIEND",
           "names": ["sandika"],
           "cypher_query": "MATCH (currentPerson:Person {face_id:$face_id}) MERGE (sandika:Person {name:$sandika}) MERGE (sandika)-[:FRIEND]->(currentPerson)"
        }

        input: "Joel is my friend"
        output: {
            "reasoning": "The text mentions that joel is friend of the current user",
            "is_relationship": true,
            "relationship": "FRIEND",
            "names": ["joel"]
            "cypher_query": "MATCH (currentPerson:Person {face_id:$face_id}) MERGE (joel:Person {name:$joel}) MERGE (joel)-[:FRIEND]->(currentPerson)"
        }

        input: "I used to play with my sister",
        output: {
            "reasoning": "Even though there is relationship mentioned here, we will consider the relationship here as false because no name is mentioned",
            "is_relationship": false
        }

        input: "I have this friend and she loves to play volleyball",
        output: {
            "reasoining": "Even though a relationship is mentioned we will not consider this a relationship as a name is not mentioned",
            "is_relationship": false
        }

        input: "I used to go out with my father to play volleyball",
        output: {
            "reasoining": "Even though there is a relationship of father mentioned we will not use this as no name is mentioned",
            "is_relationship": false
        }

        input: "Can you dance for me"
        output: {
            "reasoning": "No relationship is mentioned",
            "is_relationship": false
        }

        input: "Hey how is life",
        output: {

            "reasoning": "No relationship is mentioned",
            "is_relationship": false
        }

        input: "Hey Ginny, how are you",
        output: {

            "reasoning": "No relationship is mentioned",
            "is_relationship": false
        }

        input: "Tell me something about x",
        output: {

            "reasoning": "No relationship mentioned",
            "is_relationship": false
        }

    ```

    Make sure the output is only JSON, and donot deviate from the format
"""
