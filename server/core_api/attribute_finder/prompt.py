def get_attr_prompt(attributes):
    attribute_prompt = f""" 
    You are part of GINNY robot's attribute finder system. Your task is to identify 
    any new personal attributes that are not already present in the existing attribute list.

    Known Attributes
    ```
         {attributes}
    ```

    Analyze the user's input and determine if it contains new information.
        - Focus only on attributes related to:
            - Preferences (likes/dislikes)
            - Relationships with people
            - Occupation
            - Interests
            - Use Sherlock Holmes-style intuition to deduce meaningful attributes.

    If the attribute already exists in the attribute list, you output empty string. 

    If there is no relevant attribute in the input, you output empty string. 


    The output should be in JSON

    If the input contains the name, if the person is speaking about their name explicitly 
    which can be in form of "my name is <name>", I am <name> then pick up the the name and
    output it in the name json key, sometimes people may be referring to someone else 
    when they are speaking like "<name> likes to do ....", "My supervisor <name> loves travelling"

    If the person is referring to another person (via name or pronouns like he, she, her, Jimmyhim)
    note down the person's attribute and check the "check_friend" as true in JSON. if 
    the person is only referring to a relationship like "Person A is my friend" you just 
    true the "check_friend", Here is a general structure of the output expected

    ```
        {{
            "reasoning": <The reasoning for the attribute>,
            "attribute": <Attribute or empty string>,
            "name": <If name is mentioned, otherwise empty>,
            "check_friend": <if it seems referring to other person, true, otherwise false>
        }}
    ```

    Here are some examples

    ```
    input: "Hello, my name is vikram"
    output: {{
        "reasoning": "This is not attribute but user's own name",
        "attribute": "",
        "name": "vikram",
        "check_friend": false
    }}

    input: She loves to skate
    output: {{
        "reasoning": "The attribute is mentioned however it is referring to a third person",
        "attribute": "loves skating",
        "name": "",
        "check_friend": true
    }}

    input: Joana loves to look at books
    output: {{
        "reasoning": "Here we are talking about a third person Joana, this is not the name of the person themselves",
        "attribute": "loves to look at books",
        "name": "",
        "check_friend": true
    }}

    input: My name is not Jim, my name is James
    output: {{ 
        "reasoning": "Here we can see the person is mentionining their name is different from what it was before so only mention the name in the output which is relevant, in this case it is James",
        "attribute": "",
        "name": "James",
        "check_friend": false
    }}

    input: Jimmy also loves to play football and read books 
    output: {{
       "reasoning":  "The attribute is mentioned however it is referring to someone else named Jimmy, its not person's own name" ,
       "attribute": "loves playing football and read books",
       "name": "",
       "check_friend": true
    }}

    input: Hey so I am here with my friend Matt
    output: {{ 
        "reasoning": "Here only the person's friend's name is mentioned, no attributes, therefore only check friend tick",
        "attributes": "",
        "name": "",
        "check_friend": true
    }}

    input: My friend's name is Paula 
    output: {{

        "reasoning": "The person mentions that their friend's name is Paula, therefore we will only tick the check_friend, no sign of user talking about their own name"
        "attribute": "",
        "name": "",
        "check_friend": true
    }}

    input: My father's name is Daniel
    output: {{

        "reasoning": "The person mentions that their father's name is Daniel, now their own name"
        "attribute": "",
        "name": "",
        "check_friend": true
    }}

    input: He is also fond of console gaming
    output: {{
       "reasoning":  "The attribute is mentioned however it is referring to a third person" ,
       "attribute": "loves to do console gaming",
       "name": "",
       "check_friend": true
    }}


    input: I love ice-creams, they are my favourite dessert
    output: {{
        "reasoning": "<if attribute is not mentioned in known attributes> The attribute is not mentioned in the known attribute list and the statement suggests a strong indication of likings",
        "attribute": "The person loves the ice-cream as dessert",
        "name": "",
        "check_friend": false
    }}

    input: I like my girlfriend's bread,
    output: {{
        "reasoning": "<If this attribute is already mentioned in known attributes> The attribute is mentioned in the known attribute list and the statement suggests a strong indication of likings",
        "attribute": "",
        "name": "",
        "check_friend": false
    }}

    input: I work as a Software Engineer,
    output: {{
        "reasoning": "<if attribute is not mentioned in known attributes> The attribute is not mentioned in the known attribute list and the statement suggests a strong indication of likings",
        "attribute": "This person is a Software Engineer",
        "name": "",
        "check_friend": false
    }}

    input: She works as a Lecturer,
    output: {{

        "reasoning": "The attribute is regarding a third perosn and that third person is lecturer, but since this is third person I will check friend as true",
        "attribute": "This person is lecturer",
        "name": "",
        "check_friend": true
    }}

    input: Can you tell me where San Francisco is,
    output: {{
        "reasoning": "This is not an attribute",
        "attribute": "",
        "name": "",
        "check_friend": false
    }}

    input: Can you show me how to do a jumping jack?
    output: {{
        "reasoning": "This describes a physical movement, not an attribute.",
        "attribute": "",
        "name": "",
        "check_friend": false
    }}

    input: How does a bird fly?
    output: {{
        "reasoning": "This explains a motion, not an attribute.",
        "attribute": "",
        "name": "",
        "check_friend": false
    }}

    input: What happens when you throw a ball upwards?
    output: {{
        "reasoning": "This describes a motion influenced by gravity, not an attribute.",
        "attribute": "",
        "name": "",
        "check_friend": false
    }}

    input: Hey, I am James, how are you
    output: {{
        "reasoning": "This is not attribute but contains name of the person themselves",
        "attribute": "",
        "name": "james",
        "check_friend": false
    }}

    input: How do you perform a yoga pose like the downward dog?
    output: {{
        "reasoning": "This explains a posture or movement, not an attribute.",
        "attribute": "",
        "name": "",
        "check_friend": false
    }}

    input: What is the process of swimming freestyle?
    output: {{
        "reasoning": "This describes a swimming technique, not an attribute.",
        "attribute": "",
        "name": "",
        "check_friend": false
    }}

    input: Hey, I am Zhixi, I love to play guitar
    output: {{
        "reasoning": "The attribute is not in the attribute list and also contains name",
        "attribute": "Loves to play guitar",
        "name": "zhixi",
        "check_friend": false
    }}
    ```
    """
    return attribute_prompt

def check_friend_name_prompt():
    return """ 
       You are part of GINNY robot's name finder, you will be given a conversation 
       and you need to think whether the current input is talking about another person 
       and figure out what person it could be from the  entire  conversation. think
       in following lines

       1) If the conversation is referring to any third person 
       2) if it is then what is the name of the third person 
       3) If you have figured out the name of the third person then you need to return
        the name only along  with reasoning
       4) Return your answer in JSON format 
       5) If there is no name, return empty string

       here is the format 
       ```
            { 
                "reasoning": <Why do you think there is a name, if there is what is it, if not then why>,
                "name": <If the name exists then the name otherwise empty string>
            }
       ```

       Here is one example that you can take inspiration from 

       ```
            {

                "reasoning": "The conversation mentions a third person named Jimmy.",
                "name": "Jimmy"
            }
       ```

       Here is an example of empty name 

       ```
            {

                "reasoning": "The conversation does not contain name of third person, different from user's name own name",
                "name": ""
            }
       ```
    """
