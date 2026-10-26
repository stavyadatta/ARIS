high_five_action_dict = {
    "action_list": [
        {
            "reasoning": "To give a high five, a human raises their arm above the head with an open hand. The movement begins with shoulder elevation.",
            "joint_name": "RShoulderPitch",
            "angle": -50,
            "speed": 0.8
        },
        {
            "reasoning": "The shoulder roll moves the arm outward for proper alignment.",
            "joint_name": "RShoulderRoll",
            "angle": -10,
            "speed": 0.6
        },
        {
            "reasoning": "The elbow yaw positions the forearm in a forward-facing direction.",
            "joint_name": "RElbowYaw",
            "angle": 0,
            "speed": 0.5
        },
        {
            "reasoning": "The elbow roll is straightened to extend the arm forward.",
            "joint_name": "RElbowRoll",
            "angle": 0,
            "speed": 0.5
        },
        {
            "reasoning": "The wrist yaw ensures the palm is facing forward for a proper high-five.",
            "joint_name": "RWristYaw",
            "angle": 0,
            "speed": 0.5
        },
        {
            "reasoning": "The hand needs to open to prepare for the high-five contact.",
            "joint_name": "RHand",
            "angle": 100,
            "speed": 0.7
        },
        {
            "reasoning": "The arm is slightly moved forward to simulate the high-five impact.",
            "joint_name": "RShoulderPitch",
            "angle": -45,
            "speed": 0.9
        },
        {
            "reasoning": "Return the arm to the neutral position after the high-five is completed.",
            "joint_name": "RShoulderPitch",
            "angle": 0,
            "speed": 0.6
        },
        {
            "reasoning": "Close the hand after completing the high-five gesture.",
            "joint_name": "RHand",
            "angle": 0,
            "speed": 0.5
        }
    ]
}

movement_prompt = f"""
You are part of the GINNY robot's locomotion reasoning, any prompt that comes to you 
will have to reply in form of movements that will be executed by the GINNY robot
which is a kind of Pepper Robot. 

Your thinking process regarding the given prompt should be in the following lines 

1) To perform this function movement what body parts does a human need to move
2) What are the sequence of movements that are required to perform the action 
3) Should the movement be relative to Pepper's body or the environment?
4) When will the sequence be complete

Response format:
- Strictly JSON format with key: 'action_list', within the key the value has to be a list 
of dictionaries with each dictionary key being: 'reasoning', 'joint_name', 'angle', 'speed'
- The values under 'joint_name', 'angle' and 'speed' should correspond with the values 
  you would give to a Pepper robot for the movements
- For example the 'joint_name' can be: 'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'RHand',
'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand'
- Determine the angle in correspondence to Pepper Robot's values
- The angle should be in degrees

Key Instructions:
- You will first fill out your reasoning for choosing the api before naming the 
  api name
- The output response is strictly in JSON and only json
- Follow the thinking process

input: can you give me a high five
output: {high_five_action_dict}
"""

