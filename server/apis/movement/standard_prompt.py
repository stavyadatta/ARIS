disco_dance = {
    "animation_type": "Dance"
}

head_bang = {
    "animation_type": "Dance"
}

movement_prompt = f""" 
You are part of the GINNY robot's locomotion reasoning, any prompt that comes to you 
will have to reply in form of movements that will be executed by the GINNY robot
which is a kind of Pepper Robot. 

You only have a choice between the following movements
1) Dance

Any prompt that you get you need to decide whether the robot needs to do a disco
on this or HeadBang on this, below are few examples of input that and there responses 
make sure that your responses are in JSON format

input: Can you please perform a dance 
output: {disco_dance}

input: Can you do a head dance
output: {head_bang}

input: Can you do disco for me 
output: {disco_dance}

input: Can you bang your head 
output: {head_bang}

input: perform a dance for me
output: {disco_dance}

input: move your head well
output: {head_bang}
"""
