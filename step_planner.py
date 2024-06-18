#
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

def Plan2Action(Action, Location, Tool = "None"):


    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You are tasked with converting high level commands, to low level commands that can be executed by a parallel plate gripper. 
                          You will be given:
                            Action: A command in the form '<action>, <object> (optional), <location of object> (optional)'.
                            Location: A semantic description of the location that the grippers can move to.
                            Tool: The tools that the gripper currently holds. If none, it means that the gripper isn't holding anything
                          The High level commands will be actions that can be understood by a human.
                          The Low level commands must consist of a combination of go-to poses and gripper states (Grasp/Open). Grasp is represented by 1, Open is represented by 0.
                          If you need to perform a trajectory instead of discrete actions, do so by specifying a sequence of go-to poses.
                          Go-to poses can be specified relative to the <location of object>. You can also use deltas (X or Y or Z) relative to the <location of object>, although this won't always be necessary.
                          Robot Coordinate System:
                            By default, the parallel plate gripper is facing the positive x direction, with its base mount behind it, in the negative x direction.
                            The line joining the parallel plate grippers is along the Y axis, and Up represents the Z Axis.
                          Your response should be a series of steps in the format: Go-to: <Location> + (deltaX, deltaY, deltaZ), OR Grasp: <0/1>.
                          For Example:
                            1. Go-to: Location1 + (0, 0, 10 cm)
                            2. Grasp: 1
                          Reason out each step, and explain the intended effect for each step.
                        """
        },
        {
            "role": "user",
            "content": 
            [
            {
                "type": "text",
                "text": f"""Action: {Action},
                            Location: {Location},
                            Tool: {Tool}"""
            }
            ]
        }
    ]

    
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=prompt
    )

    return(completion.choices[0].message.content)

if __name__=="__main__":
    Action = "Flatten, dough"
    Location = "center of ball of dough"
    Tool = "Roller"
    response = Plan2Action(Action, Location, Tool)
    print(response)

