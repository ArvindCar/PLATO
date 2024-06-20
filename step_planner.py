#
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

def ProcessString(input_string):
    steps_list_part = input_string.split("Steps List:")[1].strip()
    steps = [step.split('. ', 1)[1] for step in steps_list_part.split('\n')]
    steps_nested = []
    for step in steps:
        first_split = step.split(": ", 1)
        second_split = first_split[1].split(" + ", 1)
        nested_step = [first_split[0]] + second_split
        steps_nested.append(nested_step)
    return steps_nested


def Plan2Action(Action, Location, Object = 'None', Tool = 'None'):

    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You are tasked with converting high level commands, to low level commands that can be executed by a parallel plate gripper. 
                          
                          You will be given:
                            Action: A command in the form '<action>, <object> (optional), <location of object> (optional)'.
                            Location: A semantic description of the location that the grippers can move to.
                            Object: The object that you want the robot arm to interact with. It is not currently be held by the gripper
                            Tool: The tools that the gripper currently holds. If None, it means that the gripper isn't holding anything

                          The High level commands will be actions that can be understood by a human.
                          The Low level commands must consist of a combination of Go-to poses and gripper states (Grasp/Open). Grasp is represented by 1, Open is represented by 0.
                          If you need to perform a trajectory instead of discrete actions, do so by specifying a sequence of go-to poses.
                          Go-to poses can be specified relative to the <location of object>, by using deltas (X or Y or Z) relative to the <location of object>. The deltas can also be (0, 0, 0)
                          Robot Coordinate System:
                            By default, the parallel plate gripper (along wih whatever tool it may be holding) is facing the positive x direction, with its base mount behind it, in the negative x direction.
                            The line joining the parallel plate grippers is along the Y axis, and Up represents the Z Axis.

                          Your response should be a series of steps in the format: Go-to: <Location> + (deltaX, deltaY, deltaZ), OR Grasp: <0/1>.
                          Go-to commands move the end effector, and Grasp commands close/open the gripper. Grasp commands should only be used when you want to grassp or release a tool.
                          For Example:
                            1. Go-to: Location1 + (0, 0, 0 cm)
                            2. Grasp: 1
                          Reason out each step, and explain the intended effect for each step. Assume that all upstream tasks for the given step have been completed succesffully.
                          At the end list out just the steps involved (no other information).

                          Take a look at the example below for reference:
                          <start of example>
                          [### User Input]:
                            Action: Pick-up,
                            Location: Original Position of Rolling Pin,
                            Object: Rolling Pin
                            Tool: None
                          
                          [### Expected Output]:
                            1. Grasp: 1
                            Reasoning: Since our task is just to grasp, it is assumed that all upstream tasks, like moving to the location of the rolling pin has already been completed.
                            Thus, the robot arm just needs to grasp the rolling pin
                            2. Go-to: Original Position of Rolling Pin + (0, 0, 10 cm)
                            Reasoning: Now that we have grasped the rolling pin, the next task is to pick it up. This is accomplished by a Go-to command asking the gripper to move vertically up.
                            
                            Steps List:
                              1. Grasp: 1
                              2. Go-to: Original Position of Rolling Pin + (0, 0, 10 cm)
                          <end of example>
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
                            Object: {Object},
                            Tool: {Tool}"""
            }
            ]
        }
    ]

    
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=prompt
    )
    response = completion.choices[0].message.content
    print(response)
    return(ProcessString(response))

if __name__=="__main__":
    Action = "Release"
    Location = "Workspace-center"
    Object = "None"
    Tool = "Roller"

    response = Plan2Action(Action, Location, Object, Tool)
    print(response)

