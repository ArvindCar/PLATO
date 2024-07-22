#
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO

def ProcessString(input_string):
    input_string = input_string.lower()
    steps_list_part = input_string.split("steps list:")[1].strip()
    steps = [step.split('. ', 1)[1] for step in steps_list_part.split('\n')]
    steps_nested = []
    for step in steps:
        first_split = step.split(": ", 1)
        second_split = first_split[1].split(" + ", 1)
        nested_step = [first_split[0]] + second_split
        steps_nested.append(nested_step)
    return steps_nested


def Plan2Action(Action, Location, Object = 'None', Tool = 'None', prev_steps={}):
    print("Starting Step Planner:") 

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
                            Previous Steps: This is a dictionary containing the previous inputs (keys) and outputs (values) generated by you. This is to give you some context regarding what the previous steps were. 

                          The High level commands will be actions that can be understood by a human.
                          The Low level commands must consist of a combination of Go-to poses and gripper states (Grasp/Open). Grasp is represented by 1, Open is represented by 0.
                          If you need to perform a trajectory instead of discrete actions, do so by specifying a sequence of go-to poses.
                          Go-to poses can be specified relative to the <location of object>, by using deltas (X or Y or Z) relative to the <location of object>. The deltas can also be (0, 0, 0)
                          Robot Coordinate System:
                            By default, the parallel plate gripper (along wih whatever tool it may be holding) is facing the positive x direction, with its base mount behind it, in the negative x direction.
                            The line joining the parallel plate grippers is along the Y axis, and Up represents the Z Axis.

                          Your response should be a series of steps in the format: Go-to: <Location> + (deltaX, deltaY, deltaZ cm), OR Grasp: <0/1>, OR Tilt: (ThetaX, ThetaY, ThetaZ degrees).
                          Go-to commands move the end effector, Grasp commands close/open the gripper, and Tilt commands roll/pitch/yaw the gripper. Grasp commands should only be used when you want to grasp or release a tool.
                          Keep in mind that the Tilts are calculated absolutely, not relatively. Initially, they are (0, 0, 0 degrees). Also, the angles are calculated based on the right hand thumb rule (ie. Pitching down is a positive ThetaY angle).
                          A Tilt command should have only 1 non-zero value.
                          Keep in mind that some actions will require the tool to be held at an angle, and not completely flat.
                          Any actions that you want to do should be described using these three commands, nothing else. If you want to perform more complex commands like applying forces, scooping, etc., reason them out so that they can broken down into these three fundamental building block commands.
                          
                          
                          Reason out each step, and explain the intended effect for each step. Keep in mind that you might want to end an action by liftng the object you grasped. Assume that all upstream tasks for the given step have been completed successfully.
                          At the end list out just the steps involved (no other information).

                          Take a look at the example below. Strictly follow the format of Expected Output.
                          <start of example>
                          [User Input]:
                            Action: Pick-up,
                            Location: Original Position of Rolling Pin,
                            Object: Rolling Pin
                            Tool: None
                            Previous Steps: []
                          
                          [Expected Output]:
                          Explanation:
                            1. Grasp: 0
                            Reasoning: This ensures that the gripper is open to grasp the tool (It is not currently holding any tool).
                            2. Go-to: Original Position of Rolling Pin + (0, 0, 0 cm)
                            Reasoning: In order to pick the object up, you must be at the location of the object. Since Previous Steps is empty, that means this has not been done yet.
                            3. Grasp: 1
                            Reasoning: Since we have moved to the location of the rolling pin already, the robot arm just needs to grasp the rolling pin.
                            4. Go-to: Original Position of Rolling Pin + (0, 0, 20 cm)
                            Reasoning: Now that we have grasped the rolling pin, the next task is to pick it up. This is accomplished by a Go-to command asking the gripper to move vertically up.
                            
                            Steps List: 
                              1. Grasp: 0
                              2. Go-to: Original Position of Rolling Pin + (0, 0, 0 cm)
                              3. Grasp: 1
                              4. Go-to: Original Position of Rolling Pin + (0, 0, 20 cm)
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
                            Tool: {Tool}
                            Previous Steps: {prev_steps}"""
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
    final_response = ProcessString(response)
    return(final_response)

if __name__=="__main__":
    Action = "Flatten"
    Location = "Dough"
    Object = "Dough"
    Tool = "Flattener"

    response = Plan2Action(Action, Location, Object, Tool)
    print(response)

