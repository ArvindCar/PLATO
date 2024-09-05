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


def Plan2Action(Action, Location, Description = 'None', Object = 'None', Tool = 'None', prev_steps={}):
    print("Starting Step Planner:") 

    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You are tasked with converting high level commands, to low level commands that can be executed by a parallel plate gripper. 
                          
                          You will be given:
                            Action: A command in the form '<action>, <object> (optional), <location of object> (optional)'.
                            Location: A semantic description of the location that the grippers can move to. This represents the centroid of the object
                            Description: Two arrays, describing the dimensions of the object and tool respecticely, in centimeters. If either tool or object is 'None', the respective array will be empty. Objects will be described by Length, Width and Height. Tools will be described by only Length. This length is the distance between the portion of the tool held by the gripper, and the end of the tool. You can use this information to output better Go-to positions
                            Object: The object that you want the robot arm to interact with. It is not currently be held by the gripper
                            Tool: The tools that the gripper currently holds. If None, it means that the gripper isn't holding anything
                            Previous Steps: This is a dictionary containing the previous input (keys) and outputs (values) generated by you. This is to give you some context regarding what the immediate previous step was, to help you avoid unnecessarily repeating steps. 

                          The High level commands will be actions that can be understood by a human.
                          The Low level commands must consist of a combination of Go-to poses and gripper states (Grasp/Open). Grasp is represented by 1, Open is represented by 0.
                          If you need to perform a trajectory instead of discrete actions, do so by specifying a sequence of go-to poses.
                          Go-to poses can be specified relative to the <location of object>, by using deltas (X or Y or Z) relative to the <location of object>. The deltas can also be (0, 0, 0)
                          Robot Coordinate System:
                            By default, the parallel plate gripper (along wih whatever tool it may be holding) is facing the positive x direction, with its base mount behind it, in the negative x direction.
                            The line joining the parallel plate grippers is along the Y axis, and Up represents the Z Axis.

                          Your response should be a series of steps in the format: Go-to: <Location> + (deltaX, deltaY, deltaZ cm), OR Grasp: <0/1>, OR Tilt: (ThetaX, ThetaY, ThetaZ degrees).
                          Go-to commands move the end effector, Grasp commands close/open the gripper, and Tilt commands roll/pitch/yaw the gripper. Grasp commands should only be used when you want to grasp or release a tool.
                          Keep in mind that the Tilts are calculated relatively, not absolutely. Initially, they are (0, 0, 0 degrees). Also, the angles are calculated based on the right hand thumb rule (ie. Pitching down is a positive ThetaY angle).
                          A Tilt command should have only 1 non-zero value.
                          Keep in mind that some actions will require the tool to be held at an angle, and not completely flat.
                          Any actions that you want to do should be described using these three commands, nothing else. If you want to perform more complex commands like applying forces, scooping, etc., reason them out so that they can broken down into these three fundamental building block commands.
                          
                          Reason out each step, and explain the intended effect for each step. 
                          Guidelines:
                            Keep in mind that you might want to end a pick up action by liftng the object you grasped. 
                            When releasing an object inside a container, do not go to "Original Position of the container + (0, 0, 0 cm)", as this will be inside the container. Instead, drop it from a height, ie. "Original Position of the container + (0, 0, 10 cm)"so that it is easier to open the gripper. 
                            Assume that all upstream tasks for the given step have been completed successfully.
                          At the end list out just the steps involved (no other information).

                          Keep in mind that if you want to place objects in any type of container, it is okay to drop them into the container (from 10 cm above the container)

                          Take a look at the examples below. Strictly follow the format of Expected Output.
                          <start of example>
                          [User Input]:
                            Action: Pick-up,
                            Location: Original Position of Bagel,
                            Description: [[30, 30, 10],[50]]
                            Object: Bagel
                            Tool: Spatula
                            Previous Steps: []
                          
                          [Expected Output]:
                          Explanation:
                            1. Grasp: 1
                            Reasoning: This ensures that the spatula is grasped securely by the gripper
                            2. Go-to: Original Position of Bagel + (-50, 0, 20 cm)
                            Reasoning: This ensures that the tool is positioned above an behind the bagel.
                            3. Go-to: Original Position of Bagel + (-50, 0, 0 cm)
                            Reasoning: This lowers the tool so that the flat part of the spatula is horizontal, and behind the bagel, and thus can be used to scoop it up.
                            4. Go-to: Original Position of Bagel + (10, 0, 0 cm)
                            Reasoning: This slides the flat part of the spatula under the bagel, thereby picking it up
                            5. Go-to: Original Position of Bagel + (10, 0, 10 cm)
                            Reasoning: Now that we have scooped up the bagel using the spatula, we can lift it up securely.

                          Steps List: 
                              1. Grasp: 1
                              2. Go-to: Original Position of Bagel + (-50, 0, 20 cm)
                              3. Go-to: Original Position of Bagel + (-50, 0, 0 cm)
                              4. Go-to: Original Position of Bagel + (10, 0, 0 cm)
                              5. Go-to: Original Position of Bagel + (10, 0, 10 cm)
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
                            Description: {Description}
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
    Action = "Place"
    Location = "Original Position of bowl"
    Description = '[[],[0]]'
    Object = "none"
    Tool = "onion"

    response = Plan2Action(Action, Location, Description, Object, Tool)
    print(response)
