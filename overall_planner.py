
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import re

def resize_and_return_image(input_path, max_size=512):
    with Image.open(input_path) as img:
        img.thumbnail((max_size, max_size))
        return img

def encode_image(image):
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
def ProcessString(input_string):
    input_string = input_string.lower()
    input_string = input_string.split(':')[1]
    steps = [step.strip() for step in input_string.strip().split('\n')]
    nested_list = [[re.sub(r'[^\w\s]', '', substep) for substep in step.split('. ', 1)[1].split(', ')] for step in steps]
    return nested_list
  
def OverallPlanner(Task, ObjList, PosList, ActionList, StepsList=[], step=0):

    print("Starting Overall Planner:")
    client = OpenAI()

    if step==0:

        info_prompt = {"type": "text",
                        "text": f"""Task: {Task},
                                    Objects: {ObjList},
                                    Positions: {PosList},
                                    Actions: {ActionList}
                                """
                        }
        


    
        prompt = [
            {
                "role": "system", 
                "content": """You will be given a task and a list of objects available to you to complete the task.
Your job is to give a step-by-step plan to complete the task.
Your inputs will be of the form:
    Task: The overall goal that your plan needs to achieve.
    Objects: A list of objects available to you
    Positions: A set of fixed positions in the robot workspace available for the robot to move to, consisting of semantic descriptions. You must use only these locations. You can however, mention positions relative to these positions
    Actions: A set of actions (ie. robot motion primitives) that you can use to construct your plan. You must pick your actions from this list.
Stick to these phrases exactly! Do not add any extra punctuations either.

This plan will be executed by a parallel plate gripper, so keep that in mind while constructing the plan.
Each step in your plan should roughly follow the format '<action>, <location>, <object>, <tool>'.
    action - it is the action that you want the robot arm to perform (Example: roll, flatten, push, etc.)
    location - it is the location in the workspace that you wish to go-to 
    object - it is the object that you want the robot arm to interact with. It must not currently be held by the gripper (Example: hammer, spoon, etc.)
    tool - it is the tool currently held by the gripper (Example: hammer, spoon, etc.)
For any given step, some of the above 4 parameters could be empty. (For example, when you move the gripper holding a spoon from one place to another, there is no object involved, only a tool). If any of these parameters are empty, report them as None for that step.

Example: 'Pick-up, hammer, handle, None'

As per the above example, each step should consist of just comma seperated words, no other special characters
Keep in mind that <location on object> must be a semantic description (not coordinates).
You cannot use any objects or actions not mentioned in the Objects and Actions list. 

General Guidelines:
Everytime you use an object as  a tool, place it back in its original position before moving onto the next step of the process, if the next step doesn't involve the same tool.
When you want to pick up an object just use 'Pick-up', you do not need to move to the object. This is because it handles the action of moving to the object as well as grasping it.

Take a look at the example below. Strictly follow the format of Expected Output.
<start of example>
[### User Input]:
    Task: "Place the hammer on top of the bench",
    Objects: ['hammer', 'bench'],
    Positions: ["Original Position of Hammer", "Original Position of bench"],
    Actions: ["Pick-up", "Release", "Move-to"]

[### Expected Output]:
    Overall Plan: 
    1. Pick-up, Original Position of Hammer, hammer, None
    2. Move-to, Original Position of bench, None, hammer 
    3. Release, Original Position of bench, None, hammer
    4. Move-to, Home Pose, None, None
<end of example>"""
                },
            {
                "role": "user",
                "content": 
                [
                    info_prompt
                ]
            }
        ]

    else:
        CompletedList = [StepsList[i] for i in range(step-1)]
        info_prompt = {"type": "text",
                        "text": f"""Task: {Task},
                                    Objects: {ObjList},
                                    Positions: {PosList},
                                    Actions: {ActionList},
                                    Previous Plan: {StepsList},
                                    Completed Actions: {CompletedList}
                                    Failed Action: {StepsList[step-1]}
                                """
                        }
        
        prompt = [
            {
                "role": "system", 
                "content": f"""You will be given a task, a list of objects available to you to complete the task, and the Steps that were previosuly generated by an LLM in order to complete the task.
The last action that was attempted was Step {step}: {StepsList[step-1]}, but it failed.
Your job is to replan and give a series of steps to complete the task. It can even be the same plan as the previous one, but keep in mind that the execution will begin from Step: {step}, so it might be better to plan from the start, to ensure that the step you want to be executed next in Step: {step}.

Your inputs will be of the form:
    Task: The overall goal that your plan needs to achieve.
    Objects: A list of objects available to you
    Positions: A set of fixed positions in the robot workspace available for the robot to move to, consisting of semantic descriptions.
    Actions: A set of actions (ie. robot motion primitives) that you can use to construct your plan. You must pick your actions from this list.  
    Previous Plan: The steps that were previosuly generated by an LLM in order to complete the task.
    Completed Actions: The list of steps in the previous plan that were successfully executed sequentially.
    Failed Action: The actions that failed to execute successfully.
Stick to these phrases exactly! Do not add any extra punctuations either.

This plan will be executed by a parallel plate gripper, so keep that in mind while constructing the plan.
Each step in your plan should roughly follow the format '<action>, <location>, <object>, <tool>'.
    action - it is the action that you want the robot arm to perform (Example: roll, flatten, push, etc.)
    location - it is the location in the workspace that you wish to go-to 
    object - it is the object that you want the robot arm to interact with. It must not currently be held by the gripper (Example: hammer, spoon, etc.)
    tool - it is the tool currently held by the gripper (Example: hammer, spoon, etc.)
For any given step, some of the above 4 parameters could be empty. (For example, when you move the gripper holding a spoon from one place to another, there is no object involved, only a tool). If any of these parameters are empty, report them as None for that step.


Example: 'Pick up, hammer, handle, None'

As per the above example, each step should consist of just comma seperated words, no other special characters
Keep in mind that <location on object> must be a semantic description (not coordinates).
You cannot use any objects or actions not mentioned in the Objects and Actions list. 

General Guidelines:
Everytime you use an object as  a tool, place it back in its original position before moving onto the next step of the process, if the next step doesn't involve the same tool
When you want to pick up an object just use 'Pick-up', you do not need to move to the object. This is because it handles the action of moving to the object as well as grasping it.

Take a look at the example below. Strictly follow the format of Expected Output.
<start of example>
[### User Input]:
    Task: "Place the hammer on top of the bench",
    Objects: ['hammer', 'bench'],
    Positions: ["Original Position of Hammer", "Original Position of bench"],
    Actions: ["Pick-up", "Release", "Move-to"]
    Previous Plan: [<Previous Plan>],
    Completed Actions: []
    Failed Action: [Pick-up, Original Position of Bench, hammer, None]

[### Expected Output]:
    Overall Plan: 
    1. Pick-up, Original Position of Hammer, hammer, None
    2. Move-to, Original Position of bench, None, hammer 
    3. Release, Original Position of bench, None, hammer
    4. Move-to, Original Position of Hammer, None, None
<end of example>"""
            },
            {
                "role": "user",
                "content": 
                [
                    info_prompt
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
    # image_path = "Trials/Real_table_w_tools.jpg"
    Task = "Pick up a screwdriver"
    ObjList = ["Screwdriver"]
    PosList = ["homepose", "Original Position of screwdriver"]
    ActionList = ["Push-down", "Move-to", "Grasp", "Release", "Roll", "Pour"]
    response = OverallPlanner(Task, ObjList, PosList, ActionList, StepsList=[], step=0)
    print(response)

