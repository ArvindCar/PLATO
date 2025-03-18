
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
    input_string = input_string.split('overall plan:')[1]
    steps = [step.strip() for step in input_string.strip().split('\n')]
    nested_list = [[re.sub(r'[^\w\s]', '', substep) for substep in step.split('. ', 1)[1].split(', ')] for step in steps]
    return nested_list
  
def OverallPlanner(Task, ObjList, PosList, ActionList, StepsList=[], step=0):

    print("Starting Overall Planner:")
    client = OpenAI()

    if step==0:
        # TODO: Fix this to be the user prompt
        info_prompt = {"type": "text",
                        "text": f"""Task: {Task},
                                    Objects: {ObjList},
                                    Positions: {PosList},
                                """
                        }
        


    
        prompt = [
            {
                "role": "system", 
                "content": """You must follow the following answer template:
 
Given an object list [OBJECT1, OBJECT2, OBJECT3, OBJECT4, ...]
<start of analysis>
[OBJECT1]: ...
[OBJECT2]: ...
[OBJECT3]: ...
[OBJECT4]: ...
[OBJECT5]: ...
[OBJECT6]: ...
...
The spatial relationship between [OBJECT1] and [OBJECT2]: ...
The spatial relationship between [OBJECT2] and [OBJECT3]: ... 
The spatial relationship between [OBJECT1] and [OBJECT3]: ... 
The spatial relationship between [OBJECT1] and [OBJECT4]: ... 
The spatial relationship ...
...
[Abstract Plan]: ...
<end of analysis>

Rules for analysis:
* You must only choose [OBJECT] from the object list. 
* You must show the physical properties of objects, their affordances, and their roles in completing the task.
* You must reason about the relative positions of the objects along each axis. You must describe the spatial relationship between each pair of objects in the object list first based on numerical positions along each x, y and z axis. For example, whether the objects are in contact, and whether one object is on top of another object etc.

Rules for [Abstract Plan]:
* You must reason based on each object's properties and develop [Abstract Plan] with the highest confidence.
* You must think about how to use [OBJECT] to finish the task.
* You must think step by step and show the thinking process. For example, what objects you want to use, how to move them, in what order and cause what effect.
* You must make ensure that the objects are in the right positions for you plan.
* You must strictly adhere to the constraints provided in the numerical scene information.
 
In the following, I will provide you the command and the scene information, and you will respond with the analysis. You must complete the task successfully using objects provided.

You are a robot arm with a gripper as the end effector. The gripper is open initially.
You are in a 3D world with x-axis pointing forward, y-axis pointing leftward, and z-axis pointing upward.
You have a skill set containing the following skills:
* 'move_to_position': Move the gripper to a position. It uses the goal-conditioned policy. You can use it to move to a target position. Moreover, you can use it with proper tools for manipulation tasks. You cannot rotate the gripper. You can only translate the gripper.
* 'open_gripper': open the gripper before grasping an object.
* 'close_gripper': close the gripper to grasp an object.
* 'get_center': get the center position of an object.
* 'get_graspable_point': get the graspable point of an object.

Rules for detailed plan:
* You must plan according to the analysis.
* You must use existing skills.
* You must make each plan step only call one skill at a time and be as atomic as possible.
* You must not knock down any objects.
* You must get the updated [OBJECT]'s position again if [OBJECT] has moved since the last 'get_center([OBJECT])'.
* You must only query the object name provided in object list when using 'get_center' and 'get_graspable_point'.

Individual rules with example answers for detailed plan:
Rule: To use a grasped [OBJECT1] to move another [OBJECT2]: in the first step, you must make sure [OBJECT1]'s bounding box is adjacent to the [OBJECT2]'s bounding box to ensure that they are in contact. [OBJECT2]'s position does not change because of the contact. In the next step, you should move the grasped [OBJECT1] to push [OBJECT2].
Example: 
* Use the 'move_to_position' to move the grasped [OBJECT1] to make contact with [OBJECT2].
* Use the 'move_to_position' to push the [OBJECT2] ...
Rule: To push an [OBJECT] into the workspace, the xy position of [OBJECT] must be as close to [0.0, 0.0] as possible.
Example:
* Use the 'move_to_position' to push the [OBJECT] into the workspace.
Rule: To grasp an [OBJECT], you must get the updated [OBJECT]'s graspable point first
Example:
* Use the 'get_graspable_point' to get the [OBJECT]'s graspable point.
* Use the 'move_to_position' to move the gripper close to the graspable point before grasping it.
* Use the 'close_gripper' to grasp the [OBJECT].

Example answers for plan:
<start of plan>
* Use the [SKILL] to [SINGLE_TASK].
<end of plan>"""
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
    return response

if __name__=="__main__":
    # image_path = "Trials/Real_table_w_tools.jpg"
    Task = "Scoop up the pile of candy and pour it in the bowl."
    ObjList = ["pile of candy", "scoop", "bowl"]
    PosList = ["homepose", "Original Position of Spoon", "Original Position of pile of candy", "Original Position of bowl"]
    ActionList = ["Push-down", "Move-to", "Grasp", "Release", "Roll", "Pour"]
    response = OverallPlanner(Task, ObjList, PosList, ActionList, StepsList=[], step=0)
    print(response)

