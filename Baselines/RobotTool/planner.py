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
    """
    Process the response string from the LLM to extract analysis and plan parts.
    
    Args:
        input_string: The response string from the LLM
    
    Returns:
        Tuple of (analysis_text, plan_text)
    """
    # Find the analysis section
    analysis_start = input_string.find("<start of analysis>")
    analysis_end = input_string.find("<end of analysis>")
    
    # Find the plan section
    plan_start = input_string.find("<start of plan>")
    plan_end = input_string.find("<end of plan>")
    
    # Extract the text from each section
    analysis_text = ""
    plan_text = ""
    
    if analysis_start != -1 and analysis_end != -1:
        analysis_text = input_string[analysis_start + len("<start of analysis>"):analysis_end].strip()
    
    if plan_start != -1 and plan_end != -1:
        plan_text = input_string[plan_start + len("<start of plan>"):plan_end].strip()
    
    return analysis_text, plan_text
  
def Planner(UserPrompt_description):
    print("Starting Planner:")
    client = OpenAI()

    info_prompt = {"type": "text",
                   "text": f"""{UserPrompt_description}
                           """
                  }
    
    prompt = [
        {
            "role": "system", 
            "content": """You must strictly follow the answer format below (your response will be programmatically parsed):
 
<start of analysis>
Given an object list [OBJECT1, OBJECT2, OBJECT3, OBJECT4, ...]

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

<start of plan>
* Use the [SKILL] to [SINGLE_TASK].
* Use the [SKILL] to [SINGLE_TASK].
* Use the [SKILL] to [SINGLE_TASK].
...
<end of plan>

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

IMPORTANT: Your response must have exactly two sections enclosed in the <start> and <end> tags. First the analysis section, then the plan section."""
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
    
    # Process the response to extract analysis and plan sections
    analysis, plan = ProcessString(response)
    return analysis, plan

if __name__=="__main__":
    # image_path = "Trials/Real_table_w_tools.jpg"
    UserPromp_description = """."""
    analysis, plan = Planner(UserPromp_description)
    print(analysis, plan)

