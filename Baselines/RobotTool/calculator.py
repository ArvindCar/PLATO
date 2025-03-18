
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
  
def Calculator(Task, ObjList, PosList, ActionList, StepsList=[], step=0):

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
                "content": """This part is to calculate the 3D target position of the gripper.

Common Rules:
* Calculate step by step and show the calculation process between <start of description> and <end of description>.
* Return the 3D position between <start of answer> and <end of answer>.
* You must not assume any position and directly query the updated position of the objects.
* You must only query the object name provided in object list when using 'get_center' and 'get_graspable_point'.
* The "open_gripper" and "close_gripper" do not need target positions. Return a space character between <start of answer> and <end of answer>.
* To bring an [OBJECT] into the workspace, the 3D target position should be [0.0, 0.0, object_size[2]/2].

Rule 1:
<Current Step>: Use the 'move_to_position' skill to move the [OBJECT1] to [OBJECT2].
Answer: 
<start of description>
* The [OBJECT1] is in hand. The target position should be the center of the [OBJECT2].
<end of description>
<start of answer>
The 3D target position is [OBJECT2]'s center position.
<end of answer>
Rule 2:
<Current Step>: Use the 'move_to_position' skill to push the [OBJECT] into the workspace.
Answer: 
<start of description>
To push an [OBJECT] into the workspace, the 3D target position should be [0.0, 0.0, object_size[2]/2].
<end of description>
<start of answer>
The 3D target position is [0.0, 0.0, object_size[2]/2].
<end of answer>

In the following, you will see the plan and must follow the rules.
"""
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
    Task = "Scoop up the pile of candy and pour it in the bowl."
    ObjList = ["pile of candy", "scoop", "bowl"]
    PosList = ["homepose", "Original Position of Spoon", "Original Position of pile of candy", "Original Position of bowl"]
    ActionList = ["Push-down", "Move-to", "Grasp", "Release", "Roll", "Pour"]
    response = Calculator(Task, ObjList, PosList, ActionList, StepsList=[], step=0)
    print(response)

