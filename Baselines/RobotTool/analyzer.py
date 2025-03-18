
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
  
def OverallPlanner(user_input):

    print("Starting Overall Planner:")
    client = OpenAI()

    # TODO: Fix this to be the user prompt
    info_prompt = {"type": "text",
                    "text": f"""{user_input}
                            """
                    }
    prompt = [
        {
            "role": "system", 
            "content": """Answer template:
<start of analysis>

<end of analysis>
<start of description>
The key feature is the weight of the box which is 10kg, since the robot can only push a box with a maximum weight of 5kg.
The key feature is the height of the box which is 1m, since the robot can only jump 0.5m.
<end of analysis>
 
Remember:
* Think step by step and show the steps between <start of analysis> and <end of analysis>.
* Return the key feature and its value between <start of description> and <end of description>.
* The key features are the 3D spatial information that are not directly included in the [Scene Description] but needs further calculation.

In the following, I will provide you the task description."""
                },
            {
                "role": "user",
                "content": user_input
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
    user_input = """You are in a 3D world. You are a robot arm mounted on a table. You can control the end effector's position and gripper. Object list = ['milk', 'hammer', 'stuffed_toy', 'lock']. You want to grasp the milk.

Numerical scene information:
- The position is represented by a 3D vector [x, y, z]. The axes are perpendicular to each other.
- The base of the robot arm is at [0.0, 0.0, 0.0].
- [milk]: <center>: [0.568, 0.01, 0.085]; <size>: [0.044, 0.044, 0.16]; <graspable point>: [0.568, 0.01, 0.135]. 
- [stuffed_toy]: <center>: [0.45, -0.05, 0.025]; <size>: [0.08, 0.05, 0.05]; <graspable point>: [0.45, -0.05, 0.025]. 
- [lock]: <center>: [0.35, 0.05, 0.025]; <size>: [0.04, 0.03, 0.02]; <graspable point>: [0.35, 0.05, 0.025]. 
"""
    response = OverallPlanner(user_input)

