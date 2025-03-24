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
  
def Coder(plan):

    print("Starting Coder:")
    client = OpenAI()

    info_prompt = {"type": "text",
                    "text": f"""Plan skeleton with numerical values: {plan}
                            """
                    }
    



    prompt = [
        {
            "role": "system", 
            "content": """You are a robot arm.
The robot has a skill set: ['get_center', 'get_graspable_point', 'get_size', 'move_to_position', 'open_gripper', 'close_gripper', 'get_workspace_range'].
You have a description of the plan to finish a task. We want you to turn the plan into the corresponding program with following functions:
```
def get_center(object_name):
 return object_center_position
  
```
```
def get_graspable_point(object_name):
 return object_graspable_point
  
```
```
def get_size(object_name):
 return object_size
  
```
```
def move_to_position(target_position):
```
```
def open_gripper():
```
```
def close_gripper():
```
```
def get_workspace_range(self):
 return x_min, y_min, z_min, x_max, y_max, z_max
  
```

Rules:
* Always format the code in code blocks.
* You must not leave unimplemented code blocks in your response.
* You must not leave undefined variables in your response.
* The only allowed library is numpy. Do not import or use any other library. If you use np, be sure to import numpy.
* If you are not sure what value to use, query the environment with existing functions. Do not use None for anything.
* Do not define new functions, and only use existing functions.
* You must only query the object name provided in object list when using 'get_center' and 'get_graspable_point'.

Example python code:
```
import numpy as np  # import numpy because we are using it below

x_min, y_min, z_min, x_max, y_max, z_max = get_workspace_range()
# following the detailed plan
```

In the following, I provide you the generated plan and you respond with the code."""
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
    response = Coder(Task, ObjList, PosList, ActionList, StepsList=[], step=0)
    print(response)

