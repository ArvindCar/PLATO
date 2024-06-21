# from openai import OpenAI
# client = OpenAI()
# prompt = [
#     {"role": "system", "content": "You are a high level planner, whose goal is to guide a parallel plate gripper"},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#     ]
# completion = client.chat.completions.create(
#     model='gpt-4o',
#     messages=prompt
# )

# print(completion.choices[0].message.content)

from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
from segment_anything import SamPredictor, sam_model_registry

def resize_and_return_image(input_path, max_size=512):
    with Image.open(input_path) as img:
        img.thumbnail((max_size, max_size))
        return img

def encode_image(image):
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
  
image_path = "Trials/Grid_2.png"
base64_image = encode_image(resize_and_return_image(image_path))


def main():
    
    return None

from openai import OpenAI
client = OpenAI()
prompt = [
    {
        "role": "system", 
        "content": """You are tasked with observing an image of a parallel plate gripper in the process of grasping a tool. 
                      You will be provided a side-view image of the scene, with a green grid superimposed on top of it. 
                      Grid Description:
                      Rows are represented by numbers (1-6) and columns are represented by alphabets (A-F). 
                      Each Cell position in the grid is represented by a combination of the column letter and row number. (For example, B3).

                      Your reply should contain three parts:
                        Part 1: Location of the gripper:
                          Your reply should be a cell position (for example B3). 
                          Assume that the purple dot is representative of the current gripper location.
                          Assume that the cyan dot is representative of the current object location.

                        Part 2: Is the grasp successful?
                          If the tool is currently held by the gripper, reply 'Yes'. 
                          If the tool is not held by the gripper, reply 'No'.

                        Part 3: Further action
                          If your reply to Part2 is 'No', use the grid to tell me which cell to go to in order to be able to grasp the object.
                          If your reply to Part2 is 'Yes', reply with 'None'
                      
                      Overall your answer should be of the format:
                      <cell>
                      <yes/no>
                      <cell/None>"""
    },
    {
        "role": "user",
        "content": 
        [
          {
            "type": "text",
            "text": "Is the robot grasping the tool?"
          },
          {
            "type": "image_url",
            "image_url": 
            {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "low"
            }
          }
        ]
    }
]
completion = client.chat.completions.create(
    model='gpt-4o',
    messages=prompt
)

print(completion.choices[0].message.content,"\n")
# print(type(completion.choices[0].message.content))
