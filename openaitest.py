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

def resize_and_return_image(input_path, max_size=512):
    with Image.open(input_path) as img:
        img.thumbnail((max_size, max_size))
        return img

def encode_image(image):
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
  
image_path = "Trials/Arm_no_tool_1.png"

base64_image = encode_image(resize_and_return_image(image_path))

from openai import OpenAI
client = OpenAI()
prompt = [
    {
        "role": "system", 
        "content": "You are tasked with checking if a parallel plate gripper has successfully grasped a tool. If the tool is currently held by the gripper, reply 'Yes'. If the tool is not held by the gripper, reply 'No'."
    },
    {
        "role": "user",
        "content": 
        [
          {
            "type": "text",
            "text": "Is the robot grasping the tool? (Yes/No)"
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
