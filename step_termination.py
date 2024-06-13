#
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
  
def TerminationCheck(image_path):
    base64_image = encode_image(resize_and_return_image(image_path))

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

    return(completion.choices[0].message.content,"\n")

if __name__=="__main__":
    image_path = "/home/aesee/CMU/MAIL_Lab/LLM_Tool/Arm_no_tool_1.png"
    response = TerminationCheck(image_path)
    print(response, "\n")

