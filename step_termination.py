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
  
def TerminationCheck(image_path_1, image_path_2 = None, image_path_3 = None):
    base64_image_1 = encode_image(resize_and_return_image(image_path_1))
    if image_path_2 != None:
        base64_image_2 = encode_image(resize_and_return_image(image_path_2))
    if image_path_3 != None:
        base64_image_3 = encode_image(resize_and_return_image(image_path_3))


    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You are tasked with checking if a parallel plate gripper has successfully grasped a tool, based on an egocentric view, taken from a camera near the end effector. 
                          This will give you the best idea of whether the gripper are grasping the object or not.
                          If the tool is currently held by the gripper, reply 'Yes'. 
                          If the tool is not held by the gripper, reply 'No'."""
        },
        {
            "role": "user",
            "content": 
            [
            {
                "type": "text",
                "text": "Based on the following three images, is the robot grasping the tool? (Yes/No)"
            },
            {
                "type": "image_url",
                "image_url": 
                {
                "url": f"data:image/jpeg;base64,{base64_image_1}",
                "detail": "low"
                }
            }
            # {
            #     "type": "image_url",
            #     "image_url": 
            #     {
            #     "url": f"data:image/jpeg;base64,{base64_image_2}",
            #     "detail": "low"
            #     }
            # },
            # {
            #     "type": "image_url",
            #     "image_url": 
            #     {
            #     "url": f"data:image/jpeg;base64,{base64_image_3}",
            #     "detail": "low"
            #     }
            # }
            ]
        }
    ]

    
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=prompt
    )

    return(completion.choices[0].message.content,"\n")

if __name__=="__main__":
    image_path_1 = "Trials/Termination_1_view_4.jpg"
    image_path_2 = "Trials/Termination_1_view_1.jpg"
    image_path_3 = "Trials/Termination_1_view_2.jpg"
    # response = TerminationCheck(image_path_1, image_path_2, image_path_3)
    response = TerminationCheck(image_path_1)
    print(response)

