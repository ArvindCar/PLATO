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
  
def TerminationCheck(images_path, task):
    print("Checking Step Termination:")

    image_path_1 = images_path + "/Image1.png"
    image_path_2 = images_path + "/Image2.png"
    image_path_3 = images_path + "/Image3.png"
    image_path_4 = images_path + "/Image4.png"

    base64_image_1 = encode_image(image_path_1)
    base64_image_2 = encode_image(image_path_2)
    base64_image_3 = encode_image(image_path_3)
    base64_image_4 = encode_image(image_path_4)


    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You are tasked with checking if a parallel plate gripper has successfully completed a task, based on an multi-angle views, taken from a cameras surrounding the scene. 
                          You will be given the task that was attempted by the gripper, as well as multi-angle views to help you understand the current state of the system.
                          If the task has been successfully completed, reply 1. 
                          If the task has not been successfully completed, reply 0."""
        },
        {
            "role": "user",
            "content": 
            [
            {
                "type": "text",
                "text": f"The Task assigned to the robot arm was {task}. Based on the following four images, has the robot completed the task? (1/0)"
            },
            {
                "type": "image_url",
                "image_url": 
                {
                "url": f"data:image/jpeg;base64,{base64_image_1}",
                "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": 
                {
                "url": f"data:image/jpeg;base64,{base64_image_2}",
                "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": 
                {
                "url": f"data:image/jpeg;base64,{base64_image_3}",
                "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": 
                {
                "url": f"data:image/jpeg;base64,{base64_image_4}",
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
    response = completion.choices[0].message.content
    print(response)

    return(response)

if __name__=="__main__":
    image_path_1 = "Trials/Termination_1_view_4.jpg"
    image_path_2 = "Trials/Termination_1_view_1.jpg"
    image_path_3 = "Trials/Termination_1_view_2.jpg"
    # response = TerminationCheck(image_path_1, image_path_2, image_path_3)
    response = TerminationCheck(image_path_1)
    print(response)

