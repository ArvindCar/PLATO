
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
            "content": """You will be given the side view image of a parllel plate gripper robot arm, and an object placed on a table. 
                          You are tasked with observing the image and commenting on the ability of the arm to grasp the object. 
                          If the robot arm is currently capable of gripping the object simply by closing the grippers, comment "Yes".
                          If the robot arm is currently incapable of gripping the object simply by closing the grippers, comment "No", followed by the directions in which it should move with respect to this view, in order to be aligned with the object.
                          For example, your output could be "No. Left, Down", to signify that if the grippers were to move Left and then Down, it would be able to pick up the object
                          Your answer sghould be of the form "<Yes/No>. <Comma seperated Directions>"."""
        },
        {
            "role": "user",
            "content": 
            [
            {
                "type": "text",
                "text": "Can the gripper pick up the object?"
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
    image_path = "Trials/feedback_2.jpg"
    response = TerminationCheck(image_path)
    print(response)

