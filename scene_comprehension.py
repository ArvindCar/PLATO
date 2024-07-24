
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import re
# def resize_and_return_image(input_path, max_size=512):
#     try:
#         with Image.open(input_path) as img:
#             img.thumbnail((max_size, max_size   ))
#             return img
        
#     except Exception as e:
#         print(f"Error opening image '{image_path}': {e}")
#         return None 

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# def encode_image(image):
#     with BytesIO() as buffer:
#         image.save(buffer, format="PNG")
#         return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
def ProcessString(input_string):
    split_strings = input_string.lower().split(", ")
    li = [re.sub(r'[^\w\s]', '', substring) for substring in split_strings]
    return li
  
def SceneComprehension(image_path):
    print("Starting scene Comprehension:")
    image_path_1 = image_path + "/Image2.png"
    base64_image = encode_image(image_path_1)

    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You will be given an image of a table with several objects on it. You will also be given a task which is to be performed by downstream LLMs, within this scene.
                          Your task is to observe the image and list out the various objects present on the table. 
                          Ignore any markings on the table itself.
                          Your output should be a comma seperated list of objects, in alphabetical order.
                          When listing these objects, keep in mind the context of the task itself.
                          For example if you see a deformed ball like shape on the table, and the task is to "Make a cookie", then the ball object is most likely "ball of dough".
                          To the best of your ability, describe each object in a single word/phrase."""
        },
        {
            "role": "user",
            "content": 
            [
            {
                "type": "text",
                "text": "What objects are present in the given image?"
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
    response = completion.choices[0].message.content
    responselist = ProcessString(response)
    print(response)
    return(responselist)

if __name__=="__main__":
    image_path = "/home/arvind/LLM_Tool/LLM-Tool/Save_dir/step0"
    response = SceneComprehension(image_path)
    print(response)

