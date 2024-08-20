
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import re
import ast
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
    # Parse the input string into Python objects
    parsed_input = ast.literal_eval(input_string)
    
    # Ensure the input is a list of lists
    if not all(isinstance(i, list) for i in parsed_input):
        raise ValueError("Input must be a list of lists")
    
    # Process the first list (assumed to be a list of strings)
    processed_list_1 = [re.sub(r'[^\w\s]', '', item.lower()) for item in parsed_input[0]]
    
    # Process the second list (assumed to be a list of integers)
    processed_list_2 = parsed_input[1]
    
    return processed_list_1, processed_list_2
  
def SceneComprehension(image_path, task):
    print("Starting scene Comprehension:")
    image_path_1 = image_path + "/Image2.png"
    base64_image = encode_image(image_path_1)

    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You will be given an image of a table with several objects on it. You will also be given a task which is to be performed by downstream LLMs, within this scene.
                          Your task is to observe the image and list out the various objects present on the table, ensuring the descriptions are brief and relevant to the context of the given task. It is even better if you incorporate the exact phrases from the task provided.
                          For example, if the task involves adding mustard sauce, clearly identify the object as "plastic mustard bottle" rather than just "plastic sauce bottle."
                          You are also required to provide a binary value, indicating if the object has a handle (1) or not (0).
                          Ignore any markings on the table itself.
                          Your output should 2 lists: One should be a comma seperated list of objects, in alphabetical order, and the other should be a correspodning list of their binary handle flags.
                          Try to describe the objects very briefly using the context provided. 
                          For example, if the task is related to metal objects, use the descriptor 'metal' before each object.
                          For example if you see a deformed ball like shape on the table, and the task is to "Make a cookie", then the ball object is most likely "ball of dough".
                          To the best of your ability, describe each object in a single word/phrase.
                          Example Output: (Stick to the below format exactly ie. just two lists)
                          ['plastic clay', 'plastic box', 'plastic screwdriver'], [0, 0, 1]"""
        },
        {
            "role": "user",
            "content": 
            [
            {
                "type": "text",
                "text": f"What objects are present in the given image? The task that this image is related to is {task}"
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
    return(responselist)

if __name__=="__main__":
    image_path = "/home/aesee/CMU/MAIL_Lab/LLM_Tool/Save_dir/step3"
    task = "Take the tape measure off the board"
    response, flags = SceneComprehension(image_path, task)
    print(response, flags)

