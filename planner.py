
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
  
def TerminationCheck():
    # base64_image = encode_image(resize_and_return_image(image_path))

    cookie_prompt = {"type": "text",
                     "text": """Task: Make a star shape chocolate chip cookie dough
                             Objects: Ball of cookie dough,
                                      Circle shaped cookie cutter,
                                      Star shaped cookie cutter,
                                      Rolling pin,
                                      Scissors,
                                      Hammer,
                                      Pile of chocolate chips,
                                      Spoon,
                                      Sewing needle
 
                             Positions: Original Position of Ball of cookie dough,
                                        Original Position of Circle shaped cookie cutter,
                                        Original Position of Star shaped cookie cutter,
                                        Original Position of Rolling pin,
                                        Original Position of Scissors,
                                        Original Position of Hammer,
                                        Original Position of Pile of chocolate chips,
                                        Original Position of Spoon,
                                        Original Position of Sewing needle
                                         
                                 Actions: Grasp, Push-down, Move-to, Grasp, Release, Roll, Pour  
                             """
                    }
    


    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You will be given a task and a list of objects available to you to complete the task. 
                          Your job is to give a step-by-step plan to complete the task.
                          Your inputs will be of the form:
                            Task: The overall goal that your plan needs to achieve.
                            Objects: A list of objects available to you
                            Positions: A set of fixed positions in the robot workspace available for the robot to move to, consisting of a semantic description and a set of 3D coordinates.
                            Actions: A set of actions (ie. robot motion primitives) that you can use to construct your plan. You must pick your actions from this list.

                          This plan will be executed by a parallel plate gripper, so keep that in mind while constructing the plan.
                          Each step in your plan should roughly follow the format '<action>, <object> (optional), <location on object> (optional).
                          
                          For example: 'Pick up, hammer, handle'
                          As per the above example, each step should consist of just comma seperated words, no other special characters
                          Keep in mind that <location on object> must be a semantic description (not coordinates).
                          You cannot use any objects or actions not mentioned in the Objects and Actions list. 
                          
                          General Guidelines:
                          Everytime you use a an object as  a tool, place it back in its original position before moving onto the next step of the process, if the next step doesn't involve the same tool"""
        },
        {
            "role": "user",
            "content": 
            [
                cookie_prompt
            ]
        }
    ]
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=prompt
    )

    return(completion.choices[0].message.content)

if __name__=="__main__":
    # image_path = "Trials/Real_table_w_tools.jpg"
    response = TerminationCheck()
    print(response)

