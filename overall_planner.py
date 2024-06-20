
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
    
def ProcessString(input_string):
    steps = [step.strip() for step in input_string.strip().split('\n')]
    nested_list = [step.split('. ', 1)[1].split(', ') for step in steps]
    return nested_list
  
def OverallPlanner(Task, ObjList, PosList, ActionList, StepsList=[], step=0):
    # base64_image = encode_image(resize_and_return_image(image_path))

    # cookie_prompt = {"type": "text",
    #                  "text": """Task: Make a star shape chocolate chip cookie dough
    #                          Objects: Ball of cookie dough,
    #                                   Circle shaped cookie cutter,
    #                                   Star shaped cookie cutter,
    #                                   Rolling pin,
    #                                   Scissors,
    #                                   Hammer,
    #                                   Pile of chocolate chips,
    #                                   Spoon,
    #                                   Sewing needle
 
    #                          Positions: Original Position of Ball of cookie dough,
    #                                     Original Position of Circle shaped cookie cutter,
    #                                     Original Position of Star shaped cookie cutter,
    #                                     Original Position of Rolling pin,
    #                                     Original Position of Scissors,
    #                                     Original Position of Hammer,
    #                                     Original Position of Pile of chocolate chips,
    #                                     Original Position of Spoon,
    #                                     Original Position of Sewing needle
                                         
    #                              Actions: Grasp, Push-down, Move-to, Grasp, Release, Roll, Pour  
    #                          """
    #                 }
    client = OpenAI()

    if step==0:

        info_prompt = {"type": "text",
                        "text": f"""Task: {Task},

                                    Objects: {ObjList},
    
                                    Positions: {PosList},
                                            
                                    Actions: {ActionList},
                                """
                        }
        


    
        prompt = [
            {
                "role": "system", 
                "content": """You will be given a task and a list of objects available to you to complete the task. 
                            Your job is to give a step-by-step plan to complete the task.
                            Your inputs will be of the form:
                                Task: The overall goal that your plan needs to achieve.
                                Objects: A list of objects available to you
                                Positions: A set of fixed positions in the robot workspace available for the robot to move to, consisting of semantic descriptions.
                                Actions: A set of actions (ie. robot motion primitives) that you can use to construct your plan. You must pick your actions from this list.

                            This plan will be executed by a parallel plate gripper, so keep that in mind while constructing the plan.
                            Each step in your plan should roughly follow the format '<action>, <location>, <object>, <tool>'.
                                action - it is the action that you want the robot arm to perform (Example: roll, flatten, push, etc.)
                                location - it is the location on the object that you wish to perform this action (Example: center of dough, handle of hammer, etc.)
                                object - it is the object that you want the robot arm to interact with. It must not currently be held by the gripper (Example: hammer, spoon, etc.)
                                tool - it is the tool currently held by the gripper (Example: hammer, spoon, etc.)
                            For any given step, some of the above 4 parameters could be empty. (For example, when you move the gripper holding a spoon from one place to another, there is no object involved, only a tool). If any of these parameters are empty, report them as None for that step.
                            
                            Example: 'Pick up, hammer, handle, None'

                            As per the above example, each step should consist of just comma seperated words, no other special characters
                            Keep in mind that <location on object> must be a semantic description (not coordinates).
                            You cannot use any objects or actions not mentioned in the Objects and Actions list. 
                            
                            General Guidelines:
                            Everytime you use an object as  a tool, place it back in its original position before moving onto the next step of the process, if the next step doesn't involve the same tool"""
            },
            {
                "role": "user",
                "content": 
                [
                    info_prompt
                ]
            }
        ]

    else:
        CompletedList = [StepsList[i] for i in range(step-1)]
        info_prompt = {"type": "text",
                        "text": f"""Task: {Task},

                                    Objects: {ObjList},
    
                                    Positions: {PosList},
                                            
                                    Actions: {ActionList},

                                    Previous Plan: {StepsList},

                                    Completed Actions: {CompletedList}

                                    Failed Action: {StepsList[step-1]}
                                """
                        }
        
        prompt = [
            {
                "role": "system", 
                "content": f"""You will be given a task, a list of objects available to you to complete the task, and the Steps that were previosuly generated by an LLM in order to complete the task.
                              The last action that was attempted to be done was Step {step}: {StepsList[step-1]}, but it failed.
                              Your job is to replan and give a series of steps to complete the task. It can even be the same plan as the previous one, but start from the current state, not from the start, ie. don't repeat the completed steps.
                              Your inputs will be of the form:
                                  Task: The overall goal that your plan needs to achieve.
                                  Objects: A list of objects available to you
                                  Positions: A set of fixed positions in the robot workspace available for the robot to move to, consisting of semantic descriptions.
                                  Actions: A set of actions (ie. robot motion primitives) that you can use to construct your plan. You must pick your actions from this list.  
                                  Previous Plan: The steps that were previosuly generated by an LLM in order to complete the task.
                                  Completed Actions: The list of steps in the previous plan that were successfully executed sequentially.
                                  Failed Action: The actions that failed to execute successfully.

                              This plan will be executed by a parallel plate gripper, so keep that in mind while constructing the plan.
                              Each step in your plan should roughly follow the format '<action>, <location>, <object>, <tool>'.
                                  action - it is the action that you want the robot arm to perform (Example: roll, flatten, push, etc.)
                                  location - it is the location on the object that you wish to perform this action (Example: center of dough, handle of hammer, etc.)
                                  object - it is the object that you want the robot arm to interact with. It must not currently be held by the gripper (Example: hammer, spoon, etc.)
                                  tool - it is the tool currently held by the gripper (Example: hammer, spoon, etc.)
                              For any given step, some of the above 4 parameters could be empty. (For example, when you move the gripper holding a spoon from one place to another, there is no object involved, only a tool). If any of these parameters are empty, report them as None for that step.
                              
                              Example: 'Pick up, hammer, handle, None'
                              As per the above example, each step should consist of just comma seperated words, no other special characters
                              Keep in mind that <location on object> must be a semantic description (not coordinates).
                              You cannot use any objects or actions not mentioned in the Objects and Actions list. 
                              
                              General Guidelines:
                              Everytime you use an object as  a tool, place it back in its original position before moving onto the next step of the process, if the next step doesn't involve the same tool"""
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
    return(ProcessString(response))

if __name__=="__main__":
    # image_path = "Trials/Real_table_w_tools.jpg"
    Task = "Make a star shape chocolate chip cookie dough"
    ObjList = ["Ball of cookie dough", "Circle shaped cookie cutter", "Star shaped cookie cutter", "Rolling pin", "Scissors", "Hammer", "Pile of chocolate chips", "Spoon", "Sewing needle"]
    PosList = ["workspace center", "Original Position of Ball of cookie dough","Original Position of Circle shaped cookie cutter", "Original Position of Star shaped cookie cutter", "Original Position of Rolling pin", "Original Position of Scissors", "Original Position of Hammer", "Original Position of Pile of chocolate chips", "Original Position of Spoon", "Original Position of Sewing needle"]
    ActionList = ["Push-down", "Move-to", "Grasp", "Release", "Roll", "Pour"]
    response = OverallPlanner(Task, ObjList, PosList, ActionList, StepsList=[], step=0)
    print(response)

