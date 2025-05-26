from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import re

def resize_and_return_image(input_path, max_size=512):
    with Image.open(input_path) as img:
        img.thumbnail((max_size, max_size))
        return img

def encode_image(image):
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
def ProcessString(input_string):
    """
    Process the response string from the LLM to extract descriptions, answers, and plan steps.
    
    Args:
        input_string: The response string from the LLM
    
    Returns:
        Tuple of (descriptions_list, answers_list, plans_list)
    """
    # Lists to store extracted data
    descriptions = []
    answers = []
    plans = []
    
    # Extract description sections
    description_pattern = r"<start of description>(.*?)<end of description>"
    description_matches = re.findall(description_pattern, input_string, re.DOTALL)
    descriptions = [match.strip() for match in description_matches]
    
    # Extract answer sections
    answer_pattern = r"<start of answer>(.*?)<end of answer>"
    answer_matches = re.findall(answer_pattern, input_string, re.DOTALL)
    answers = [match.strip() for match in answer_matches]
    
    # Extract numbered plan steps
    # Look for the pattern: "1. **Text**" or similar numbered patterns
    plan_pattern = r"\d+\.\s+\*\*(.*?)\*\*"
    plan_matches = re.findall(plan_pattern, input_string)
    
    if plan_matches:
        plans = [match.strip() for match in plan_matches]
    else:
        # Try alternative pattern without bold formatting
        plan_pattern = r"\d+\.\s+(Use the.*?)\."
        plan_matches = re.findall(plan_pattern, input_string)
        
        if plan_matches:
            plans = [match.strip() for match in plan_matches]
        else:
            # If no numbered steps found, try finding bullet points
            plan_pattern = r"\*\s+(Use the.*?)\."
            plan_matches = re.findall(plan_pattern, input_string)
            plans = [match.strip() for match in plan_matches]
    
    # If still no plans found, look for any step starting with "Use the"
    if not plans:
        plan_pattern = r"(Use the[^\.]*\.)"
        plan_matches = re.findall(plan_pattern, input_string)
        plans = [match.strip() for match in plan_matches]
    
    # If plans are still empty, create a list of placeholder plans matching the number of descriptions
    if not plans and descriptions:
        plans = [f"Step {i+1}" for i in range(len(descriptions))]
    
    return descriptions, answers, plans

def Calculator(Lstar, H):
    print("Starting Calculator:")
    client = OpenAI()

    info_prompt = {"type": "text",
                    "text": f"""Description:\n{Lstar},
                                Plan:\n{H},
                            """
                    }

    prompt = [
        {
            "role": "system", 
            "content": """This part is to calculate the 3D target position of the gripper.

Common Rules:
* Calculate step by step and show the calculation process between <start of description> and <end of description>.
* Return the 3D position between <start of answer> and <end of answer>.
* You must not assume any position and directly query the updated position of the objects.
* You must only query the object name provided in object list when using 'get_center' and 'get_graspable_point'.
* The "open_gripper" and "close_gripper" do not need target positions. Return a space character between <start of answer> and <end of answer>.
* To bring an [OBJECT] into the workspace, the 3D target position should be [0.0, 0.0, object_size[2]/2].

IMPORTANT FORMAT INSTRUCTIONS:
For each step in the plan, your response MUST follow this exact format:

1. **Use the [SKILL] to [SINGLE_TASK].**

   <start of description>
   [Calculation process]
   <end of description>
   <start of answer>
   [Position coordinates or space for gripper actions]
   <end of answer>

2. **Use the [SKILL] to [SINGLE_TASK].**

   <start of description>
   [Calculation process]
   <end of description>
   <start of answer>
   [Position coordinates or space for gripper actions]
   <end of answer>

And so on for each step in the plan.

Example format:
1. **Use the 'get_center' to get the cup's center position.**

   <start of description>
   This step doesn't require a target position calculation as we are just getting the cup's center position.
   <end of description>
   <start of answer>
    
   <end of answer>

2. **Use the 'move_to_position' to move to the cup's center.**

   <start of description>
   The cup's center position is [0.3, 0.0, 0.1] as provided in the scene information.
   <end of description>
   <start of answer>
   [0.3, 0.0, 0.1]
   <end of answer>

Rule 1:
<Current Step>: Use the 'move_to_position' skill to move the [OBJECT1] to [OBJECT2].
Answer: 
<start of description>
* The [OBJECT1] is in hand. The target position should be the center of the [OBJECT2].
<end of description>
<start of answer>
The 3D target position is [OBJECT2]'s center position.
<end of answer>

Rule 2:
<Current Step>: Use the 'move_to_position' skill to push the [OBJECT] into the workspace.
Answer: 
<start of description>
To push an [OBJECT] into the workspace, the 3D target position should be [0.0, 0.0, object_size[2]/2].
<end of description>
<start of answer>
The 3D target position is [0.0, 0.0, object_size[2]/2].
<end of answer>

In the following, you will see the plan and must follow the rules.
"""
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
    
    # Process the response to extract descriptions, answers, and plans as separate lists
    descriptions, answers, plans = ProcessString(response)
    return descriptions, answers, plans

if __name__=="__main__":
    Lstar = "."
    H = "."
    response = Calculator(Lstar, H)
    print(response)

