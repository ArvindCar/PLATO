
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
    
from openai import OpenAI
import ast
import re

def ProcessResponse(input_string):
    """
    Process the response from the LLM into a structured format.
    Expected input is a dictionary mapping object names to their grasp points.
    """
    try:
        # Parse the input string into Python objects
        parsed_input = ast.literal_eval(input_string)
        
        # Ensure the input is a dictionary
        if not isinstance(parsed_input, dict):
            raise ValueError("Input must be a dictionary mapping objects to grasp points")
        
        return parsed_input
    except Exception as e:
        print(f"Error processing response: {e}")
        return {}

def GraspPointDetection(objects, positions, dimensions):
    """
    Determine appropriate grasp points for a list of objects based on their positions and dimensions.
    
    Args:
        objects: List of object names
        positions: List of object positions (x, y, z)
        dimensions: List of object dimensions (width, height, depth)
        
    Returns:
        Dictionary mapping object names to their grasp points
    """
    print("Starting Grasp Point Detection:")
    
    # Create combined data for the prompt
    object_data = []
    for i in range(len(objects)):
        object_data.append({
            "name": objects[i],
            "position": positions[i],
            "dimensions": dimensions[i]
        })
    
    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You are an expert in robotic grasping. Your task is to determine the optimal grasp points for various objects.
                          
                          For each object, you will be provided:
                          1. The name/description of the object
                          2. Its position in 3D space (x, y, z)
                          3. Its dimensions (width, height, depth)
                          
                          Based on this information, determine the best point to grasp each object. Consider:
                          - The shape and function of the object
                          - The presence of handles or natural grasp points
                          - Stability during grasping
                          - Accessibility of the grasp point
                          
                          Your output should be a Python dictionary where:
                          - Keys are the object names
                          - Values are [x, y, z] coordinates representing the best grasp points
                          
                          The grasp point coordinates should be relative to the object's position.
                          For example, if an object position is [10, 20, 0] and you suggest grasping 2 units to the right,
                          the grasp point would be [12, 20, 0].
                          
                          Example output:
                          {
                              "cup": [10.5, 20.0, 2.5],
                              "plate": [15.0, 18.0, 0.5]
                          }"""
        },
        {
            "role": "user",
            "content": f"Please determine the optimal grasp points for the following objects:\n{object_data}"
        }
    ]
    
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=prompt 
    )
    
    response = completion.choices[0].message.content
    grasp_points = ProcessResponse(response)
    return grasp_points

if __name__ == "__main__":
    # Example usage
    objects = ["cup", "plate", "scissors"]
    positions = [[10, 20, 0], [15, 18, 0], [8, 22, 0]]
    dimensions = [[5, 10, 5], [20, 1, 20], [15, 2, 5]]
    
    grasp_points = GraspPointDetection(objects, positions, dimensions)
    print(grasp_points)
  
from openai import OpenAI
import ast
import re

def ProcessResponse(input_string):
    """
    Process the response from the LLM into a structured format.
    Expected input is a dictionary mapping object names to their grasp points.
    """
    try:
        # Parse the input string into Python objects
        parsed_input = ast.literal_eval(input_string)
        
        # Ensure the input is a dictionary
        if not isinstance(parsed_input, dict):
            raise ValueError("Input must be a dictionary mapping objects to grasp points")
        
        return parsed_input
    except Exception as e:
        print(f"Error processing response: {e}")
        return {}

def GraspPointDetection(objects, positions, dimensions):
    """
    Determine appropriate grasp points for a list of objects based on their positions and dimensions.
    
    Args:
        objects: List of object names
        positions: List of object positions (x, y, z)
        dimensions: List of object dimensions (width, height, depth)
        
    Returns:
        Dictionary mapping object names to their grasp points
    """
    print("Starting Grasp Point Detection:")
    
    # Create combined data for the prompt
    object_data = []
    for i in range(len(objects)):
        object_data.append({
            "name": objects[i],
            "position": positions[i],
            "dimensions": dimensions[i]
        })
    
    client = OpenAI()
    prompt = [
        {
            "role": "system", 
            "content": """You are an expert in robotic grasping. Your task is to determine the optimal grasp points for various objects.
                          
                          For each object, you will be provided:
                          1. The name/description of the object
                          2. Its position in 3D space (x, y, z)
                          3. Its dimensions (width, height, depth)
                          
                          Based on this information, determine the best point to grasp each object. Consider:
                          - The shape and function of the object
                          - The presence of handles or natural grasp points
                          - Stability during grasping
                          - Accessibility of the grasp point
                          
                          Your output should be a Python dictionary where:
                          - Keys are the object names
                          - Values are [x, y, z] coordinates representing the best grasp points
                          
                          The grasp point coordinates should be relative to the object's position.
                          For example, if an object position is [10, 20, 0] and you suggest grasping 2 units to the right,
                          the grasp point would be [12, 20, 0].
                          
                          Example output:
                          {
                              "cup": [10.5, 20.0, 2.5],
                              "plate": [15.0, 18.0, 0.5]
                          }"""
        },
        {
            "role": "user",
            "content": f"Please determine the optimal grasp points for the following objects:\n{object_data}"
        }
    ]
    
    completion = client.chat.completions.create(
        model='gpt-4o',
        messages=prompt 
    )
    
    response = completion.choices[0].message.content
    grasp_points = ProcessResponse(response)
    return grasp_points

if __name__ == "__main__":
    # Example usage
    objects = ["cup", "plate", "scissors"]
    positions = [[10, 20, 0], [15, 18, 0], [8, 22, 0]]
    dimensions = [[5, 10, 5], [20, 1, 20], [15, 2, 5]]
    
    grasp_points = GraspPointDetection(objects, positions, dimensions)
    print(grasp_points)

if __name__=="__main__":
    # image_path = "Trials/Real_table_w_tools.jpg"
    user_input = """You are in a 3D world. You are a robot arm mounted on a table. You can control the end effector's position and gripper. Object list = ['milk', 'hammer', 'stuffed_toy', 'lock']. You want to grasp the milk.

Numerical scene information:
- The position is represented by a 3D vector [x, y, z]. The axes are perpendicular to each other.
- The base of the robot arm is at [0.0, 0.0, 0.0].
- [milk]: <center>: [0.568, 0.01, 0.085]; <size>: [0.044, 0.044, 0.16]; <graspable point>: [0.568, 0.01, 0.135]. 
- [stuffed_toy]: <center>: [0.45, -0.05, 0.025]; <size>: [0.08, 0.05, 0.05]; <graspable point>: [0.45, -0.05, 0.025]. 
- [lock]: <center>: [0.35, 0.05, 0.025]; <size>: [0.04, 0.03, 0.02]; <graspable point>: [0.35, 0.05, 0.025]. 
"""
    response = Analyzer(user_input)

