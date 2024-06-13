from PIL import Image
import os

# Function to resize and save image
def resize_and_save_image(input_path, output_path, max_size=512):
    with Image.open(input_path) as img:
        img.thumbnail((max_size, max_size))
        img.save(output_path)

# Example usage
if __name__ == "__main__":
    # Specify input and output paths
    input_image_path = "/home/aesee/CMU/MAIL_Lab/LLM_Tool/Arm_no_tool_1.png"  # Replace with your input image path
    output_image_path = "/home/aesee/CMU/MAIL_Lab/LLM_Tool/Arm_no_tool_1_resized.png"  # Replace with your output image path

    # Resize and save the image
    resize_and_save_image(input_image_path, output_image_path)
