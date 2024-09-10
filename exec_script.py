from frankapy import FrankaArm
import robomail.vision as vis
from PIL import Image
import numpy as np
import os
import pyrealsense2 as rs
import sys
import ast
from datetime import datetime
import queue
import threading
import cv2
import re

sys.path.append('/home/arvind/LLM_Tool/LLM-Tool')
from overall_planner import OverallPlanner
from scene_comprehension import SceneComprehension
from step_termination import TerminationCheck
from step_planner import Plan2Action

sys.path.append('/home/arvind/LLM_Tool/SAM')
from object_segmentation import get_centroid

sys.path.append('/home/arvind/LLM_Tool/grasping/grasping/os_tog/notebooks')
from grasping import do_grasp


def rotate_x(matrix, angle):
    """
    Rotates a 3D matrix around the x-axis by a given angle.

    Parameters:
    matrix (numpy.ndarray): The input matrix to be rotated (3x3 or 4x4).
    angle (float): The angle in radians.

    Returns:
    numpy.ndarray: The rotated matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    return np.dot(rotation_matrix, matrix)

def rotate_y(matrix, angle):
    """
    Rotates a 3D matrix around the y-axis by a given angle.

    Parameters:
    matrix (numpy.ndarray): The input matrix to be rotated (3x3 or 4x4).
    angle (float): The angle in radians.

    Returns:
    numpy.ndarray: The rotated matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    return np.dot(rotation_matrix, matrix)

def rotate_z(matrix, angle):
    """
    Rotates a 3D matrix around the z-axis by a given angle.

    Parameters:
    matrix (numpy.ndarray): The input matrix to be rotated (3x3 or 4x4).
    angle (float): The angle in radians.

    Returns:
    numpy.ndarray: The rotated matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    return np.dot(rotation_matrix, matrix)


def run_command(act, feature, deltas, fa):
    """
    Executes the given low level command

    Parameters:
    act: The type of low level command (go-to/grasp/tilt)
    feature: parameters of the low level command
    deltas: parameters of the low level command
    fa: FrankaArm instance

    Returns:
    """
    if act == 'go-to':
        pose = fa.get_pose()
        print("Feature: ",feature)
        print("Deltas: ", deltas)
        # pose.translation[:2] = feature[:2] + deltas[:2]/100
        # fa.goto_pose(pose)
        pose.translation = feature + deltas/100
        pose.translation[2] = max(pose.translation[2], 0)
        fa.goto_pose(pose)
        
    elif act == "grasp" or act == "open":
        if feature == '0':
            target_width = min(max(0.03, fa.get_gripper_width()) + 0.005, 0.075)
            fa.goto_gripper(width=target_width)
            print("Opening gripper to width: ", target_width)
        elif feature == '1':
            fa.goto_gripper(width=0.0, grasp=True, force=15.0)

    elif act == "tilt":
        print("Feature: ", feature)
        angles = re.findall(r'(\d+)', feature)  # Finds all numeric parts
        angles = [float(angle) for angle in angles]  # Convert to float
        print("Angles: ", angles)

        # Ensure you have exactly 3 components
        if len(angles) != 3:
            raise ValueError("Feature string does not have exactly three components.")
        
        pose = fa.get_pose()
        home_rotation = pose.rotation
        
        # Tilt around the x-axis
        if angles[0] != 0:
            pose.rotation = rotate_x(home_rotation, np.radians(angles[0]))
        
        # Tilt around the y-axis
        elif angles[1] != 0:
            pose.rotation = rotate_y(home_rotation, np.radians(angles[1]))
        
        # Tilt around the z-axis
        elif angles[2] != 0:
            pose.rotation = rotate_z(home_rotation, np.radians(angles[2]))
        fa.goto_pose(pose)
    return

def exec_experiment(fa, cam2, cam3, cam4, cam5, save_dir, Task, ActionList, home_pose, done_queue):    
    # cam6 = vis.CameraClass(cam_number=6) # This is the overview camera
    # W = 1280
    # H = 800
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_device('152522250441')
    # config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    # pipeline.start(config)

    # img1, _, _, _, _ = cam1.get_next_frame()
    img2, dimg2, pc2, _, _ = cam2.get_next_frame()
    img3, dimg3, pc3, _, _ = cam3.get_next_frame()
    img4, dimg4, pc4, _, _ = cam4.get_next_frame()
    img5, dimg5, pc5, _, _ = cam5.get_next_frame()

    # img6, _, _, _, _ = cam6.get_next_frame()
    
    save_path = save_dir + '/step0'
    max_size = 512

    ImgList = [img2, img3, img4, img5]
    DepthList = [dimg2, dimg3, dimg4, dimg5]
    os.makedirs(save_path, exist_ok=True)

    for j, (img_arr, dimg_arr) in enumerate(zip(ImgList,DepthList)):
        img = Image.fromarray(img_arr)
        img.thumbnail((max_size, max_size))
        img.save(save_path + f"/Image{j+2}.png")

        dimg = Image.fromarray(dimg_arr)
        if dimg.mode not in ['L', 'RGB']:  # Check if mode is unsupported
            dimg = dimg.convert('L') 
        dimg.thumbnail((max_size, max_size))
        dimg.save(save_path + f"/Depth{j+2}.png")



    # Query Scene comp, get list of objects

    ObjList, HandleFlags = SceneComprehension(save_path, Task)
    with open(save_dir + '/ObjList', 'a') as file:
            file.write(", ".join(ObjList) + "\n")
            file.write(str(HandleFlags) + "\n")

    HandleDict = dict(zip(ObjList, HandleFlags))
    PosList = [f"original position of {obj}" for obj in ObjList]
    PosList.append("home pose")
    # PosList.extend([f"{obj}" for obj in ObjList])
    print("Object List: ", ObjList)
    print("Handle Flags: ", HandleFlags)
    
    #TODO: Query Point-LLM, to get positions
    DescList = {obj: [] for obj in ObjList}
    ObjLocList = []
    for obj in ObjList:
        centroid, description = get_centroid(cam2, cam3, cam4, cam5, obj, save_pc = True, save_path=save_dir, viz = False)
        ObjLocList.append(centroid)
        DescList[obj] = description*100
    DescList['none'] = []
    ObjLocList.append(home_pose.translation)
    LocDict = dict(zip(PosList, ObjLocList))

    prev_steps = {}
    #Pass query and list of objects to planner 
    StepsList = OverallPlanner(Task, ObjList, PosList, ActionList)
    num_steps = len(StepsList)
    for i in range(num_steps):
        with open(save_dir + '/StepsList', 'a') as file:
            file.write(f"Step {i}: ")
            file.write(", ".join(StepsList[i-1]) + "\n")

    #Iterate through list of steps and query step planner to get got-to poses
    i=1
    while i<=num_steps:
        print(f"Executing Step {i}: {StepsList[i-1]}")
        with open(save_path + '/CommandList', 'a') as file:
            file.write("Step: ")
            file.write(", ".join(StepsList[i-1]) + "\n")
            file.write("Actions:" + "\n")

        Action, Location, Positioning, Object, Tool = StepsList[i-1]

        if Tool == 'none': # Currently nothing is grasped
            fa.open_gripper()
            print("Nothing is grasped")
            CommandList = Plan2Action(Action, Location, Positioning, [[0], []], Object, Tool, prev_steps)
        elif HandleDict[Tool] == 1: # Currently grasped object is a tool
            print("Currently grasped object is a tool")
            CommandList = Plan2Action(Action, Location, Positioning [DescList[Object], [max(DescList[Tool])]], Object, Tool, prev_steps)
        else: # Currently grasped object is not a tool
            print('Currently grasped object is not a tool')
            CommandList = Plan2Action(Action, Location, Positioning, [DescList[Object], [0]], Object, Tool, prev_steps) # This means that the "Tool" has no handle

        with open(save_path + '/CommandList', 'a') as file:
            for step in CommandList:
                file.write(", ".join(step) + "\n")
        
        if Action == "pickup" and HandleDict[Object] == 1:
            with open(save_path + '/CommandList', 'a') as file:
                file.write("Affordance Model was queried" + "\n")
            # Query SAM to get centroid position of Object
            global_pos, _ = get_centroid(cam2, cam3, cam4, cam5, Object, save_pc = False, save_path = save_dir, viz = False)
            print("centroid: ",global_pos) # [ 0.5317093  -0.10578898 -0.0108681 ]
            pose = fa.get_pose()
            pose.translation = global_pos + np.array([0.05, 0, 0]) # Observation Offset
            pose.translation[2] = 0.5
            fa.goto_pose(pose)
            do_grasp(save_path, query_tool = Object, query_task='pickup', fa=fa)
        elif Location == "home pose":
            fa.reset_joints()
        else:
            #Query the steps to actions LLM
            for command in CommandList:
                print(command)
                act = command[0]
                feature = command[1]
                deltas = None
                if act == 'go-to':
                    print("Go-to Command Identified. Accessing LocDict")
                    tuple_string = command[2]
                    tuple_elements = tuple_string.strip('()').split(',')
                    tuple_numbers = [float(element.strip().split()[0]) for element in tuple_elements]
                    deltas = np.array(tuple_numbers)
                    feature = LocDict[feature]
                run_command(act, feature, deltas, fa)


        # Termination check
        img2, dimg2, pc2, _, _ = cam2.get_next_frame()
        img3, dimg3, pc3, _, _ = cam3.get_next_frame()
        img4, dimg4, pc4, _, _ = cam4.get_next_frame()
        img5, dimg5, pc5, _, _ = cam5.get_next_frame()

        ImgList = [img2, img3, img4, img5]
        DepthList = [dimg2, dimg3, dimg4, dimg5]
        save_path = save_dir + f'/step{i}'
        os.makedirs(save_path, exist_ok=True)

        for j, (img_arr, dimg_arr) in enumerate(zip(ImgList,DepthList)):
            img = Image.fromarray(img_arr)
            img.thumbnail((max_size, max_size))
            img.save(save_path + f"/Image{j+2}.png")

            dimg = Image.fromarray(dimg_arr)
            if dimg.mode not in ['L', 'RGB']:  # Check if mode is unsupported
                dimg = dimg.convert('L') 
                dimg.thumbnail((max_size, max_size))
            dimg.save(save_path + f"/Depth{j+2}.png")

        #TODO: Query pointllm to get new positions and make a new LocDict
        # NewPos = []
        # NewPosList = [f"New Position of {obj}" for obj in NewPosList]


        # if TerminationCheck(save_path, Action):
        inputs = (Action, Location, Object, Tool)
        prev_steps.clear()
        prev_steps[inputs] = CommandList
        print(prev_steps)
        i+=1

        # else:
        #     #TODO: Replan using OverallPlanner LLM
        #     StepsList = OverallPlanner(Task, ObjList, PosList, ActionList, StepsList, i)
    done_queue.put("Done")

def video_loop(cam_pipeline, save_path, done_queue):
    print("Starting video Recording")
    # Use 'mp4v' codec for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    video_resolution = (600, 900)

    # Save video as MP4
    out = cv2.VideoWriter(save_path + '/video.mp4', fourcc, 30.0, video_resolution)
    # frame_save_counter = 0
    
    # Record until main loop is complete
    while done_queue.empty():
        frames = cam_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # Crop and rotate the image to just show elevated stage area
        cropped_image = color_image[100:700, 220:1120, :]
        rotated_image = cv2.rotate(cropped_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        out.write(rotated_image)
    
    cam_pipeline.stop()
    out.release()


if __name__ == "__main__":
    #TODO:
    Task = "Scoop up the candy"

    #TODO: Change this to be generated by an LLM
    ActionList = ["Pick-up", "Push-down", "Move-to", "Release", "Roll", "Scoop", "Pour"]

    #TODO:
    base_dir = "/home/arvind/LLM_Tool/Save_dir/"
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    new_dir_name = f"{Task}_{timestamp}"
    save_dir = os.path.join(base_dir, new_dir_name)

    os.makedirs(save_dir, exist_ok=True)

    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    home_pose = fa.get_pose()
    # cam1 = vis.CameraClass(cam_number=1)
    cam2 = vis.CameraClass(cam_number=2)
    cam3 = vis.CameraClass(cam_number=3)
    cam4 = vis.CameraClass(cam_number=4)
    cam5 = vis.CameraClass(cam_number=5)

    # initialize camera 6 pipeline
    W = 1280
    H = 800
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('152522250441')
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipeline.start(config)

    done_queue = queue.Queue()

    main_thread = threading.Thread(target=exec_experiment, args=(fa, cam2, cam3, cam4, cam5, save_dir, Task, ActionList, home_pose, done_queue))
    video_thread = threading.Thread(target=video_loop, args=(pipeline, save_dir, done_queue))

    main_thread.start()
    video_thread.start()
