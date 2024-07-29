
from frankapy import FrankaArm
import robomail.vision as vis
from PIL import Image
import numpy as np
import os
import pyrealsense2 as rs
import sys
import ast

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
    if act == 'go-to':
        pose = fa.get_pose()
        print("Feature: ",feature)
        print("Deltas: ", deltas)
        pose.translation[:2] = feature[:2] + deltas[:2]/100
        fa.goto_pose(pose)
        pose.translation = feature + deltas/100
        fa.goto_pose(pose)
        
    elif act == "grasp":
        if feature == '0':
            fa.open_gripper()
        elif feature == '1':
            fa.goto_gripper(width=0.0, grasp=True, force=15.0)

    elif act == "tilt":
        if feature[0] != '0':
            pose = fa.get_pose()
            home_rotation = pose.rotation
            angle = ast.literal_eval(feature[0])
            pose.rotation = rotate_x(home_rotation, np.radians(angle)) 
        elif feature[1] != '0':
            pose = fa.get_pose()
            angle = ast.literal_eval(feature[1])
            pose.rotation = rotate_y(home_rotation, np.radians(angle))
        elif feature[2] != '0':
            pose = fa.get_pose()
            angle = ast.literal_eval(feature[2])
            pose.rotation = rotate_z(home_rotation, np.radians(angle)) 
    return


if __name__ == "__main__":
    #TODO:
    Task = "Pick up apple and put it in bowl"

    #TODO: Change this to be generated by an LLM
    ActionList = ["Pick-up", "Push-down", "Move-to", "Release", "Roll", "Scoop", "Pour"]

    #TODO:
    save_dir = "/home/arvind/LLM_Tool/Save_dir"

    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()
    home_pose = fa.get_pose()
    # cam1 = vis.CameraClass(cam_number=1)
    cam2 = vis.CameraClass(cam_number=2)
    cam3 = vis.CameraClass(cam_number=3)
    cam4 = vis.CameraClass(cam_number=4)
    cam5 = vis.CameraClass(cam_number=5)

    # cam6 = vis.CameraClass(cam_number=6) # This is the overview camera

    # W = 1280
    # H = 800
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_device('152522250441')
    # config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    # pipeline.start(config)

    # img1, _, _, _, _ = cam1.get_next_frame()
    img2, _, pc2, _, _ = cam2.get_next_frame()
    img3, _, pc3, _, _ = cam3.get_next_frame()
    img4, _, pc4, _, _ = cam4.get_next_frame()
    img5, _, pc5, _, _ = cam5.get_next_frame()

    # img6, _, _, _, _ = cam6.get_next_frame()

    ImgList = [img2, img3, img4, img5]
    
    save_path = save_dir + '/step0'
    os.makedirs(save_path, exist_ok=True)
    max_size = 512


    for i, img_arr in enumerate(ImgList):
        img = Image.fromarray(img_arr)
        img.thumbnail((max_size, max_size))
        img.save(save_path + f"/Image{i+2}.png")


    # Query Scene comp, get list of objects

    ObjList = SceneComprehension(save_path, Task)
    PosList = [f"original position of {obj}" for obj in ObjList]
    PosList.append("home pose")
    # PosList.extend([f"{obj}" for obj in ObjList])
    print("Object List: ", ObjList)
    
    #TODO: Query Point-LLM, to get positions

    ObjLocList = []
    for obj in ObjList:
        ObjLocList.append(get_centroid(cam2, cam3, cam4, cam5, obj))
    ObjLocList.append(home_pose.translation)
    LocDict = dict(zip(PosList, ObjLocList))

    prev_steps = {}
    #Pass query and list of objects to planner 
    StepsList = OverallPlanner(Task, ObjList, PosList, ActionList)
    num_steps = len(StepsList)
    

    #Iterate through list of steps and query step planner to get got-to poses
    i=1
    while i<=num_steps:
        print(f"Executing Step {i}: {StepsList[i-1]}")
        Action = StepsList[i-1][0]
        Location = StepsList[i-1][1]
        Object = StepsList[i-1][2]
        Tool = StepsList[i-1][3]
        print(Action)
        CommandList = Plan2Action(Action, Location, Object, Tool, prev_steps)
        #TODO: Write a condition to check if the next action is a grasp action
        if Action == "pickup":
            # Query SAM to get centroid position of Object
            global_pos = get_centroid(cam2, cam3, cam4, cam5, Object)
            pose = fa.get_pose()
            pose.translation = global_pos + np.array([0.05, 0, 0]) # Observation Offset
            pose.translation[2] = 0.5
            fa.goto_pose(pose)
            do_grasp(save_path, query_tool = Object, query_task='pickup')
            pose = fa.get_pose()
            pose.translation[2] += 0.2
            fa.goto_pose(pose)
        if Location == "home pose":
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
                    tuple_numbers = [int(element.strip().split()[0]) for element in tuple_elements]
                    deltas = np.array(tuple_numbers)
                    feature = LocDict[feature]
                run_command(act, feature, deltas, fa)


        # Termination check
        # img1, _, _, _, _ = cam1.get_next_frame()
        img2, _, pc2, _, _ = cam2.get_next_frame()
        img3, _, pc3, _, _ = cam3.get_next_frame()
        img4, _, pc4, _, _ = cam4.get_next_frame()
        img5, _, pc5, _, _ = cam5.get_next_frame()

        ImgList = [img2, img3, img4, img5]
        save_path = save_dir + f'/step{i}'
        os.makedirs(save_path, exist_ok=True)

        for j, img_arr in enumerate(ImgList):
            img = Image.fromarray(img_arr)
            img.thumbnail((max_size, max_size))
            img.save(save_path + f"/Image{j+2}.png")

        #TODO: Query pointllm to get new positions and make a new LocDict
        # NewPos = []
        # NewPosList = [f"New Position of {obj}" for obj in NewPosList]


        # if TerminationCheck(save_path, Action):
        inputs = (Action, Location, Object, Tool)

        prev_steps[inputs] = CommandList
        i+=1

        # else:
        #     #TODO: Replan using OverallPlanner LLM
        #     StepsList = OverallPlanner(Task, ObjList, PosList, ActionList, StepsList, i)

