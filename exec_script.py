from overall_planner import OverallPlanner
from scene_comprehension import SceneComprehension
from step_termination import TerminationCheck
from step_planner import Plan2Action
from frankapy import FrankaArm
import robomail.vision as vis
from PIL import Image
import numpy as np


def run_command(act, feature, deltas, fa):
    if act == 'Go-to':
        pose = fa.get_pose()
        pose.translation = feature + deltas
        fa.goto_pose(pose)
        
    if act == 'Grasp':
        if feature == 0:
            fa.open_gripper()
        elif feature == 1:
            fa.goto_gripper(width=0.0, grasp=True)
        
    return


if __name__ == "__main__":
    #TODO:
    Task = "Place the tape measure on top of the board"

    #TODO: Change this to be generated by an LLM
    ActionList = ["Grasp", "Push-down", "Move-to", "Grasp", "Release", "Roll", "Scoop", "Pour"]

    #TODO:
    save_dir = "/home/arvind/LLM_Tool/LLM-Tool/Save_dir"

    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()

    cam1 = vis.CameraClass(cam_number=1)
    cam2 = vis.CameraClass(cam_number=2)
    cam3 = vis.CameraClass(cam_number=3)
    cam4 = vis.CameraClass(cam_number=4)
    cam5 = vis.CameraClass(cam_number=5)

    img1, _, _, _ = cam1._get_next_frame()
    img2, _, pc2, _ = cam2._get_next_frame()
    img3, _, pc3, _ = cam3._get_next_frame()
    img4, _, pc4, _ = cam4._get_next_frame()
    img5, _, pc5, _ = cam5._get_next_frame()

    ImgList = [img1, img2, img3, img4, img5]
    
    save_path = save_dir + '/step0'
    max_size = 512


    for i, img_arr in enumerate(ImgList):
        img = Image.fromarray(img_arr)
        img.thumbnail((max_size, max_size))
        img.save(save_path + f"/Image{i+1}.png")


    # Query Scene comp, get list of objects

    ObjList = SceneComprehension(save_path)
    PosList = [f"Original Position of {obj}" for obj in ObjList]
    print(ObjList)
    
    #TODO: Query Point-LLM, to get positions

    ObjLocList = [np.array([0.5, 0.1, 0.1]), np.array([0.7, 0.3, 0.1]), np.array([0.2, 0.4, 0.1])]
    LocDict = dict(zip(PosList, ObjLocList))

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

        glob_pos = LocDict[Location]

        #Query the steps to actions LLM
        
        CommandList = Plan2Action(Action, Location, Object, Tool)
        

        #TODO: Execute the actions   
        for command in CommandList:
            act = command[0]
            feature = command[1]
            deltas = None
            if act == 'Got-to':
                tuple_string = command[2]
                tuple_elements = tuple_string.strip('()').split(',')
                tuple_numbers = [int(element.strip().split()[0]) for element in tuple_elements]
                deltas = np.array(tuple_numbers)
                feature = LocDict[feature]
            run_command(act, feature, deltas, fa)


        # Termination check
        img1, _, _, _ = cam1._get_next_frame()
        img2, _, pc2, _ = cam2._get_next_frame()
        img3, _, pc3, _ = cam3._get_next_frame()
        img4, _, pc4, _ = cam4._get_next_frame()
        img5, _, pc5, _ = cam5._get_next_frame()

        ImgList = [img1, img2, img3, img4, img5]
        save_path = save_dir + f'/step{i}'

        for j, img_arr in enumerate(ImgList):
            if j!=4:
                img = Image.fromarray(img_arr)
                img.thumbnail((max_size, max_size))
                img.save(save_path + f"/Image{j+1}.png")

        #TODO: Query pointllm to get new positions and make a new LocDict
        # NewPos = []
        # NewPosList = [f"New Position of {obj}" for obj in NewPosList]


        if TerminationCheck(img1):
            i+=1

        else:
            #TODO: Replan using OverallPlanner LLM
            StepsList = OverallPlanner(Task, ObjList, PosList, ActionList, StepsList, i)

