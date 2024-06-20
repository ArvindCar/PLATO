from overall_planner import OverallPlanner
from scene_comprehension import SceneComprehension
from step_termination import TerminationCheck
from frankapy import FrankaArm
import robomail.vision as vis
from PIL import Image



if __name__ == "__main__":
    #TODO:
    Task = "Make a star shape chocolate chip cookie dough"

    #TODO:
    ActionList = ["Grasp", "Push-down", "Move-to", "Grasp", "Release", "Roll", "Scoop", "Pour"]

    #TODO:
    save_dir = "/path/to/save/dir"

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
        img.save(save_path)


    #Query Scene comp, get list of objects

    ObjList = SceneComprehension(save_path)
    PosList = [f"Original Position of {obj}" for obj in ObjList]
    print(ObjList)
    
    # #TODO: Query Point-LLM, to get positions

    # ObjLocList = []
    # LocDict = dict(zip(PosList, ObjLocList))

    # #TODO: Pass query and list of objects to planner 
    # StepsList = OverallPlanner(Task, ObjList, PosList, ActionList)
    # num_steps = len(StepsList)

    # #TODO: Iterate through list of steps and query step planner to get got-to poses
    # i=1
    # while i<=num_steps:
    #     print(f"Executing Step {i}: {StepsList[i-1]}")

    #     #TODO: Query the steps to actions LLM

    #     #TODO: Execute the actions

    #     # Termination check
    #     cam1 = vis.CameraClass(cam_number=1)
    #     cam2 = vis.CameraClass(cam_number=2)
    #     cam3 = vis.CameraClass(cam_number=3)
    #     cam4 = vis.CameraClass(cam_number=4)
    #     cam5 = vis.CameraClass(cam_number=5)

    #     img1, _, _, _ = cam1._get_next_frame()
    #     img2, _, pc2, _ = cam2._get_next_frame()
    #     img3, _, pc3, _ = cam3._get_next_frame()
    #     img4, _, pc4, _ = cam4._get_next_frame()
    #     img5, _, pc5, _ = cam5._get_next_frame()

    #     ImgList = [img1, img2, img3, img4, img5]
    #     save_path = save_dir + f'/step{i}'

    #     for i, img_arr in enumerate(ImgList):
    #         img = Image.fromarray(img_arr)
    #         img.thumbnail((max_size, max_size))
    #         img.save(save_path)

    #     #TODO: Query pointllm to get new positions and make a new LocDict
    #     NewPos = []
    #     NewPosList = [f"New Position of {obj}" for obj in NewPosList]


    #     if TerminationCheck(img1):
    #         i+=1

    #     else:
    #         #TODO: Replan using OverallPlanner LLM
    #         StepsList = OverallPlanner(Task, ObjList, PosList, ActionList, StepsList, i)
