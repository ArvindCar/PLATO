from frankapy import FrankaArm
import numpy as np
if __name__=="__main__":
    fa = FrankaArm()
    # Bottle
    # centroid = np.array([0.56509244, -0.11072098,  0.1345986])
    # translation = np.array([0.569686, -0.10920805, 0.02136894])
    # rotation = np.array([[-0.40493783,0.8029171,-0.4374351 ], [-0.08902445,0.44151926,0.8928244 ], [0.91, .4004808,-0.1073086 ]])

    # Tape Measure
    centroid = np.array([0.53454995, 0.13211356, 0.04396592])
    translation = np.array([0.5214546,  0.12441766, 0.01789842])
    rotation = np.array([[ 7.5592965e-02, -2.4835473e-01,  9.6571505e-01], [ 2.8120205e-01, -9.2386740e-01, -2.5960428e-01], [ 9.5666665e-01,  2.9118532e-01, -1.2728115e-08]])

    translation[2] = 2 * centroid[2] - translation[2]   
    # reflection_matrix = np.array([
    #     [-1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, 1]
    # ])
    transform_matrix = np.array([
        [-1, 0, 0],
        [0, 0, -1],
        [0, -1, 0]
    ])
    # rotation_matrix_y = np.array([
    #     [0, 0, -1],
    #     [0, 1, 0],
    #     [1, 0, 0]
    # ])
    rotation = rotation
    pose = fa.get_pose()
    print("Tran: ", translation)
    print("Rotation: ", rotation)
    pose.translation = translation
    pose.rotation = rotation
    fa.goto_pose(pose)
