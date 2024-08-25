from frankapy import FrankaArm
import numpy as np
import robomail.vision as vis
import copy
import cv2

fa = FrankaArm()
fa.reset_joints()
fa.open_gripper
pose = fa.get_pose()
# global_pos = np.array([ 0.52269747, -0.00996165, -0.00627307])
# pose.translation = global_pos + np.array([0.05, 0, 0]) # Observation Offset
# pose.translation[2] = 0.5
# fa.goto_pose(pose)
# global transform_gripcam, transform_gripcam1, verts, pc_gripcam
# gripper_cam = vis.CameraClass(1, W = 1280, H = 720)
# transform_gripcam = gripper_cam.get_cam_extrinsics()
# cimage, dimage, pc_gripcam, verts, _ = gripper_cam.get_next_frame()
# cv2.imwrite('/home/arvind/Downloads/test.png', cimage)
# pc_gripcam.transform(transform_gripcam)
# pose = fa.get_pose()
# translation = np.array(pose.translation)
# correction = np.array([-0.010, -0.030, -0.005])
# Rx_0 = np.array([[1, 0, 0], 
#                     [0, 1, 0], 
#                     [0, 0, 1]])
# transform_gripcam1 = copy.deepcopy(transform_gripcam)
# transform_gripcam1[:3,:3] = Rx_0
# transform_gripcam1[:3, -1] = translation + correction

x1, y1, w, h, t = ['359.45297', '393.9288', '54.047974', '32.03473', '0.1308997']

# x1 = float(x1)
# x1 = int(round(x1)+320)

# y1 = float(y1)
# y1 = int(round(720-(y1+120)))
t = float(t)

# center_of_gripper = np.array(verts[int(y1)][int(x1)])
# homog_center = np.append(center_of_gripper, 1)
# rotated_center_of_gripper = transform_gripcam @ homog_center
# rotated_center_of_gripper = transform_gripcam1 @ rotated_center_of_gripper
# rotated_center_of_gripper[0] -= 0.05
# rotated_center_of_gripper[1] += 0.05
# rotated_center_of_gripper[2] = 0.01

# print(rotated_center_of_gripper[:3])

# pose = fa.get_pose()
R = np.array([
    [np.cos(t), -np.sin(t), 0],
    [np.sin(t),  np.cos(t), 0],
    [0, 0, 1]
])
pose.translation = np.array([0.39991895, -0.05284854,  0.07])
pose.rotation = pose.rotation @ R
fa.goto_pose(pose)
fa.goto_gripper(width=0.0, grasp=True, force=5)