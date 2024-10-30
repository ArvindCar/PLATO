import time
import numpy as np
import open3d as o3d
import robomail.vision as vis
import cv2
import warnings
warnings.filterwarnings("ignore")
import json
import subprocess
from frankapy import FrankaArm
import copy
import csv
import math
import matplotlib.pyplot as plt

fa = FrankaArm()
pose = fa.get_pose()
centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
centroid_sphere.paint_uniform_color([1, 0, 0])
centroid_sphere.translate(np.array([0.52818618, 0.03389224, -0.00697554]))
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
gripper_cam = vis.CameraClass(1, W = 1280, H = 720)
transform_gripcam = gripper_cam.get_cam_extrinsics()
cimage, dimage, pc_gripcam, verts, pinhole_camera_intrinsic = gripper_cam.get_next_frame()
# o3d.visualization.draw_geometries([pc_gripcam, coordinate_frame])
# pc_gripcam.transform(transform_gripcam)
# o3d.visualization.draw_geometries([pc_gripcam, coordinate_frame])

pose = fa.get_pose()
translation = np.array(pose.translation)
correction = np.array([-0.05, 0.010, -0.025])
Rx_0 = np.array([[1, 0, 0], 
                    [0, 1, 0], 
                    [0, 0, 1]])
Ry_180 = np.array([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1]
            ])
transform_gripcam1 = copy.deepcopy(transform_gripcam)
transform_gripcam1[:3,:3] = Ry_180
transform_gripcam1[:3, -1] = translation + correction
# pc_gripcam.transform(transform_gripcam1)
rgbd_seg = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(cimage), o3d.geometry.Image(dimage), convert_rgb_to_intensity=False)
pc_seg = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_seg, pinhole_camera_intrinsic)
pc_seg = pc_seg.transform(transform_gripcam)
pc_seg = pc_seg.transform(transform_gripcam1)


cam2 = vis.CameraClass(2)
transform_2w = cam2.get_cam_extrinsics()
cimage2, dimage2, pc2, verts2, cam_intrinsic2  = cam2.get_next_frame()
pc2.transform(transform_2w)
o3d.visualization.draw_geometries([pc_seg, coordinate_frame, pc2, centroid_sphere])
fa.close_gripper()
pose.translation = [0.52818618, 0.03389224, -0.00697554]
ws_correction = [-0.035, 0.015, 0]
pose.translation += ws_correction
fa.goto_pose(pose)