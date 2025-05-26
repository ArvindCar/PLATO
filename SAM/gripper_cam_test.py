import time
import numpy as np
import open3d as o3d
import robomail.vision as vis
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sam3d import get_bbox, SAM
import os, sys, requests, argparse
import robomail.vision as vis
import cv2
from frankapy import FrankaArm

if __name__ == "__main__":
    fa = FrankaArm()
    pose = fa.get_pose()
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    translation = np.array(pose.translation)
    correction = np.array([0.010, -0.030, -0.005])
    Rx_180 = np.array([[1, 0, 0], 
                       [0, -1, 0], 
                       [0, 0, -1]])
    cam1 = vis.CameraClass(1, W = 1280, H=720)
    transform_1w = cam1.get_cam_extrinsics()
    cimage1, _, pc1, _, _  = cam1.get_next_frame()
    o3d.visualization.draw_geometries([pc1, coordinate_frame])

    pc1 = pc1.transform(transform_1w)
    print(type(transform_1w))
    transform_1w[:3,:3] = Rx_180
    transform_1w[:3, -1] = translation + correction

    print(transform_1w)
    pc1 = pc1.transform(transform_1w)

    # cimage_path = '/home/arvind/LLM_Tool/SAM/cimage_gripper.png'
    # pil_cimage = Image.fromarray(cimage1)
    # pil_cimage.save(cimage_path)
    
    # R_combined = Rz_90 @ Rx_180

    
    o3d.visualization.draw_geometries([pc1, coordinate_frame])


    cam2 = vis.CameraClass(2)
    cam3 = vis.CameraClass(3)
    cam4 = vis.CameraClass(4)
    cam5 = vis.CameraClass(5)

    # get the camera to world transforms
    transform_2w = cam2.get_cam_extrinsics()# .matrix()
    transform_3w = cam3.get_cam_extrinsics()# .matrix()
    transform_4w = cam4.get_cam_extrinsics()# .matrix()
    transform_5w = cam5.get_cam_extrinsics()# .matrix()

    # initialize the 3D vision code
    pcl_vis = vis.Vision3D()

    # this requires a chnage to camera.py: get_next_frame() should also return the cam_instrinsic
    cimage2, dimage2, pc2, verts2, cam_intrinsic2  = cam2.get_next_frame()
    cimage3, dimage3, pc3, verts3, cam_intrinsic3 = cam3.get_next_frame()
    cimage4, dimage4, pc4, verts4, cam_intrinsic4 = cam4.get_next_frame()
    cimage5, dimage5, pc5, verts5, cam_intrinsic5 = cam5.get_next_frame()
    verts = verts2 + verts3 + verts4 + verts5
    #dimg = [dimage2, dimage3, dimage4, dimage5]
    cam_instrinsic = [cam_intrinsic2, cam_intrinsic3, cam_intrinsic4, cam_intrinsic5]
    depth_image = [dimage2, dimage3, dimage4, dimage5]


# transform cameras
    pc2.transform(transform_2w)
    pc3.transform(transform_3w)
    pc4.transform(transform_4w)
    pc5.transform(transform_5w)

# combine into single pointcloud
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = pc3.points
    pointcloud.colors = pc3.colors
    pointcloud.points.extend(pc2.points)
    pointcloud.colors.extend(pc2.colors)
    pointcloud.points.extend(pc4.points)
    pointcloud.colors.extend(pc4.colors)
    pointcloud.points.extend(pc5.points)
    pointcloud.colors.extend(pc5.colors)
    pointcloud.points.extend(pc1.points)
    pointcloud.colors.extend(pc1.colors)
    points = np.asarray(pointcloud.points)
    colors = np.asarray(pointcloud.colors)
    xmin = 0
    xmax = 2.5
    ymin = -0.6
    ymax = 0.6
    zmin = -0.1
    zmax = 0.5
    mask = (points[:, 0] > xmin) & (points[:, 0] < xmax) & (points[:, 1] > ymin) & (points[:, 1] < ymax) & (points[:, 2] > zmin) & (points[:, 2] < zmax)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pointcloud, coordinate_frame])

