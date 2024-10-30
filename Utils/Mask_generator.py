from frankapy import FrankaArm
import robomail.vision as vis
import open3d as o3d
import numpy as np
import copy
import cv2

if __name__ == '__main__':
    fa = FrankaArm()
    # pose = fa.get_pose()
    # pose.translation[0] += 0.2
    # fa.goto_pose(pose)
    gripper_cam = vis.CameraClass(1, W = 1280, H = 720)
    transform_gripcam = gripper_cam.get_cam_extrinsics()
    cimage, dimage, pc_gripcam, verts, _ = gripper_cam.get_next_frame()
    cimage = cv2.cvtColor(cimage, cv2.COLOR_BGR2RGB)
    cv2.imwrite('/home/arvind/LLM_Tool/spoon.png', cimage)

    # pc_gripcam.transform(transform_gripcam)
    # pose = fa.get_pose()
    # translation = np.array(pose.translation)
    # correction = np.array([-0.010, -0.030, -0.005])
    # Rx_0 = np.array([[1, 0, 0], 
    #                  [0, 1, 0], 
    #                  [0, 0, 1]])
    # transform_gripcam1 = copy.deepcopy(transform_gripcam)
    # transform_gripcam1[:3,:3] = Rx_0
    # transform_gripcam1[:3, -1] = translation + correction
    # pc_gripcam.transform(transform_gripcam1)

    # cam2 = vis.CameraClass(2)
    # cam3 = vis.CameraClass(3)
    # cam4 = vis.CameraClass(4)
    # cam5 = vis.CameraClass(5)
    # transform_2w = cam2.get_cam_extrinsics()# .matrix()
    # transform_3w = cam3.get_cam_extrinsics()# .matrix()
    # transform_4w = cam4.get_cam_extrinsics()# .matrix()
    # transform_5w = cam5.get_cam_extrinsics()# .matrix()
    # cimage2, dimage2, pc2, verts2, cam_intrinsic2  = cam2.get_next_frame()
    # cimage3, dimage3, pc3, verts3, cam_intrinsic3 = cam3.get_next_frame()
    # cimage4, dimage4, pc4, verts4, cam_intrinsic4 = cam4.get_next_frame()
    # cimage5, dimage5, pc5, verts5, cam_intrinsic5 = cam5.get_next_frame()
    # pc2.transform(transform_2w)
    # pc3.transform(transform_3w)
    # pc4.transform(transform_4w)
    # pc5.transform(transform_5w)
    # pointcloud = o3d.geometry.PointCloud()
    # pointcloud.points = pc3.points
    # pointcloud.colors = pc3.colors
    # pointcloud.points.extend(pc2.points)
    # pointcloud.colors.extend(pc2.colors)
    # pointcloud.points.extend(pc4.points)
    # pointcloud.colors.extend(pc4.colors)
    # pointcloud.points.extend(pc5.points)
    # pointcloud.colors.extend(pc5.colors)

    # points = np.asarray(pointcloud.points)
    # colors = np.asarray(pointcloud.colors)
    # xmin, xmax = 0, 1
    # ymin, ymax = -0.5, 0.5
    # zmin, zmax = 0, 0.4
    # mask = (points[:, 0] > xmin) & (points[:, 0] < xmax) & (points[:, 1] > ymin) & (points[:, 1] < ymax) & (points[:, 2] > zmin) & (points[:, 2] < zmax)
    # points = points[mask]
    # colors = colors[mask]
    # pointcloud.points = o3d.utility.Vector3dVector(points)
    # pointcloud.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pointcloud])