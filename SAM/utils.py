import cv2
import open3d as o3d
import numpy as np
import pyrealsense2 as rs

def corresponding_verts(mask_image_path, verts):
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    segmented_verts = []
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i][j] == 255:
                segmented_verts.append(verts[i, j])
    return segmented_verts

def segment_corresponding_pc(verts, pc_path):
    point_cloud = o3d.io.read_point_cloud(pc_path)
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    for vert in verts:
        #distances = np.linalg.norm(points - vert, axis=1)
        #closest_point_index = np.argmin(distances)
        colors[verts] = [1,1,1]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])
    file_path = 'pointcloudsegmented.ply'
    o3d.io.write_point_cloud(file_path, point_cloud)
    return file_path



if __name__ == '__main__':
    corresponding_verts()
    #pixel_vertex = verts[479, 784]#pixel_point[1], pixel_point[0]]
    #print(pixel_vertex)
    point_cloud = rs.pointcloud()

    