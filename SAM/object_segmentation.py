import time
import numpy as np
import open3d as o3d
import robomail.vision as vis
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sam3d import get_bbox, SAM
import os, sys, requests, argparse
from sklearn.neighbors import NearestNeighbors
from frankapy import FrankaArm
from matplotlib import pyplot as plt

def close_window_after_delay(window, delay):
    time.sleep(delay)
    window.close()

def remove_noise(points, colors, neighbors=40, std_ratio=0.2):
    nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    mean_dist = np.mean(distances, axis=1)
    std_dist = np.std(mean_dist)
    
    threshold_dist = np.mean(mean_dist) + std_ratio * std_dist
    
    clean_points = points[mean_dist < threshold_dist]
    clean_colors = colors[mean_dist < threshold_dist]
    return clean_points, clean_colors

def clear_directory(directory_path):
    """
    Deletes all contents of the specified directory without deleting the directory itself.

    :param directory_path: Path to the directory to be cleared
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
    return

def get_pointcloud_segmented_region(mask_image, depth_image, pinhole_camera_intrinsic):
    mask_image = Image.open(mask_image)
    mask_image = np.array(mask_image)
    mask_image = np.expand_dims(mask_image, axis=-1)
    mask_image = np.repeat(mask_image, 3, axis=-1)
    mask_image = mask_image.astype(np.uint8)
    rgbd_seg = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(mask_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)
    pc_seg = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_seg, pinhole_camera_intrinsic)
    return pc_seg

def update_or_append_obj(obj_list, new_dict, dist_threshold, logits_list, cam_idx, mask_idx):
    for i, old_obj in enumerate(obj_list):
        if np.linalg.norm(old_obj["mean"][:2] - new_dict["mean"][:2]) < dist_threshold:
            print(f'Object {cam_idx, mask_idx} within threshold for old object {old_obj["pcs"]}')
            old_obj["pcs"].append([cam_idx, mask_idx])
            old_obj["score"] += logits_list[cam_idx][0][mask_idx]
            return 
    obj_list.append(new_dict)
    return


def get_centroid(cam2, cam3, cam4, cam5, arg_textprompt, save_pc = False, save_path=None, viz = False):
    # initialize the cameras
    clear_directory('/home/arvind/LLM_Tool/SAM/try_images')
    import robomail.vision as vis # if not this line thinks of vis as a local variable...
    if cam2 == []:
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
    # o3d.visualization.draw_geometries([pointcloud])

    

    image_pairs = [
        (cimage2, dimage2),
        (cimage3, dimage3),
        (cimage4, dimage4),
        (cimage5, dimage5)
    ]
    cimage_paths = []
    for i, (cimage, dimage) in enumerate(image_pairs, start=2):
        cimage_path = f'/home/arvind/LLM_Tool/SAM/try_images/cimage{i}.png'
        dimg_path = f'/home/arvind/LLM_Tool/SAM/try_images/dimage{i}.png'
        pil_cimage = Image.fromarray(cimage)
        pil_dimage = Image.fromarray(dimage)
        pil_cimage.save(cimage_path)
        pil_dimage.save(dimg_path)
        cimage_paths.append(cimage_path)

    print(f'Capture scene done! Trying to mask {arg_textprompt}')

    #perform sam
    get_associated_bbox = get_bbox()
    segment_class = SAM()
    logits_list = [[] for _ in range(4)]
    for i, cimg_path in enumerate(cimage_paths):
        bbox_list, logits = get_associated_bbox.__run__(cimg_path, arg_textprompt)
        logits_list[i].append(logits)
        # print(f"Logits are: cam{i}", logits)
        # print(logits_list[0][1])
        img_path = f'/home/arvind/LLM_Tool/SAM/try_images/mask{i}'
        segment_class.get_mask(cimg_path, bbox_list, img_path)
    print('Masking done')
    
    pointcloud_seg = o3d.geometry.PointCloud()
    pc_segs = [[] for _ in range(4)]
    for i in range(len(cam_instrinsic)):
        idx = 0
        while True:
            mask_path = f'/home/arvind/LLM_Tool/SAM/try_images/mask{i}_{idx}.png'
            if not os.path.exists(mask_path):
                break
            pc_seg = get_pointcloud_segmented_region(mask_path, depth_image[i], cam_instrinsic[i])
            pc_segs[i].append(pc_seg)
            idx += 1
    # print(pc_segs)

    means = [[] for _ in range(4)]
    obj_points = [[] for _ in range(4)]
    obj_colors = [[] for _ in range(4)]
    obj_list = []
    dist_threshold = 0.05
    white_threshold = 0.9
    cropped_pcs = [[] for _ in range(4)]
    new_pc_segs = []
    # print(logits_list)
    print("=====")
    for i, _ in enumerate(pc_segs[0]):
        pointcloud_seg = pc_segs[0][i].transform(transform_2w)
        pointcloud = pc2
        points = np.asarray(pointcloud_seg.points)
        colors = np.asarray(pointcloud_seg.colors)
        og_colors = np.asarray(pointcloud.colors)
        white_mask = (colors[:, 0] > white_threshold) & (colors[:, 1] > white_threshold) & (colors[:, 2] > white_threshold)
        obj_points[0].append(points[white_mask])
        obj_colors[0].append(og_colors[white_mask])
        new_pointcloud_seg = o3d.geometry.PointCloud()
        new_pointcloud_seg.points = o3d.utility.Vector3dVector(obj_points[0][-1])
        new_pointcloud_seg.colors = o3d.utility.Vector3dVector(obj_colors[0][-1])
        cropped_pcs[0].append(new_pointcloud_seg)
        new_dict = {"mean":np.mean(obj_points[0][i], axis=0), "pcs":[[0, i]], "score":logits_list[0][0][i]}
        obj_list.append(new_dict)
        # print("obj list: ", obj_list)
    

    for i, _ in enumerate(pc_segs[1]):
        pointcloud_seg =pc_segs[1][i].transform(transform_3w)
        pointcloud = pc3
        points = np.asarray(pointcloud_seg.points)
        colors = np.asarray(pointcloud_seg.colors)
        og_colors = np.asarray(pointcloud.colors)
        white_mask = (colors[:, 0] > white_threshold) & (colors[:, 1] > white_threshold) & (colors[:, 2] > white_threshold)
        obj_points[1].append(points[white_mask])
        obj_colors[1].append(og_colors[white_mask])
        new_pointcloud_seg = o3d.geometry.PointCloud()
        new_pointcloud_seg.points = o3d.utility.Vector3dVector(obj_points[1][-1])
        new_pointcloud_seg.colors = o3d.utility.Vector3dVector(obj_colors[1][-1])
        cropped_pcs[1].append(new_pointcloud_seg)
        new_dict = {"mean":np.mean(obj_points[1][i], axis=0), "pcs":[[1, i]], "score":logits_list[1][0][i]}
        update_or_append_obj(obj_list, new_dict, dist_threshold, logits_list, 1, i)
        # print("obj list: ", obj_list)


    for i, _ in enumerate(pc_segs[2]):
        pointcloud_seg =pc_segs[2][i].transform(transform_4w)
        pointcloud = pc4
        points = np.asarray(pointcloud_seg.points)
        colors = np.asarray(pointcloud_seg.colors)
        og_colors = np.asarray(pointcloud.colors)
        white_mask = (colors[:, 0] > white_threshold) & (colors[:, 1] > white_threshold) & (colors[:, 2] > white_threshold)
        obj_points[2].append(points[white_mask])
        obj_colors[2].append(og_colors[white_mask])
        new_pointcloud_seg = o3d.geometry.PointCloud()
        new_pointcloud_seg.points = o3d.utility.Vector3dVector(obj_points[2][-1])
        new_pointcloud_seg.colors = o3d.utility.Vector3dVector(obj_colors[2][-1])
        cropped_pcs[2].append(new_pointcloud_seg)
        new_dict = {"mean":np.mean(obj_points[2][i], axis=0), "pcs":[[2, i]], "score":logits_list[2][0][i]}
        update_or_append_obj(obj_list, new_dict, dist_threshold, logits_list, 2, i)
        # print("obj list: ", obj_list)
    

    for i, _ in enumerate(pc_segs[3]):
        pointcloud_seg = pc_segs[3][i].transform(transform_5w)
        pointcloud = pc5
        points = np.asarray(pointcloud_seg.points)
        colors = np.asarray(pointcloud_seg.colors)
        og_colors = np.asarray(pointcloud.colors)
        white_mask = (colors[:, 0] > white_threshold) & (colors[:, 1] > white_threshold) & (colors[:, 2] > white_threshold)
        obj_points[3].append(points[white_mask])
        obj_colors[3].append(og_colors[white_mask])
        new_pointcloud_seg = o3d.geometry.PointCloud()
        new_pointcloud_seg.points = o3d.utility.Vector3dVector(obj_points[3][-1])
        new_pointcloud_seg.colors = o3d.utility.Vector3dVector(obj_colors[3][-1])
        cropped_pcs[3].append(new_pointcloud_seg)
        new_dict = {"mean":np.mean(obj_points[3][i], axis=0), "pcs":[[3, i]], "score":logits_list[3][0][i]}
        update_or_append_obj(obj_list, new_dict, dist_threshold, logits_list, 3, i)
        # print("obj list: ", obj_list)
    obj_list.sort(key = lambda x: x["score"])
    for i, idxs in enumerate(obj_list[0]["pcs"]):
        new_pc_segs.append(cropped_pcs[idxs[0]][idxs[1]])
    # print("New pc segs", new_pc_segs)
    pointcloud_seg = o3d.geometry.PointCloud()
    pointcloud_seg.points = new_pc_segs[0].points
    pointcloud_seg.colors = new_pc_segs[0].colors
    for i in range(1, len(new_pc_segs)):
        pointcloud_seg.points.extend(new_pc_segs[i].points)
        pointcloud_seg.colors.extend(new_pc_segs[i].colors)

    # # pointcloud_seg = pcl_vis.remove_background(pointcloud_seg, center= np.array([0.5, 0, 0]),radius=0.5)

    # points = np.asarray(pointcloud_seg.points)
    # colors = np.asarray(pointcloud_seg.colors)
    # og_colors = np.asarray(pointcloud.colors)

    # # Define the threshold for white points
    #   # Adjust this value if needed

    # # Filter out the white points and discard the black ones
    # white_mask = (colors[:, 0] > white_threshold) & (colors[:, 1] > white_threshold) & (colors[:, 2] > white_threshold)

    # # Select the points and colors that are white
    # filtered_points = points[white_mask]
    # filtered_colors = og_colors[white_mask]
    # cropped = pcl_vis.remove_background(pointcloud, radius=0.5, center=np.array([0.6,0,0]))

    # o3d.io.write_point_cloud(f'../anygrasp_sdk/grasp_detection/seg_pcall_{arg_textprompt}.ply', cropped)



    # Create a new point cloud with the filtered points and updated colors
    # pointcloud_seg = o3d.geometry.PointCloud()
    # pointcloud_seg.points = o3d.utility.Vector3dVector(filtered_points)
    # pointcloud_seg.colors = o3d.utility.Vector3dVector(filtered_colors)

    # centroid = np.mean(filtered_points, axis=0)

    # Create a small sphere at the centroid for visualization
    # centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    # centroid_sphere.paint_uniform_color([1, 0, 0])

    o3d.io.write_point_cloud(f'../anygrasp_sdk/grasp_detection/seg_pc_{arg_textprompt}.ply', pointcloud_seg)

    # pc2 = pcl_vis.remove_background(pc2, radius=0.3, center=np.array([0.6,0,0]))
    # o3d.visualization.draw_geometries([pointcloud_seg])

    points = np.asarray(pointcloud_seg.points)
    colors = np.asarray(pointcloud_seg.colors)
    # for i in range(20):
    #     print(i)
    clean_points, clean_colors = remove_noise(points, colors)
    pointcloud_clean = o3d.geometry.PointCloud()
    pointcloud_clean.points = o3d.utility.Vector3dVector(clean_points)
    pointcloud_clean.colors = o3d.utility.Vector3dVector(clean_colors)

    mean = np.mean(clean_points, axis=0)
    centered_point_cloud = clean_points - mean

    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_point_cloud.T)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvalues and eigenvectors by the magnitude of the eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Rotate the point cloud to align with the principal axes
    rotated_point_cloud = np.dot(centered_point_cloud, eigenvectors)

    # Step 3: Calculate Bounding Box
    min_coords = np.min(rotated_point_cloud, axis=0)
    max_coords = np.max(rotated_point_cloud, axis=0)

    # Dimensions of the bounding box
    dimensions = np.round([max_coords - min_coords], 2)[0]
    print(dimensions)

    


    if save_pc:
        if save_path == None:
            print("Warning!!! Enter save_path as an argument if you want to save the PCs")
        else:
            o3d.io.write_point_cloud(save_path + f'/{arg_textprompt}.ply', pointcloud_seg)
            o3d.io.write_point_cloud(save_path + f'/{arg_textprompt}_clean.ply', pointcloud_clean)
            # vis1 = o3d.visualization.Visualizer()
            # vis1.create_window()
            # vis1.get_render_option().background_color = np.array([1, 1, 1])
            # vis1.add_geometry(pointcloud_seg)
            # vis1.poll_events()
            # vis1.update_renderer()
            # pc_img = vis1.capture_screen_float_buffer(True)
            # plt.imsave(save_path + f'/{arg_textprompt}.png', pc_img)
            # vis1.remove_geometry(pointcloud_seg)
            # vis1.add_geometry(pointcloud_clean)
            # vis1.poll_events()
            # vis1.update_renderer()
            # pc_img = vis1.capture_screen_float_buffer(True)
            # plt.imsave(save_path + f'/{arg_textprompt}_clean.png', pc_img)
            # vis1.remove_geometry(pointcloud_clean)
            # vis1.destroy_window()


    centroid = np.mean(np.asarray(pointcloud_seg.points), axis=0)

    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    centroid_sphere.paint_uniform_color([1, 0, 0])
    centroid_sphere.translate(centroid)

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(pointcloud_clean)
    # vis.add_geometry(centroid_sphere)
    # # vis.run()
    # thread = threading.Thread(target=close_window_after_delay, args=(vis, 3))
    # thread.start()

    # # Run the visualization
    # vis.run()
    if viz == True:
        o3d.visualization.draw_geometries([pointcloud_clean, centroid_sphere])

    # fa = FrankaArm()
    # pose = fa.get_pose()
    centroid[0] = centroid[0] - 0.02
    centroid[1] = centroid[1] + 0.02
    centroid[2] = centroid[2] - 0.01
    # pose.translation = centroid
    # fa.goto_pose(pose)
    return centroid, dimensions


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--textprompt", type= str, required=True)
    args.add_argument("--savepc", action='store_true')
    args = args.parse_args()
    # if args.checkimg:
    centroid, _ = get_centroid([], [], [], [], args.textprompt, args.savepc)