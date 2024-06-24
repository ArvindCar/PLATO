import time
import numpy as np
import open3d as o3d
import robomail.vision as vis
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from sam3d import get_bbox, SAM

def get_pointcloud_segmented_region(mask_image, depth_image, pinhole_camera_intrinsic):
    mask_image = Image.open(mask_image)
    mask_image = np.array(mask_image)
    mask_image = np.expand_dims(mask_image, axis=-1)
    mask_image = np.repeat(mask_image, 3, axis=-1)
    mask_image = mask_image.astype(np.uint8)

    """depth_image = Image.open(depth_image)
    depth_image = np.array(depth_image)
    depth_image = np.expand_dims(depth_image, axis=-1)
    depth_image = np.repeat(depth_image, 3, axis=-1)
    depth_image = depth_image.astype(np.uint8)
    """
    #print(depth_image.shape)
    rgbd_seg = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(mask_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)
    pc_seg = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_seg, pinhole_camera_intrinsic)
    return pc_seg

def capture_scene(arg_textprompt, arg_checkimages=True):
    # initialize the cameras
    import robomail.vision as vis # if not this line thinks of vis as a local variable...
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

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    _, depth_image2, pc2, _, cam_intrinsic2 = cam2.get_next_frame()
    _, depth_image3, pc3, _, cam_intrinsic3 = cam3.get_next_frame()
    _, depth_image4, pc4, _, cam_intrinsic4 = cam4.get_next_frame()
    _, depth_image5, pc5, _, cam_intrinsic5 = cam5.get_next_frame()

    # transform cameras
    pc2.transform(transform_2w)
    pc3.transform(transform_3w)
    pc4.transform(transform_4w)
    pc5.transform(transform_5w)


    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = pc3.points
    pointcloud.colors = pc3.colors
    pointcloud.points.extend(pc2.points)
    pointcloud.colors.extend(pc2.colors)
    pointcloud.points.extend(pc4.points)
    pointcloud.colors.extend(pc4.colors)
    pointcloud.points.extend(pc5.points)
    pointcloud.colors.extend(pc5.colors)

    pointcloud = pcl_vis.remove_background(pointcloud, radius=1.15) 

    vis.add_geometry(pointcloud)
    vis.poll_events()
    vis.update_renderer()
    i=0

    
    #while True:
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
    pointcloud.points = pc3.points
    pointcloud.colors = pc3.colors
    pointcloud.points.extend(pc2.points)
    pointcloud.colors.extend(pc2.colors)
    pointcloud.points.extend(pc4.points)
    pointcloud.colors.extend(pc4.colors)
    pointcloud.points.extend(pc5.points)
    pointcloud.colors.extend(pc5.colors)

    pointcloud = pcl_vis.remove_background(pointcloud, radius=0.95, center=np.array([0.6,0,0]))

    if arg_checkimages:
        image_pairs = [
            (cimage2, dimage2),
            (cimage3, dimage3),
            (cimage4, dimage4),
            (cimage5, dimage5)
        ]
        cimage_paths = []
        for i, (cimage, dimage) in enumerate(image_pairs, start=2):
            cimage_path = f'try_images/cimage{i}.png'
            dimg_path = f'try_images/dimage{i}.png'
            pil_cimage = Image.fromarray(cimage)
            pil_dimage = Image.fromarray(dimage)
            pil_cimage.save(cimage_path)
            pil_dimage.save(dimg_path)
            cimage_paths.append(cimage_path)
            #print(cimage_path)

    vis.update_geometry(pointcloud)
    vis.poll_events()
    vis.update_renderer()
    #print(pointcloud[0,0,0])
    o3d.io.write_point_cloud('temp_pointcloud.ply', pointcloud)

    print('Capture scene done! Trying to mask the images')

    #perform sam
    get_associated_bbox = get_bbox()
    segment_class = SAM()
    for i, cimg_path in enumerate(cimage_paths):
        print(cimg_path)
        bbox = get_associated_bbox.__run__(cimg_path, arg_textprompt)
        img_path = f'try_images/mask{i}.png'
        segment_class.get_mask(cimg_path, bbox, img_path)
        """segmented_verts = cimage_paths(img_path, verts)
        if i == 0:
            vertices = segmented_verts
        else:
            vertices = vertices + segmented_verts"""
    print('Masking done')
    
    pointcloud_seg = o3d.geometry.PointCloud()
    pc_segs = []
    for i in range(len(cam_instrinsic)):
        pc_segs.append(get_pointcloud_segmented_region(f'try_images/mask{i}.png', depth_image[i], cam_instrinsic[i]))

    pc_segs[0].transform(transform_2w)
    pc_segs[1].transform(transform_3w)
    pc_segs[2].transform(transform_4w)
    pc_segs[3].transform(transform_5w)
    pointcloud_seg = o3d.geometry.PointCloud()
    pointcloud_seg.points = pc_segs[0].points
    pointcloud_seg.colors = pc_segs[0].colors
    pointcloud_seg.points.extend(pc_segs[1].points)
    pointcloud_seg.colors.extend(pc_segs[1].colors)
    pointcloud_seg.points.extend(pc_segs[2].points)
    pointcloud_seg.colors.extend(pc_segs[2].colors)
    pointcloud_seg.points.extend(pc_segs[3].points)
    pointcloud_seg.colors.extend(pc_segs[3].colors)
    print(np.asarray(pointcloud_seg))
    pointcloud_seg = pcl_vis.remove_background(pointcloud_seg, center= np.array([0.5, 0, 0]),radius=0.5)

    o3d.io.write_point_cloud('seg_pc.ply', pointcloud_seg)
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud('seg_pc.ply')
    #print(pcd)
    #print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])



    return cimage_paths, verts, cam_instrinsic, depth_image