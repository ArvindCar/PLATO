from livestream import capture_scene, get_pointcloud_segmented_region
from sam3d import get_bbox, SAM
import os, sys, requests, argparse
from utils import corresponding_verts, segment_corresponding_pc
import numpy as np
import open3d as o3d


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--checkimg", type=bool, default=False)
    args.add_argument("--textprompt", type= str, required=True)
    args = args.parse_args()
    if args.checkimg:
        cimg_paths, verts, cam_instrinsic, depth_img = capture_scene(args.textprompt)
    
   