#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, opencvToTorch
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import os
WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    image = cv2.imread(cam_info.image, cv2.IMREAD_UNCHANGED)
    orig_w, orig_h = image.shape[1], image.shape[0]
    
    dir_name = os.path.dirname(os.path.dirname(cam_info.image)) 
    if os.path.exists(os.path.join(dir_name, "masks")):
        mask_file = os.path.join(dir_name, "masks", os.path.basename(cam_info.image))
        segmask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED) 
    else:
        segmask = None

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        MAX_RES = 1280 * 720
        if args.resolution == -1:
            if orig_w*orig_h > MAX_RES:
                global WARNED
                if not WARNED:
                    print(f"[ INFO ] Encountered quite large input images (>{int(MAX_RES/1000)}K pixels), rescaling to {int(MAX_RES/1000)}K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w*orig_h / MAX_RES
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        
    if not segmask is None:
        segmask = opencvToTorch(segmask, resolution,segmentation=True, interpolation=cv2.INTER_NEAREST)

    if image.ndim == 3 and image.shape[2] > 3:
        resized_image_rgb = opencvToTorch(image[...,:3], resolution, interpolation=cv2.INTER_LINEAR)
        loaded_mask = opencvToTorch(image[...,3], resolution, interpolation=cv2.INTER_NEAREST)
        gt_image = resized_image_rgb
        assert False, "Test this please"
    else:
        resized_image_rgb = opencvToTorch(image, resolution, cv2.INTER_LINEAR)
        loaded_mask = None
        gt_image = resized_image_rgb

    gt_image = resized_image_rgb[:3, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask, seg_mask=segmask,
                  image_name=cam_info.image_name, uid=id, resolution=resolution, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in tqdm(enumerate(cam_infos), total=len(cam_infos), desc="Loading cameras"):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera, is_train_camera: bool = True):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'img_path' : camera.image_path,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width),
        'is_train_camera' : is_train_camera
    }
    return camera_entry
