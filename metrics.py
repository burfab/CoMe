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

from pathlib import Path
import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import LPIPSEval
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from utils.flip import LDRFLIPLoss
import cv2
import numpy as np

def readImages(renders_dir, gt_dir, masks_dir=None):
    renders = []
    gts = []
    image_names = []
    masks = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = None
        if not masks_dir is None:
            mask = cv2.imread(masks_dir / fname, cv2.IMREAD_UNCHANGED)
            if mask.dtype == np.uint8: mask = mask / 255
                
            if mask.ndim == 3: mask = mask.squeeze(-1)
            assert mask.ndim == 2
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        masks.append(None if mask is None else torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda())
        image_names.append(fname)
    return renders, gts, masks, image_names


def dilate(binary_image, kernel_size=3):
    """
    binary_image: (H, W) tensor with values {0,1} or bool
    returns: dilated image of shape (H, W)
    """
    pad = kernel_size // 2

    if binary_image.ndim == 2:
        x = binary_image.float()[None, None]  # (1,1,H,W)
    if binary_image.ndim == 3:
        x = binary_image.float()[None]  # (1,1,H,W)
    else: x = binary_image.float()

    # max pooling = dilation
    y = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )

    if binary_image.ndim == 3: y = y[0]
    if binary_image.ndim == 2: y = y[0]
    return y.to(binary_image.dtype)


def erode(binary_image, kernel_size=3):
    """
    binary_image: (H, W) tensor with values {0,1} or bool
    returns: eroded image of shape (H, W)
    """
    pad = kernel_size // 2

    if binary_image.ndim == 2:
        x = binary_image.float()[None, None]  # (1,1,H,W)
    if binary_image.ndim == 3:
        x = binary_image.float()[None]  # (1,1,H,W)
    else: x = binary_image.float()

    # erosion = min pooling
    # min(x) = -max(-x)
    y = -F.max_pool2d(
        -x,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )
    
    if binary_image.ndim == 3: y = y[0]
    if binary_image.ndim == 2: y = y[0]
    return y.to(binary_image.dtype)

def closing(binary_image, kernel_size):
    return erode(dilate(binary_image,kernel_size), kernel_size)
def opening(binary_image, kernel_size):
    return dilate(erode(binary_image,kernel_size), kernel_size)


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    flip = LDRFLIPLoss()
    lpips = LPIPSEval(net_type='vgg', device='cuda')

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / args.dataset
            pointcloud_dir = Path(scene_dir) / "point_cloud"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                mask_dir = method_dir / "masks_rendered"
                
                if not os.path.exists(gt_dir) or not os.path.exists(renders_dir) or (not os.path.exists(mask_dir) and args.use_masks):
                    print("\tNot computing metrics as no renders found")
                    continue
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, mask_dir)
                if len(renders) == 0:
                    print("\tNo images found")

                ssims = []
                psnrs = []
                lpipss = []
                flips = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    gt = gts[idx]
                    im = renders[idx]
                    mask = masks[idx] > args.mask_th
                    if args.use_masks:
                        if args.close_mask_kernel > 0:
                            mask = closing(mask, args.close_mask_kernel)
                        if args.erode_mask_kernel > 0:
                            mask = erode(mask, args.erode_mask_kernel)
                        
                        gt = gt * mask
                        im = im * mask
                        
                    ssims.append(ssim(im,gt, mask=mask))
                    psnrs.append(psnr(im, gt, mask=mask))
                    lpipss.append(lpips.criterion(im, gt, mask=mask).squeeze())
                    flips.append(flip(im, gt, mask=mask).mean().item())
                    
                # load number of gaussians
                # with open(os.path.join(pointcloud_dir, f"iteration_{method.split('_')[-1]}", "num_gaussians.json"), 'r') as fp:
                #     num_gaussians = json.load(fp)["num_gaussians"]

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("  FLIP : {:>12.7f}".format(torch.tensor(flips).mean(), ".5"))
                # print("  NUM  : {:>12}".format(num_gaussians))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "FLIPS": torch.tensor(flips).mean().item(),
                                                        # "NUM": num_gaussians}
                                                    })
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "FLIPS": {name: fl for fl, name in zip(torch.tensor(flips).tolist(), image_names)}}
                                                        )

            with open(scene_dir + f"/{args.dataset}_results_full.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/{args.dataset}_per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            print("\tReason: ", e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--dataset', '-d', type=str, default="test")
    parser.add_argument('--use_masks', action="store_true")
    parser.add_argument('--erode_mask_kernel', type=int, default=3)
    parser.add_argument('--close_mask_kernel', type=int, default=3)
    parser.add_argument('--mask_th', type=float, default=0.1)
    args = parser.parse_args()
    evaluate(args.model_paths)
