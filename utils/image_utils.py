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

import torch

def mse(img1, img2, mask=None):
    diff2 = (img1 - img2) ** 2

    if mask is not None:
        mask = mask.expand_as(diff2)
        return (diff2 * mask).sum(dim=(1,2,3), keepdim=True) / \
               (mask.sum(dim=(1,2,3), keepdim=True) + 1e-8)

    return diff2.mean(dim=(1,2,3), keepdim=True)


def psnr(img1, img2, mask=None):
    err = mse(img1, img2, mask)
    return -10 * torch.log10(err+1e-8)
