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
import cv2
import sys
from datetime import datetime
import numpy as np
import random
import torch.nn.functional as F

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def opencvToTorch(cv_image, resolution, convert_bgr_to_rgb=True, segmentation=False, interpolation = cv2.INTER_LINEAR):
    """
    cv_image: numpy array (H, W, C) or (H, W)
    resolution: (width, height)
    returns: torch tensor (C, H, W) in [0,1] float32
    """

    # Resize (cv2 expects (width, height))
    resized = cv2.resize(cv_image, resolution, interpolation=interpolation)

    # Convert BGR -> RGB if needed
    if not segmentation and (convert_bgr_to_rgb and len(resized.shape) == 3):
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # To torch float [0,1]
    if not segmentation:
        tensor = torch.from_numpy(resized).float() / 255.0
    else:
        tensor = torch.from_numpy(resized)

    if len(tensor.shape) == 3:
        return tensor.permute(2, 0, 1).contiguous()
    else:
        # grayscale
        return tensor.unsqueeze(0).contiguous()

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def get_reset_expon_lr_func(
lr_init, 
lr_final, 
reset_step, 
reset_lr_init, 
lr_delay_steps=0, 
lr_delay_mult=1.0, 
max_steps=1000000
):
    """
    Exponential decay that resets at reset_step to reset_lr_init, 
    then decays to lr_final by max_steps.
    """
    
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0

        # Determine phase
        if step < reset_step:
            # Phase 1: Initial decay
            current_lr_init = lr_init
            # We treat the first phase as if it's aiming for lr_final over max_steps
            t = np.clip(step / max_steps, 0, 1)
        else:
            # Phase 2: Reset and faster decay
            current_lr_init = reset_lr_init
            # Recalculate 't' to start from 0 at reset_step and hit 1 at max_steps
            # This makes the decay "faster" because it has less time to reach lr_final
            t = np.clip((step - reset_step) / (max_steps - reset_step), 0, 1)

        # Learning rate delay logic (standard reverse cosine)
        if lr_delay_steps > 0 and step < lr_delay_steps:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        # Log-linear interpolation (Exponential decay)
        # Using current_lr_init which changes after reset_step
        log_lerp = np.exp(np.log(current_lr_init) * (1 - t) + np.log(lr_final) * t)
        
        return delay_rate * log_lerp

    return helper 
    

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def robust_sigma_inv(
    g_scales: torch.Tensor,
    g_rotation: torch.Tensor,
    return_invscale_rot: bool = False,
) -> torch.Tensor:
    """
    Compute the robust inverse covariance matrix for an anisotropic Gaussian,
    given its scales and rotations.

    Specifically, computes S^-1 @ R^T @ R @ S^-1 = (S^-1 @ R^T) @ (S^-1 @ R^T)^T,
    where S is a scaling (diagonal) matrix and R is a rotation matrix.

    Args:
        g_scales (torch.Tensor): Shape (N, 3) or (B, k, 3)
            The scaling components (standard deviations along each axis).
        g_rotation (torch.Tensor): Shape (N, 4) or (B, k, 4)
            The quaternion rotation of each Gaussian.

    Returns:
        torch.Tensor: The inverse covariance matrices with shape (N, 3, 3) or (B, k, 3, 3).
    """
    using_batches = g_scales.ndim == 3
    if using_batches:
        B, k, _ = g_scales.shape
        g_scales = g_scales.view(-1, 3)
        g_rotation = g_rotation.view(-1, 4)
    # Build S^-1 @ R^T
    M = build_scaling_rotation(
        s=1. / g_scales,
        r=g_rotation,
    ).transpose(-1, -2)  # (..., 3, 3)
    # Compute the full inverse covariance matrix
    sigma_inv = M.transpose(-1, -2) @ M  # (..., 3, 3)
    if using_batches:
        sigma_inv = sigma_inv.view(B, k, 3, 3)
        M = M.view(B, k, 3, 3)
    if return_invscale_rot:
        return sigma_inv, M
    else:
        return sigma_inv

def robust_gaussian_eval_shifted_points(
    shifted_points: torch.Tensor,
    gaussian_invscale_rot: torch.Tensor,
    gaussian_opacity: torch.Tensor,
):
    """
    Evaluate the Gaussian density at given shifted points.

    Args:
        shifted_points (torch.Tensor): The shifted points. Shape (N, 3).
        gaussian_invscale_rot (torch.Tensor): The inverse scale and rotation of the Gaussians. Shape (N, 3, 3).
        gaussian_opacity (torch.Tensor): The opacity of the Gaussians. Shape (N, 1).
    """
    N = shifted_points.shape[0]

    # M @ (x - mu)
    transformed_shifts = torch.bmm(
        gaussian_invscale_rot,  # (N, 3, 3)
        shifted_points.unsqueeze(-1),  # (N, 3, 1)
    ).squeeze(-1)  # (N, 3)

    dist_sq = (transformed_shifts ** 2).sum(dim=-1, keepdim=True) # (N, 1)
    gaussian_density = gaussian_opacity * torch.exp(
        -0.5 * dist_sq
    )  # (N, 1)

    return gaussian_density

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

        # this is required for torch.compile to work
        def isatty(self):
            if hasattr(old_f, "isatty"):
                return old_f.isatty()
            return False

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    eps = 1e-8
    ab = ab / ab.norm(dim=-1, keepdim=True).clamp_min(eps) 
    return ab
