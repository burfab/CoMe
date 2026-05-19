# copy from 2DGS
import math
import torch
import numpy as np

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

threshold = 2
def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.cross(dx, dy, dim=-1)
    
    #w = torch.clamp_max(1.0 / (dx*dx + dy*dy), 2.0).detach()
    #boundary_mask = torch.abs(dx[0,:]) > threshold | torch.abs(dy[0,:]) > threshold
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

def central_diff(image, ignore_inval = None, op=torch.eq, return_squared_norm=False, scale_x=1, scale_y=1):
    """
        image
    """
    output = torch.zeros_like(image)[:,:,0]
    
    dx = torch.cat([image[1:-1, 2:] - image[1:-1, :-2]], dim=1) * scale_x
    dy = torch.cat([image[2:, 1:-1] - image[:-2, 1:-1]], dim=0) * scale_y
    
    if ignore_inval is not None:
        if isinstance(ignore_inval,torch.Tensor) and ignore_inval.ndim == 1: ignore_inval = ignore_inval.reshape(1,1,-1)
        valid_dx = (
            ~(op(image[1:-1, 2:], ignore_inval)).all(-1) &
            ~(op(image[1:-1, :-2], ignore_inval)).all(-1)
        )

        valid_dy = (
            ~(op(image[2:, 1:-1], ignore_inval)).all(-1) &
            ~(op(image[:-2, 1:-1], ignore_inval)).all(-1)
        )

        output[1:-1, 1:-1] = (
            (dx*dx).sum(dim=-1) * valid_dx +
            (dy*dy).sum(dim=-1) * valid_dy
        )
    else:
        output[1:-1, 1:-1] = (
            (dx*dx).sum(dim=-1) +
            (dy*dy).sum(dim=-1)
        )
    if not return_squared_norm: return output.sqrt()
    return output

import torch

def central_diff_normals(
    image,
    ignore_inval=None,
    op=torch.eq,
    scale_x=1.0,
    scale_y=1.0,
    epsilon=0.005,           # Threshold for angular variation
    epsilon_type="hinge",  # Options: "hard", "hinge", "soft_gate"
):
    """
    Central-difference angular variation for normal maps with epsilon insensitivity.

    Args:
        image: H x W x 3 tensor of unit normals
        ignore_inval: optional invalid value marker
        op: comparison op for invalid masking
        scale_x, scale_y: Optional derivative scaling factors.
        epsilon: Angular variation threshold (1 - cos(theta)). 
                 Small values like 0.001 to 0.005 are good starting points.
        epsilon_type: How to handle values below epsilon:
            - "hard": Exactly 0 below epsilon; original value above it (gradient jump).
            - "hinge": Exactly 0 below epsilon; shifts the loss down above it (smooth gradient).
            - "soft_gate": Smoothly suppresses gradients for variations smaller than epsilon.
    Returns:
        H x W tensor
    """
    output = torch.zeros_like(image[..., 0])

    # Neighbor pairs
    nx1 = image[1:-1, 2:]
    nx0 = image[1:-1, :-2]

    ny1 = image[2:, 1:-1]
    ny0 = image[:-2, 1:-1]

    # Angular smoothness: 1 - dot(n1, n2)
    dx = (1.0 - (nx1 * nx0).sum(dim=-1).clamp(-1.0, 1.0)) * scale_x
    dy = (1.0 - (ny1 * ny0).sum(dim=-1).clamp(-1.0, 1.0)) * scale_y

    # --- Apply Epsilon Thresholding ---
    if epsilon > 0.0:
        if epsilon_type == "hard":
            # Direct 0-out. Note: Can cause slight optimization oscillations at the boundary.
            dx = torch.where(dx < epsilon, torch.zeros_like(dx), dx)
            dy = torch.where(dy < epsilon, torch.zeros_like(dy), dy)
            
        elif epsilon_type == "hinge":
            # Standard L1-style dead-zone (SVR style). Smooth gradient everywhere except exactly at epsilon.
            dx = torch.clamp(dx - epsilon, min=0.0)
            dy = torch.clamp(dy - epsilon, min=0.0)
            
        elif epsilon_type == "soft_gate":
            # Multiplies loss by a ramp (0 to 1) below epsilon.
            # This turns the loss quadratic near 0, giving a perfectly smooth downweighting.
            weight_x = torch.clamp(dx / epsilon, max=1.0)
            weight_y = torch.clamp(dy / epsilon, max=1.0)
            dx = dx * weight_x
            dy = dy * weight_y

    if ignore_inval is not None:
        if isinstance(ignore_inval, torch.Tensor) and ignore_inval.ndim == 1:
            ignore_inval = ignore_inval.reshape(1, 1, -1)

        valid_dx = ~(op(nx1, ignore_inval)).all(-1) & ~(op(nx0, ignore_inval)).all(-1)
        valid_dy = ~(op(ny1, ignore_inval)).all(-1) & ~(op(ny0, ignore_inval)).all(-1)

        output[1:-1, 1:-1] = (dx * valid_dx + dy * valid_dy)
    else:
        output[1:-1, 1:-1] = dx + dy

    return output



class DepthNormalConsistencyLoss(torch.nn.Module):
    def __init__(self, intrinsics: tuple, grazing_threshold: float = 1e-3):
        """
        Args:
            intrinsics: (fx, fy, cx, cy) camera intrinsics
            grazing_threshold: Ignores gradients at sharp silhouettes or grazing angles
        """
        super().__init__()
        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.grazing_threshold = grazing_threshold

    def forward(self, rendered_depth: torch.Tensor, rendered_normals: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            rendered_depth: (1, H, W) or (H, W) tensor from 3DGS rasterizer
            rendered_normals: (3, H, W) camera-space normals (OpenCV convention: +Z forward, +Y down)
            mask: (1, H, W) or (H, W) optional foreground/valid opacity mask
        """
        if rendered_depth.ndim == 2:
            rendered_depth = rendered_depth.unsqueeze(0)
        if mask is not None and mask.ndim == 2:
            mask = mask.unsqueeze(0)
            
        _, H, W = rendered_depth.shape
        device = rendered_depth.device
        dtype = rendered_depth.dtype

        # 1. Pixel Ray Vectors
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        u = ((x - self.cx) / self.fx).unsqueeze(0)
        v = ((y - self.cy) / self.fy).unsqueeze(0)

        # 2. Predicted Target Gradients from Rendered Normals
        nx, ny, nz = -rendered_normals[0:1], -rendered_normals[1:2], -rendered_normals[2:3]
        
        N_dot_r = nx * u + ny * v + nz
        # Clamp near zero to avoid explosion
        N_dot_r = torch.where(
            N_dot_r.abs() < 1e-6,
            torch.where(N_dot_r >= 0, 1e-6, -1e-6),
            N_dot_r
        )

        # Analytical log-depth target gradients
        target_gx = -nx / (self.fx * N_dot_r)
        target_gy = -ny / (self.fy * N_dot_r)

        # 3. Actual Gradients from Rendered Log-Depth
        L = torch.log(torch.clamp(rendered_depth, min=1e-5))
        
        actual_dx = torch.zeros_like(L)
        actual_dy = torch.zeros_like(L)
        actual_dx[..., :, :-1] = L[..., :, 1:] - L[..., :, :-1]
        actual_dy[..., :-1, :] = L[..., 1:, :] - L[..., :-1, :]

        # 4. Handle Masking and Edge Weighting
        if mask is None:
            mask = torch.ones_like(L)
            
        # Only compute loss across valid neighboring pixel edges
        W_x = mask[..., :, 1:] * mask[..., :, :-1]
        W_y = mask[..., 1:, :] * mask[..., :-1, :]

        # Ignore grazing angles where normal is perpendicular to camera ray
        valid_grad = (N_dot_r.abs() > self.grazing_threshold)
        W_x = W_x * valid_grad[..., :, 1:] * valid_grad[..., :, :-1]
        W_y = W_y * valid_grad[..., 1:, :] * valid_grad[..., :-1, :]

        # 5. Compute Weighted L1 or L2 Mismatch Loss
        loss_x = torch.abs(actual_dx[..., :, :-1] - target_gx[..., :, :-1]) * W_x
        loss_y = torch.abs(actual_dy[..., :-1, :] - target_gy[..., :-1, :]) * W_y

        # Normalize by the number of active edges to keep loss scaled properly
        total_edges = W_x.sum() + W_y.sum() + 1e-6
        
        return (loss_x.sum() + loss_y.sum()) / total_edges
