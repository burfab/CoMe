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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_reset_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.reloc_utils import compute_relocation_cuda
from utils.deform_utils import Deformation
from utils import densify_utils 
from diff_gaussian_rasterization._C import compute_filter_3d
from typing import List
from einops import einsum
import trimesh
import warnings
from arguments import BoundingSetting, MeshingSettings
from scene.appearance_network import VastGaussianAppearanceEmbedding

from scene.cameras import Camera

def init_cdf_mask(importance, thres=1.0):
    importance = importance.flatten()   
    if thres!=1.0:
        percent_sum = thres
        vals,idx = torch.sort(importance+(1e-6))
        cumsum_val = torch.cumsum(vals, dim=0)
        split_index = ((cumsum_val/vals.sum()) > (1-percent_sum)).nonzero().min()
        split_val_nonprune = vals[split_index]

        non_prune_mask = importance>split_val_nonprune 
    else: 
        non_prune_mask = torch.ones_like(importance).bool()
        
    return non_prune_mask

@torch.no_grad()
def get_frustum_mask_batched(points: torch.Tensor, cameras: List[Camera], near: float = 0.02, far: float = 1e6):
    
    N = 200_000
    
    mask = torch.empty(0, device='cuda', dtype=torch.bool)
    number_of_batches = np.ceil(len(points)/N).astype(int)
    for i in range(number_of_batches):        
        mask = torch.cat((mask, get_frustum_mask(points[N*i: N * (i+1)], cameras, near, far)))
    return mask
    
@torch.no_grad()
def get_frustum_mask(points: torch.Tensor, cameras: List[Camera], near: float = 0.02, far: float = 1e6):
    H, W = cameras[0].image_height, cameras[0].image_width

    intrinsics = torch.stack(
        [
            torch.Tensor(
                [[cam.focal_x, 0, W / 2],
                 [0, cam.focal_y, H / 2],
                 [0, 0, 1]]
            ) for cam in cameras
        ], 
        dim=0
    ).to(points.device)

    # full_proj_matrices: (n_view, 4, 4)
    view_matrices = torch.stack(
        [cam.world_view_transform for cam in cameras], dim=0
    ).transpose(1, 2)

    ones = torch.ones_like(points[:, 0]).unsqueeze(-1)
    # homo_points: (N, 4)
    homo_points = torch.cat([points, ones], dim=-1)

    # uv_points: (n_view, N, 4, 4)
    # Apply batch matrix multiplication to get uv_points for all cameras
    view_points = einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
    view_points = view_points[:, :, :3]

    uv_points = einsum(intrinsics, view_points, "n_view b c, n_view N c -> n_view N b")

    z = uv_points[:, :, -1:]
    uv_points = uv_points[:, :, :2] / z
    u, v = uv_points[:, :, 0], uv_points[:, :, 1]

    # Optionally, we can apply near-far culling
    # Apply near-far culling
    depth = view_points[:, :, -1]
    cull_near_fars = (depth >= near) & (depth <= far)

    # Apply frustum mask
    mask = torch.any(cull_near_fars & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1), dim=0)
    return mask
class GaussianModel:

    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]
    
    def get_view2gaussian(self, viewmatrix, deformation=Deformation()):
        r = self._rotation
        norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
        q = r / norm[:, None]
        
        xyz = self.get_xyz
        scales = self.get_scaling_with_3D_filter
        
        xyz, q, scales = deformation.apply(xyz, q, scales, self.scaling_activation)
        
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
    
        rots = R
        N = xyz.shape[0]
        G2W = torch.zeros((N, 4, 4), device='cuda')
        G2W[:, :3, :3] = rots # TODO check if we need to transpose here
        G2W[:, :3, 3] = xyz
        G2W[:, 3, 3] = 1.0
        
        viewmatrix = viewmatrix.transpose(0, 1)
        G2V = viewmatrix @ G2W
        
        R = G2V[:, :3, :3]
        t = G2V[:, :3, 3]
        
        t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
        V2G = torch.zeros((N, 4, 4), device='cuda')
        V2G[:, :3, :3] = R.transpose(1, 2)
        V2G[:, :3, 3] = t2
        V2G[:, 3, 3] = 1.0
        
        # transpose view2gaussian to match glm in CUDA code
        V2G = V2G.transpose(2, 1).contiguous()
        
        # precompute results to reduce computation and IO
        S_inv_square = 1.0 / (scales ** 2)
        R = V2G[:, :3, :3].transpose(1, 2)
        t2 = V2G[:, 3:, :3]
        
        C = torch.sum((t2 ** 2) * S_inv_square[:, None, :], dim=2)
        S_inv_square_R = S_inv_square[:, :, None] * R
        B = t2 @ S_inv_square_R
        Sigma = R.transpose(1, 2) @ S_inv_square_R
        merged = torch.cat([Sigma[:, :, 0], Sigma[:, 1:, 1], Sigma[:, 2:, 2], B.squeeze(), C], dim=1)
        
        return merged

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.segmentation_activation = torch.exp
        self.segmentation_inverse_activation = lambda x: torch.log(x.clamp_min(1e-6))


    def __init__(self, sh_degree : int, use_SBs : bool = False):
        self.cnt_learned_normals_features = 4
        self.segmentation_dimension = 8
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._learned_normals_features = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.use_SBs = use_SBs
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.filter_3D = torch.empty(0)
        self.tmp_radii = None 
        
        
        
          
        # [NEW] Online Accumulation Tensors
        self.accum_eta = torch.empty(0)
        self.accum_view_count = torch.empty(0)
        self.max_eta_3ch = torch.empty(0)
        self.accum_weights_valid = torch.empty(0)
        self.densify_count = torch.empty(0)  # Track how many times GS has been normally densified
        
        # [NEW] Multiview Consistency Attributes
        self.eta_high_count = torch.empty(0)   # Count of views with high eta
        self.eta_high_sum_3ch = torch.empty(0) # Sum of eta_3ch for high eta views
        self.eta_mid_count = torch.empty(0)    # Count of views with mid eta
        self.eta_mid_sum_3ch = torch.empty(0)  # Sum of eta_3ch for mid eta views
        self.eta_low_count = torch.empty(0)    # Count of views with low eta 
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._confidence,
            self._learned_normals_features,
            self._segmentation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.filter_3D
        )
    
    def restore(self, model_args, training_args, mesh_args, appearance_net):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._confidence,
        self._learned_normals_features,
        self._segmentation,
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.filter_3D) = model_args
        self.training_setup(training_args, mesh_args, appearance_net)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.clean_nans()
        
        
        
    def clean_nans(self):
        isnan = torch.isnan(self._xyz).any(-1) | torch.isnan(self._scaling).any(-1) | torch.isnan(self._rotation).any(-1)
        self.prune_points(isnan)

    # setter for scaling
    def set_scaling(self, new_scales):
        self._scaling = self.scaling_inverse_activation(new_scales)

    # setter for opacity
    def set_opacity(self, new_opacity):
        self._opacity = self.inverse_opacity_activation(new_opacity)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    def get_scaling_n_opacity_with_3D_filter(self, use_mip_filter:bool=True):
        if use_mip_filter:
            opacity = self.opacity_activation(self._opacity)
            scales = self.get_scaling
            scales_square = torch.square(scales)
            det1 = scales_square.prod(dim=1)
            scales_after_square = scales_square + torch.square(self.filter_3D) 
            det2 = scales_after_square.prod(dim=1) 
            coef = torch.sqrt(det1 / det2)
            scales = torch.sqrt(scales_after_square)
            return scales, opacity * coef[..., None]
        else:
            return self.get_scaling_with_3D_filter, self.get_opacity_with_3D_filter
    
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_confidence(self):
        return self._confidence
    
    @property
    def get_learned_normals(self):
        return self.convert_features_to_normals(True, None)
    
    @property
    def get_smallest_axis(self):
        # Get scaling
        scaling = self.get_scaling_with_3D_filter  # (N, 3)
        
        # Get index of axis with smallest scaling
        min_axis_idx = torch.argmin(scaling, dim=-1, keepdim=True)  # (N, 1)
        
        # Get rotation matrices
        rotation_matrices = build_rotation(self._rotation)  # (N, 3, n_axes)
        
        # Get smallest axis as the corresponding column of the rotation matrix
        smallest_axis = torch.gather(
            input=rotation_matrices,  # (N, 3, n_axes)
            dim=-1,
            index=min_axis_idx.unsqueeze(1).repeat(1, 3, 1),  # (N, 3, 1)
        ).squeeze(-1)  # (N, 3)
        
        return smallest_axis 
    
    #Warning, these normals are not normalized even if normalize is True!!! normalize just normalizes the directions
    def convert_features_to_normals(self, normalize: bool = True, use_smallest_axis: bool = None):
        assert self.cnt_learned_normals_features
        
        # If None, falls back to default behavior:
        # - If n_gaussian_features == 1, use the smallest axis
        # - If n_gaussian_features == 4, use the full learnable normals
        if use_smallest_axis is None:
            assert self.cnt_learned_normals_features in [1, 4], "Invalid number of Gaussian features"
            use_smallest_axis = self.cnt_learned_normals_features == 1
        
        # Check that the number of Gaussian features is consistent with the chosen mode
        if use_smallest_axis:
            assert self.cnt_learned_normals_features == 1
        else:
            assert self.cnt_learned_normals_features == 4
            
        # Get the Gaussian features
        features = self._learned_normals_features
        
        # Get the normal directions
        if use_smallest_axis:
            normal_directions = self.get_smallest_axis  # (N_gaussians, 3)
        else:
            normal_directions = features[:, :3]  # (N_gaussians, 3)
            if normalize:
                normal_directions = torch.nn.functional.normalize(normal_directions, dim=-1)

        # Get the normal signs
        normal_signs = torch.tanh(features[:, -1:])  # (N_gaussians, 1)

        # Get the normal vectors by multiplying the directions and signs
        feature_normals = normal_directions * normal_signs  # (N_gaussians, 3)
        return feature_normals 
    
    
    @property
    def get_segmentation(self):
        return self.segmentation_activation(self._segmentation)

    @torch.no_grad()
    def compute_3D_filter(self, cameras, CUDA=True):
        print("Computing 3D filter")
        if not CUDA:
            #TODO consider focal length and image width
            xyz = self.get_xyz
            distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
            valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
            
            # we should use the focal length of the highest resolution camera
            focal_length = 0.
            for camera in cameras:

                # transform points to camera space
                R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
                T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
                # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
                xyz_cam = xyz @ R + T[None, :]
                
                xyz_to_cam = torch.norm(xyz_cam, dim=1)
                
                # project to screen space
                valid_depth = xyz_cam[:, 2] > 0.2 # TODO remove hard coded value
                
                
                x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
                z = torch.clamp(z, min=0.001)
                
                x = x / z * camera.focal_x + camera.image_width / 2.0
                y = y / z * camera.focal_y + camera.image_height / 2.0
                
                # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
                
                # use similar tangent space filtering as in the paper
                in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
                
            
                valid = torch.logical_and(valid_depth, in_screen)
                
                # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
                distance[valid] = torch.min(distance[valid], z[valid])
                valid_points = torch.logical_or(valid_points, valid)
                if focal_length < camera.focal_x:
                    focal_length = camera.focal_x
            
            distance[~valid_points] = distance[valid_points].max()
            
            #TODO remove hard coded value
            #TODO box to gaussian transform
            filter_3D = distance / focal_length * (0.2 ** 0.5)
            self.filter_3D = filter_3D[..., None]
        else:
            viewmatrices_torch = torch.stack([c.world_view_transform for c in cameras]).cuda()
            # initialize to (-1), if the filter is negative in the end, we know the point was never observed
            filter_3d_cuda = torch.ones_like(self._opacity) * -1
            compute_filter_3d(
                self.get_xyz,
                viewmatrices_torch,
                cameras[0].image_width, cameras[0].image_height,
                cameras[0].focal_x.item(), cameras[0].focal_y.item(),
                filter_3d_cuda
            )
            
            filter_3d_cuda[filter_3d_cuda < -0.2] = filter_3d_cuda.max()
            self.filter_3D = filter_3d_cuda

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, MCMC_init : bool, add_bbox: bool):
        self.spatial_lr_scale = spatial_lr_scale
        
        if add_bbox: pcd = densify_utils.add_bbox_faces(pcd)
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        
        

        if self.use_SBs:
            pcd_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()

            spherical_betas_paramscount = 3 + self.max_sh_degree * 6
            features = torch.zeros((pcd_color.shape[0], spherical_betas_paramscount)).float().cuda()
            features[:, :3] = pcd_color
            features[:, 3:] = 0.0
        else:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        
        if MCMC_init:
            scales = self.scaling_inverse_activation(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
        else:
            scales = self.scaling_inverse_activation(torch.sqrt(dist2))[...,None].repeat(1, 3) 
            
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        
        if MCMC_init:
            opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        else:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        if self.use_SBs:
            self._features_dc = nn.Parameter(features[:,0:3].contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,3:spherical_betas_paramscount].contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # per-Gaussian confidence
        # initialize to 0 so initial confidence is 1
        self._confidence = nn.Parameter(
            torch.zeros_like(self._opacity)
        ).requires_grad_(True)
        
        self._learned_normals_features = nn.Parameter(
            torch.zeros((fused_point_cloud.shape[0], self.cnt_learned_normals_features)).to(self._opacity.device).float().requires_grad_(True)
        ).requires_grad_(True)
        
        #per-Gaussian segmentation
        self._segmentation = nn.Parameter(
            torch.randn((fused_point_cloud.shape[0], self.segmentation_dimension)).to(self._opacity.device).float()
        ).requires_grad_(True)
        
        

    def training_setup(self, training_args, mesh_args, appearance_net):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        
       
        # [NEW] Online Accumulation Tensors
        self.accum_eta = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.accum_view_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_eta_3ch = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.accum_weights_valid = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.densify_count = torch.zeros((self.get_xyz.shape[0]), dtype=torch.int32, device="cuda")
        
        # [NEW] Multiview Consistency Attributes
        self.eta_high_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.eta_high_sum_3ch = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.eta_mid_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.eta_mid_sum_3ch = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.eta_low_count = torch.zeros((self.get_xyz.shape[0]), device="cuda")
 
        

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._confidence], 'lr': training_args.confidence_lr, "name": "confidence"},
            {'params': [self._segmentation], 'lr': training_args.segmentation_lr, "name": "segmentation"},

        ]
        if self.cnt_learned_normals_features > 0:
            l+=[{'params': [self._learned_normals_features], 'lr': training_args.learned_normals_lr, "name": "learned_normals_features"}]
        if appearance_net is not None:
            if isinstance(appearance_net, VastGaussianAppearanceEmbedding):
                l += [
                    {'params': [appearance_net._appearance_embeddings], 'lr': mesh_args.appearance_lr_init, "name": "appearance embedding"},
                    {'params': appearance_net.appearance_network.parameters(), 'lr': mesh_args.appearance_lr_init, "name": "appearance net"}, #, 'weight_decay': 0.01}
                ]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        if training_args.position_lr_reset_interval <= 0: 
            self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                                            max_steps=training_args.position_lr_max_steps)
        else:
            reset_value = (
                training_args.position_lr_init + 
                training_args.position_lr_reset_interval * 
                (training_args.position_lr_init - training_args.position_lr_final) / 
                training_args.position_lr_max_steps * 
                self.spatial_lr_scale
            ) 
            self.xyz_scheduler_args = get_reset_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                              reset_step=training_args.position_lr_reset_interval, 
                                                              reset_lr_init=reset_value,
                                                            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                                            max_steps=training_args.position_lr_max_steps)
        
        self.appearance_scheduler_args = get_expon_lr_func(
            lr_init=mesh_args.appearance_lr_init,
            lr_final=mesh_args.appearance_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)
        

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        pos_lr = 0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                pos_lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = pos_lr
            if "appearance" in param_group["name"]:
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
        return pos_lr 
        
        

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        if self.use_SBs:
            for i in range(self._features_dc.shape[1]):
                l.append('f_dc_{}'.format(i)) 
            for i in range(self._features_rest.shape[1]):
                l.append('f_rest_{}'.format(i)) 
        else:
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('confidence')
        for i in range(self.cnt_learned_normals_features):
            l.append(f'learned_normals_features_{i}')
        for i in range(self.segmentation_dimension):
            l.append(f'segmentation_{i}')
        l.append('filter_3D')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        if self.use_SBs:
            f_dc = self._features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        else:
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        confidence = self._confidence.detach().cpu().numpy()
        learned_normals_features = self._learned_normals_features.detach().cpu().numpy()
        segmentation = self._segmentation.detach().cpu().numpy()
        filter_3D = self.filter_3D.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, confidence, learned_normals_features, segmentation, filter_3D), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def decay_opacity(self, val=0.999):
        opacities_new = inverse_sigmoid(self.get_opacity * val)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
    def reset_confidence(self):
        optimizable_tensors = self.replace_tensor_to_optimizer(torch.zeros_like(self._confidence), "confidence")
        self._confidence = optimizable_tensors["confidence"]
        
    @torch.no_grad()
    def reset_learned_normal_features(self, reset_directions: bool = True, reset_signs: bool = True):
        assert self.cnt_learned_normals_features > 0
        assert self.cnt_learned_normals_features  in [1,4]
        
        new_normal_features = self._learned_normals_features.clone()
        # Reset the normal signs to 0
        if reset_signs:
            new_normal_features[:, -1:] = 0.0
        
        # Reset the normal directions to shortest Gaussian axis
        if reset_directions:
            if self.cnt_learned_normals_features == 1:
                pass
            elif self.cnt_learned_normals_features == 4:
                # Reset the normal directions to shortest Gaussian axis
                #   > Get min scale
                scale = self.get_scaling_with_3D_filter  # (N_gaussians, 3)
                min_scaling_idx = torch.argmin(scale, dim=-1, keepdim=True)  # (N_gaussians, 1)
                #   > Get rotation matrix
                rotation_matrices = build_rotation(self._rotation)  # (N_gaussians, 3, 3)
                #   > Get column of rotation matrix corresponding to min scale
                gaussian_shortest_axis = torch.gather(
                    input=rotation_matrices,  # (N_gaussians, 3, 3)
                    index=min_scaling_idx.unsqueeze(1).repeat(1, 3, 1),  # (N_gaussians, 3, 1)
                    dim=2,
                ).squeeze(-1)  # (N_gaussians, 3)
                
                new_normal_features[:, :3] = gaussian_shortest_axis
            
            else:
                raise ValueError(f"Invalid number of Gaussian features: {self.cnt_learned_normals_features}") 
            
            
        optimizable_tensors = self.replace_tensor_to_optimizer(new_normal_features, "learned_normals_features")
        self._learned_normals_features = optimizable_tensors["learned_normals_features"]

    def reset_opacity(self):
        # reset opacity by considering 3D filter
        current_opacity_with_filter = self.get_opacity_with_3D_filter
        opacities_new = torch.min(current_opacity_with_filter, torch.ones_like(current_opacity_with_filter)*0.01)
        
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)
        opacities_new = opacities_new / coef[..., None]
        opacities_new = self.inverse_opacity_activation(opacities_new)

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
        
    def _get_tetra_points_blobs_as_spokes(
        self, 
        downsample_ratio:float=None, 
        return_sdf_values:bool=False, 
        xyz_idx:torch.Tensor=None, 
        verbose:bool=False,
        scale_points_with_downsample_ratio:bool=True,
        scale_points_factor:float=None,
        opacity_threshold:float=None,
        override_opacity:torch.Tensor=None,
        return_min_scales:bool=False,
        point_shifts:torch.Tensor=None,
    ):
        """
        Get the tetra points of the Gaussian model.

        Args:
            downsample_ratio (float, optional): The ratio to downsample the tetra points. Defaults to None.
            return_sdf_values (bool, optional): Whether to return the SDF values. Defaults to False.
            xyz_idx (torch.Tensor, optional): The indices of the tetra points to return. 
                If opacity_threshold is provided, xyz_idx should index points that are not filtered out by the opacity threshold,
                such that xyz_idx.max() < (self.get_opacity_with_3D_filter > opacity_threshold).sum().
                Defaults to None. 
                Overrides downsample_ratio if both are provided.
            verbose (bool, optional): Whether to print verbose information. Defaults to False.
            scale_points_with_downsample_ratio (bool, optional): Whether to scale the points with the downsample ratio. 
                Defaults to True. Overrides scale_points_factor if both are provided.
            scale_points_factor (float, optional): The factor to scale the points. Defaults to None.
            opacity_threshold (float, optional): The opacity threshold to filter the points. Defaults to None.
            override_opacity (torch.Tensor, optional): The opacities to use for the tetra points. 
            return_min_scales (bool, optional): Whether to return the minimum scales of the vertices. Defaults to False.
            point_shifts (torch.Tensor, optional): The shifts to apply to the points. Defaults to None. Has shape (N_points, 9, 3).
        Raises:
            ValueError: If SDF values are not used but return_sdf_values is True.

        Returns:
            vertices (torch.Tensor): The vertices of the tetra points.
            vertices_scale (torch.Tensor): The scale of the vertices.
            sdf_values (torch.Tensor, optional): The SDF values of the tetra points.
        """
        M = trimesh.creation.box()
        M.vertices *= 2
        
        use_downsample_ratio = (downsample_ratio is not None) and (downsample_ratio < 1.0)
        use_xyz_idx = xyz_idx is not None
        if verbose:
            print(f"[INFO] Downsample ratio: {downsample_ratio}.")
            
        xyz = self.get_xyz
        scale = self.get_scaling_with_3D_filter * 3.
        rots = build_rotation(self._rotation)
        if return_sdf_values:
            if not self.use_sdf_values:
                raise ValueError("SDF values are not used")
            sdf_values = self.get_sdf_values
        
        # Filter points with small opacity
        if (opacity_threshold is not None) and (opacity_threshold > 0.0):
            if override_opacity is not None:
                opacity = override_opacity
                if verbose:
                    print(f"[INFO] Using provided opacity values.")
            else:
                opacity = self.get_opacity_with_3D_filter
            mask = (opacity > opacity_threshold).squeeze()
            xyz = xyz[mask]
            scale = scale[mask]
            rots = rots[mask]
            if return_sdf_values:
                sdf_values = sdf_values[mask]
                
            if verbose:
                print(f"[INFO] Number of tetra points after opacity threshold: {xyz.shape[0]}.")
            
            # Update downsample ratio
            if use_downsample_ratio:
                downsample_ratio = min(downsample_ratio * self._xyz.shape[0] / xyz.shape[0], 1.0)
                use_downsample_ratio = downsample_ratio < 1.0
                if verbose:
                    print(f"[INFO] Updated downsample ratio: {downsample_ratio}.")
        
        if use_downsample_ratio or use_xyz_idx:
            if use_xyz_idx:
                downsample_ratio = xyz_idx.shape[0] / xyz.shape[0]
                if verbose:
                    print(f"[INFO] Using provided xyz_idx to downsample tetra points, with ratio {downsample_ratio}.")
            else:
                xyz_idx = torch.randperm(xyz.shape[0])[:int(xyz.shape[0] * downsample_ratio)]
                if verbose:
                    print(f"[INFO] Downsampling tetra points by {downsample_ratio}.")
                
            xyz = xyz[xyz_idx]
            scale = scale[xyz_idx]
            if scale_points_with_downsample_ratio:
                scale = scale / (downsample_ratio ** (1/3))
            elif scale_points_factor is not None:
                scale = scale * scale_points_factor
            rots = rots[xyz_idx]
            if return_sdf_values:
                sdf_values = sdf_values[xyz_idx]
            if verbose:
                print(f"[INFO] Number of tetra points after downsampling: {xyz.shape[0] * self.n_pivots_per_gaussian}.")
        
        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)  # (N_points, 3, 8)
        
        # Add point shifts if provided
        if point_shifts is not None:
            vertices = torch.cat(
                [
                    vertices,  # (N_points, 3, 8)
                    torch.zeros_like(vertices[:, :, :1]),  # (N_points, 3, 1)
                ], 
                dim=-1,
            )
            vertices = vertices + point_shifts.permute(0, 2, 1)  # (N_points, 3, 9)

            # scale vertices first
            vertices = vertices * scale.unsqueeze(-1)  # (N_points, 3, 9)
            vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)  # (N_points, 3, 9)
            vertices = vertices.permute(0, 2, 1)  # (N_points, 9, 3)
            
            # Reshape + concatenate centers at the end
            vertices = torch.cat(
                [
                    vertices[:, :-1, :].reshape(-1, 3),  # (N_points * 8, 3)
                    vertices[:, -1, :],  # (N_points, 3)
                ],
                dim=0,
            )  # (N_points * 9, 3)
            
        else:
            # scale vertices first
            vertices = vertices * scale.unsqueeze(-1)
            vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
            vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
            # concat center points
            vertices = torch.cat([vertices, xyz], dim=0)
        
        if return_min_scales: scale_min = scale.min(dim=-1, keepdim=True)[0]
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        
        if return_min_scales:
            print(f"[INFO] Returning min scales in tetra points computation.")
            scale_corner_min = scale_min.repeat(1, 8).reshape(-1, 1)
            vertices_scale_min = torch.cat([scale_corner_min, scale_min], dim=0)
            return vertices, vertices_scale, vertices_scale_min
        if return_sdf_values:
            return vertices, vertices_scale, sdf_values
        else:
            return vertices, vertices_scale
        
    def get_tetra_points_blobs_as_spokes(
        self, 
        let_gradients_flow:bool=False,
        **kwargs
    ):
        if let_gradients_flow:
            return self._get_tetra_points_blobs_as_spokes(**kwargs)
        else:
            with torch.no_grad():
                return self._get_tetra_points_blobs_as_spokes(**kwargs) 

    @torch.no_grad()
    def get_tetra_points(self, views: List[Camera], meshing_settings : MeshingSettings):
        M = trimesh.creation.box()
        M.vertices *= 2
        
        rots = build_rotation(self._rotation)
        xyz = self.get_xyz
        
        # tight opacity bounding, as in StopThePop (in comment)
        match meshing_settings.bounding:
            case BoundingSetting.SIGMA_3:
                scale = self.get_scaling_with_3D_filter * 3.
            case BoundingSetting.SIGMA_333:
                scale = self.get_scaling_with_3D_filter * 3.33
            case BoundingSetting.STP:
                scale = self.get_scaling_with_3D_filter * torch.sqrt(2. * torch.log(255. * self.get_opacity_with_3D_filter))
            #torch.sqrt(2 * torch.log(255 * self.get_opacity_with_3D_filter))
        # filter points with small opacity (as done for bicycle in GOF)
        if meshing_settings.opacity_cutoff_tetra > 0.:
            opacity = self.get_opacity_with_3D_filter
            mask = (opacity > meshing_settings.opacity_cutoff_tetra).squeeze(-1)
            xyz = xyz[mask]
            scale = scale[mask]
            rots = rots[mask]

        # if we still have > 2.5M Gaussians, just pick the largest ones
        N_MAX_GAUSSIANS = 2_500_000
        if xyz.shape[0] > N_MAX_GAUSSIANS:
            all_scales = scale.mean(dim=-1)
            top_scales = all_scales.topk(N_MAX_GAUSSIANS).indices
            xyz = xyz[top_scales]
            scale = scale[top_scales]
            rots = rots[top_scales]

        vertices = M.vertices.T    
        vertices = torch.from_numpy(vertices).float().cuda().unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        # scale vertices first
        vertices = vertices * scale.unsqueeze(-1)
        vertices = torch.bmm(rots, vertices).squeeze(-1) + xyz.unsqueeze(-1)
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous()
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0)
        
        # scale is not a good solution but use it for now
        scale = scale.max(dim=-1, keepdim=True)[0]
        scale_corner = scale.repeat(1, 8).reshape(-1, 1)
        vertices_scale = torch.cat([scale_corner, scale], dim=0)
        
        # Mask out vertices outside of context views
        if meshing_settings.near_far_culling:
            vertex_mask = get_frustum_mask_batched(vertices, views, meshing_settings.near, meshing_settings.far)
            return vertices[vertex_mask], vertices_scale[vertex_mask]
        else:
            return vertices, vertices_scale
  

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        self.use_SBs = len(extra_f_names) in {12, 18, 24, 30}  
        
        segmentation_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("segmentation_")]
        segmentation_names = sorted(segmentation_names, key = lambda x: int(x.split('_')[-1]))
        self.segmentation_dimension = len(segmentation_names)
        
        learned_normals_features_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("learned_normals_features_")]
        learned_normals_features_names = sorted(learned_normals_features_names, key = lambda x: int(x.split('_')[-1]))
        self.cnt_learned_normals_features = len(learned_normals_features_names)

        filter_3D = None
        if "filter_3D" in plydata.elements[0]:
            filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
            self.filter_3D = nn.Parameter(torch.tensor(filter_3D, dtype=torch.float, device="cuda").requires_grad_(True))
        else:
            warnings.warn("3D Filter was not loaded (wasn't in ply file), and should be precomputed with training cameras")
        confidence = None
        if "confidence" in plydata.elements[0]:
            confidence = np.asarray(plydata.elements[0]["confidence"])[..., np.newaxis]
            self._confidence = nn.Parameter(torch.tensor(confidence, dtype=torch.float, device="cuda").requires_grad_(True))
            
        learned_normals_features = None if self.cnt_learned_normals_features == 0 else np.zeros((xyz.shape[0], self.cnt_learned_normals_features))
        for idx, attr_name in enumerate(learned_normals_features_names):
            learned_normals_features[:, idx] = np.asarray(plydata.elements[0][attr_name])
        if self.cnt_learned_normals_features > 0:
            self._learned_normals_features = nn.Parameter(torch.tensor(learned_normals_features, dtype=torch.float, device="cuda").requires_grad_(True))
            
        
        segmentation = None if self.segmentation_dimension == 0 else np.zeros((xyz.shape[0], self.segmentation_dimension))
        for idx, attr_name in enumerate(segmentation_names):
            segmentation[:, idx] = np.asarray(plydata.elements[0][attr_name])
        if self.segmentation_dimension > 0:
            self._segmentation = nn.Parameter(torch.tensor(segmentation, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.use_SBs:
            features_dc = np.zeros((xyz.shape[0], 3))        
            features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        else:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])


        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        if self.use_SBs:
            features_extra = features_extra.reshape((features_extra.shape[0], len(extra_f_names)))
        else:
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.use_SBs:
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        else:
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "appearance" in group["name"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None:
                    print("WARNING, stored state is None")
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                    break
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "appearance" in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._confidence = optimizable_tensors["confidence"]
        self._learned_normals_features = optimizable_tensors["learned_normals_features"]
        self._segmentation = optimizable_tensors["segmentation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]

        
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]
        
        if self.filter_3D is not None and self.filter_3D.numel() > 0:
            self.filter_3D = self.filter_3D[valid_points_mask]
        
         # [NEW] Prune Accumulators
        if self.accum_eta.numel() > 0:
            self.accum_eta = self.accum_eta[valid_points_mask]
            self.accum_view_count = self.accum_view_count[valid_points_mask]
            self.max_eta_3ch = self.max_eta_3ch[valid_points_mask]
            self.accum_weights_valid = self.accum_weights_valid[valid_points_mask]
        if self.densify_count.numel() > 0:
            self.densify_count = self.densify_count[valid_points_mask]
        # [NEW] Prune Multiview Consistency Attributes
        if self.eta_high_count.numel() > 0:
            self.eta_high_count = self.eta_high_count[valid_points_mask]
            self.eta_high_sum_3ch = self.eta_high_sum_3ch[valid_points_mask]
            self.eta_mid_count = self.eta_mid_count[valid_points_mask]
            self.eta_mid_sum_3ch = self.eta_mid_sum_3ch[valid_points_mask]
            self.eta_low_count = self.eta_low_count[valid_points_mask] 
        

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "appearance" in group["name"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_confidence, new_learned_normals_features, new_segmentation, new_tmp_radii, new_filter3d,reset_params=True):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "confidence" : new_confidence,
        "learned_normals_features" : new_learned_normals_features,
        "segmentation" : new_segmentation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._confidence = optimizable_tensors["confidence"]
        self._learned_normals_features = optimizable_tensors["learned_normals_features"]
        self._segmentation = optimizable_tensors["segmentation"]
        if self.tmp_radii is None:
            self.tmp_radii = torch.zeros((self.get_xyz.shape[0] - new_tmp_radii.shape[0]), device="cuda")
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii)) 
        
        if self.filter_3D is not None and self.filter_3D.numel() > 0:
            self.filter_3D = torch.cat((self.filter_3D, new_filter3d), dim=0) 
        
        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            
        # [NEW] Resize Accumulators (Append zeros for new points)
        # Note: We append zeros because new points haven't been seen yet.
        n_new = new_xyz.shape[0]
        self.accum_eta = torch.cat((self.accum_eta, torch.zeros(n_new, device="cuda")))
        self.accum_view_count = torch.cat((self.accum_view_count, torch.zeros(n_new, device="cuda")))
        self.max_eta_3ch = torch.cat((self.max_eta_3ch, torch.zeros((n_new, 3), device="cuda")))
        self.accum_weights_valid = torch.cat((self.accum_weights_valid, torch.zeros(n_new, device="cuda")))
        # Note: densify_count appends zeros, caller may overwrite with inherited values
        self.densify_count = torch.cat((self.densify_count, torch.zeros(n_new, dtype=torch.int32, device="cuda")))
        # [NEW] Resize Multiview Consistency Attributes
        self.eta_high_count = torch.cat((self.eta_high_count, torch.zeros(n_new, device="cuda")))
        self.eta_high_sum_3ch = torch.cat((self.eta_high_sum_3ch, torch.zeros((n_new, 3), device="cuda")))
        self.eta_mid_count = torch.cat((self.eta_mid_count, torch.zeros(n_new, device="cuda")))
        self.eta_mid_sum_3ch = torch.cat((self.eta_mid_sum_3ch, torch.zeros((n_new, 3), device="cuda")))
        self.eta_low_count = torch.cat((self.eta_low_count, torch.zeros(n_new, device="cuda")))
 
            
    def densify_and_split_structgs(self, metric_mask, max_eta_3ch=None, scale_power=1.0):
        """
        Splits Gaussians with Analytic Anisotropic 3-Channel Guidance.
        UPDATED: Computes kx, ky, kz separately based on sampling theory.
        k ~ sqrt(eta) reflects the sampling rate required to resolve the frequency violation.
        
        Args:
            scale_power: Exponent for scale division (new_scale = old_scale / k^scale_power).
                         1.0 = linear division, 2.0 = more aggressive shrinking.
        """
        new_xyz_list = []
        new_f_dc_list = []
        new_f_rest_list = []
        new_opacity_list = []
        new_scaling_list = []
        new_rotation_list = []
        new_confidence_list = []
        new_learned_normals_features_list = []
        new_segmentation_list = []
        new_radii_list = []
        new_filter_3D_list = []
        
        # [NEW] Accumulators for new stats
        new_accum_eta_list = []
        new_accum_view_count_list = []
        new_max_eta_3ch_list = []
        new_densify_count_list = []  # Inherit parent's count + 1

        n_total = self.get_xyz.shape[0]
        
        # Handle Padding
        if metric_mask.shape[0] < n_total:
            padding = torch.zeros(n_total - metric_mask.shape[0], dtype=torch.bool, device="cuda")
            metric_mask = torch.cat((metric_mask, padding))
            
        if max_eta_3ch is not None and max_eta_3ch.shape[0] < n_total:
            padding = torch.zeros((n_total - max_eta_3ch.shape[0], 3), device="cuda")
            max_eta_3ch = torch.cat((max_eta_3ch, padding))
            
        # Indices of points to split
        split_indices = torch.nonzero(metric_mask).squeeze(1)
        # Handle single-point case: ensure split_indices is always 1D
        if split_indices.dim() == 0:
            split_indices = split_indices.unsqueeze(0)
        if split_indices.numel() == 0:
            return

        # Prepare parameters for split candidates
        current_scales = self.get_scaling[split_indices]
        current_rots = self._rotation[split_indices]
        current_xyz = self.get_xyz[split_indices]
        
        # --- Determine k (splits per axis) ---
        if max_eta_3ch is not None:
            # Frequency-based splitting
            etavals = max_eta_3ch[split_indices] # [K, 3]
            
            # Sampling Theory:
            # eta is the frequency energy ratio (~ (sigma * omega_max)^2 ).
            # To resolve the aliasing, we need to reduce sigma by factor k such that sigma_new * omega_max <= 1.
            # sigma_new = sigma / k => (sigma/k)^2 * omega^2 <= 1 => eta / k^2 <= 1 => k >= sqrt(eta).
            
            # We calculate k separately for each axis.
            # Clamp min=2 (no split) and max=8 (limit VRAM usage).
            ks = torch.sqrt(torch.clamp(etavals, min=1.0)).ceil().int()
            # ks = etavals.ceil().int()
            ks = torch.clamp(ks, min=1)
            
        else:
            # Gradient-based split fallback (Standard 3DGS behavior)
            # Splits the longest axis by a factor of 2 (creates 2 gaussians)
            # Here we simulate that by setting k=2 on the max axis.
            max_scale_vals, max_scale_indices = torch.max(current_scales, dim=1)
            ks = torch.ones((current_scales.shape[0], 3), dtype=torch.int, device="cuda")
            ks.scatter_(1, max_scale_indices.unsqueeze(1), 2)
        
        # Group by configuration of (kx, ky, kz)
        # --- Analytic Split Logic (Vectorized) ---
        # Total splits N = kx * ky * kz
        N_per_point = ks.prod(dim=1) # [M]
        
        # Repeat parent attributes for each child
        # [Sum(N), ...]
        repeats = N_per_point
        
        # 1. Expand Parent Attributes
        # new_scaling = self.scaling_inverse_activation(current_scales / ks.float()).repeat_interleave(repeats, dim=0)
        new_scaling = self.scaling_inverse_activation(current_scales / ks.float()**scale_power).repeat_interleave(repeats, dim=0)
        new_rotation = current_rots.repeat_interleave(repeats, dim=0)
        new_features_dc = self._features_dc[split_indices].repeat_interleave(repeats, dim=0)
        new_features_rest = self._features_rest[split_indices].repeat_interleave(repeats, dim=0)
        new_opacity = self._opacity[split_indices].repeat_interleave(repeats, dim=0)
        new_radii = self.tmp_radii[split_indices].repeat_interleave(repeats, dim=0)
        new_filter_3D = self.filter_3D[split_indices].repeat_interleave(repeats, dim=0)
        new_densify_count = (self.densify_count[split_indices] + 1).repeat_interleave(repeats)
        new_confidence = self._confidence[split_indices].repeat_interleave(repeats, dim=0)
        new_learned_normals_features = self._learned_normals_features[split_indices].repeat_interleave(repeats, dim=0)
        new_segmentation = self._segmentation[split_indices].repeat_interleave(repeats, dim=0)

        # 2. Generate Grid Coordinates
        # We need to generate indices (ix, iy, iz) for each child j of parent p
        # where 0 <= ix < kx_p, etc.
        
        # create a flat range of indices [0, 1, ..., Sum(N)-1]
        total_children = repeats.sum()
        # computes start index for each parent in the flattened array
        # starts[i] = sum(N[:i])
        starts = torch.cumsum(repeats, dim=0) - repeats
        
        # expand starts to match children: [0, 0, ..., 0, 1, 1, ..., 1] 
        # (but we want the start index value)
        starts_expanded = starts.repeat_interleave(repeats)
        
        # local_index = global_index - start_index_of_parent
        child_indices = torch.arange(total_children, device="cuda") - starts_expanded
        
        # Retrieve repeated k values for modulo operations
        ks_repeated = ks.repeat_interleave(repeats, dim=0) # [Sum(N), 3]
        kx_r = ks_repeated[:, 0]
        ky_r = ks_repeated[:, 1]
        kz_r = ks_repeated[:, 2]
        
        # Decode (ix, iy, iz) from child_indices
        # index = ix * (ky*kz) + iy * kz + iz 
        # This follows the meshgrid order (indexing='ij') where Z varies fastest
        iz = child_indices % kz_r
        iy = (child_indices // kz_r) % ky_r
        ix = child_indices // (ky_r * kz_r)
        
        # Convert integer indices to centered coordinates
        # coord = i - (k-1)/2.0
        grid_x = ix.float() - (kx_r.float() - 1.0) / 2.0
        grid_y = iy.float() - (ky_r.float() - 1.0) / 2.0
        grid_z = iz.float() - (kz_r.float() - 1.0) / 2.0
        
        grid_flat = torch.stack([grid_x, grid_y, grid_z], dim=1) # [Sum(N), 3]
        
        # 3. Compute Offsets
        # divisions = sigma_new * sqrt(12)
        # sigma_new = sigma_old / k
        # We already computed new scales (pre-activation), but we need the raw sigma_new for offset calc.
        # current_scales is [M, 3]. 
        scales_repeated = current_scales.repeat_interleave(repeats, dim=0)
        stds_new_repeated = scales_repeated / ks_repeated.float()
        separations = stds_new_repeated * (12**0.5)
        
        local_offsets = separations * grid_flat # [Sum(N), 3]
        
        # 4. Rotate Offsets
        # rots_sub = current_rots.repeat_interleave(repeats, dim=0) -> new_rotation (already computed)
        R_sub = build_rotation(new_rotation)
        
        # bmm needs [B, 3, 3] x [B, 3, 1] -> [B, 3, 1]
        # local_offsets.unsqueeze(-1) is [Sum(N), 3, 1]
        world_offsets = torch.bmm(R_sub, local_offsets.unsqueeze(-1)).squeeze(-1)
        
        new_xyz = current_xyz.repeat_interleave(repeats, dim=0) + world_offsets

        # 5. Append to lists (now just single tensors)
        new_xyz_list.append(new_xyz)
        new_scaling_list.append(new_scaling)
        new_rotation_list.append(new_rotation)
        new_f_dc_list.append(new_features_dc)
        new_f_rest_list.append(new_features_rest)
        new_opacity_list.append(new_opacity)
        new_radii_list.append(new_radii)
        new_confidence_list.append(new_confidence)
        new_learned_normals_features_list.append(new_learned_normals_features)
        new_segmentation_list.append(new_segmentation)
        new_filter_3D_list.append(new_filter_3D)
        new_densify_count_list.append(new_densify_count)
        
        # Accumulators (zeros)
        n_new_total_sub = new_xyz.shape[0]
        new_accum_eta_list.append(torch.zeros(n_new_total_sub, device="cuda"))
        new_accum_view_count_list.append(torch.zeros(n_new_total_sub, device="cuda"))
        new_max_eta_3ch_list.append(torch.zeros((n_new_total_sub, 3), device="cuda"))

        # Prune Original Points
        total_prune_mask = torch.zeros(n_total, dtype=torch.bool, device="cuda")
        total_prune_mask[split_indices] = True
        
        if len(new_xyz_list) > 0:
            self.densification_postfix(
                torch.cat(new_xyz_list),
                torch.cat(new_f_dc_list),
                torch.cat(new_f_rest_list),
                torch.cat(new_opacity_list),
                torch.cat(new_scaling_list),
                torch.cat(new_rotation_list),
                torch.cat(new_confidence_list),
                torch.cat(new_learned_normals_features_list),
                torch.cat(new_segmentation_list),
                torch.cat(new_radii_list),
                torch.cat(new_filter_3D_list)
            )

            n_new_total = self.get_xyz.shape[0]
            n_added = n_new_total - n_total
            
            # [NEW] Update densify_count for new children (inherited count + 1)
            # densification_postfix appends zeros, we overwrite with inherited values
            self.densify_count = torch.cat((self.densify_count[:n_total], torch.cat(new_densify_count_list)))
            
            full_prune_filter = torch.cat((total_prune_mask, torch.zeros(n_added, device="cuda", dtype=bool)))
            self.prune_points(full_prune_filter)

    def expand_undersized_gs(self, tau_expand, max_eta_3ch):
        """
        Analytically expands undersized Gaussians to the exact correct scale.
        
        Theory:
        - eta = (sigma * omega)^2 where sigma is scale, omega is max spatial frequency
        - To satisfy Nyquist (eta = 1): sigma_target = sigma_old / sqrt(eta)
        - In log-space: log(sigma_target) = log(sigma_old) - 0.5 * log(eta)
        """
        if max_eta_3ch is None:
            return

        # Identify undersized axes (eta < tau_expand and eta > 0)
        undersized_mask = (max_eta_3ch < tau_expand) & (max_eta_3ch > 0)
        
        if not undersized_mask.any():
            return

        with torch.no_grad():
            # Direct analytical update: log_scale_new = log_scale_old - 0.5 * log(eta)
            # This makes eta_new = 1.0 exactly
            eta_vals = torch.clamp(max_eta_3ch[undersized_mask], min=1e-6)
            delta_log_scale = -0.5 * torch.log(eta_vals)  # Direct correction
            
            delta = torch.zeros_like(self._scaling)
            delta[undersized_mask] = delta_log_scale
            
            new_scaling = self._scaling + delta
            
            # Replace in optimizer
            optimizable_tensors = self.replace_tensor_to_optimizer(new_scaling, "scaling")
            self._scaling = optimizable_tensors["scaling"] 
            
            
    def densify_and_clone_structgs(self, metric_mask, filter):
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_filter_3D = self.filter_3D[selected_pts_mask]
        new_confidence = self._confidence[selected_pts_mask]
        new_learned_normals_features = self._learned_normals_features[selected_pts_mask]
        new_segmentation = self._segmentation[selected_pts_mask]
        
        # [NEW] Clone inherits parent's densify_count (no increment)
        parent_counts = self.densify_count[selected_pts_mask]
        n_old = self.get_xyz.shape[0]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_confidence, new_learned_normals_features, new_segmentation, new_tmp_radii, new_filter_3D)
        
        # Overwrite the zeros appended by densification_postfix with inherited counts
        n_new = new_xyz.shape[0]
        self.densify_count[n_old:n_old+n_new] = parent_counts


    def densify_and_prune_structgs(self, max_screen_size, min_opacity, extent, radii, args, 
                                 importance_score=None, pruning_score=None, 
                                 custom_split_mask=None, custom_prune_mask=None, 
                                 viewspace_points_indices=None,
                                 max_eta_3ch=None, use_abs_grad=False
                                 ): 
        conf_split = torch.clamp(torch.exp(self._confidence).squeeze(-1), min=1e-6, max=1.0).detach()
        densify_grad_threshold_fixed = args.densify_grad_threshold / conf_split
        
        grad_vars = (self.xyz_gradient_accum / self.denom)
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = (self.xyz_gradient_accum_abs / self.denom)
        grads_abs[grads_abs.isnan()] = 0.0
        
        ratio = (torch.norm(grad_vars, dim=-1) >= densify_grad_threshold_fixed).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        # if this value is absurdly high (as it is, we effectively will not use absolute gradients)
        if not use_abs_grad:
            Q = 1e4
        if (Q == 0).item():
            assert(False)
        
        Q_fixed = Q / conf_split
        
        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= densify_grad_threshold_fixed, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= Q_fixed, True, False)
         
        
        
        # --- MERGING RULES ---
        full_split_mask = torch.zeros_like(grad_qualifiers, dtype=torch.bool)
        full_prune_mask = torch.zeros_like(grad_qualifiers, dtype=torch.bool)
        
        if custom_split_mask is not None:
            if viewspace_points_indices is not None:
                full_split_mask[viewspace_points_indices] = custom_split_mask
            else:
                full_split_mask = custom_split_mask
            
        if custom_prune_mask is not None:
            if viewspace_points_indices is not None:
                full_prune_mask[viewspace_points_indices] = custom_prune_mask
            else:
                full_prune_mask = custom_prune_mask

        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.percent_dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.percent_dense*extent

        final_split_mask = torch.logical_and(
            torch.logical_or(full_split_mask, grad_qualifiers_abs), 
            split_qualifiers
        )
        
        final_clone_mask = torch.logical_and(
            torch.logical_or(full_split_mask, grad_qualifiers), 
            # grad_qualifiers,
            clone_qualifiers
        )

        # 4. Execute Split/Clone
        metric_mask = importance_score > args.importance_score_threshold if importance_score is not None else torch.ones_like(final_clone_mask)

        self.densify_and_clone_structgs(metric_mask, final_clone_mask)
        
        # Call Analytic Split
        combined_split_mask = torch.logical_and(metric_mask, final_split_mask)
        self.densify_and_split_structgs(combined_split_mask, max_eta_3ch=max_eta_3ch, scale_power=args.ks_scale_power)

        # --- Pruning ---
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # Apply custom prune mask
        n_new = prune_mask.shape[0]
        n_old = full_prune_mask.shape[0]
        if n_new > n_old:
            padding = torch.zeros(n_new - n_old, dtype=torch.bool, device="cuda")
            full_prune_mask = torch.cat([full_prune_mask, padding])
            
        prune_mask = torch.logical_or(prune_mask, full_prune_mask)

        if pruning_score is not None:
            scores = 1 - pruning_score 
            to_remove = torch.sum(prune_mask)
            remove_budget = int(0.5 * to_remove)

            if remove_budget:
                n_init_points = self.get_xyz.shape[0]
                padded_importance = torch.zeros((n_init_points), dtype=torch.float32)
                padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
                selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool, device="cuda")
                sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
                selected_pts_mask[sampled_indices] = True
                final_prune = torch.logical_and(prune_mask, selected_pts_mask)
                self.prune_points(final_prune)
        else:
            self.prune_points(prune_mask)
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache() 
        
    def final_prune_structgs(self, min_opacity, pruning_score = None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""
        prune_mask = (self.get_opacity < min_opacity).squeeze() 
        scores_mask = pruning_score > 0.9
        final_prune = torch.logical_or(prune_mask, scores_mask)
        self.prune_points(final_prune) 

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        
        # confidence-based thresholds
        conf_split = torch.clamp(torch.exp(self._confidence).squeeze(-1), min=1e-6, max=1.0).detach()
        grad_threshold = grad_threshold / conf_split
        grad_abs_threshold = grad_abs_threshold / conf_split
        
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        if self.use_SBs:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1)
        else:
            new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
            new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)

        
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_confidence = self._confidence[selected_pts_mask].repeat(N,1)
        new_learned_normals_features = self._learned_normals_features[selected_pts_mask].repeat(N,1)
        new_segmentation = self._segmentation[selected_pts_mask].repeat(N,1)
        
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_filter_3D = self.filter_3D[selected_pts_mask].repeat(N, 1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_confidence, new_learned_normals_features, new_segmentation, new_tmp_radii, new_filter_3D)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

# The following code is based on Gaussian Opacity Fields (https://github.com/autonomousvision/gaussian-opacity-fields):
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/5245b20e5d11acd6d1ff5af4b890dc2bedd99693/scene/gaussian_model.py#L631
    def densify_and_clone(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, clone_with_sampling=False):
        # confidence-based thresholds
        conf_split = torch.clamp(torch.exp(self._confidence).squeeze(-1), min=1e-6, max=1.0).detach()
        grad_threshold = grad_threshold / conf_split
        grad_abs_threshold = grad_abs_threshold / conf_split
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        if clone_with_sampling:
            # sample a new gaussian instead of fixing position
            # TODO: maybe it makes sense to move along the direction of maximum variance
            # also, can we somehow derive something that makes sure the density at the max of the Gaussians is somewhat preserved?
            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_confidence = self._confidence[selected_pts_mask]
        new_learned_normals_features = self._learned_normals_features[selected_pts_mask]
        new_segmentation = self._segmentation[selected_pts_mask]
        
        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_filter_3D = self.filter_3D[selected_pts_mask] 
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_confidence, new_learned_normals_features, new_segmentation, new_tmp_radii, new_filter_3D)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, abs_grad_for_densification=False, clone_with_sampling=False):
        self.tmp_radii = radii
 
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
        
        # if this value is absurdly high (as it is, we effectively will not use absolute gradients)
        if not abs_grad_for_densification:
            Q = 1e4
        if (Q == 0).item():
            assert(False)

        before = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent, clone_with_sampling)
        clone = self._xyz.shape[0]
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        split = self._xyz.shape[0]


        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # print(f"Prune {torch.sum(prune_mask)} points due to low opacity")
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # print(f"Prune {torch.sum(big_points_vs)} points due to big screen size")
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # print(f"Prune {torch.sum(big_points_ws)} points due to big scale")
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        
        prune = self._xyz.shape[0]        
        
        tmp_radii = self.tmp_radii
        self.tmp_radii = None 
        
        torch.cuda.empty_cache()
        
        return clone - before, split - clone, split - prune
    

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
        self.denom[update_filter] += 1
        
# The following code is based on 3DGS-MCMC (https://github.com/ubc-vision/3dgs-mcmc):
# https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/scene/gaussian_model.py#L411
    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation,
            "confidence" : self._confidence,
            "learned_normals_features" : self._learned_normals_features,
            "segmentation" : self._segmentation
            }
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # handle params for the appearance embedding
            if 'appearance' in group['name']:
                continue

            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 
        self._confidence = optimizable_tensors["confidence"]
        self._learned_normals_features = optimizable_tensors["learned_normals_features"]
        self._segmentation = optimizable_tensors["segmentation"]
        torch.cuda.empty_cache()
        return optimizable_tensors
    
    def _update_params(self, idxs, ratio):
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=self.get_opacity[idxs, 0],
            scale_old=self.get_scaling[idxs],
            N=ratio[idxs, 0].to(torch.int32) + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))
        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]
    
    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio
    
    def relocate_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return
        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]
        self.replace_tensors_to_optimizer(inds=reinit_idx) 
        
    def reclone_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]
        if alive_indices.shape[0] <= 0:
            return
        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])
        
        selected_pts_mask = torch.zeros((self._opacity.shape[0],)).bool().cuda()
        selected_pts_mask[reinit_idx] = True
        
        new_xyz = self._xyz[selected_pts_mask]

        # sample a new gaussian instead of fixing position
        stds = self.get_scaling[selected_pts_mask]
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask])
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
        
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        
    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0
        probs = self.get_opacity.squeeze(-1) 
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)
        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)
        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)
        return num_gs


    # interesction_preserving with visibility_culling
    def culling_with_interesction_preserving(self, views, render_simp):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()
        accum_area_max = torch.zeros(self._xyz.shape[0]).cuda()


        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self)
            accum_weights = render_pkg["max_weights"]

            imp_score=imp_score+accum_weights            

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)


            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1


        non_prune_mask = init_cdf_mask(importance=imp_score, thres=0.99) 

        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, non_prune_mask==False)
        self.prune_points(prune_mask) 


    # interesction_sampling with visibility_culling
    def culling_with_interesction_sampling(self, views, render_simp):

        imp_score = torch.zeros(self._xyz.shape[0]).cuda()

        count_rad = torch.zeros((self._xyz.shape[0],1)).cuda()
        count_vis = torch.zeros((self._xyz.shape[0],1)).cuda()

        for view in views:
            render_pkg = render_simp(view, self)
            accum_weights = render_pkg["max_weights"]

            imp_score=imp_score+accum_weights

            non_prune_mask = init_cdf_mask(importance=accum_weights, thres=0.99)


            count_rad[render_pkg["radii"]>0] += 1
            count_vis[non_prune_mask] += 1


        prob = imp_score/imp_score.sum()
        prob = prob.cpu().numpy()

        # TODO: fix sampling factor
        factor=0.6
        N_xyz=self._xyz.shape[0]
        num_sampled=int(N_xyz*factor*((prob!=0).sum()/prob.shape[0]))
        indices = np.random.choice(N_xyz, size=num_sampled, 
                                    p=prob, replace=False)

        non_prune_mask = np.zeros(N_xyz, dtype=bool)
        non_prune_mask[indices] = True

        prune_mask = (count_vis<=1)[:,0]
        prune_mask = torch.logical_or(prune_mask, torch.tensor(non_prune_mask==False, device='cuda'))
        self.prune_points(prune_mask) 