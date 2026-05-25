from typing import Dict, Any, Tuple, Optional, List, Callable
import numpy as np
import torch
from argparse import Namespace
from arguments import PipelineParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.regularization.normal_field import (
    get_pivots_from_normals,
    get_signed_distance_to_depthmap,
    get_gaussian_std_in_direction,
)
from utils.densification.normal_error import compute_normal_error
from utils.geometry_utils import depth_to_normal, depth_to_normal_with_mask
from utils.general_utils import build_rotation


def initialize_normal_field(
    scene,
) -> Dict[str, Any]:
    normal_field_state = {}
    return normal_field_state

def reset_normal_field_state_at_next_iteration(
    normal_field_state: Dict[str, Any],
) -> Dict[str, Any]:
    return normal_field_state


@torch.no_grad()
def densify_normal_field(
    iteration: int, 
    gaussians: GaussianModel, 
    cameras: List[Camera],
    scene: Scene, 
    pipe: PipelineParams, 
    background: torch.Tensor, 
    kernel_size: float, 
    config: Dict[str, Any],
    normal_field_state: Dict[str, Any], 
    render_func: Callable, 
    args: Namespace,
    maintain_constant_volume: bool = False,
):
    # Get Gaussian normals
    gaussian_normals = gaussians.convert_features_to_normals()  # (N_gaussians, 3)
    gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)  # (N_gaussians, 3)
    
    # Compute normal errors
    normal_errors = compute_normal_error(
        gaussians=gaussians,
        cameras=cameras,
        render_func=render_func,
        pipe=pipe,
        background=background,
        method=config["densification_normalization_method"],  # "count" or "area" or "none"
        normal_to_use=config["densification_normal_to_use"],  # "rendered" or "median_depth" or "expected_depth"
    )  # (N_gaussians,)
    
    # Compute normal errors quantile
    normal_errors_quantile = torch.quantile(normal_errors, q=1. - config["densification_normal_errors_quantile"])
    
    # Densification mask
    densification_mask = normal_errors > normal_errors_quantile  # (N_gaussians,)

    # If N_max_gaussians is set, cap the number of new Gaussians
    if getattr(args, 'N_max_gaussians', None) is not None:
        n_current = gaussians._xyz.shape[0]
        n_allowed = args.N_max_gaussians - n_current
        if n_allowed <= 0:
            print("[WARNING] Maximum Number of Gaussians reached. Skipping Densification.")
            return  # Already at or above cap, skip densification entirely
        n_selected = densification_mask.sum().item()
        if n_selected > n_allowed:
            # Keep only the top n_allowed Gaussians by normal error
            candidate_indices = densification_mask.nonzero(as_tuple=True)[0]
            top_indices = candidate_indices[normal_errors[candidate_indices].topk(n_allowed).indices]
            densification_mask = torch.zeros_like(densification_mask)
            densification_mask[top_indices] = True
            print(f"[WARNING] Capping the number of gaussians to {args.N_max_gaussians}.")

    # Adjust scale of Gaussians to be densified. The idea is to divide the volume of the densified Gaussian by 2,
    # while taking into account the direction of the normal.
    if maintain_constant_volume:
        #   > First, we compute the local basis of the Gaussian
        local_basis = build_rotation(
            r=gaussians._rotation[densification_mask]  # (N_gaussians_to_densify, 3, n_vectors_in_basis)
        ).transpose(-1, -2)  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
        
        #   > Then, we compute the projections of the normals on the local basis
        projections_on_local_basis = (
            gaussian_normals[densification_mask].unsqueeze(1)  # (N_gaussians_to_densify, 1, 3)
            * local_basis  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
        ).sum(dim=-1)  # (N_gaussians_to_densify, n_vectors_in_basis)
        
        #   > We compute the logarithm of the adjustment factors
        log_adjustment_factors = np.log(1. / 2.) * projections_on_local_basis ** 2
        
        #   > Adjust the scaling of the Gaussians
        gaussians._scaling[densification_mask] = gaussians._scaling[densification_mask] + log_adjustment_factors
    
    # Compute xyz of cloned Gaussians as same xyz minus a small multiple of the normal
    new_xyz = gaussians._xyz[densification_mask]  # (N_new_gaussians, 3)
    new_normals = - gaussian_normals[densification_mask]  # (N_new_gaussians, 3)
    normal_stds = get_gaussian_std_in_direction(
        directions=new_normals.unsqueeze(1),  # (N_new_gaussians, 1, 3)
        gaussian_scaling=gaussians.get_scaling_with_3D_filter[densification_mask].detach(), 
        gaussian_rotation=gaussians._rotation[densification_mask].detach(),
        normalize_directions=False,
    )  # (N_gaussians, 1)
    # FIXME: What is the best factor to use here?
    delta = 0.1
    # delta = 1.0
    # delta = np.sqrt(3.)
    # new_xyz = new_xyz + 0.01 * normal_stds * new_normals
    # new_xyz = new_xyz + 1. * normal_stds * new_normals
    # new_xyz = new_xyz + 3. * normal_stds * new_normals
    # new_xyz = new_xyz + 0.05 * normal_stds * new_normals  # best so far?
    new_xyz = new_xyz + delta * normal_stds * new_normals
    
    # Compute normal features of cloned Gaussians to obtain the opposite normal
    new_gaussian_features = gaussians._gaussian_features[densification_mask]  # (N_new_gaussians, n_features)
    new_gaussian_features[:, -1:] = -new_gaussian_features[:, -1:]
    
    # Update xyz of densified Gaussians to be xyz plus a small multiple of the normal
    gaussians._xyz[densification_mask] = (
        gaussians._xyz[densification_mask]
        + delta * normal_stds * gaussian_normals[densification_mask]
    )
    
    # Densify Gaussians
    gaussians.densify_and_clone_from_mask(
        selected_pts_mask=densification_mask,
        new_xyz=new_xyz,
        new_gaussian_features=new_gaussian_features,
    )
    

@torch.no_grad()
def prune_non_maximal_gaussians(
    gaussians: GaussianModel,
    cameras: List[Camera],
    pipe: PipelineParams,
    background: torch.Tensor,
):
    is_maximal = torch.zeros(
        gaussians._xyz.shape[0],
        dtype=torch.bool,
        device=gaussians._xyz.device
    )
    
    for i_cam in range(len(cameras)):
        render_pkg = render_depth(
            viewpoint_camera=cameras[i_cam], 
            pc=gaussians, 
            pipe=pipe, 
            bg_color=background,
            culling=None
        )
        
        max_idx = render_pkg["gidx"].unique()
        is_maximal[max_idx] = True
        
    gaussians.prune_points(~is_maximal)