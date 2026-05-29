#adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
from typing import List
import numpy as np
import torch
import einops
from functools import partial
from scene import Scene
import os
import random
from argparse import ArgumentParser
from arguments import SplattingSettings,ModelParams, PipelineParams, get_combined_args, MeshingSettings
from gaussian_renderer import GaussianModel
import trimesh
from scene.cameras import Camera
from scene.mesh import return_delaunay_tets
from gaussian_renderer import render, integrate, compute_transmittance, ExtendedSettings, GlobalSortOrder
from utils.regularization.sdf.depth_fusion import evaluate_mesh_colors_all_vertices
from utils.extraction.pivots import get_intersecting_pivots_from_normals, get_pivots_by_scores, sample_random_pivots, get_searched_pivots
from utils.extraction.mesh import extract_mesh, compute_isosurface_value_from_depth
from utils.camera_utils import get_cameras_spatial_extent
from utils.geometry_utils import transform_points_world_to_view
import time
from tqdm import tqdm
import gc
import open3d as o3d



def refine_intersections_with_binary_search(
    end_points:torch.Tensor,
    end_sdf:torch.Tensor,
    sdf_function, # A callable
    n_binary_steps:int,
) -> torch.Tensor:
    """
    Refine the intersected isosurface points with binary search.
    
    Args:
        end_points (torch.Tensor): The end points. (N_verts, 2, 3)
        end_sdf (torch.Tensor): The SDF values at the end points. (N_verts, 2, 1)
        sdf_function (Callable): The SDF function. Takes a tensor of points and returns the SDF values.
        n_binary_steps (int): The number of binary steps.
        
    Returns:
        refined_points (torch.Tensor): The refined points. (N_verts, 3)
    """
    
    left_points = end_points[:, 0, :].clone()  # (N_verts, 3)
    right_points = end_points[:, 1, :].clone()  # (N_verts, 3)
    left_sdf = end_sdf[:, 0, :].clone()  # (N_verts, 1)
    right_sdf = end_sdf[:, 1, :].clone()  # (N_verts, 1)
    points = (left_points + right_points) / 2  # (N_verts, 3)

    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        
        mid_sdf = sdf_function(mid_points)
        mid_sdf = mid_sdf.unsqueeze(-1)
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
        
        points = (left_points + right_points) / 2  # (N_verts, 3)

        torch.cuda.empty_cache()
        gc.collect()
        
    return points



@torch.no_grad()
def get_frustum_mask_batched(points: torch.Tensor, cameras, near: float = 0.02, far: float = 1e6):
    
    N = 200_000
    
    mask = torch.empty(0, device='cuda', dtype=torch.bool)
    number_of_batches = np.ceil(len(points)/N).astype(int)
    for i in range(number_of_batches):        
        mask = torch.cat((mask, get_frustum_mask(points[N*i: N * (i+1)], cameras, near, far)))
    return mask
    
@torch.no_grad()
def get_frustum_mask(points: torch.Tensor, cameras, near: float = 0.02, far: float = 1e6):
    H, W = cameras[0].image_height, cameras[0].image_width

    intrinsics = torch.stack(
        [
            torch.Tensor(
                [[cam.focal_x, 0, cam.image_width/2],
                 [0, cam.focal_y, cam.image_height/2],
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
    view_points = einops.einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
    view_points = view_points[:, :, :3]

    uv_points = einops.einsum(intrinsics, view_points, "n_view b c, n_view N c -> n_view N b")

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


@torch.no_grad()
def evaluate_vacancy_sof(
    points, 
    views, 
    gaussians, 
    pipeline, 
    background, 
    kernel_size, 
    splat_args: ExtendedSettings, 
    znear = 0.02,
    zfar = 1e6,
):
    """
    Evaluate a lower bound of the vacancy field of the scene.

    Args:
        points (torch.Tensor): Points to evaluate the vacancy field at. (N, 3)
        views (List[Camera]): Views to evaluate the vacancy field at.
        gaussians (GaussianModel): Gaussian model to evaluate the vacancy field at.
        pipeline (_type_): _description_
        background (_type_): _description_
        kernel_size (_type_): _description_
        splat_args (ExtendedSettings): _description_
        znear (float, optional): _description_. Defaults to 0.02.
        zfar (_type_, optional): _description_. Defaults to 1e6.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    vacancy = torch.zeros((points.shape[0]), dtype=torch.float32, device="cuda")
    
    for _, view in enumerate(tqdm(views, desc="Vacancy evaluation progress")):
        frustum_mask = get_frustum_mask(points=points, cameras=[view], near=znear, far=zfar)
        
        ret = compute_transmittance(
            points3D=points[frustum_mask],  # (N_frustum, 3)
            viewpoint_camera=view, 
            pc=gaussians, 
            pipe=pipeline, 
            bg_color=background, 
            kernel_size=kernel_size, 
            scaling_modifier=1.0, 
            subpixel_offset=None, 
            splat_args=splat_args, 
        )
        
        vacancy[frustum_mask] = torch.maximum(
            vacancy[frustum_mask],  # (N_frustum,)
            ret["transmittance"],  # (N_frustum,)
        )
        
    return vacancy


@torch.no_grad()
def evaluate_vacancy_sof_fast(
    points, 
    views, 
    gaussians, 
    pipeline, 
    background, 
    kernel_size, 
    splat_args: ExtendedSettings, 
    znear = 0.02,
    zfar = 1e6,
    permute_views = True,
):
    """
    Evaluate if the vacancy at a point is greater than a threshold.
    Returns a boolean mask indicating which points have a vacancy value greater than the threshold in splat_args.meshing_settings.transmittance_threshold.

    Args:
        points (torch.Tensor): Points to evaluate the vacancy field at. (N, 3)
        views (List[Camera]): Views to evaluate the vacancy field at.
        gaussians (GaussianModel): Gaussian model to evaluate the vacancy field at.
        pipeline (_type_): _description_
        background (_type_): _description_
        kernel_size (_type_): _description_
        splat_args (ExtendedSettings): _description_
        znear (float, optional): _description_. Defaults to 0.02.
        zfar (_type_, optional): _description_. Defaults to 1e6.
        permute_views (bool, optional): Whether to permute the list of views. Defaults to True.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: Boolean mask indicating which points have a vacancy value greater than the threshold. Shape (N,).
    """
    vacancy = torch.zeros((points.shape[0]), dtype=torch.bool, device="cuda")
    update_idx = torch.arange(points.shape[0], device="cuda")
    
    # Permute the list of views
    if permute_views:
        permuted_indices = np.random.permutation(len(views))
        views_to_use = [views[i] for i in permuted_indices]
    else:
        views_to_use = views
    
    for _, view in enumerate(tqdm(views_to_use, desc="Vacancy evaluation progress")):
        # Get current remaining points to update
        N_current = update_idx.shape[0]
        current_points = points[update_idx]  # (N_current, 3)
        
        # Get frustum mask for current view
        frustum_mask = get_frustum_mask(points=current_points, cameras=[view], near=znear, far=zfar)  # (N_current,)
        
        # Compute transmittance of points that are in the frustum
        ret = compute_transmittance(
            points3D=current_points[frustum_mask],  # (N_frustum, 3)
            viewpoint_camera=view, 
            pc=gaussians, 
            pipe=pipeline, 
            bg_color=background, 
            kernel_size=kernel_size, 
            scaling_modifier=1.0, 
            subpixel_offset=None, 
            splat_args=splat_args, 
        )
        
        # Compute mask of points that pass the threshold
        passing_mask = torch.zeros((N_current), dtype=torch.bool, device="cuda")  # (N_current,)
        passing_mask[frustum_mask] = ret["transmittance"] > splat_args.meshing_settings.transmittance_threshold  # (N_frustum,)
        
        # Get indices of points that pass the threshold
        pass_idx = update_idx[passing_mask]  # (N_pass,)
        
        # Update vacancy of points that pass the threshold
        vacancy[pass_idx] = True  # (N_pass,)
        
        # Update update_idx to only include points that did not pass the threshold yet
        update_idx = update_idx[~passing_mask]  # (N_current - N_pass,)
        
    return vacancy

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


@torch.no_grad()
def evaluation_validation(view, points, inside):
    if view.seg_mask is None:
        return inside
    else:
        mask = view.seg_mask > 0
    R = torch.from_numpy(view.R).float().to(points.device)
    T = torch.from_numpy(view.T).float().to(points.device)
    points_cam = points @ R + T
    pts2d = points_cam[:, :2] / points_cam[:, 2:]
    pts2d = torch.addcmul(
        pts2d.new_tensor(
            [
                (view.image_width / 2 * 2.0 + 1.0) / view.image_width - 1.0,
                (view.image_height / 2 * 2.0 + 1.0) / view.image_height - 1.0,
            ]
        ),
        pts2d.new_tensor([view.focal_x * 2.0 / view.image_width, view.focal_y * 2.0 / view.image_height]),
        pts2d,
    )
    sampled_mask = torch.nn.functional.grid_sample(mask.float()[None].cuda(), pts2d[None, None], align_corners=True)
    return (sampled_mask.squeeze() > 0.5) & inside


@torch.no_grad()
def compute_valid_mask_single_view(
    fov_camera: Camera, points: torch.Tensor, znear=0.1,
) -> torch.Tensor:
    # Get parameters
    points_shape = points.shape
    Fx = fov_camera.focal_x
    Fy = fov_camera.focal_y
    Cx = fov_camera.image_width/2
    Cy = fov_camera.image_height/2
    H = fov_camera.image_height
    W = fov_camera.image_width
    
    # Transform points to camera space
    points_in_camera_space = transform_points_world_to_view(
        points=points.view(1, -1, 3),
        cameras=[fov_camera],
    ).squeeze(0)  # (N, 3)
    
    # Compute point projections
    pts_projections = torch.stack(
        [
            points_in_camera_space[:,0] * Fx / points_in_camera_space[:,2] + Cx,
            points_in_camera_space[:,1] * Fy / points_in_camera_space[:,2] + Cy
        ],
        -1
    ).float()
    
    # Compute frustum mask
    mask = (
        (pts_projections[:, 0] > 0) 
        & (pts_projections[:, 0] < W) 
        & (pts_projections[:, 1] > 0) 
        & (pts_projections[:, 1] < H) 
        & (points_in_camera_space[:,2] > znear)
    )
    
    return mask.view(points_shape[:-1])


@torch.no_grad()
def compute_valid_mask(points, views):
    any_valid = []
    chunk_size = 20_000_000
    
    # Get scene radius
    scene_radius = get_cameras_spatial_extent(cameras=views)['radius'].item()
    znear = 0.02 * scene_radius
    
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        # Initialize valid mask as False
        any_valid_chunk = torch.zeros(point_chunk.shape[0], dtype=torch.bool, device="cuda")

        # Iterate over views
        for view in tqdm(views, desc="Rendering progress"):
            # Compute frustum mask for single view
            inside = compute_valid_mask_single_view(fov_camera=view, points=point_chunk, znear=znear).view(-1)
            assert inside.shape == any_valid_chunk.shape
            
            # Combine with GT mask if available
            valid_points = evaluation_validation(view, point_chunk, inside)
            
            # Update valid mask
            any_valid_chunk = torch.logical_or(any_valid_chunk, valid_points)

        any_valid.append(any_valid_chunk)

    return torch.cat(any_valid)


@torch.no_grad()
def marching_tetrahedra_with_binary_search(
    model_path: str, 
    views: List[Camera], 
    scene: Scene, 
    gaussians: GaussianModel, 
    pipeline: PipelineParams, 
    background: torch.Tensor, 
    kernel_size: float, 
    args,
    splat_args: ExtendedSettings
):  
    if args.dtype == "int32":
        index_dtype = torch.int32
    elif args.dtype == "int64":
        index_dtype = torch.int64
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")
    print(f"[INFO] Using {index_dtype} for indexing.")
    
    # Get scene spatial extent
    scene_radius = get_cameras_spatial_extent(cameras=views)['radius'].item()
    print(f"[INFO] Scene radius: {scene_radius}")
    
    # Define frustum parameters
    apply_frustum_culling = True
    standard_scale = 6.
    frustum_near = 0.02 * scene_radius / standard_scale
    frustum_far = 1e6 * scene_radius / standard_scale
    if apply_frustum_culling:
        print(f"[INFO] Using frustum culling with znear={frustum_near} and zfar={frustum_far}")

    transmittance_threshold = 0.5 + args.isosurface_value
    print(f"[INFO] Using transmittance threshold: {transmittance_threshold}")
    args.isosurface_value = 0.0
    
    @torch.no_grad()
    def sdf_function(points):
        # splat_args = ExtendedSettings()
        splat_args.sort_settings.sort_order = GlobalSortOrder.MIN_Z_BOUNDING
        splat_args.meshing_settings.alpha_early_stop = True
        splat_args.meshing_settings.transmittance_threshold = transmittance_threshold
        
        is_vacant = evaluate_vacancy_sof_fast(
            points=points,
            views=views, 
            gaussians=gaussians, 
            pipeline=pipeline, 
            background=background, 
            kernel_size=kernel_size, 
            splat_args=splat_args, 
            znear=frustum_near,
            zfar=frustum_far,
            permute_views=True,
        )  # (N,)
        
        occupancy = 1. - is_vacant.float()  # (N,)
        return 0.5 - occupancy.view(-1)  # (N,)    
    
    # Compute best isosurface value using depth points
    if args.compute_automatically_isosurface_value:
        print("[INFO] Computing isosurface value automatically...")
        
        # Compute depth maps
        depth_maps = []
        for i_cam in tqdm(range(len(views)), desc="Computing depth maps"):
            render_pkg = render(
                viewpoint_camera=views[i_cam],
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,
                splat_args=splat_args,
            )
            depth_i = render_pkg["render"][6]
            depth_maps.append(depth_i)
        
        # Compute isosurface value from depth points
        sdf_isosurface_value = compute_isosurface_value_from_depth(
            cameras=views,
            depth=depth_maps,
            sdf_function=sdf_function,
            n_depth_points=1_000_000,
            reduction="median",
        )
        
        # Delete depth maps
        del depth_maps
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        
    else:
        print(f"[INFO] Using provided isosurface value.")
        sdf_isosurface_value = args.isosurface_value
    
    # Adjust isosurface value
    print(f"[INFO] Adjusting SDF isosurface value to {sdf_isosurface_value}...")
    def sdf_function_wrapper(points, **kwargs):
        return sdf_function(points, **kwargs) - sdf_isosurface_value
    
    # Batchify the sdf function if necessary
    def batchified_sdf_function(points, **kwargs):
        all_sdf = []
        n_points = points.shape[0]
        
        if n_points > args.n_points_per_sdf_evaluation:
            n_batches = (n_points + args.n_points_per_sdf_evaluation - 1) // args.n_points_per_sdf_evaluation
        else:
            n_batches = 1
            
        for i_batch in range(n_batches):
            start_idx = i_batch * args.n_points_per_sdf_evaluation
            end_idx = min(start_idx + args.n_points_per_sdf_evaluation, n_points)
            batch_points = points[start_idx:end_idx]
            batch_sdf = sdf_function_wrapper(batch_points, **kwargs)
            all_sdf.append(batch_sdf)
        
        return torch.cat(all_sdf, dim=0)
    
    # Get pivots
    if args.use_tetra_points:
        print("[INFO] Using tetra points. Switching to 9 pivots.")
        args.n_pivots = 9
        pivots, pivot_scales = gaussians.get_tetra_points_blobs_as_spokes(let_gradients_flow=False)
        pivots = pivots.view(-1, 3)
        pivot_scales = pivot_scales.view(-1, 1)
        pivot_sdfs = batchified_sdf_function(pivots)
    else:
        if args.use_scores:
            pivots = get_pivots_by_scores(
                gaussians=gaussians,
                cameras=views,
                pipe=pipeline,
                background=background,
                score_ratio_threshold=args.score_threshold,
                kernel_size=0.0,
            ).view(-1, 3)
            pivot_results = pivots, torch.ones_like(pivots[:, 0])
            
        elif args.random_pivots:
            pivots, _ = sample_random_pivots(
                n_pivots_per_gaussian=args.n_pivots, 
                gaussians=gaussians, 
                sample_radius=args.random_radius,
                sdf_function=batchified_sdf_function,
            )
            pivot_results = pivots, torch.ones_like(pivots[:, 0])
            
        elif args.use_searched_pivots:
            pivots, _ = get_searched_pivots(
                gaussians, 
                search_iter=args.search_iter, 
                sdf_function=batchified_sdf_function, 
                std_factor=args.std_factor,
                step_size=args.search_step_size,
            )
            pivot_results = pivots, torch.ones_like(pivots[:, 0])
        
        else:
            pivot_results = get_intersecting_pivots_from_normals(
                n_pivots=args.n_pivots,
                gaussians=gaussians,
                normals=None,
                std_factor=args.std_factor,
                use_smallest_axis_as_normal = args.use_smallest_axis_as_normal,
                sdf_function=batchified_sdf_function if args.use_intersecting_pivots else None,
            )
        
        # Compute SDF values for pivots
        if args.use_intersecting_pivots:
            pivots, pivot_scales, pivot_sdfs = pivot_results
            pivots = pivots.view(-1, 3)
            pivot_scales = pivot_scales.view(-1, 1)
            pivot_sdfs = pivot_sdfs.view(-1)
        else:
            pivots, pivot_scales = pivot_results
            pivots = pivots.view(-1, 3)
            pivot_scales = pivot_scales.view(-1, 1)
            pivot_sdfs = batchified_sdf_function(pivots)
    
    # Compute valid mask
    if args.use_valid_mask:
        valid_mask = compute_valid_mask(points=pivots, views=views)
        pivot_sdfs[torch.logical_not(valid_mask)] = 0.5
        print(f"[INFO] Using valid mask for marching tetrahedra with shape {valid_mask.shape}")
        print(f"[INFO] Switching SDF values of invalid points to 0.5")
    else:
        valid_mask = None
        print("[INFO] Not using valid mask for marching tetrahedra")
    
    # Compute Delaunay triangulation
    t0 = time.time()
    tets = return_delaunay_tets(pivots, method="tetranerf").cpu()
    t1 = time.time()
    print(f"[INFO] Computed {tets.shape[0]} tets with Delaunay triangulation: {t1 - t0}s")
    
    # Extract mesh
    mesh, details = extract_mesh(
        delaunay_tets=tets.cuda(),
        pivots=pivots,
        pivots_sdf=pivot_sdfs,
        pivots_colors=None,
        pivots_scale=pivot_scales,
        filter_large_edges=args.filter_large_edges,
        collapse_large_edges=args.collapse_large_edges,
        return_details=True,
        mtet_on_cpu=args.mtet_on_cpu,
        valid=valid_mask,
    )
    torch.cuda.empty_cache()
    
    # Binary search
    if args.n_binary_steps > 0:        
        end_points = details['end_points']
        end_sdf = details['end_sdf']
        verts = refine_intersections_with_binary_search(
            end_points=end_points,
            end_sdf=end_sdf,
            sdf_function=batchified_sdf_function,
            n_binary_steps=args.n_binary_steps,
        )
        mesh.verts = verts

    # Compute vertex colors
    print("[INFO] Computing vertex colors...")
    verts_colors = evaluate_mesh_colors_all_vertices(
        views=views, 
        mesh=mesh,
        masks=None,
        use_scalable_renderer=True,
    ).view(-1, 3)
    
    # Create mesh
    mesh = trimesh.Trimesh(
        vertices=mesh.verts.cpu().numpy(), 
        faces=mesh.faces.cpu().numpy(), 
        vertex_colors=(verts_colors.cpu().numpy() * 255).astype(np.uint8), 
        process=False
    )

    # Export mesh
    if transmittance_threshold != 0.5:
        iso_suffix = f"_transmittance_threshold_{transmittance_threshold}"
    elif args.isosurface_value != 0:
        iso_suffix = f"_iso_{args.isosurface_value}"
    else: # isosurface_value == 0
        iso_suffix = ""

    mesh_save_path = os.path.join(model_path,f"test/ours_{args.iteration}/mesh_{args.n_pivots}pivots{iso_suffix}.ply")
    if args.use_scores:
        mesh_save_path = mesh_save_path.replace(".ply", "_scores.ply")
    if args.use_searched_pivots:
        mesh_save_path = mesh_save_path.replace(".ply", "_searched.ply")
    mesh.export(mesh_save_path)
    print(f"Mesh saved to:\n{mesh_save_path}")
    
    # Postprocess
    if args.postprocess:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))

        print("[INFO] Start postprocessing to remove floaters and disconnected parts.")
        mesh = post_process_mesh(o3d_mesh, 1)
        print(f"[INFO] Postprocessing done.")
        
        mesh_save_path = mesh_save_path.replace(".ply", "_post.ply")
        o3d.io.write_triangle_mesh(mesh_save_path, mesh)
        print(f"[INFO] Postprocessed mesh saved to: \n{mesh_save_path}")


@torch.no_grad()
def main(
    dataset : ModelParams, 
    pipeline : PipelineParams, 
    args,
    splat_args: ExtendedSettings
):
    # Load scene and Gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"))
    print(f"[INFO] Loaded Gaussian Model from {os.path.join(dataset.model_path, 'point_cloud', f'iteration_{args.iteration}', 'point_cloud.ply')}")
    print(f"[INFO]    > Number of Gaussians: {gaussians._xyz.shape[0]}")
    
    # Background color and kernel size
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    try:
        kernel_size = dataset.kernel_size
    except:
        print("No kernel size found in dataset, using 0.0")
        kernel_size = 0.0
    
    # Extract mesh
    marching_tetrahedra_with_binary_search(
        model_path=dataset.model_path, 
        views=scene.getTrainCameras(), 
        scene=scene, 
        gaussians=gaussians, 
        pipeline=pipeline, 
        background=background, 
        kernel_size=kernel_size, 
        args=args,
        splat_args=splat_args
    )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)

    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    
    # SDF function to use
    parser.add_argument("--n_points_per_sdf_evaluation", default=999_999_999, type=int)
    parser.add_argument("--dtype", default="int32", choices=["int32", "int64"])
    
    # Pivots
    parser.add_argument("--n_pivots", default=2, type=int)
    parser.add_argument("--use_smallest_axis_as_normal", action="store_true")
    parser.add_argument("--std_factor", default=3.0, type=float)
    parser.add_argument("--use_intersecting_pivots", action="store_true")
    parser.add_argument("--use_tetra_points", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    #   > Score-based pivots
    parser.add_argument("--use_scores", action="store_true")
    parser.add_argument("--score_threshold", default=0.75, type=float)
    #   > Random pivots spawned from Gaussians
    parser.add_argument("--random_pivots", action="store_true")
    parser.add_argument("--random_radius", default=2.0, type=float)
    #   > Refined pivots by searching in the direction of the normal
    parser.add_argument("--use_searched_pivots", action="store_true")
    parser.add_argument("--search_iter", default=10, type=int)
    parser.add_argument("--search_step_size", default=1.0, type=float)
    
    # Extraction
    parser.add_argument("--mtet_on_cpu", action="store_true")
    parser.add_argument("--use_valid_mask", action="store_true")
    parser.add_argument("--filter_large_edges", action="store_true")
    parser.add_argument("--collapse_large_edges", action="store_true")

    # Integration
    parser.add_argument("--n_binary_steps", default=8, type=int)
    parser.add_argument("--isosurface_value", default=-9999., type=float)  # 0.2 is a good value for GOF and occupancy
    
    # Postprocessing
    parser.add_argument("--postprocess", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # Change args if use scores
    if args.use_scores or args.random_pivots or args.use_searched_pivots:
        if args.random_pivots:
            print("[INFO] Using random pivots.")
        elif args.use_scores:
            print("[INFO] Using scores for pivots.")
        elif args.use_searched_pivots:
            print("[INFO] Using searched pivots.")
        args.use_tetra_points = False
        args.use_intersecting_pivots = False
        args.filter_large_edges = False
        args.collapse_large_edges = False
    
    # For integration mode
    args.compute_automatically_isosurface_value = (args.isosurface_value <= -9999.)
    
    splat_args = ss.get_settings(args)
    main(
        model.extract(args), 
        pipeline.extract(args), 
        args,
        splat_args=splat_args
    )
    