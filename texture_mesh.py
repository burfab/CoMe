#adopted from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/extract_mesh.py
import cv2
import numpy as np
import torch
import open3d as o3d
from scene import Scene
import os
from arguments import ModelParams, PipelineParams, OptimizationParams, SplattingSettings
import random
from argparse import ArgumentParser, BooleanOptionalAction
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import trimesh
from scene.mesh import Meshes, MeshRenderer, ScalableMeshRenderer, MeshRasterizer
from utils.regularization.sdf.depth_fusion import frustum_cull_mesh
from tqdm import tqdm
import gc
from utils.loss_utils import l1_loss
from decoupled_fused_ssim import fused_ssim
from random import randint

from gaussian_renderer import render

def main(
    dataset : ModelParams, 
    pipeline : PipelineParams, 
    splat_args: SplattingSettings,
    args,
):
    # Get device
    device = torch.device(torch.cuda.current_device())
    splat_args.render_geometry = args.render_geometry
    
    # Load scene and Gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}", "point_cloud.ply"))
    print(f"[INFO] Loaded Gaussian Model from {os.path.join(dataset.model_path, 'point_cloud', f'iteration_{args.iteration}', 'point_cloud.ply')}")
    print(f"[INFO]    > Number of Gaussians: {gaussians._xyz.shape[0]}")
    
    # Get cameras
    cameras = scene.getTrainCameras()
    
    # Background color and kernel size
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    try:
        kernel_size = dataset.kernel_size
    except:
        print("No kernel size found in dataset, using 0.0")
        kernel_size = 0.0
    
    # Render images with Gaussians
    images = []
    gaussians.active_sh_degree = args.sh_degree_for_texturing
    print(f"[INFO] Set SH degree to {gaussians.active_sh_degree}")
    for i_cam in range(len(cameras)):
        with torch.no_grad():
            render_pkg = render(
                viewpoint_camera=cameras[i_cam],
                pc=gaussians,
                pipe=pipeline,
                bg_color=background,splat_args=splat_args
            )
            images.append(render_pkg['render'].cpu().detach())
        
    # Load mesh
    mesh_dir = os.path.join(args.model_path, args.subfolder_prefix + "_" + str(args.iteration))
    mesh_path = os.path.join(mesh_dir, args.mesh_name)
    mesh_filename = os.path.basename(mesh_path)
    mesh_name, mesh_ext = os.path.splitext(mesh_filename)
    mesh_extension = mesh_ext.lstrip('.')
    print(f"[INFO] Loading mesh from {mesh_path}")
    print(f"          > Mesh name: {mesh_name}")
    print(f"          > Mesh extension: {mesh_extension}")
    mesh = trimesh.load(mesh_path)
    
    # Get mesh args
    verts = torch.from_numpy(mesh.vertices).float().to(device="cuda")
    faces = torch.from_numpy(mesh.faces).to(device="cuda")
    _verts_colors = torch.from_numpy(mesh.visual.vertex_colors).float().to(device="cuda")[:, :3] / 255.0  # (N, 3)
    _verts_colors = torch.clamp(_verts_colors, min=1e-6, max=1.0 - 1e-6)
    # logit(x) = log(x / (1 - x)) <= inverse sigmoid
    _verts_colors = torch.log(_verts_colors/ (1.0 - _verts_colors)) 
    
    _verts_colors = torch.nn.Parameter(_verts_colors, requires_grad=True).to(device="cuda")
    print(f"[INFO] Vertex colors shape: {_verts_colors.shape}")
    print(f"[INFO] Vertex colors max: {_verts_colors.max()}, min: {_verts_colors.min()}")
    
    # Instantiates parameters and optimizer
    l = [{'params': [_verts_colors], 'lr': args.lr, "name": "verts_colors"}]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    print(f"[INFO] Learnable parameters:")
    for param_group in optimizer.param_groups:
        print(param_group["name"], param_group["lr"], param_group["params"][0].shape)
    
    # Define mesh renderer
    if args.use_scalable_renderer and False:
        renderer = ScalableMeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras)
        )
    else:
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras)
        )
        
    # Texture refinement
    viewpoint_idx_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(args.n_iter), desc="Texture refinement progress")
    
    print(f"[INFO] Starting texture refinement with {args.n_iter} iterations")
    visited = torch.zeros(verts.shape[0], dtype=torch.bool, device="cuda")
 
    for i_iter in range(args.n_iter):
        # Get updated mesh
        mesh = Meshes(
            verts=verts,
            faces=faces,
            verts_colors=0. + torch.sigmoid(_verts_colors),
        )
        
        # Get random viewpoint
        if not viewpoint_idx_stack:
            viewpoint_idx_stack = list(range(len(cameras)))
        _random_view_idx = randint(0, len(viewpoint_idx_stack)-1)
        viewpoint_idx = viewpoint_idx_stack.pop(_random_view_idx)
        
        # Render frustum-culled mesh
        rendered_image = renderer(
            mesh=frustum_cull_mesh(mesh, cameras[viewpoint_idx]),  # FIXME: Add znear
            cam_idx=viewpoint_idx,
            return_depth=True,
            return_normals=True,
            use_antialiasing=True,  
        )
        mesh_rgb = rendered_image['rgb'].squeeze(0).permute(2, 0, 1)
        
        gt_image = images[viewpoint_idx][:3].to(device)
        depth = images[viewpoint_idx][6].to(device)
        normals = images[viewpoint_idx][3:6].to(device)
        mask = depth.detach() > 0
        
        lssim, cslssim = fused_ssim((gt_image*mask[None,...]).unsqueeze(0), (mesh_rgb*mask[None,...]).unsqueeze(0), gs_image_mapped=(mesh_rgb*mask[None,...]).unsqueeze(0))
        Ll1 = l1_loss(mesh_rgb * mask[None,...], gt_image*mask[None,...])
        LSSIM = ((torch.ones_like(lssim) - lssim) * mask).mean()
        loss = ((1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - LSSIM))
        
        # 1. Compute means along the channel dimension (dim=0)
        red_channel = mesh_rgb.mean(0)
        blue_channel = gt_image.mean(0)
        green_channel = torch.zeros_like(red_channel)

        # 2. Stack to form an RGB image [H, W, C]
        # PyTorch typically processes as RGB, but OpenCV window rendering uses BGR
        # For cv2.imshow to display Red as Red and Blue as Blue, stack in BGR order: [Blue, Green, Red]
        vis_image = torch.stack([blue_channel, green_channel, red_channel], dim=-1)
        cv2.imshow("WND", vis_image.detach().cpu().numpy())
        cv2.waitKey(1)
        
        loss.backward()

        with torch.no_grad():
            if _verts_colors.grad is not None:
                visited |= (_verts_colors.grad.abs().sum(-1) > 0)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Periodically propagate colors to unvisited verts
        if i_iter % 10 == 0 and (~visited).any():
            with torch.no_grad():
                colors = torch.sigmoid(_verts_colors)
                neighbor_sum = torch.zeros_like(colors)
                neighbor_count = torch.zeros(colors.shape[0], 1, device="cuda")
                v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
                for src, dst in [(v1,v0),(v2,v0),(v0,v1),(v2,v1),(v0,v2),(v1,v2)]:
                    neighbor_sum.index_add_(0, dst, colors[src])
                    neighbor_count.index_add_(0, dst,
                        torch.ones(src.shape[0], 1, device="cuda"))
                
                unvisited = ~visited
                neighbor_avg = (neighbor_sum / neighbor_count.clamp(min=1)).clamp(1e-6, 1-1e-6)
                _verts_colors[unvisited] = torch.log(
                    neighbor_avg[unvisited] / (1.0 - neighbor_avg[unvisited])
                )
                visited = torch.zeros(verts.shape[0], dtype=torch.bool, device="cuda")
        
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            if i_iter % 5 == 0:
                postfix_dict = {}
                postfix_dict["Loss"] = f"{ema_loss_for_log:.{7}f}"
                progress_bar.set_postfix(postfix_dict)
                progress_bar.update(5)
        
        if i_iter % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
    print(f"[INFO] Texture refinement completed")
    
    # Create mesh
    with torch.no_grad():
        mesh = trimesh.Trimesh(
            vertices=mesh.verts.cpu().numpy(), 
            faces=mesh.faces.cpu().numpy(), 
            vertex_colors=(torch.sigmoid(_verts_colors).detach().clamp(0., 1.).cpu().numpy() * 255).astype(np.uint8), 
            process=False
        )
    output_path = os.path.join(mesh_dir, mesh_name + f"_texture_refined_{i_iter}.{mesh_extension}")
    mesh.export(output_path)
    print(f"[INFO] Texture refined mesh saved to {output_path}")
    cv2.waitKey(-1)
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)

    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mesh_name", default="tsdf.ply")
    parser.add_argument("--subfolder_prefix", default="ours")
    
    # texture refinement
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--sh_degree_for_texturing", type=int, default=0)
    parser.add_argument("--use_scalable_renderer", action=BooleanOptionalAction, default=True)
    parser.add_argument("--render_geometry", action=BooleanOptionalAction, default=True)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    splat_args = ss.get_settings(args)
    main(
        model.extract(args), 
        pipeline.extract(args), 
        splat_args,
        args,
    )
    