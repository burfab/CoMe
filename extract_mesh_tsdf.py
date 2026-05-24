import torch
from scene import Scene
import os
import copy
from os import makedirs
from gaussian_renderer import render
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import open3d as o3d
import open3d.core as o3c
import cv2
import math
from arguments import ModelParams, PipelineParams, OptimizationParams, SplattingSettings
from torchvision.utils import save_image
from utils.vis_utils import apply_depth_colormap
from utils import segmentation_utils
import torch.nn.functional as F

def select_diverse_prototypes(features, k):
    # features: (N, C) already normalized

    selected = torch.zeros(features.shape[0], dtype=torch.bool).cuda()

    # pick first: highest norm (or random)
    idx = torch.argmax(features.norm(dim=1))
    selected[idx] = True

    for _ in range(k - 1):
        selected_feats = features[selected,:]  # (m, C)

        sim = torch.nn.CosineSimilarity(2)(features[~selected], selected_feats.unsqueeze(1))
        min_dist = (1 - sim).min(dim=0).values  # cosine distance

        idx = torch.argmax(min_dist)
        features[idx]
        selected[idx] = True

    return features[selected]  # (k, C)
    
    
def train_segmentation_model(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, splat_args, top_k, min_pixnum=100):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    
    splat_args.blend_extra_features = gaussians.segmentation_dimension
    segmenter = segmentation_utils.Segmenter(gaussians.segmentation_dimension, 4).to(gaussians._xyz.device)
    optim = torch.optim.AdamW(segmenter.parameters(), lr=1e-3, weight_decay=1e-4)
    batch_size = 1
    num_iters = 2000
    display_class = 0
    cv2.namedWindow("WINDOW_SEGMENTATION", cv2.WINDOW_NORMAL)
    for i in enumerate(tqdm(range(0, num_iters, batch_size), desc="Segmentation train progress")):
        inputs = []
        targets = []
        for j in range(batch_size):
            with torch.no_grad():
                rand_index = random.randint(0, len(views)-1)
                view = views[rand_index]
                rendering = render(view, gaussians, pipeline, background, splat_args=splat_args)["render"]
                gt_segmentation = view.seg_mask.cuda()
                gt_segmentation = gt_segmentation.detach().long()
                feature_map = torch.clamp_min(rendering[-gaussians.segmentation_dimension:rendering.shape[0],:,:].detach(),1e-6).log()
                inputs.append(feature_map)
                targets.append(gt_segmentation.squeeze())
        
        y = segmenter(torch.stack(inputs,dim=0))
        loss = torch.nn.functional.cross_entropy(y, torch.stack(targets, dim=0),reduction="mean", label_smoothing=0.05)
        loss.backward()
        print(f"Loss: {loss.item(): 3.5f}   \r", end="")
        optim.step()
        optim.zero_grad()
        
        preds = torch.nn.functional.softmax(y, dim=1)
        im = (preds[0, display_class%segmenter.n_classes].detach().cpu().numpy().squeeze() * 255).astype(np.uint8)
        cv2.imshow("WINDOW_SEGMENTATION", im)
        
        key = cv2.waitKey(1)
        if key == ord(' '): display_class+=1
        #torch.save(feature_map.detach().cpu(), f"/tmp/feats_{view.uid}.pth")
        #torch.save(gt_segmask.detach().cpu(), f"/tmp/gt_{view.uid}.pth")
        assert feature_map.shape[0] == gaussians.segmentation_dimension
            
    
    key = 0
    while key != 27:
        key = cv2.waitKey(-1)
        if key == ord(' '): display_class+=1
        im = (preds[0, display_class%segmenter.n_classes].detach().cpu().numpy().squeeze() * 255).astype(np.uint8)
        cv2.imshow("WINDOW_SEGMENTATION", im)
    cv2.destroyAllWindows()
            
    splat_args.blend_extra_features = 0
    return segmenter

        
def tsdf_fusion(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, splat_args, voxel_size, seg_network=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    makedirs(render_path, exist_ok=True)
    o3d_device = o3d.core.Device("CUDA:0")
    # = 0.002
    ALPHA_THRESH=0.0
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000,
            device=o3d_device)
    
    os.makedirs('depths', exist_ok=True)
    
    with torch.no_grad():
        index = 0
        
        ignore_classes = []
        if not seg_network is None: 
            splat_args.blend_extra_features = gaussians.segmentation_dimension
            ignore_classes = [0,2]
        
        
        views_subsampled = []
        sdf_trunc = 6.0
        #for i in range(0, len(views), 10): views_subsampled.append(views[i])
        views_subsampled = views
        cv2.namedWindow("WIN", cv2.WINDOW_NORMAL)
        for _, view in enumerate(tqdm(views_subsampled, desc="Rendering progress")):
            
            rendering = render(view, gaussians, pipeline, background, splat_args=splat_args#, kernel_size=kernel_size
                               )["render"]
            
            depth = rendering[6:7, :, :]
            alpha = rendering[7:8, :, :]
            rgb = rendering[:3, :, :]
            max_depth = torch.max(rendering[6, :, :])
            min_depth = torch.min(rendering[6, :, :])
            
            if seg_network is not None and False:
                feature_map_logits = gaussians.segmentation_inverse_activation(rendering[-gaussians.segmentation_dimension:rendering.shape[0],:,:].detach())
                segmentation = seg_network(feature_map_logits.unsqueeze(0)).squeeze(0).argmax(0,keepdim=True)
                gt = (view.original_image.cpu().detach().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                for ignore_class in ignore_classes:
                    mask_segmentation = segmentation != ignore_class
                    depth[~mask_segmentation] = 0
                    
                    mask_segmentation_big = cv2.resize(mask_segmentation.squeeze().detach().cpu().numpy().astype(np.uint8) * 255,(gt.shape[1], gt.shape[0]))
                    gt[~(mask_segmentation_big > 0),:] = gt[~(mask_segmentation_big>0), :] * 0.3
                    
                cv2.imshow("WIN", gt)
                cv2.waitKey(1)
            
            
            save_image(apply_depth_colormap(depth.permute(1,2,0), None, min_depth, max_depth).permute(-1,0,1), f'depths/out_depth{index}.png')
            save_image(rgb, f'depths/rgb_{index}.png')
            
            index += 1
            
            if view.gt_alpha_mask is not None:
                assert(False)
                depth[(view.gt_alpha_mask < ALPHA_THRESH)] = 0
            
            depth[(alpha < ALPHA_THRESH)] = 0
            if depth[depth != 0].numel() < depth.numel() * 0.1: continue 
            
            W = view.image_width
            H = view.image_height
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, 0, 1]]).float().cuda().T
            intrins =  (view.projection_matrix @ ndc2pix)[:3,:3].T
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=W,
                height=H,
                cx = intrins[0,2].item(),
                cy = intrins[1,2].item(), 
                fx = intrins[0,0].item(), 
                fy = intrins[1,1].item()
            )
            
            extrinsic = np.asarray((view.world_view_transform.T).cpu().numpy())
            
            o3d_color = o3d.t.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_depth = o3d.t.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C"))
            o3d_color = o3d_color.to(o3d_device)
            o3d_depth = o3d_depth.to(o3d_device)

            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)#.to(o3d_device)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)#.to(o3d_device)
            
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                o3d_depth, intrinsic, extrinsic, 1.0, float(sdf_trunc))

            vbg.integrate(frustum_block_coords, o3d_depth, o3d_color, intrinsic,
                          intrinsic, extrinsic, 1.0, float(sdf_trunc))
            
        print("Extract Mesh")
        mesh = vbg.extract_triangle_mesh().to_legacy()
        print(f"Mesh Extracted: {render_path}/tsdf.ply")
        # write mesh
        o3d.io.write_triangle_mesh(f"{render_path}/tsdf.ply", mesh)
        print("Cluster connected triangles")
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
                mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area) 
                
        mesh_0 = copy.deepcopy(mesh)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < (cluster_n_triangles.max() * 0.25)
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        o3d.io.write_triangle_mesh(f"{render_path}/tsdf_largest_component.ply", mesh_0)
        o3d.visualization.draw_geometries([mesh_0]) 
        mesh_0 = mesh_0.remove_unreferenced_vertices()
        mesh_simplified = mesh_0.simplify_quadric_decimation(target_number_of_triangles=30_000)
        o3d.io.write_triangle_mesh(f"{render_path}/tsdf_largest_component_down.ply", mesh_simplified)
        
        
            
            
def extract_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, splat_args, voxel_size: float):
    with torch.no_grad():
        seg_network_path = os.path.join(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "segmentation_network.pth"))
        seg_network = segmentation_utils.Segmenter.from_checkpoint(torch.load(seg_network_path)).cuda()
        dataset.init_type = "sfm"
        dataset.depths = ""
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
        train_cameras = scene.getTrainCameras()
    
        gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = 0 # dataset.kernel_size
        
        # hotfix
        pipeline.convert_SBs_python = gaussians.use_SBs
        
        cams = train_cameras
        gaussians.compute_3D_filter(cams)
        tsdf_fusion(dataset.model_path, "test", iteration, cams, gaussians, pipeline, background, kernel_size, splat_args, voxel_size=voxel_size, seg_network=seg_network)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--voxel_size", type=float, default=0.002)
    args = get_combined_args(parser)

    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    splat_args = ss.get_settings(args)
    extract_mesh(model.extract(args), args.iteration, pipeline.extract(args), splat_args, voxel_size=args.voxel_size)