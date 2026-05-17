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
import os
from importlib import import_module
import torch
import json
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, SplattingSettings, OptimizationParams, SplattingSettings, MeshingParams
from utils.depth_utils import depths_to_points, depth_to_normal, central_diff, DepthNormalConsistencyLoss
from utils.vis_utils import gui_visualize, export_image
from utils import segmentation_utils
from utils import deform_utils
from utils import densify_utils
from scene.gaussian_model import build_scaling_rotation
from diff_gaussian_rasterization import ExtendedSettings, DebugVisualization, DebugVisualizationType
from decoupled_fused_ssim import fused_ssim
import numpy as np
from scene.appearance_network import AppearanceEmbedding, VastGaussianAppearanceEmbedding, SSIMDecoupledAppearanceEmbedding
from functools import partial
import copy
from scene.densifier import AbsGradDensifier, MCMCDensifier, MSv2AbsGradDensifier, CustomDensifier
import warnings

RED = '\033[31m'
RESET = '\033[0m'

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# TODO: can we precompute this? should be easy enough (to store as well)
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if lr_init == 0:
            return 0
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
        return (delay_rate * log_lerp)

    return helper

def training(dataset, opt, pipe : PipelineParams, mesh : MeshingParams, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, splat_args: ExtendedSettings):
    import time
    start_event = time.time()
    
    first_iter = 0
    # TODO: reintroduce tensorboard to log how many Gaussians we densify, etc.
    tb_writer = prepare_output_and_logger(dataset, splat_args, opt, pipe, mesh)
    gaussians = GaussianModel(dataset.sh_degree, use_SBs=pipe.convert_SBs_python)
    scene = Scene(dataset, gaussians, MCMC_init=mesh.cap_max != -1, add_bbox_faces=opt.add_bbox_faces)
    trainCameras = scene.getTrainCameras().copy() 
    
    segmentation_classes_set = set()
    for c in scene.getTrainCameras():
        uniques_seg_mask = torch.unique(c.seg_mask.cuda().detach()).cpu().tolist()
        for seg_class in uniques_seg_mask: segmentation_classes_set.add(seg_class)
    
    segmentation_network = segmentation_utils.Segmenter(gaussians.segmentation_dimension, len(segmentation_classes_set)).cuda()
    segmentation_network_optim = torch.optim.Adam(
        [
            {
                "params": segmentation_network.parameters(),
                "lr": opt.segmentation_network_lr,
            }
        ]
    )
    
    number_views_for_deform_model = np.max([c.uid for c in trainCameras]).item()+1
    
    
    if mesh.use_vastgaussian_appearance:
        appearance_embedding = VastGaussianAppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim, lambda_l2=opt.lambda_l2)
    elif mesh.use_ssimdecoupled_appearance:
        appearance_embedding = SSIMDecoupledAppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim, lambda_l2=opt.lambda_l2)
    else:
        warnings.warn("Unknown appearance embedding, using default (No Appearance Embedding)")
        appearance_embedding = AppearanceEmbedding(num_views=len(trainCameras), lambda_ssim=opt.lambda_dssim, lambda_l2=opt.lambda_l2)
    gaussians.training_setup(opt, mesh, appearance_embedding)
    gaussians.clean_nans()
    deform_cfg = deform_utils.make_deform_config(gaussians, scene, opt, number_views_for_deform_model)
    deform_model = deform_utils.DeformModel(deform_cfg) 
    deform_model.training_setting()
    
    
    if checkpoint:
        (model_params, first_iter, (_appearance_embedding), (_segmentation_network, _segmentation_network_optim), _deform_model) = torch.load(checkpoint, weights_only=False)
        appearance_embedding.restore(_appearance_embedding)
        gaussians.restore(model_params, opt, mesh, appearance_embedding)
        segmentation_network = segmentation_utils.Segmenter.from_checkpoint(_segmentation_network)
        segmentation_network_optim.load_state_dict(_segmentation_network_optim)
        deform_model = deform_utils.DeformModel.restore(_deform_model, first_iter)
        

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # TODO: same strategy as for the appearance embedding
    
    if True:
        densifier = CustomDensifier(gaussians, opt, mesh, dataset, pipe, scene.cameras_extent)
    elif mesh.use_msv2_simplification:
        densifier = MSv2AbsGradDensifier(gaussians, opt, mesh, dataset, pipe)
    elif mesh.cap_max == -1:
        densifier = AbsGradDensifier(gaussians, opt, mesh, dataset, pipe)
    else:
        densifier = MCMCDensifier(gaussians, opt, mesh, dataset, pipe)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx
    # because I did this error once in the past
    del camera, idx
        
    # at first, we don't need the opacity
    splat_args.render_opacity = False
    
    structure_tensor_cache = densify_utils.precompute_structure_tensors(scene, opt.st_mode, opt.st_levels)
    class BatchStats:
        # [NEW] Batch Training: accumulate gradients from multiple cameras
        def __init__(self, max_batch_size, device="cuda"):
            self.batch_loss = 0.0
            self.batch_rgb_loss = 0.0
            self.last_radii = None
            self.last_visibility_filter = None
            self.batch_size = 0
            self.max_batch_size = max_batch_size
        def complete(self):
            return self.batch_size == self.max_batch_size
        def on_frame(self, loss, rgb_loss, radii, visibility_filter):
            self.batch_loss += loss.item()
            self.batch_rgb_loss += rgb_loss.item()
            self.last_radii = radii
            self.last_visibility_filter = visibility_filter
            self.batch_size+=1
            
    
    masks_selfgenerated = {}
    
    temp_lambdas = {
        "occupation_lambda": 10,
        "variational_depth_normal_fusion_lambda": 5e-1,
        "depth_smoothness": 5e-2 * scene.cameras_extent
        }
    batch = None
    
    for iteration in checkpoint_iterations: assert (iteration % opt.batch_size) == 0, "Must be a multiple of batch size"
        
    
    gaussians.compute_3D_filter(cameras=trainCameras, CUDA=not pipe.compute_filter3D_python)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if batch is None or batch.complete():
            batch = BatchStats(opt.batch_size,gaussians.get_xyz.device)
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, message = network_gui.receive()
                if custom_cam != None:
                    with torch.no_grad():
                        debugVis = DebugVisualization(**message["debug_data"])
                        deform_time = message.get("deform_time_point", -1.0)
                        deformation = deform_model.deformation(gaussians, deform_time) if deform_time >= 0 else deform_utils.Deformation()
                        
                        render_segmentation = message["custom_message"].startswith("segmentation")
                        splat_args.blend_extra_features = gaussians.segmentation_dimension
                        
                        gaussians_mask = None
                        if message["custom_message"] == "no_bg_gaussians":
                            with torch.no_grad():
                                gaussians_classified = segmentation_utils.classify_gaussians(gaussians, segmentation_network).argmax(-1, keepdim=True)
                                gaussians_mask = (gaussians_classified == 1) | (gaussians_classified == 3)
                        
                        net_image = render(custom_cam, gaussians, pipe, background, message["scaling_modifier"], splat_args=splat_args, debugVis=debugVis, deformation=deformation, gaussians_mask = gaussians_mask)["render"]
                    if debugVis.type == 0 or debugVis.type == DebugVisualizationType.CONFIDENCE:
                        rgb_image = net_image[:3]
                        if message["render_appearance_embedding"] or message["custom_message"] == "appearance":
                            rgb_image = appearance_embedding.appearance_mapping(rgb_image, message["camera_idx"])
                        image = gui_visualize(
                            render_cam=custom_cam,
                            alpha=net_image[7:8],
                            distortion=net_image[8:9],
                            depth=net_image[6:7],
                            normal=net_image[3:6],
                            confidence=net_image[10:11],
                            render=rgb_image,
                            color_variance=net_image[11:12],
                            normal_variance=net_image[12:13],
                            other_args=message,
                            occupation=net_image[13:14],
                            occupation2=net_image[14:15],
                            segmentation=None if not render_segmentation else net_image[-gaussians.segmentation_dimension:],
                            segmentation_network = segmentation_network
                        )
                    else:
                        image = net_image[:3]

                    image = torch.clamp(image, 0., 1.)
                    net_image_bytes = memoryview((image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                net_image_bytes = memoryview((image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, json.dumps({"method_dir": dataset.model_path}))
                if bool(message["train"]) and ((iteration < int(opt.iterations)) or not bool(message["keep_alive"])):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None
                
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        if iteration % opt.self_generated_masks_interval == 0 and iteration > 0 or (iteration == first_iter and first_iter >= opt.self_generated_masks_interval):
            if iteration > opt.densify_until_iter: appearance_embedding.lambda_l2 = 0
            with torch.no_grad():
                all_cameras = scene.getTrainCameras().copy()
                import cv2
                cv2.namedWindow("MASK", cv2.WINDOW_NORMAL)
                for c in tqdm(all_cameras):
                    render_pkg = render(c, gaussians, pipe, bg, splat_args=splat_args, gt_color=None, deformation=deform_utils.Deformation(), extract_final_T=True)
                    mask = 1-render_pkg["final_T"].squeeze(0)
                    masks_selfgenerated[c.uid] = mask.detach().cpu() #torch.nn.functional.sigmoid((mask.detach()- 0.9) * 20).cpu()  
                    D = render_pkg["render"][6].detach().cpu()
                    N = render_pkg["render"][3:6].detach().cpu()
                    C = render_pkg["render"][:3].detach().cpu()
                    C2 = c.original_image.detach().cpu()
                    K = torch.tensor([[c.focal_x, 0, c.image_width/2],
                                      [0, c.focal_y, c.image_height/2],
                                      [0,0,1]
                                      ]).cpu().float()
                    torch.save((D,N,C, C2, c.world_view_transform.cpu().detach(),K),f"/tmp/test/{c.uid}.pth")
                    cv2.imshow("MASK",(masks_selfgenerated[c.uid].numpy()*255).astype(np.uint8))
                    cv2.waitKey(1)
                cv2.destroyAllWindows()
        if iteration == opt.reset_confidence_iteration: gaussians.reset_confidence()
                

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # Pop from the start of the FPS-sorted list
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if opt.camera_sampling == "fps":
                viewpoint_stack = densify_utils.sampling_cameras(viewpoint_stack, mode="fps", num_cams=len(viewpoint_stack))
            else:
                import random
                random.shuffle(viewpoint_stack)
        viewpoint_cam = viewpoint_stack.pop(0)
        if len(viewpoint_stack) == 0: viewpoint_stack = None
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration > mesh.distortion_from_iter and mesh.lambda_opacity_field > 0.0:
            splat_args.render_opacity = True

        gt_image = viewpoint_cam.original_image.cuda()
        # not sure we need detach here
        
        render_segmentation = (iteration % opt.contrastive_interval == 0 or iteration % opt.spatial_similarity_interval == 0)
        splat_args.blend_extra_features = gaussians.segmentation_dimension if render_segmentation else 0
        
        if iteration >= opt.deform_first_step:
            deformation = deform_model.deformation(gaussians, viewpoint_cam.uid/(number_views_for_deform_model-1))
        else: deformation = deform_utils.Deformation()
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, splat_args=splat_args, gt_color=gt_image.detach(), deformation=deformation)
        rendering, viewspace_point_tensor, visibility_filter, radii, cov2D = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["cov2D"]
        

        opacity = rendering[7]
        image = rendering[:3, :, :]
        gt_segmentation = viewpoint_cam.seg_mask.cuda().squeeze(0).long()
        
        mask = (masks_selfgenerated[viewpoint_cam.uid] if viewpoint_cam.uid in masks_selfgenerated else (gt_segmentation > 0).float().detach()).cuda()
        mask = mask #* (mask > 0.5)
        
        # custom variance losses
        variance = rendering[11, :, :]
        normal_variance = rendering[12, :, :]
      
        confidence_pp_rgb_loss_mean = None
        confidence_scaled_rgb_loss_mean = None
        confidence_log_term_mean = None
        confidence_neg_alpha_log_term_mean = None
        
        gt_image = gt_image * mask + bg[:, None, None] * (1-mask)
        
        # TODO: don't mean the SSIM
        if mesh.color_confidence and iteration >= mesh.color_confidence_from_iter:  

            # Use a higher minimum to avoid numerical instability in log and gradient computation
            # log(1e-3) ≈ -6.9, which gives gradient of ~200 instead of ~200,000 at minimum
            # This prevents gradient explosion while maintaining the regularization effect
            confidence = torch.clamp(rendering[10, :, :], min=1e-3, max=5.0).unsqueeze(0)

            pp_rgb_loss = appearance_embedding(image, gt_image, viewpoint_cam.idx, mask)           
            alpha = mesh.color_confidence_max

            # Numerically stable: higher minimum clamp prevents extreme gradients
            # Gradient w.r.t. confidence: rgb_loss_og - alpha/confidence
            # At confidence=1e-3: alpha/confidence = 0.2/1e-3 = 200 (reasonable)
            # At confidence=1e-6: alpha/confidence = 0.2/1e-6 = 200,000 (problematic)
            rgb_loss = (pp_rgb_loss * confidence - alpha * torch.log(confidence) * mask)

            # Confidence-specific terms for TensorBoard diagnostics.
            confidence_pp_rgb_loss_mean = pp_rgb_loss.mean()
            confidence_scaled_rgb_loss_mean = (pp_rgb_loss * confidence).mean()
            confidence_log_term_mean = torch.log(confidence).mean()
            confidence_neg_alpha_log_term_mean = (-alpha * torch.log(confidence)).mean()
            # TODO: confidence into a CUDA kernel for speed
        else:
            rgb_loss = appearance_embedding(image, gt_image, viewpoint_cam.idx, mask)
        
        seg_loss_obj = torch.tensor(0,dtype=torch.float).cuda().requires_grad_(True)
        seg_loss_obj_3d = torch.tensor(0,dtype=torch.float).cuda().requires_grad_(True)
        seg_network_loss = torch.tensor(0,dtype=torch.float).cuda().requires_grad_(True)
        #Segmentation Loss
        if render_segmentation:
            #gt_segmask = segmentation_utils.set_bg_to_one_and_class_borders_to_zero(gt_segmentation)
            gt_segmask = gt_segmentation
            feature_map = rendering[-gaussians.segmentation_dimension:rendering.shape[0],:,:]
            assert feature_map.shape[0] == gaussians.segmentation_dimension
            
            if iteration >= opt.segmentation_network_first_step:
                #y_seg_network,seg_network_reg = segmentation_network.forward_with_reg_loss(torch.clamp_min(feature_map.detach().unsqueeze(0),1e-6).log(), True)
                y_seg_network,seg_network_reg = segmentation_network.forward_with_reg_loss(gaussians.segmentation_inverse_activation(feature_map).unsqueeze(0), True)
                seg_network_loss = segmentation_utils.structured_hinge_segmentation(gt_image.unsqueeze(0).detach(), y_seg_network, gt_segmask.unsqueeze(0))*1e-3 + seg_network_reg * 1e-4
            # Clustering Loss
            if iteration % opt.contrastive_interval == 0 and False:
                feature_map = (gaussians.segmentation_inverse_activation(feature_map)).permute(1, 2, 0)
                gt_segmasks = gt_segmask.long()
                id_unique_list, n_i_list = segmentation_utils.get_unique_id_list(gt_segmasks, opt.min_pixnum, segmentation_classes_set)
                seg_loss_obj = segmentation_utils.contrastive_2d_loss(gt_segmasks, feature_map, id_unique_list, n_i_list, segmentation_network.get_prototypes(),lambda_val=opt.contrastive_lambda)

            # Spatial-similarity Loss
            if iteration % opt.spatial_similarity_interval == 0 and False:
                features3d = gaussians.get_segmentation
                seg_loss_obj_3d = segmentation_utils.spatial_loss(gaussians.get_xyz.squeeze().detach().clone(), features3d, k_pull=opt.k_pull, k_push=opt.k_push, lambda_pull=opt.lambda_pull, lambda_push=opt.lambda_push, max_points=opt.reg_max_points, sample_size=opt.reg_sample_size) 
                
            
        occupation = rendering[13:14]
        occupation2 = rendering[14:15]
        
        
            
        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        distortion_loss = distortion_map.mean()
        
        # depth normal consistency
        depth = rendering[6, :, :]
        if depth.isnan().sum() > 0:
            print("DEPTH IS NAN!!!!!")
            depth[depth.isnan()] = 0.0
        
        
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
        
        mask_no_normal1 = (depth_normal == torch.zeros_like(depth_normal[:,0:1,0:1])).all(0,keepdim=True).detach()
        mask_no_normal2 = (render_normal == torch.zeros_like(render_normal[:,0:1,0:1])).all(0,keepdim=True).detach()
        mask_no_normal = mask_no_normal1 | mask_no_normal2
        
        # c2w = (viewpoint_cam.world_view_transform.T).inverse()
        # if we only need the rotation, why bother with the inverse
        c2w = (viewpoint_cam.world_view_transform)
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        
        nabla_I = central_diff(gt_image.permute(1,2,0)).cuda()
        
        normal_error = (1 - (render_normal_world * depth_normal).sum(dim=0))
        depth_normal_loss = (normal_error * (~mask_no_normal.squeeze(0) * mask)).mean()
        
        lambda_distortion = mesh.lambda_distortion if iteration >= mesh.distortion_from_iter else 0.0
        lambda_depth_normal = mesh.lambda_depth_normal if iteration >= mesh.depth_normal_from_iter else 0.0
        
        depth_smoothness_loss = central_diff(depth.unsqueeze(0).permute(1,2,0), ignore_inval=torch.zeros_like(depth[0,0].unsqueeze(0)))
        depth_smoothness_loss = (depth_smoothness_loss*(mask * (depth>0))).mean()
        lambda_depth_smoothness = temp_lambdas["depth_smoothness"] if iteration >= mesh.depth_normal_from_iter else 0.0
            
        # Normal regularization (smoothness)
        normal_loss = central_diff(render_normal.permute(1,2,0), ignore_inval = torch.zeros_like(render_normal[:,0,0])) * torch.exp(-nabla_I)
        normal_loss = (normal_loss*(mask * ~mask_no_normal2)).mean()
        lambda_normal = mesh.lambda_smoothness if iteration >= mesh.depth_normal_from_iter else 0.0

        lambda_opacity_field = mesh.lambda_opacity_field if iteration >= mesh.distortion_from_iter else 0.0
        opa_loss = ((opacity - 0.5)*mask)**2

        #Ll1opacity_smoothness = central_diff(rendering[7][..., None]) * torch.exp(-nabla_I)
        opa_loss = opa_loss.mean()
        
        lambda_extent = mesh.lambda_extent if iteration >= mesh.distortion_from_iter else 0.0
        extent_loss = rendering[9]
        extent_loss = (extent_loss*mask).mean()
        
        rgb_loss_mean = rgb_loss.mean()
        
        lambda_variance = mesh.lambda_variance if iteration >= mesh.variance_from_iter else 0.0
        lambda_normal_variance = mesh.lambda_normal_variance if iteration >= mesh.normal_variance_from_iter else 0.0
        variance_loss = (variance*mask).mean()
        normal_variance_loss = (normal_variance*mask).mean()
        
        #freq_loss = densify_utils.frequency_loss_simple(image, structure_tensor_cache[viewpoint_cam.image_name])
        
        occupation_loss = temp_lambdas["occupation_lambda"] * ((occupation * (1-mask))**2).mean() #if iteration >= mesh.distortion_from_iter else 0
        
        
        normal_consistency_loss = DepthNormalConsistencyLoss((viewpoint_cam.focal_x, viewpoint_cam.focal_y, viewpoint_cam.image_width/2, viewpoint_cam.image_height/2))
        loss_variational_depth_normal_fusion = normal_consistency_loss(depth, render_normal, mask)
        
        # Final loss
        #TODO: Try Variational Depth-Normal Fusion
        
        if iteration < opt.position_lr_max_steps:
            loss =  rgb_loss_mean + \
                    depth_normal_loss    * lambda_depth_normal + \
                    distortion_loss      * lambda_distortion +  \
                    normal_loss          * lambda_normal + \
                    opa_loss             * lambda_opacity_field + \
                    extent_loss          * lambda_extent + \
                    variance_loss        * lambda_variance + \
                    normal_variance_loss * lambda_normal_variance + \
                    (mesh.opacity_reg * torch.abs(gaussians.get_opacity).mean() if mesh.opacity_reg > 0 else 0.0) + \
                    (mesh.scale_reg * (gaussians.get_scaling * gaussians.get_scaling).mean() if mesh.scale_reg > 0 else 0.0) + \
                    (mesh.min_scale_reg * torch.min(gaussians.get_scaling, dim=-1).values.mean() if mesh.min_scale_reg > 0 else 0.0) + \
                    seg_loss_obj + \
                    seg_loss_obj_3d + \
                    occupation_loss  + \
                    (temp_lambdas["variational_depth_normal_fusion_lambda"] if iteration >= mesh.distortion_from_iter else 0) * loss_variational_depth_normal_fusion + \
                    lambda_depth_smoothness * depth_smoothness_loss
                    #freq_loss * opt.lambda_freq
        else:
            loss = rgb_loss_mean
                
        loss.backward(retain_graph=True)
        seg_network_loss.backward()
        
        
        
        # [NEW] Online Accumulation (per camera in batch)
        
        batch.on_frame(loss, rgb_loss_mean, radii, visibility_filter)
        with torch.no_grad():
            if iteration < opt.densify_until_iter and iteration % 10 == 0:
                densify_utils.update_freq_stats_online(viewpoint_cam, gaussians, cov2D, visibility_filter, structure_tensor_cache, viewspace_point_tensor=viewspace_point_tensor, grad_threshold= opt.freq_grad_threshold, transmittance_threshold=opt.freq_transmittance_threshold, opacity_threshold=opt.freq_opacity_threshold, eta_compute_mode=opt.eta_compute_mode) 
        
        
        
        batch_complete_or_last_iter = batch.complete() or iteration == opt.iterations
        
        iter_end.record()
        with torch.no_grad():
            if batch_complete_or_last_iter:
                # Progress bar
                ema_loss_for_log = 0.4 * (batch.batch_loss/max(batch.batch_size, 1)) + 0.6 * ema_loss_for_log
                if (iteration//min(opt.batch_size,5)) % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{len(gaussians._xyz)}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

            # Log and save
            if iteration % 10 == 0 or iteration == opt.iterations:
                training_report(
                    tb_writer=tb_writer,
                    iteration=iteration,
                    rgb_loss=rgb_loss_mean,
                    total_loss=loss,
                    elapsed_ms=iter_start.elapsed_time(iter_end),
                    depth_normal_loss=depth_normal_loss,
                    lambda_depth_normal=lambda_depth_normal,
                    distortion_loss=distortion_loss,
                    lambda_distortion=lambda_distortion,
                    normal_loss=normal_loss,
                    lambda_normal=lambda_normal,
                    opacity_loss=opa_loss,
                    lambda_opacity_field=lambda_opacity_field,
                    extent_loss=extent_loss,
                    lambda_extent=lambda_extent,
                    variance_loss=variance_loss,
                    lambda_variance=lambda_variance,
                    confidence_pp_rgb_loss_mean=confidence_pp_rgb_loss_mean,
                    confidence_scaled_rgb_loss_mean=confidence_scaled_rgb_loss_mean,
                    confidence_log_term_mean=confidence_log_term_mean,
                    confidence_neg_alpha_log_term_mean=confidence_neg_alpha_log_term_mean,
                    seg_loss_obj_3d=None if iteration % opt.spatial_similarity_interval != 0 else seg_loss_obj_3d,
                    seg_loss_obj= None if iteration % opt.contrastive_interval != 0 else seg_loss_obj
                )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, appearance_embedding.capture(), segmentation_network.capture(), deform_model.capture())
                

            # Densification (AbsGrad or MCMC)
            if batch_complete_or_last_iter:
                temp_splat_args = copy.deepcopy(splat_args)
                temp_splat_args.consider_max_weight = True
                render_simp = partial(render, pipe=pipe, bg_color=background, splat_args=temp_splat_args)
                densifier.densify(
                    iteration=iteration,
                    visibility_filter=visibility_filter,
                    radii=radii,
                    viewspace_point_tensor=viewspace_point_tensor,
                    cameras_extent=scene.cameras_extent,
                    trainCameras=trainCameras,
                    render_simp=render_simp
                )

                # Optimizer step
                if iteration < opt.iterations:
                    if iteration < opt.position_lr_max_steps:
                        gaussians.optimizer.step()
                    
                    if iteration >= opt.segmentation_network_first_step:
                        segmentation_network_optim.step()
                        
                    deform_model.optimizer_step(iteration)
                        
                    deform_model.optimizer_zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    segmentation_network_optim.zero_grad(set_to_none = True)
                    densifier.postfix(xyz_lr=xyz_lr)
            #Not entirely correct with batch size, just make sure its a multiple of batch size
            if (iteration in checkpoint_iterations) and iteration != first_iter:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration, appearance_embedding.capture(), (segmentation_network.capture(), segmentation_network_optim.state_dict()), deform_model.capture()), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    end_event = time.time() 
    
    print(f'Training in {end_event - start_event :.4f} seconds!')

def prepare_output_and_logger(args, settings: ExtendedSettings, opt, pipe, mesh):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
        
    # write config file
    with open(os.path.join(args.model_path, "config.json"), 'w') as config_json:
        json.dump(settings.to_dict(), config_json)

    # write output config files for opt, pipe, mesh
    with open(os.path.join(args.model_path, "mesh_args"), 'w') as f:
        f.write(str(Namespace(**vars(mesh))))
    with open(os.path.join(args.model_path, "rem_args"), 'w') as f:
        f.write(str(Namespace(**{**vars(opt), **vars(pipe)})))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_log_dir = os.path.join(args.model_path, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_log_dir)

        scene_name = os.path.basename(os.path.normpath(args.model_path))
        tb_writer.add_text("run/scene_name", scene_name, 0)
        tb_writer.add_text("run/model_path", args.model_path, 0)
        tb_writer.add_text(
            "run/config",
            (
                f"iterations={opt.iterations}\n"
                f"lambda_dssim={opt.lambda_dssim}\n"
                f"lambda_distortion={mesh.lambda_distortion}\n"
                f"lambda_depth_normal={mesh.lambda_depth_normal}\n"
                f"lambda_smoothness={mesh.lambda_smoothness}\n"
                f"lambda_opacity_field={mesh.lambda_opacity_field}\n"
                f"lambda_extent={mesh.lambda_extent}\n"
                f"lambda_variance={mesh.lambda_variance}"
            ),
            0,
        )
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    iteration,
    rgb_loss,
    total_loss,
    elapsed_ms,
    depth_normal_loss,
    lambda_depth_normal,
    distortion_loss,
    lambda_distortion,
    normal_loss,
    lambda_normal,
    opacity_loss,
    lambda_opacity_field,
    extent_loss,
    lambda_extent,
    lambda_variance,
    variance_loss,
    confidence_pp_rgb_loss_mean=None,
    confidence_scaled_rgb_loss_mean=None,
    confidence_log_term_mean=None,
    confidence_neg_alpha_log_term_mean=None,
    seg_loss_obj = None,
    seg_loss_obj_3d = None,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss/rgb_loss", rgb_loss.item(), iteration)
        tb_writer.add_scalar("train_loss/total_loss", total_loss.item(), iteration)
        tb_writer.add_scalar("timing/iter_time_ms", elapsed_ms, iteration)

        tb_writer.add_scalar("regularization/depth_normal", depth_normal_loss.item(), iteration)
        tb_writer.add_scalar("regularization/distortion", distortion_loss.item(), iteration)
        tb_writer.add_scalar("regularization/normal_smoothness", normal_loss.item(), iteration)
        tb_writer.add_scalar("regularization/opacity_field", opacity_loss.item(), iteration)
        tb_writer.add_scalar("regularization/extent", extent_loss.item(), iteration)
        tb_writer.add_scalar("regularization/variance", variance_loss.item(), iteration)
        
        if lambda_depth_normal > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/depth_normal",
                (depth_normal_loss * lambda_depth_normal).item(),
                iteration,
            )
        if lambda_distortion > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/distortion",
                (distortion_loss * lambda_distortion).item(),
                iteration,
            )
        if lambda_normal > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/normal_smoothness",
                (normal_loss * lambda_normal).item(),
                iteration,
            )
        if lambda_opacity_field > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/opacity_field",
                (opacity_loss * lambda_opacity_field).item(),
                iteration,
            )
        if lambda_extent > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/extent",
                (extent_loss * lambda_extent).item(),
                iteration,
            )
        if lambda_variance > 0.0:
            tb_writer.add_scalar(
                "weighted_regularization/variance",
                (variance_loss * lambda_variance).item(),
                iteration,
            )

        if confidence_pp_rgb_loss_mean is not None:
            tb_writer.add_scalar(
                "confidence_terms/pp_rgb_loss_mean",
                confidence_pp_rgb_loss_mean.item(),
                iteration,
            )
        if confidence_scaled_rgb_loss_mean is not None:
            tb_writer.add_scalar(
                "confidence_terms/pp_rgb_loss_scaled_mean",
                confidence_scaled_rgb_loss_mean.item(),
                iteration,
            )
        if confidence_log_term_mean is not None:
            tb_writer.add_scalar(
                "confidence_terms/log_confidence_mean",
                confidence_log_term_mean.item(),
                iteration,
            )
        if confidence_neg_alpha_log_term_mean is not None:
            tb_writer.add_scalar(
                "confidence_terms/neg_alpha_log_confidence_mean",
                confidence_neg_alpha_log_term_mean.item(),
                iteration,
            )
            
        if seg_loss_obj is not None:
            tb_writer.add_scalar(
                "segmentation_terms/loss_obj",
                seg_loss_obj.item(),
                iteration,
            )
        if seg_loss_obj_3d is not None:
            tb_writer.add_scalar(
                "segmentation_terms/loss_obj_3d",
                seg_loss_obj_3d.item(),
                iteration,
            )
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    mp = MeshingParams(parser)
    ss = SplattingSettings(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    splat_args = ss.get_settings(args)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), mp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, splat_args)

    # All done
    print("\nTraining complete.")
