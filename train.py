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
import torch.nn.functional as F
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
from utils.depth_utils import depths_to_points, depth_to_normal, depth_to_normal_scharr, central_diff, central_diff_normals, DepthNormalConsistencyLoss, intrinsics_from_view
from utils.vis_utils import gui_visualize, export_image
from utils import segmentation_utils
from utils import deform_utils
from utils import densify_utils
import utils.normal_field 
import utils.multiview
import pytorch3d.structures
import pytorch3d.ops
from scene.gaussian_model import build_scaling_rotation
from diff_gaussian_rasterization import ExtendedSettings, DebugVisualization, DebugVisualizationType
from decoupled_fused_ssim import fused_ssim
import numpy as np
from scene.appearance_network import AppearanceEmbedding, VastGaussianAppearanceEmbedding, SSIMDecoupledAppearanceEmbedding
from functools import partial
import copy
from scene.densifier import AbsGradDensifier, MCMCDensifier, MSv2AbsGradDensifier, CustomDensifier, NormalDensifier
import warnings

RED = '\033[31m'
RESET = '\033[0m'

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
    
   
def dilate(binary_image, kernel_size=3):
    pad = kernel_size // 2

    x = binary_image.float()[None, None]

    y = F.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )

    return y[0, 0]


def erode(binary_image, kernel_size=3):
    pad = kernel_size // 2

    x = binary_image.float()[None, None]

    y = -F.max_pool2d(
        -x,
        kernel_size=kernel_size,
        stride=1,
        padding=pad
    )

    return y[0, 0] 
    
def closing(binary_image, kernel_size):
    return erode(dilate(binary_image,kernel_size), kernel_size)

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


        

#n in 3xH,W, mask in 1xHxW
def curvature_prior(normal_map, mask):
    """
    Computes the Anisotropic TV-L1 norm on a surface normal map.
    
    Args:
        normal_map: Tensor of shape (3, H, W)
        mask: Optional boolean Tensor of shape (H, W)
    """
    # Vertical differences (along H-axis / dim 1)
    diff_v = torch.abs(normal_map[:, 1:, :] - normal_map[:, :-1, :])
    
    # Horizontal differences (along W-axis / dim 2)
    diff_h = torch.abs(normal_map[:, :, 1:] - normal_map[:, :, :-1])
    
    if mask is not None:
        # Create masks for valid adjacent pairs
        mask_v = mask[1:, :] & mask[:-1, :]
        mask_h = mask[:, 1:] & mask[:, :-1]
        
        # Masking a (3, H, W) tensor with an (H, W) boolean mask 
        # yields a (3, N) tensor, which we then sum up
        loss_v = (diff_v * mask_v[None,...]).mean()
        loss_h = (diff_h * mask_h[None,...]).mean()
        
        # Total valid items = (number of valid pairs) * 3 channels
        tv_loss = (loss_v + loss_h)
    else:
        # Default mean reduction across the entire tensor
        tv_loss = diff_v.mean() + diff_h.mean()
        
    return tv_loss


class LossPart:
    def __init__(self, key, lamda, first_iter=0, last_iter=None, cond=lambda iter: True, loss_key:int=0):
        self._key = key
        self._value = torch.tensor(0,dtype=torch.float).cuda().requires_grad_(True)
        self._lamda = lamda
        self._first_iter = first_iter
        self._last_iter = last_iter
        self._cond = cond
        self._loss_key = loss_key
        
    def __repr__(self):
        return (
            f"'{self._loss_key}){self._key}': " + \
            f"{self._value.item():5.5f}, "
            f"lambda={self._lamda:5.5f}, "
            f"first_iter={self._first_iter}, "
            f"last_iter={self._last_iter}"
            f")"
        ) 
        
    def lamda_is_zero(self): return self._lamda == 0
    def get_value_raw(self): return self._value
    def get_key(self): return self._key
    def get_value(self): return self._value * self._lamda
    def set_value(self, val): self._value = val
    def get_loss_key(self): return self._loss_key
    def reset_value(self): self._value = torch.tensor(0,dtype=torch.float).cuda().requires_grad_(True)
    def set_lamda(self, lamda): self._lamda = lamda
    def is_active(self, iteration):
        cond_iter = (iteration >= self._first_iter and (True if self._last_iter is None else iteration <= self._last_iter))
        cond_lamda = self._lamda != 0.0
        return cond_iter and cond_lamda and self._cond(iteration)
            
class LossCollection:
    def __init__(self, opt, mesh, temp_lambdas):
        self.losses = {}
        self.rgb = LossPart("rgb", 1.0, first_iter=0, last_iter=None)
        self.seg_obj = LossPart("seg_obj", 1.0, first_iter=opt.contrastive_interval, last_iter=None, cond=lambda iter: (iter % opt.contrastive_interval == 0) and False)
        self.seg_obj3d = LossPart("seg_obj3d", 1.0, first_iter=opt.spatial_similarity_interval, last_iter=None, cond=lambda iter: (iter % opt.spatial_similarity_interval == 0) and False)
        self.seg_network = LossPart("seg_network", 1.0, first_iter=0, last_iter=None, cond=lambda iter: (iter % opt.spatial_similarity_interval == 0) or (iter % opt.contrastive_interval == 0) and False, loss_key=1)
        self.curvature = LossPart("curvature", temp_lambdas["curvature_lambda"], first_iter=mesh.distortion_from_iter, last_iter=None)
        self.depth_normal = LossPart("depth_normal", mesh.lambda_depth_normal, first_iter=mesh.depth_normal_from_iter, last_iter=None)
        self.smoothness_normal = LossPart("smoothness_normal", mesh.lambda_smoothness, first_iter=mesh.depth_normal_from_iter, last_iter=None)
        self.opacity = LossPart("opacity", mesh.lambda_opacity_field, first_iter=mesh.distortion_from_iter, last_iter=None)
        self.extent = LossPart("extent", mesh.lambda_extent, first_iter=mesh.distortion_from_iter, last_iter=None)
        self.distortion = LossPart("distortion", mesh.lambda_distortion, first_iter=mesh.distortion_from_iter, last_iter=None)
        self.learned_normal = LossPart("learned_normal", temp_lambdas["learned_normal_error"], first_iter=mesh.normal_field_from_iter, last_iter=None, cond=lambda iter: True and mesh.use_normal_field)
        self.variance = LossPart("variance", mesh.lambda_variance, mesh.variance_from_iter, None)
        self.normal_variance = LossPart("normal_variance", mesh.lambda_normal_variance, mesh.normal_variance_from_iter, None)
        self.occupation = LossPart("occupation", temp_lambdas["occupation_lambda"], 0, None)
        self.occupation2 = LossPart("occupation2", temp_lambdas["occupation2_lambda"], mesh.depth_normal_from_iter, None, cond=lambda iter: opt.binary_search_depth)
        self.occupation_variance = LossPart("occupation_variance", temp_lambdas["occupation_var_lambda"], mesh.variance_from_iter, None)
        self.variance = LossPart("variance", mesh.lambda_variance, mesh.variance_from_iter, None)
        self.opacity_reg = LossPart("opacity_reg", mesh.opacity_reg, 0, None)
        self.scale_reg = LossPart("scale_reg", mesh.scale_reg, 0, None)
        self.min_scale_reg = LossPart("min_scale_reg", mesh.min_scale_reg, 0, None)
        self.points_3d_reg = LossPart("points_3d_reg", temp_lambdas["points_3d_reg"], mesh.depth_normal_from_iter, None)
        
        self._register_members()

    def _register_members(self):
        self.losses = {}

        for name, value in vars(self).items():
            if isinstance(value, LossPart):
                key = value.get_key()

                if key in self.losses:
                    raise ValueError(f"Duplicate loss key: {key}")
                self.losses[key] = value
    
    def register_loss_parts(self,parts: LossPart): 
        for p in parts:
            assert p.get_key() not in self.losses
            self.losses[p.get_key()] = p
    def register_loss_part(self,part: LossPart): 
        self.register_loss_parts([part])
    def unregister_loss_part(self,key): 
        assert key in self.losses
        self.losses.pop(key)
    def reset_values(self):
        for key in self.losses:
            self.losses[key].reset_value()
    def compute(self, iteration, loss_key=0):
        val = torch.tensor(0,dtype=torch.float).cuda().requires_grad_(True)
        for key in self.losses:
            part = self.losses[key]
            assert part.get_key() == key, "Key not matching"
            if part.get_loss_key() != loss_key: continue
            if part.is_active(iteration):
                val=val+part.get_value()
        return val



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
    
    normal_field_config = {}
    normal_densifier = NormalDensifier(gaussians, opt, mesh, dataset, pipe, scene.cameras_extent, normal_field_config)
    
    if not mesh.use_structure_tensor_densification:
        opt.lambda_l2 = 0
    if mesh.use_structure_tensor_densification:
        densifier = CustomDensifier(gaussians, opt, mesh, dataset, pipe, scene.cameras_extent, [int(opt.densify_until_iter*0.5), int(opt.densify_until_iter * 0.95), int(opt.densify_until_iter*1.25)])
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
            self.batch_rgb_loss += rgb_loss.get_value().item()
            self.last_radii = radii
            self.last_visibility_filter = visibility_filter
            self.batch_size+=1
            
    
    masks_selfgenerated = {}
    
    temp_lambdas = {
        "occupation_lambda": 0.1 * 0,
        "occupation2_lambda": 0.00,
        "occupation_var_lambda": 0.00,
        "variational_depth_normal_fusion_lambda": 0.000,
        "depth_smoothness": 0.0 * (1/scene.cameras_extent),
        "surface_L_lambda": 0,
        "learned_normal_error": (1.0 if mesh.use_normal_field else 0) * mesh.lambda_depth_normal * 0.6, #thats what they have even though they set it to 0.05 ,but they also have some ratio thats 0.6
        "detach_depth_normal": False,
        "curvature_lambda": 0.1 * 0,
        "points_3d_reg": 0.0
        }
    batch = None
    
    for iteration in checkpoint_iterations: assert (iteration % opt.batch_size) == 0, "Must be a multiple of batch size"
        
    
    gaussians.compute_3D_filter(cameras=trainCameras, CUDA=not pipe.compute_filter3D_python)
    
    
    
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    
    multi_view_state = utils.multiview.initialize_multiview_regularization(scene, pipe, 0.0, mesh)
    
    for iteration in range(first_iter, opt.iterations + 1):        
        loss_collection = LossCollection(opt, mesh, temp_lambdas)
        loss_collection.reset_values()
        
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
                        splat_args.render_learned_normals = message["custom_message"].startswith("learned_normals")
                        splat_args.blend_extra_features = (gaussians.segmentation_dimension if render_segmentation else 0) + (3 if splat_args.render_learned_normals else 0)
                        
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
                            learned_normals=net_image[15:18] if splat_args.render_learned_normals else None,
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
        normal_field_kick_on = (iteration >= mesh.normal_field_from_iter) and mesh.use_normal_field
        if iteration == mesh.normal_field_from_iter: gaussians.reset_learned_normal_features()
        if iteration == opt.reset_confidence_iteration: gaussians.reset_confidence()
        if normal_field_kick_on: splat_args.render_learned_normals = True
        if iteration > opt.densify_until_iter: appearance_embedding.lambda_l2 = 0
        
                    
        if iteration % opt.self_generated_masks_interval == 0 and iteration > 0 or (iteration == first_iter and first_iter >= opt.self_generated_masks_interval):
            with torch.no_grad():
                all_cameras = scene.getTrainCameras().copy()
                import cv2
                cv2.namedWindow("MASK", cv2.WINDOW_NORMAL)
                for c in tqdm(all_cameras):
                    splat_args.render_learned_normals = False
                    splat_args.render_geometry = opt.binary_search_depth
                    splat_args.blend_extra_features = 0
                    render_pkg = render(c, gaussians, pipe, bg, splat_args=splat_args, gt_color=None, deformation=deform_utils.Deformation(), extract_final_T=True)
                    mask = 1.0-render_pkg["final_T"].squeeze(0)
                    masks_selfgenerated[c.uid] = closing(mask.detach()>0.15, 5).float().cpu() #torch.nn.functional.sigmoid((mask.detach()- 0.9) * 20).cpu()  
                    if os.path.exists("/tmp/test") and os.path.isdir("/tmp/test"):
                        D = render_pkg["render"][6].detach().cpu()
                        N = render_pkg["render"][3:6].detach().cpu()
                        C = render_pkg["render"][:3].detach().cpu()
                        C2 = c.original_image.detach().cpu()
                        K = intrinsics_from_view(c).cpu()
                        torch.save((D,N,C, C2, c.world_view_transform.cpu().detach(),K),f"/tmp/test/{c.uid}.pth")
                    cv2.imshow("MASK",(masks_selfgenerated[c.uid].numpy()*255).astype(np.uint8))
                    cv2.waitKey(1)
            cv2.destroyAllWindows()
            
                

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            assert opt.iterations >= 15_000
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
            
        
        splat_args.render_geometry = False
        if (iteration >= mesh.depth_normal_from_iter and mesh.lambda_depth_normal > 0.0) or normal_field_kick_on:
            splat_args.render_geometry = opt.binary_search_depth
        if iteration >= mesh.distortion_from_iter and mesh.lambda_opacity_field > 0.0 and not splat_args.render_geometry:
            splat_args.render_opacity = True

        gt_image = viewpoint_cam.original_image.cuda()
        
        #disabled for now 
        render_segmentation = (loss_collection.seg_obj.is_active(iteration) or loss_collection.seg_obj3d.is_active(iteration) or loss_collection.seg_network.is_active(iteration)) and False
        assert not render_segmentation, "CUDA KERNEL NEEDS TO BE ADAPTED AGAIN"
        splat_args.blend_extra_features = (gaussians.segmentation_dimension if render_segmentation else 0 + (3 if splat_args.render_learned_normals else 0))
        
        if iteration >= opt.deform_first_step:
            deformation = deform_model.deformation(gaussians, viewpoint_cam.uid/(number_views_for_deform_model-1))
        else: deformation = deform_utils.Deformation()
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, splat_args=splat_args, gt_color=gt_image.detach(), deformation=deformation)
        rendering, viewspace_point_tensor, visibility_filter, radii, cov2D = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["cov2D"]
        
        
        #multiview_render_fn = partial(render, splat_args=splat_args, gt_color=None, deformation=deformation)
        #losses["multiview"] = utils.multiview.compute_multiview_regularization(iteration, scene, render_pkg, viewpoint_cam, viewpoint_cam.idx, gaussians, multiview_render_fn, pipe, bg, mesh, multi_view_state, 0.0)

        image = rendering[:3, :, :]
        gt_segmentation = viewpoint_cam.seg_mask.cuda().squeeze(0).long()
        
        mask = (masks_selfgenerated[viewpoint_cam.uid] if viewpoint_cam.uid in masks_selfgenerated else (gt_segmentation > 0).float().detach()).cuda()
        mask = mask #* (mask > 0.5)
        
        # custom variance losses
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
            loss_collection.rgb.set_value((pp_rgb_loss * confidence - alpha * torch.log(confidence) * mask).mean())

            # Confidence-specific terms for TensorBoard diagnostics.
            confidence_pp_rgb_loss_mean = pp_rgb_loss.mean()
            confidence_scaled_rgb_loss_mean = (pp_rgb_loss * confidence).mean()
            confidence_log_term_mean = torch.log(confidence).mean()
            confidence_neg_alpha_log_term_mean = (-alpha * torch.log(confidence)).mean()
            # TODO: confidence into a CUDA kernel for speed
        else:
            loss_collection.rgb.set_value(appearance_embedding(image, gt_image, viewpoint_cam.idx, mask).mean())
        
        
        ### SEGMENTATION_LOSS
        if render_segmentation:
            #gt_segmask = segmentation_utils.set_bg_to_one_and_class_borders_to_zero(gt_segmentation)
            gt_segmask = gt_segmentation
            feature_map = rendering[-gaussians.segmentation_dimension:rendering.shape[0],:,:]
            assert feature_map.shape[0] == gaussians.segmentation_dimension
            
            if iteration >= opt.segmentation_network_first_step:
                #y_seg_network,seg_network_reg = segmentation_network.forward_with_reg_loss(torch.clamp_min(feature_map.detach().unsqueeze(0),1e-6).log(), True)
                y_seg_network,seg_network_reg = segmentation_network.forward_with_reg_loss(gaussians.segmentation_inverse_activation(feature_map).unsqueeze(0), True)
                loss_collection.seg_network.set_value(segmentation_utils.structured_hinge_segmentation(gt_image.unsqueeze(0).detach(), y_seg_network, gt_segmask.unsqueeze(0))*1e-3 + seg_network_reg * 1e-4)
            # Clustering Loss
            if iteration % opt.contrastive_interval == 0 and False:
                feature_map = (gaussians.segmentation_inverse_activation(feature_map)).permute(1, 2, 0)
                gt_segmasks = gt_segmask.long()
                id_unique_list, n_i_list = segmentation_utils.get_unique_id_list(gt_segmasks, opt.min_pixnum, segmentation_classes_set)
                loss_collection.seg_obj.set_value(segmentation_utils.contrastive_2d_loss(gt_segmasks, feature_map, id_unique_list, n_i_list, segmentation_network.get_prototypes(),lambda_val=opt.contrastive_lambda))

            # Spatial-similarity Loss
            if iteration % opt.spatial_similarity_interval == 0 and False:
                features3d = gaussians.get_segmentation
                loss_collection.seg_obj3d.set_value(segmentation_utils.spatial_loss(gaussians.get_xyz.squeeze().detach().clone(), features3d, k_pull=opt.k_pull, k_push=opt.k_push, lambda_pull=opt.lambda_pull, lambda_push=opt.lambda_push, max_points=opt.reg_max_points, sample_size=opt.reg_sample_size) )
                
        ### END_SEGMENTATION_LOSS
            
        c2w = (viewpoint_cam.world_view_transform)
        render_normal = rendering[3:6]
        depth = rendering[6]
        opacity = rendering[7]
        distortion_map = rendering[8, :, :]
        extent = rendering[9]
        variance = rendering[11]
        normal_variance = rendering[12]
        occupation = rendering[13:14]
        occupation2 = rendering[14:15]
        occupation_var = occupation2 - occupation*occupation
        learned_normals = None
        if splat_args.render_learned_normals:
            #Don't normalize them!
            learned_normals = rendering[15:18]
        
        # depth normal consistency
        if depth.isnan().sum() > 0:
            print("DEPTH IS NAN!!!!!")
            depth[depth.isnan()] = 0.0
            
        
        
        #for tensor board we keep extra variables apart from is_active
        compute_curvature_loss = loss_collection.curvature.is_active(iteration)
        compute_learned_normal_loss = loss_collection.learned_normal.is_active(iteration)
        compute_depth_normal_loss = loss_collection.depth_normal.is_active(iteration)
        compute_smoothness_loss = loss_collection.smoothness_normal.is_active(iteration)
        compute_distortion_loss = loss_collection.distortion.is_active(iteration)
        compute_extent_loss = loss_collection.extent.is_active(iteration)
        compute_opacity_loss = loss_collection.opacity.is_active(iteration)
        compute_variance_loss = loss_collection.variance.is_active(iteration)
        compute_normal_variance_loss = loss_collection.normal_variance.is_active(iteration)
        compute_occupation_loss = loss_collection.occupation.is_active(iteration)
        compute_occupation2_loss = loss_collection.occupation2.is_active(iteration)
        compute_occupation_variance_loss = loss_collection.occupation_variance.is_active(iteration)
        compute_opacity_reg = loss_collection.opacity_reg.is_active(iteration)
        compute_scale_reg = loss_collection.scale_reg.is_active(iteration)
        compute_min_scale_reg = loss_collection.min_scale_reg.is_active(iteration)
        compute_points_3d_reg = loss_collection.points_3d_reg.is_active(iteration)
        
        depth_normal_needed = compute_learned_normal_loss or compute_depth_normal_loss or compute_curvature_loss
        render_normal_world_needed = compute_depth_normal_loss
        render_normal_needed = compute_smoothness_loss
        nabla_I_needed = compute_smoothness_loss
        
        if depth_normal_needed:
            depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None,...],cam_space=False, mask= depth>0)
            depth_normal = depth_normal.permute(2, 0, 1)
            mask_no_normal_depth = (depth_normal == torch.zeros_like(depth_normal[:,0:1,0:1])).all(0,keepdim=True).detach()
        else: 
            depth_normal = None
            mask_no_normal_depth = None
            
        if compute_points_3d_reg:
            points_3d = depths_to_points(viewpoint_cam, depth[None, ...], cam_space=False)
            pcls = pytorch3d.structures.Pointclouds(points_3d[points_3d[:,2]>0].unsqueeze(0))
            pcls_gaussian = pytorch3d.structures.Pointclouds(gaussians.get_xyz.unsqueeze(0))
            points_3d_reg = pytorch3d.ops.ball_query(pcls.points_packed().unsqueeze(0), pcls_gaussian.points_packed().unsqueeze(0), K=5, radius=0.1,skip_points_outside_cube=True, return_nn=False).dists.mean()
            loss_collection.points_3d_reg.set_value(points_3d_reg)
        
        if render_normal_needed or render_normal_world_needed:
            render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)
            mask_no_normal_render = (render_normal == torch.zeros_like(render_normal[:,0:1,0:1])).all(0,keepdim=True).detach()
            render_normal_world = c2w[:3, :3] @ render_normal.reshape(3, -1)
            render_normal_world = render_normal_world.reshape(3, *render_normal.shape[1:])
            if not render_normal_needed: render_normal = None
        else:
            render_normal = None
            render_normal_world = None
            mask_no_normal_render = None
            
        nabla_I = None if not nabla_I_needed else central_diff(gt_image.permute(1,2,0)).cuda()
        
        
        if compute_depth_normal_loss:
            normal_error = (1 - (render_normal_world * depth_normal).sum(dim=0))
            epsilon_depth_normal_error = 0.000  # Tune this (e.g., 0.001 to 0.005)
            if epsilon_depth_normal_error != 0:
                normal_error = torch.clamp(normal_error - epsilon_depth_normal_error, min=0.0)
            loss_collection.depth_normal.set_value((normal_error * (~(mask_no_normal_depth | mask_no_normal_render).squeeze(0) * mask)).mean())
            
            
        if compute_learned_normal_loss:
            learned_normals_world = c2w[:3, :3] @ learned_normals.reshape(3, -1)
            learned_normals_world = learned_normals_world.reshape(3, *learned_normals.shape[1:])
            loss_collection.learned_normal.set_value(
                ((1 - (learned_normals_world * (depth_normal if not temp_lambdas["detach_depth_normal"] else depth_normal.detach())).sum(dim=0)) * (~(mask_no_normal_depth).squeeze(0) * mask)).mean()
            )
        
        
        if compute_curvature_loss: loss_collection.curvature.set_value(((curvature_prior(depth_normal, ~mask_no_normal_depth[0]))*mask).mean())
            
        # Normal regularization (smoothness)
        if compute_smoothness_loss:
            loss_collection.smoothness_normal.set_value(((central_diff(render_normal.permute(1,2,0), ignore_inval = torch.zeros_like(render_normal[:,0,0]), return_squared_norm=True) * torch.exp(-nabla_I)) *(mask * ~mask_no_normal_render)).mean())

        if compute_opacity_loss: loss_collection.opacity.set_value((((opacity - 0.5)*mask)**2).mean())
        if compute_extent_loss: loss_collection.extent.set_value((extent*mask).mean())
        if compute_distortion_loss: loss_collection.distortion.set_value((distortion_map*mask).mean())
        
        if compute_variance_loss: loss_collection.variance.set_value((variance*mask).mean())
        if compute_normal_variance_loss: loss_collection.normal_variance.set_value((normal_variance*mask).mean())
        
        
        #freq_loss = densify_utils.frequency_loss_simple(image, structure_tensor_cache[viewpoint_cam.image_name])
        
        if compute_occupation_loss: loss_collection.occupation.set_value(((occupation * (1-mask))**2).mean())
        if compute_occupation2_loss: loss_collection.occupation2.set_value((((occupation2.detach()-depth)**2 * (mask))).mean())
        if compute_occupation_variance_loss: loss_collection.occupation_variance.set_value(((occupation_var * (mask))**2).mean())
        
        if compute_opacity_reg: loss_collection.opacity_reg.set_value(gaussians.get_opacity.abs().mean())
        if compute_scale_reg: loss_collection.scale_reg.set_value((gaussians.get_scaling * gaussians.get_scaling).mean())
        if compute_min_scale_reg: loss_collection.min_scale_reg.set_value(torch.min(gaussians.get_scaling, dim=-1).values.mean())
        
        
        loss = loss_collection.compute(iteration,loss_key=0)
        seg_network_loss = loss_collection.compute(iteration, loss_key=1)
        
        loss.backward(retain_graph=True)
        seg_network_loss.backward(retain_graph=True)
        
        
        
        # [NEW] Online Accumulation (per camera in batch)
        batch.on_frame(loss, loss_collection.rgb, radii, visibility_filter)
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
                    loss_collection=loss_collection,
                    total_loss=loss,
                    elapsed_ms=iter_start.elapsed_time(iter_end),
                    confidence_pp_rgb_loss_mean=confidence_pp_rgb_loss_mean,
                    confidence_scaled_rgb_loss_mean=confidence_scaled_rgb_loss_mean,
                    confidence_log_term_mean=confidence_log_term_mean,
                    confidence_neg_alpha_log_term_mean=confidence_neg_alpha_log_term_mean,
                )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, appearance_embedding.capture(), segmentation_network.capture(), deform_model.capture())
                

            # Densification (AbsGrad or MCMC)
            gaussians_have_changed = False
            if batch_complete_or_last_iter:
                temp_splat_args = copy.deepcopy(splat_args)
                temp_splat_args.consider_max_weight = True
                render_simp = partial(render, pipe=pipe, bg_color=background, splat_args=temp_splat_args)
                gaussians_have_changed, computed_filter3d = densifier.densify(
                    iteration=iteration,
                    visibility_filter=visibility_filter,
                    radii=radii,
                    viewspace_point_tensor=viewspace_point_tensor,
                    cameras_extent=scene.cameras_extent,
                    trainCameras=trainCameras,
                    render_simp=render_simp
                )
                
                ##CODE INSERT
                
                # ---Normal Field Densification---
                if mesh.use_normal_field and normal_densifier is not None:
                    gaussians_have_changed_normal_densifier, computed_filter3d_normal_densifier = normal_densifier.densify(iteration, scene.getTrainCameras().copy(), bg, 
                                                                                                            None, None, normal_field_kick_on, normal_field_config)
                    assert not computed_filter3d_normal_densifier, "Should do it as it doesn't know what the other densifier did"
                    assert not gaussians_have_changed_normal_densifier and gaussians_have_changed, "Hmm, how about the 3d filter"
                    gaussians_have_changed = gaussians_have_changed or gaussians_have_changed_normal_densifier
                    
                            
                            

                if gaussians_have_changed and not computed_filter3d: 
                    gaussians.compute_3D_filter(scene.getTrainCameras().copy(), CUDA=not pipe.compute_filter3D_python)
                    if normal_densifier is not None and mesh.use_normal_field: normal_densifier.reset_normal_field_state_at_next_iteration()
                elif opt.densify_until_iter < iteration and iteration % 100 == 0:
                    gaussians.compute_3D_filter(scene.getTrainCameras().copy(), CUDA=not pipe.compute_filter3D_python)
                    
                #END CODE INSERT

                # Optimizer step
                if iteration < opt.iterations:
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
    loss_collection: LossCollection,
    total_loss,
    elapsed_ms,
    confidence_pp_rgb_loss_mean=None,
    confidence_scaled_rgb_loss_mean=None,
    confidence_log_term_mean=None,
    confidence_neg_alpha_log_term_mean=None,
):
    if tb_writer:
        for key in loss_collection.losses:
            part = loss_collection.losses[key]
            tb_writer.add_scalar(f"loss_parts/{key}", part.get_value_raw().item(), iteration)
            tb_writer.add_scalar(f"loss_parts_weighted/{key}", part.get_value().item(), iteration)
        tb_writer.add_scalar("train_loss/total_loss", total_loss.item(), iteration)
        tb_writer.add_scalar("timing/iter_time_ms", elapsed_ms, iteration)

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
