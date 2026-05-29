from scene import GaussianModel
from arguments import OptimizationParams, PipelineParams, MeshingParams, ModelParams
from scene.gaussian_model import build_scaling_rotation
import torch
import numpy as np
from scene import Scene

class Densifier:
    gaussians : GaussianModel = None
    def __init__(self, gaussians : GaussianModel, opt : OptimizationParams, mp : MeshingParams, dataset : ModelParams, pipe : PipelineParams):
        self.gaussians = gaussians
        self.opt = opt
        self.mp = mp
        self.dataset = dataset
        self.pipe = pipe
    
    def densify(self, iteration : int, **kwargs):
        raise NotImplementedError()
    
    def postfix(self, xyz_lr : float, **kwargs):
        raise NotImplementedError()
    
# The following densification logic is based on Gaussian Opacity Fields (https://github.com/autonomousvision/gaussian-opacity-fields):
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/5245b20e5d11acd6d1ff5af4b890dc2bedd99693/train.py#L253
class AbsGradDensifier(Densifier):   
    def densify(self, iteration: int, **kwargs):
        visibility_filter = kwargs.get("visibility_filter")
        radii = kwargs.get("radii")
        viewspace_point_tensor = kwargs.get("viewspace_point_tensor")
        cameras_extent = kwargs.get("cameras_extent")
        trainCameras = kwargs.get("trainCameras")
        
        opt = self.opt
        mesh = self.mp
        dataset = self.dataset
        
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #GOF: use 0.05 min opacity instead of 0.005
                self.gaussians.densify_and_prune(opt.densify_grad_threshold, mesh.prune_threshold, cameras_extent, size_threshold, radii,
                                            abs_grad_for_densification=mesh.abs_grad_for_densification,
                                            clone_with_sampling=mesh.clone_with_sampling)
                # we need to compute the 3D filter here for reasons (see reset_opacity())
                self.gaussians.compute_3D_filter(trainCameras, CUDA=not self.pipe.compute_filter3D_python)
                
            if mesh.opacity_decay == 0 and iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                self.gaussians.reset_opacity()
        
            if mesh.opacity_decay != 0 and iteration % 50 == 0 and iteration > opt.densify_from_iter:
                self.gaussians.decay_opacity(mesh.opacity_decay)
                
        if iteration % 100 == 0 and iteration > opt.densify_until_iter and iteration < opt.iterations - 100:
            self.gaussians.compute_3D_filter(trainCameras, CUDA=not self.pipe.compute_filter3D_python)
                
    def postfix(self, xyz_lr : float, **kwargs):
        pass
    
# The following densification logic is based on 3DGS-MCMC (https://github.com/ubc-vision/3dgs-mcmc):
# https://github.com/ubc-vision/3dgs-mcmc/blob/7b4fc9f76a1c7b775f69603cb96e70f80c7e6d13/train.py#L124
class MCMCDensifier(Densifier):
    def densify(self, iteration: int, **kwargs):
        opt = self.opt
        mesh = self.mp
        dataset = self.dataset
        
        trainCameras = kwargs.get("trainCameras")
        
        if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            dead_mask = (self.gaussians.get_opacity <= 0.005).squeeze(-1)
            self.gaussians.relocate_gs(dead_mask=dead_mask)
            self.gaussians.add_new_gs(cap_max=mesh.cap_max)
            
        # Mip-Splatting
        if iteration > opt.densify_from_iter and iteration % 100 == 0 and iteration < opt.iterations - 100:
            self.gaussians.compute_3D_filter(cameras=trainCameras, CUDA=not self.pipe.compute_filter3D_python)
    
    def op_sigmoid(self, x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    def postfix(self, xyz_lr : float, **kwargs):
        mesh = self.mp
        
        L = build_scaling_rotation(self.gaussians.get_scaling, self.gaussians.get_rotation)
        actual_covariance = L @ L.transpose(1, 2)
        
        noise = torch.randn_like(self.gaussians._xyz) * (self.op_sigmoid(1- self.gaussians.get_opacity))*mesh.noise_lr*xyz_lr
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        self.gaussians._xyz.add_(noise)

# The following densification logic is based on Gaussian Opacity Fields (https://github.com/autonomousvision/gaussian-opacity-fields):
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/5245b20e5d11acd6d1ff5af4b890dc2bedd99693/train.py#L253
class MSv2AbsGradDensifier(Densifier):   
    def densify(self, iteration: int, **kwargs):
        visibility_filter = kwargs.get("visibility_filter")
        radii = kwargs.get("radii")
        viewspace_point_tensor = kwargs.get("viewspace_point_tensor")
        cameras_extent = kwargs.get("cameras_extent")
        trainCameras = kwargs.get("trainCameras")
        render_simp = kwargs.get("render_simp")
        
        opt = self.opt
        mesh = self.mp
        dataset = self.dataset
        
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #GOF: use 0.05 min opacity instead of 0.005
                self.gaussians.densify_and_prune(opt.densify_grad_threshold, mesh.prune_threshold, cameras_extent, size_threshold, radii,
                                            abs_grad_for_densification=mesh.abs_grad_for_densification,
                                            clone_with_sampling=mesh.clone_with_sampling)
                # we need to compute the 3D filter here for reasons (see reset_opacity())
                self.gaussians.compute_3D_filter(trainCameras, CUDA=not self.pipe.compute_filter3D_python)
                
            if mesh.opacity_decay == 0 and iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                self.gaussians.reset_opacity()
        
            if mesh.opacity_decay != 0 and iteration % 50 == 0 and iteration > opt.densify_from_iter:
                self.gaussians.decay_opacity(mesh.opacity_decay)
        else:
            if iteration == 15000:
                self.gaussians.culling_with_interesction_preserving(trainCameras, render_simp)
                torch.cuda.empty_cache()
            elif iteration == 20000:
                self.gaussians.culling_with_interesction_sampling(trainCameras, render_simp)
                torch.cuda.empty_cache()
                    
        if iteration % 100 == 0 and iteration > opt.densify_until_iter and iteration < opt.iterations - 100:
            self.gaussians.compute_3D_filter(trainCameras, CUDA=not self.pipe.compute_filter3D_python)
    
    def postfix(self, xyz_lr : float, **kwargs):
        pass
    
    
    
class NormalDensifier(Densifier):   
    def __init__(self, gaussians : GaussianModel, opt : OptimizationParams, mp : MeshingParams, dataset : ModelParams, pipe : PipelineParams, cameras_extent:float, normal_field_config:dict):
        super().__init__(gaussians, opt, mp, dataset, pipe)
        self.cameras_extent = cameras_extent
        self.normal_field_config = normal_field_config
        self.initialize_normal_field()
    @torch.no_grad()
    def prune_non_maximal_gaussians(
        self,
        cameras,
        background: torch.Tensor,
        render_depth
    ):
        is_maximal = torch.zeros(
            self.gaussians._xyz.shape[0],
            dtype=torch.bool,
            device=self.gaussians._xyz.device
        )
        
        for i_cam in range(len(cameras)):
            render_pkg = render_depth(
                viewpoint_camera=cameras[i_cam], 
                pc=self.gaussians, 
                pipe=self.pipe, 
                bg_color=background,
                culling=None
            )
            
            max_idx = render_pkg["gidx"].unique()
            is_maximal[max_idx] = True
            
        self.gaussians.prune_points(~is_maximal)
        gaussians_have_changed = True
        
        return gaussians_have_changed
        
    def reset_normal_field_state_at_next_iteration(self):
        pass
    
    def initialize_normal_field(self) -> dict:
        self.normal_field_state = {}


    @torch.no_grad()
    def densify_normal_field(
        self,
        cameras,
        background: torch.Tensor, 
        config: dict,
        render_func, 
    ):
        maintain_constant_volume = config["maintain_constant_volume"]
        from utils.normal_field import compute_normal_error, get_gaussian_std_in_direction, build_rotation
        # Get Gaussian normals
        gaussian_normals = self.gaussians.convert_features_to_normals()  # (N_gaussians, 3)
        gaussian_normals = torch.nn.functional.normalize(gaussian_normals, dim=-1)  # (N_gaussians, 3)
        
        # Compute normal errors
        normal_errors = compute_normal_error(
            gaussians=self.gaussians,
            cameras=cameras,
            render_func=render_func,
            pipe=self.pipe,
            background=background,
            method=config["densification_normalization_method"],  # "count" or "area" or "none"
            normal_to_use=config["densification_normal_to_use"],  # "rendered" or "median_depth" or "expected_depth"
        )  # (N_gaussians,)
        
        # Compute normal errors quantile
        normal_errors_quantile = torch.quantile(normal_errors, q=1. - config["densification_normal_errors_quantile"])
        
        # Densification mask
        densification_mask = normal_errors > normal_errors_quantile  # (N_gaussians,)

        do_densify=True
        gaussians_changed = False
        # If N_max_gaussians is set, cap the number of new Gaussians
        if self.mp.cap_max > 0:
            n_current = self.gaussians._xyz.shape[0]
            n_allowed = self.mp.cap_max - n_current
            if n_allowed <= 0:
                print("[WARNING] Maximum Number of Gaussians reached. Skipping Densification.")
                do_densify = False
                # Already at or above cap, skip densification entirely
            if n_allowed > 0:
                n_selected = densification_mask.sum().item()
                if n_selected > n_allowed:
                    # Keep only the top n_allowed Gaussians by normal error
                    candidate_indices = densification_mask.nonzero(as_tuple=True)[0]
                    top_indices = candidate_indices[normal_errors[candidate_indices].topk(n_allowed).indices]
                    densification_mask = torch.zeros_like(densification_mask)
                    densification_mask[top_indices] = True
                    print(f"[WARNING] Capping the number of gaussians to {self.mp.cap_max}.")
                    
        if not do_densify: return gaussians_changed
                    
        # Adjust scale of Gaussians to be densified. The idea is to divide the volume of the densified Gaussian by 2,
        # while taking into account the direction of the normal.
        if maintain_constant_volume:
            #   > First, we compute the local basis of the Gaussian
            local_basis = build_rotation(
                r=self.gaussians._rotation[densification_mask]  # (N_gaussians_to_densify, 3, n_vectors_in_basis)
            ).transpose(-1, -2)  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
            
            #   > Then, we compute the projections of the normals on the local basis
            projections_on_local_basis = (
                gaussian_normals[densification_mask].unsqueeze(1)  # (N_gaussians_to_densify, 1, 3)
                * local_basis  # (N_gaussians_to_densify, n_vectors_in_basis, 3)
            ).sum(dim=-1)  # (N_gaussians_to_densify, n_vectors_in_basis)
            
            #   > We compute the logarithm of the adjustment factors
            log_adjustment_factors = np.log(1. / 2.) * projections_on_local_basis ** 2
            
            #   > Adjust the scaling of the Gaussians
            self.gaussians._scaling[densification_mask] = self.gaussians._scaling[densification_mask] + log_adjustment_factors
        
        # Compute xyz of cloned Gaussians as same xyz minus a small multiple of the normal
        new_xyz = self.gaussians._xyz[densification_mask]  # (N_new_gaussians, 3)
        new_normals = - gaussian_normals[densification_mask]  # (N_new_gaussians, 3)
        normal_stds = get_gaussian_std_in_direction(
            directions=new_normals.unsqueeze(1),  # (N_new_gaussians, 1, 3)
            gaussian_scaling=self.gaussians.get_scaling_with_3D_filter[densification_mask].detach(), 
            gaussian_rotation=self.gaussians._rotation[densification_mask].detach(),
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
        new_gaussian_features = self.gaussians._learned_normals_features[densification_mask]  # (N_new_gaussians, n_features)
        new_gaussian_features[:, -1:] = -new_gaussian_features[:, -1:]
        
        # Update xyz of densified Gaussians to be xyz plus a small multiple of the normal
        self.gaussians._xyz[densification_mask] = (
            self.gaussians._xyz[densification_mask]
            + delta * normal_stds * gaussian_normals[densification_mask]
        )
        
        # Densify Gaussians
        self.gaussians.densify_and_clone_for_learned_normals(
            selected_pts_mask=densification_mask,
            new_xyz=new_xyz,
            new_learned_normals=new_gaussian_features,
        ) 
        gaussians_changed = True
        
        return gaussians_changed
    
    def densify(self,iteration, cameras, bg, render_func, render_depth_func, normal_field_kick_on, normal_field_config):
        dens_cond_1 = (
            normal_field_kick_on 
            and normal_field_config["use_densification"]
        )
        dens_cond_2 = (
            (iteration+1 >= normal_field_config["start_iter_densification"])
            and (iteration+1 <= normal_field_config["end_iter_densification"])
        )
        dens_cond_3 = (
            (iteration+1 - normal_field_config["start_iter_densification"]) % normal_field_config["densify_every_n_iterations"] == 0
        )
        gaussians_have_changed = False
        if dens_cond_1 and dens_cond_2 and dens_cond_3:
            print(f"[INFO] Densifying normal field at iteration {iteration+1}.")
            print(f"        > Using normalization method: {normal_field_config['densification_normalization_method']}.")
            print(f"        > Using normal computed from: {normal_field_config['densification_normal_to_use']}.")
            print(f"        > Using normal errors quantile: {normal_field_config['densification_normal_errors_quantile']}.")
            print(f"        > Maintaining constant volume: {normal_field_config['maintain_constant_volume']}.")
            print(f"        > Number of Gaussians before densification: {self.gaussians._xyz.shape[0]}.")
            gaussians_have_changed = gaussians_have_changed or self.densify_normal_field(cameras, bg, normal_field_config, render_func)
            
            
            if normal_field_config["reset_normals_after_densification"]:
                print(f"[INFO] Resetting normal features after densification.")
                self.gaussians.reset_normal_features(
                    reset_directions=normal_field_config["reset_normal_directions"],
                    reset_signs=normal_field_config["reset_normal_signs"],
                )
                
                
        # ---Normal field pruning---
        prune_cond_1 = (
            normal_field_kick_on 
            and normal_field_config["use_pruning"]
        )
        prune_cond_2 = (
            (iteration+1 >= normal_field_config["start_iter_pruning"])
            and (iteration+1 <= normal_field_config["end_iter_pruning"])
        )
        prune_cond_3 = (
            (iteration+1 - normal_field_config["start_iter_pruning"]) % normal_field_config["prune_every_n_iterations"] == 0
        )
        if prune_cond_1 and prune_cond_2 and prune_cond_3:
            print(f"[INFO] Pruning non-maximal Gaussians at iteration {iteration+1}.")
            print(f"        > Number of Gaussians before pruning: {self.gaussians._xyz.shape[0]}.")
            gaussians_have_changed = gaussians_have_changed or self.prune_non_maximal_gaussians(
                cameras=cameras,
                background=bg,
                render_depth=render_depth_func
            )
            print(f"        > Number of Gaussians after pruning: {self.gaussians._xyz.shape[0]}.")


        return gaussians_have_changed
        
    
class CustomDensifier(Densifier):   
    def __init__(self, gaussians : GaussianModel, opt : OptimizationParams, mp : MeshingParams, dataset : ModelParams, pipe : PipelineParams, cameras_extent:float,prune_iterations = [4000, 8000]):
        super().__init__(gaussians, opt, mp, dataset, pipe)
        self.prune_iterations = prune_iterations
        self.cameras_extent = cameras_extent
    
    
    def densify(self, iteration: int, **kwargs):
        gaussians_changed = False
        visibility_filter = kwargs.get("visibility_filter")
        radii = kwargs.get("radii")
        viewspace_point_tensor = kwargs.get("viewspace_point_tensor")
        cameras_extent = kwargs.get("cameras_extent")
        trainCameras = kwargs.get("trainCameras")
        render_simp = kwargs.get("render_simp")
        
        opt = self.opt
        mesh = self.mp
        dataset = self.dataset
        
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
            is_normal_densification = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
            is_warmup_densification = opt.warmup_densification and iteration > opt.densify_from_iter and iteration % 100 == 0 and not is_normal_densification
            
            

            if is_warmup_densification:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #GOF: use 0.05 min opacity instead of 0.005
                self.gaussians.densify_and_prune(opt.densify_grad_threshold, mesh.prune_threshold, cameras_extent, size_threshold, radii,
                                            abs_grad_for_densification=mesh.abs_grad_for_densification,
                                            clone_with_sampling=mesh.clone_with_sampling)
                gaussians_changed = True
            elif is_normal_densification:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                
                # 1. Calculate Gradients (Standard 3DGS metric)
                grads = self.gaussians.xyz_gradient_accum / self.gaussians.denom
                grads[grads.isnan()] = 0.0
                
                # 2. Create Gradient Mask 
                # [CRITICAL] We MUST use this to prevent 7M points. 
                # Only split if the geometry is struggling (high error).
                is_grad_high = torch.norm(grads, dim=-1) >= 1e-5

                # 3. [NEW] Multiview Consistency Criterion
                # Compute ratios of high/low eta counts
                valid_mask = self.gaussians.accum_view_count > 0
                
                high_ratio = torch.zeros_like(self.gaussians.accum_view_count)
                low_ratio = torch.zeros_like(self.gaussians.accum_view_count)
                high_ratio[valid_mask] = self.gaussians.eta_high_count[valid_mask] / self.gaussians.accum_view_count[valid_mask]
                low_ratio[valid_mask] = self.gaussians.eta_low_count[valid_mask] / self.gaussians.accum_view_count[valid_mask]
                
                # 4. Split if consistently high eta across views 
                split_mask = (high_ratio > opt.split_ratio_threshold) & is_grad_high
                
                # 5. Compute average high eta 3ch for densification guidance
                avg_high_eta_3ch = torch.zeros_like(self.gaussians.max_eta_3ch)
                has_high = self.gaussians.eta_high_count > 0
                avg_high_eta_3ch[has_high] = self.gaussians.eta_high_sum_3ch[has_high] / self.gaussians.eta_high_count[has_high].unsqueeze(1)
                max_high_eta = self.gaussians.max_eta_3ch
                # 6. Prune if consistently low eta across views 
                prune_mask = (low_ratio > opt.prune_ratio_threshold) & valid_mask

                # [NEW] Expand undersized Gaussians
                self.gaussians.expand_undersized_gs(
                    tau_expand=opt.tau_expand,
                    max_eta_3ch=avg_high_eta_3ch
                )

                # Pass the AVG high eta to the splitting function for analytic guidance
                self.gaussians.densify_and_prune_structgs(
                    max_screen_size=size_threshold,
                    min_opacity=mesh.prune_threshold, # 0.005 is a good default
                    extent=self.cameras_extent,
                    radii=radii,
                    args=opt,
                    importance_score=self.gaussians.accum_view_count,
                    pruning_score=None,
                    custom_split_mask=split_mask,
                    custom_prune_mask=prune_mask,
                    viewspace_points_indices=None,
                    max_eta_3ch=max_high_eta,  # Use max high eta for shaping the split
                    use_abs_grad=mesh.abs_grad_for_densification
                )
                
                # Reset accumulators
                self.gaussians.accum_eta.zero_()
                self.gaussians.accum_view_count.zero_()
                self.gaussians.max_eta_3ch.zero_()
                self.gaussians.accum_weights_valid.zero_()
                # Reset multiview consistency accumulators
                self.gaussians.eta_high_count.zero_()
                self.gaussians.eta_high_sum_3ch.zero_()
                self.gaussians.eta_mid_count.zero_()
                self.gaussians.eta_mid_sum_3ch.zero_()
                self.gaussians.eta_low_count.zero_()
                gaussians_changed = True
                
            if mesh.opacity_decay == 0 and iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                self.gaussians.reset_opacity()
        
            if mesh.opacity_decay != 0 and iteration % 50 == 0 and iteration > opt.densify_from_iter:
                self.gaussians.decay_opacity(mesh.opacity_decay)
                
            if iteration in self.prune_iterations:
                prune_mask = (self.gaussians.get_opacity < mesh.prune_threshold).squeeze()
                self.gaussians.prune_points(prune_mask)
                gaussians_changed = True
                pass
                
        else:
            if iteration == 15000:
                self.gaussians.culling_with_interesction_preserving(trainCameras, render_simp)
                gaussians_changed = True
                torch.cuda.empty_cache()
            elif iteration == 20000:
                self.gaussians.culling_with_interesction_sampling(trainCameras, render_simp)
                gaussians_changed = True
                torch.cuda.empty_cache()
                    
        return gaussians_changed
            
            
    def postfix(self, xyz_lr : float, **kwargs):
        pass
    