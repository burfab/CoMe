from scene import GaussianModel
from arguments import OptimizationParams, PipelineParams, MeshingParams, ModelParams
from scene.gaussian_model import build_scaling_rotation
import torch

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
    
    
class CustomDensifier(Densifier):   
    def __init__(self, gaussians : GaussianModel, opt : OptimizationParams, mp : MeshingParams, dataset : ModelParams, pipe : PipelineParams, cameras_extent:float,prune_iterations = [4000, 8000]):
        super().__init__(gaussians, opt, mp, dataset, pipe)
        self.prune_iterations = prune_iterations
        self.cameras_extent = cameras_extent
    
    
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
            
            is_normal_densification = iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0
            is_warmup_densification = opt.warmup_densification and iteration > opt.densify_from_iter and iteration % 100 == 0 and not is_normal_densification
            
            

            if is_warmup_densification:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #GOF: use 0.05 min opacity instead of 0.005
                self.gaussians.densify_and_prune(opt.densify_grad_threshold, mesh.prune_threshold, cameras_extent, size_threshold, radii,
                                            abs_grad_for_densification=mesh.abs_grad_for_densification,
                                            clone_with_sampling=mesh.clone_with_sampling)
                # we need to compute the 3D filter here for reasons (see reset_opacity())
                self.gaussians.compute_3D_filter(trainCameras, CUDA=not self.pipe.compute_filter3D_python)
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
                # gaussians.expand_undersized_gs(
                #     tau_expand=opt.tau_expand,
                #     max_eta_3ch=avg_high_eta_3ch
                # )

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
                
            if mesh.opacity_decay == 0 and iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                self.gaussians.reset_opacity()
        
            if mesh.opacity_decay != 0 and iteration % 50 == 0 and iteration > opt.densify_from_iter:
                self.gaussians.decay_opacity(mesh.opacity_decay)
                
            if iteration in self.prune_iterations:
                prune_mask = (self.gaussians.get_opacity < mesh.prune_threshold).squeeze()
                self.gaussians.prune_points(prune_mask)
                pass
                
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
    