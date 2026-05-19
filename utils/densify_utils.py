import torch
import torchvision
from utils.sh_utils import RGB2SH
import torch.functional as F
from utils.loss_utils import l1_loss


def frequency_loss(means2D, cov2D, st_map, H, W):
    # Normalize coordinates to [-1,1] for grid_sample
    nx = (means2D[:,0] / (W - 1)) * 2 - 1
    ny = (means2D[:,1] / (H - 1)) * 2 - 1
    grid = torch.stack([nx, ny], dim=-1).view(1,1,-1,2)

    # Sample structure tensor
    Sxx, Sxy, Syy = F.grid_sample(st_map, grid, align_corners=True, padding_mode='border').view(3, -1)

    # Compute directional energy eta = u^T J u for the Gaussian principal axis directions
    a, b, c = cov2D[:,0], cov2D[:,1], cov2D[:,2]  # ellipse parameters

    # 1. Compute Determinant (Area^2)
    det = a * c - b * b
    
    # 2. Compute Linear Scale (Radius approx)
    # sqrt(det) = Area (s^2). sqrt(sqrt(det)) = Radius (s).
    sigma_area = torch.sqrt(det + 1e-6)
    sigma_linear = torch.sqrt(sigma_area + 1e-6) 

    # 3. Frequency Threshold (Wavelength in pixels)
    # Ensure st_map is normalized so freq_target is in meaningful units (e.g., 0.0 to 1.0)
    freq_target = torch.sqrt(Sxx + Syy + 1e-6) 
    
    wavelength_limit = 1.0 / (freq_target + 1e-6)

    # 4. Loss: Penalize if Radius > Wavelength
    # Using 'sigma_linear' ensures we compare pixels to pixels
    loss = torch.relu(sigma_linear - wavelength_limit).mean()

    return loss


def fast_gaussian_blur(img, kernel_size, sigma):
    """
    Optimized Gaussian blur that downsamples for large sigma values to save computation.
    Includes safety checks to prevent kernel size from exceeding image dimensions.
    """
    if sigma < 0.01:
        return img

    B, C, H, W = img.shape
    
    # Safety: If sigma is extremely large, return global average
    if sigma > max(H, W):
        return img.mean(dim=(2,3), keepdim=True).expand(-1, -1, H, W)

    if kernel_size is None:
        kernel_size = int(2 * 4 * sigma + 1) | 1

    if sigma <= 2.0:
        # Safety: kernel size cannot exceed image dimensions for reflect padding
        kernel_size = min(kernel_size, (H - 1) * 2 - 1, (W - 1) * 2 - 1)
        if kernel_size < 3:
            return img
        return torchvision.transforms.functional.gaussian_blur(img, kernel_size, sigma)
    
    # For large sigma, we can blur a downsampled version and then upsample.
    scale = int(sigma)
    scale = min(scale, H // 4, W // 4)
    
    if scale <= 1:
        kernel_size = min(kernel_size, (H - 1) * 2 - 1, (W - 1) * 2 - 1)
        if kernel_size < 3:
            return img
        return torchvision.transforms.functional.gaussian_blur(img, kernel_size, sigma)
    
    # Target resolution
    target_h, target_w = max(2, H // scale), max(2, W // scale)
    
    # Downsample
    img_down = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    # Blurred Sigma in downsampled space
    sigma_down = sigma / (H / target_h) # Use actual scale factors
    k_size_down = int(2 * 4 * sigma_down + 1) | 1
    
    # Safety cap on downsampled kernel
    k_size_down = min(k_size_down, (target_h - 1) * 2 - 1, (target_w - 1) * 2 - 1)
    
    # Blur
    if k_size_down >= 3:
        img_blur_down = torchvision.transforms.functional.gaussian_blur(img_down, k_size_down, sigma_down)
    else:
        # If still too small, the image is basically just the blurred version already
        img_blur_down = img_down
    
    # Upsample back
    return F.interpolate(img_blur_down, size=(H, W), mode='bilinear', align_corners=False)

def get_structure_tensor_torch(image_tensor, sigma=1.0, rho=1.0): 
    """
    Computes Multi-Channel Structure Tensor (Di Zenzo's Method).
    Summing energy across RGB channels avoids missing iso-luminant edges.
    """
    B, C, H, W = image_tensor.shape
    
    # 1. Gaussian Blur (Pre-smoothing)
    # Important: Apply to all channels (RGB) independently
    k_size = int(2 * 4 * sigma + 1)
    if k_size % 2 == 0: k_size += 1
    
    # blur applies to all channels automatically if shape is (B, C, H, W)
    img_smooth = fast_gaussian_blur(image_tensor, k_size, sigma)

    # 2. Sobel Derivatives (Vectorized for RGB)
    # We repeat the kernel for each channel and use groups=C to keep channels separate
    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float, device=image_tensor.device).view(1,1,3,3)
    sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float, device=image_tensor.device).view(1,1,3,3)
    
    # Repeat kernel for C channels (e.g., 3 for RGB)
    weight_x = sobel_x_kernel.repeat(C, 1, 1, 1)
    weight_y = sobel_y_kernel.repeat(C, 1, 1, 1)

    # Ix, Iy will be shape (B, C, H, W)
    Ix = F.conv2d(img_smooth, weight_x, padding=1, groups=C)
    Iy = F.conv2d(img_smooth, weight_y, padding=1, groups=C)

    # 3. Compute Products per Channel
    Ixx_c = Ix**2
    Ixy_c = Ix*Iy
    Iyy_c = Iy**2

    # 4. SUM ACROSS CHANNELS (Di Zenzo's Method)
    # We sum the energy of R, G, and B. 
    # Result is (B, 1, H, W)
    Ixx = Ixx_c.sum(dim=1, keepdim=True)
    Ixy = Ixy_c.sum(dim=1, keepdim=True)
    Iyy = Iyy_c.sum(dim=1, keepdim=True)

    # 5. Window Integration (Structure Tensor smoothing)
    # Now we smooth the summed energy
    k_size_rho = int(2 * 4 * rho + 1)
    if k_size_rho % 2 == 0: k_size_rho += 1
    
    Sxx = fast_gaussian_blur(Ixx, k_size_rho, rho)
    Sxy = fast_gaussian_blur(Ixy, k_size_rho, rho)
    Syy = fast_gaussian_blur(Iyy, k_size_rho, rho)

    # 6. Normalize
    # Normalizing by 9.0 (typical Sobel magnitude scaling) + max value is usually good practice
    # so thresholds don't drift wildly with brightness.
    magnitude = Sxx + Syy
    max_val = torch.amax(magnitude, dim=(1, 2, 3), keepdim=True) + 1e-6 # Use batch-wise maximum
    
    Sxx = Sxx / max_val
    Sxy = Sxy / max_val
    Syy = Syy / max_val

    # Output: (B, 3, H, W) where channels are Sxx, Sxy, Syy
    return torch.cat([Sxx, Sxy, Syy], dim=1)

def get_multiscale_structure_tensor_v1(image_tensor, levels=3, base_sigma=1., octave_step=1.5, smoothing_factor=1.0, power_factor=3.0, aggregation_mode='average'): 
    """
    Computes an Improved Multi-scale Structure Tensor.
    Analyzes the image at multiple scales (octaves) to determine the dominant frequency.
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    
    # 1. Compute Base Orientation (finest scale)
    base_st = get_structure_tensor_torch(image_tensor, sigma=base_sigma)
    Sxx_base = base_st[:, 0:1]
    Sxy_base = base_st[:, 1:2]
    Syy_base = base_st[:, 2:3]

    # 2. Analyze Frequency Response and Aggregate Component-wise
    import math
    current_smooth = image_tensor
    
    accum_Sxx = torch.zeros((B, 1, H, W), device=device)
    accum_Sxy = torch.zeros((B, 1, H, W), device=device)
    accum_Syy = torch.zeros((B, 1, H, W), device=device)
    accum_weight = torch.zeros((B, 1, H, W), device=device)
    
    sigma_accum = 0.0
    
    # We use a lower power factor to allow more "mixing" of scales
    # This helps curved structures (captured at one scale) to influence 
    # the anisotropy of even the sharpest edges.
    # power_factor = 2.0  <-- REMOVED override to respect argument (default 3.0) 
    
    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)
        target_sigma = base_sigma * (octave_step ** i)
        
        sigma_inc = math.sqrt(max(1e-6, target_sigma**2 - sigma_accum**2))
        next_smooth = fast_gaussian_blur(current_smooth, None, sigma_inc)
        
        # Band Response (Difference of Gaussians)
        band_response = (current_smooth - next_smooth).pow(2).sum(dim=1, keepdim=True).sqrt()
        
        # Spatial Smoothing for Low-Frequency Context
        if i > 0 and smoothing_factor > 0:
            smoothing_sigma = target_sigma * 2.0
            band_response = fast_gaussian_blur(band_response, None, smoothing_sigma)
        
        # Integrate structural components at this scale
        # Larger rho (3.0 * sigma) is critical for seeing curvature
        integration_rho = target_sigma * 3.0 
        k_size_rho = int(2 * 4 * integration_rho + 1) | 1
        
        Sxx_i = fast_gaussian_blur(Sxx_base, k_size_rho, integration_rho)
        Sxy_i = fast_gaussian_blur(Sxy_base, k_size_rho, integration_rho)
        Syy_i = fast_gaussian_blur(Syy_base, k_size_rho, integration_rho)
        
        # Normalize to orientation-only
        trace_i = Sxx_i + Syy_i + 1e-6
        # Apply orientation component at this scale
        Sxx_norm_i = Sxx_i / trace_i
        Sxy_norm_i = Sxy_i / trace_i
        Syy_norm_i = Syy_i / trace_i
        
        # Weight by band energy
        weight_i = band_response.pow(power_factor)
        
        # We want the final sum to have Trace = freq^2
        # freq_i = band_freq
        target_wavelength_sq_inv = band_freq**2
        
        accum_Sxx += Sxx_norm_i * weight_i * target_wavelength_sq_inv
        accum_Sxy += Sxy_norm_i * weight_i * target_wavelength_sq_inv
        accum_Syy += Syy_norm_i * weight_i * target_wavelength_sq_inv
        accum_weight += weight_i
        
        current_smooth = next_smooth
        sigma_accum = target_sigma
        
    # Final normalization
    final_Sxx = accum_Sxx / (accum_weight + 1e-6)
    final_Sxy = accum_Sxy / (accum_weight + 1e-6)
    final_Syy = accum_Syy / (accum_weight + 1e-6)
    
    return torch.cat([final_Sxx, final_Sxy, final_Syy], dim=1)

def get_multiscale_structure_tensor_v2(image_tensor, levels=3, base_sigma=1., octave_step=1.5, smoothing_factor=1.0, power_factor=3.0, aggregation_mode='average'): 
    """
    Computes a True Multi-scale Structure Tensor.
    Unlike v1, this version computes the structure tensor at each scale level
    based on the blurred image at that level, rather than blurring a pre-computed
    base tensor. This properly captures orientation and frequency at each scale.
    """
    B, C, H, W = image_tensor.shape
    device = image_tensor.device
    
    import math
    current_smooth = image_tensor
    
    accum_Sxx = torch.zeros((B, 1, H, W), device=device)
    accum_Sxy = torch.zeros((B, 1, H, W), device=device)
    accum_Syy = torch.zeros((B, 1, H, W), device=device)
    accum_weight = torch.zeros((B, 1, H, W), device=device)
    
    sigma_accum = 0.0
    
    for i in range(levels):
        band_freq = 1.0 / (octave_step ** i)
        target_sigma = base_sigma * (octave_step ** i)
        
        # Progressively blur the image to reach target_sigma
        sigma_inc = math.sqrt(max(1e-6, target_sigma**2 - sigma_accum**2))
        next_smooth = fast_gaussian_blur(current_smooth, None, sigma_inc)
        
        # Band Response (Difference of Gaussians) - measures energy at this scale
        band_response = (current_smooth - next_smooth).pow(2).sum(dim=1, keepdim=True).sqrt()
        
        # Spatial Smoothing for Low-Frequency Context
        if i > 0 and smoothing_factor > 0:
            smoothing_sigma = target_sigma * 2.0
            band_response = fast_gaussian_blur(band_response, None, smoothing_sigma)
        
        # TRUE MULTI-SCALE: Compute structure tensor on the blurred image at this level
        # Use appropriate sigma and rho for this scale
        scale_sigma = target_sigma
        integration_rho = target_sigma * 3.0  # Larger rho is critical for seeing curvature
        
        # Compute structure tensor directly on the current smoothed image
        st_i = get_structure_tensor_torch(next_smooth, sigma=scale_sigma, rho=integration_rho)
        Sxx_i = st_i[:, 0:1]
        Sxy_i = st_i[:, 1:2]
        Syy_i = st_i[:, 2:3]
        
        # Normalize to orientation-only
        trace_i = Sxx_i + Syy_i + 1e-6
        Sxx_norm_i = Sxx_i / trace_i
        Sxy_norm_i = Sxy_i / trace_i
        Syy_norm_i = Syy_i / trace_i
        
        # Weight by band energy
        weight_i = band_response.pow(power_factor)
        
        # We want the final sum to have Trace = freq^2
        target_wavelength_sq_inv = band_freq**2
        
        accum_Sxx += Sxx_norm_i * weight_i * target_wavelength_sq_inv
        accum_Sxy += Sxy_norm_i * weight_i * target_wavelength_sq_inv
        accum_Syy += Syy_norm_i * weight_i * target_wavelength_sq_inv
        accum_weight += weight_i
        
        current_smooth = next_smooth
        sigma_accum = target_sigma
        
    # Final normalization
    final_Sxx = accum_Sxx / (accum_weight + 1e-6)
    final_Sxy = accum_Sxy / (accum_weight + 1e-6)
    final_Syy = accum_Syy / (accum_weight + 1e-6)
    
    return torch.cat([final_Sxx, final_Sxy, final_Syy], dim=1)
    
def frequency_loss_simple(rendered_image, st_map):
    pred_st = get_structure_tensor_torch(rendered_image.unsqueeze(0)) 
    loss= l1_loss(pred_st, st_map)  
    return loss

def estimate_required_gaussians(image, base_density=0.01, detail_sensitivity=0.5, get_multiscale_structure_tensor_fn=None):
    """
    Estimates the theoretical number of Gaussians needed for a single view.
    
    Args:
        image: (B, C, H, W) input image tensor (0-1 range).
        base_density: Minimum GS per pixel for flat regions (0.01 = 1 GS per 10x10 block).
        detail_sensitivity: Scaling factor for frequency contribution.
        
    Returns:
        estimated_count: Integer estimation of GS count.
    """
    B, C, H, W = image.shape
    total_pixels = H * W
    
    # 1. Compute Structure Tensor (B, 3, H, W) -> channels are Sxx, Sxy, Syy
    # This uses the multiscale function
    st_map = get_multiscale_structure_tensor_fn(image)
    
    # 2. Extract Energy (Trace = Sxx + Syy)
    # Sxx is at index 0, Syy is at index 2
    Sxx = st_map[:, 0, :, :]
    Syy = st_map[:, 2, :, :]
    
    # The trace represents the squared magnitude of local variations.
    # Higher trace = Higher Frequency = Needs more primitives.
    local_energy = Sxx + Syy
    
    # 3. Integrate Energy
    # Sum of energy across the image
    total_structure_energy = local_energy.sum().item()
    
    # 4. Apply Heuristic Formula
    # Base count: Even flat walls need *some* Gaussians to exist (coverage).
    count_coverage = total_pixels * base_density
    
    # Detail count: Extra Gaussians needed for edges/textures.
    # Since st_map is normalized to ~1.0 max, 
    # we assume max energy requires ~1.0 GS/pixel scaling.
    count_detail = total_structure_energy * detail_sensitivity
    
    estimated_count = int(count_coverage + count_detail)
    
    return estimated_count, count_coverage, count_detail

 
from .graphics_utils import BasicPointCloud
import numpy as np
def add_bbox_faces(pcl: BasicPointCloud, N = 16):
    # [NEW] Sample new GS to cover the 6 faces of bounding box at a fixed resolution
    with torch.no_grad():
        print("Sampling 6 faces of bounding box...")
        xyz = torch.from_numpy(pcl.points).cuda()
        colors = torch.from_numpy(pcl.colors).cuda()
        min_bound = xyz.min(dim=0)[0] * 5
        max_bound = xyz.max(dim=0)[0] * 5
        
        new_xyz_list = []
        
        # X-faces
        y_range = torch.linspace(min_bound[1], max_bound[1], N, device="cuda")
        z_range = torch.linspace(min_bound[2], max_bound[2], N, device="cuda")
        Y, Z = torch.meshgrid(y_range, z_range, indexing='ij')
        Y = Y.reshape(-1)
        Z = Z.reshape(-1)
        
        X_min = torch.full_like(Y, min_bound[0])
        new_xyz_list.append(torch.stack([X_min, Y, Z], dim=-1))
        
        X_max = torch.full_like(Y, max_bound[0])
        new_xyz_list.append(torch.stack([X_max, Y, Z], dim=-1))
        
        # Y-faces
        x_range = torch.linspace(min_bound[0], max_bound[0], N, device="cuda")
        X, Z = torch.meshgrid(x_range, z_range, indexing='ij')
        X = X.reshape(-1)
        Z = Z.reshape(-1)
        
        Y_min = torch.full_like(X, min_bound[1])
        new_xyz_list.append(torch.stack([X, Y_min, Z], dim=-1))
        
        Y_max = torch.full_like(X, max_bound[1])
        new_xyz_list.append(torch.stack([X, Y_max, Z], dim=-1))
        
        # Z-faces
        X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        
        Z_min = torch.full_like(X, min_bound[2])
        new_xyz_list.append(torch.stack([X, Y, Z_min], dim=-1))
        
        Z_max = torch.full_like(X, max_bound[2])
        new_xyz_list.append(torch.stack([X, Y, Z_max], dim=-1))
        
        all_new_xyz = torch.cat(new_xyz_list, dim=0)
        num_new = all_new_xyz.shape[0]
        
        grey = torch.tensor([0.5, 0.5, 0.5], device="cuda").unsqueeze(0).repeat(num_new, 1)
        pcl_new = BasicPointCloud(torch.cat([xyz,all_new_xyz],dim=0).cpu().detach().numpy(), torch.cat([colors, grey], dim=0).cpu().detach().numpy(), None)
        return pcl_new
    

def precompute_structure_tensors(scene, st_mode = "v1", st_levels=4):
    torch.cuda.empty_cache()
    from time import time
    start = time()
    structure_tensor_cache = {}
     # Group cameras by resolution
    cameras_by_res = {}
    for cam in scene.getTrainCameras():
        res = (cam.image_height, cam.image_width)
        if res not in cameras_by_res:
            cameras_by_res[res] = []
        cameras_by_res[res].append(cam)

    # Process each resolution group in batches
    for res, cams in cameras_by_res.items():
        pixel_count = res[0] * res[1]
        batch_size = 100

        print(f"  Processing {len(cams)} views at {res[1]}x{res[0]} (Batch size: {batch_size})")
        for i in range(0, len(cams), batch_size):
            batch_cams = cams[i : i + batch_size]
            
            img_batch = torch.stack([cam.original_image.cuda()*(cam.seg_mask.cuda()>0) for cam in batch_cams], dim=0)
            
            with torch.no_grad():
                if st_mode == "v1":
                    st_batch = get_multiscale_structure_tensor_v1(img_batch, levels=st_levels)
                elif st_mode == "v2":
                    st_batch = get_multiscale_structure_tensor_v2(img_batch, levels=st_levels)
                
                for j, cam in enumerate(batch_cams):
                    st_map = st_batch[j:j+1] # Keep (1, 3, H, W) shape
                    structure_tensor_cache[cam.image_name] = st_map.cpu()
                    
                    # Normalize for visualization (Sxx, Sxy, Syy can be large, so we normalize per image)
                    # st_vis = st_map.clone().detach()
                    # st_vis = (st_vis - st_vis.min()) / (st_vis.max() - st_vis.min() + 1e-6)
                    # torchvision.utils.save_image(st_vis, os.path.join(st_vis_dir, f"{cam.image_name}.png"))
    print(f"Pre-computing Structure Tensors took {time() - start:.4f} seconds") 
    
    return structure_tensor_cache


import torch
import torch.nn.functional as F
from PIL import ImageFilter
from .loss_utils import l1_loss
from .densify_utils import frequency_loss_simple, get_structure_tensor_torch
import torchvision.transforms as transforms
from utils.general_utils import build_rotation
import random


def sampling_cameras(my_viewpoint_stack, mode="fps", num_cams=60, weights=None):
    ''' 
    Sample a given number of cameras from the viewpoint stack.
    
    Args:
        my_viewpoint_stack: List of camera viewpoints
        mode: "random" for random sampling, "fps" for farthest point sampling
    
    Returns:
        camlist: List of sampled cameras
    '''

    num_cams = num_cams
    
    if mode == "random":
        # Original random sampling
        camlist = []
        for _ in range(num_cams):
            loc = random.randint(0, len(my_viewpoint_stack) - 1)
            camlist.append(my_viewpoint_stack.pop(loc))
        return camlist
    
    elif mode == "fps":
        # Farthest Point Sampling based on camera positions
        num_cams = min(num_cams, len(my_viewpoint_stack))
        
        # Extract camera positions from world_view_transform
        # The camera position in world space is -R^T @ t where [R|t] is the view matrix
        camera_positions = []
        for cam in my_viewpoint_stack:
            # world_view_transform is [4, 4], extract rotation and translation
            w2c = cam.world_view_transform.transpose(0, 1)  # [4, 4]
            R = w2c[:3, :3]  # [3, 3]
            t = w2c[:3, 3]   # [3]
            # Camera position in world coordinates: -R^T @ t
            cam_pos = -R.T @ t
            camera_positions.append(cam_pos)
        
        camera_positions = torch.stack(camera_positions)  # [N, 3]
        
        # Farthest Point Sampling
        N = len(my_viewpoint_stack)
        selected_indices = []
        distances = torch.full((N,), float('inf'), device=camera_positions.device)
        
        # Start with a random camera
        current_idx = random.randint(0, N - 1)
        selected_indices.append(current_idx)
        
        for _ in range(num_cams - 1):
            # Update distances to the nearest selected point
            current_pos = camera_positions[current_idx]
            dists_to_current = torch.norm(camera_positions - current_pos, dim=1)
            distances = torch.minimum(distances, dists_to_current)
            
            # Exclude already selected points
            distances[selected_indices] = -1
            
            # Select the farthest point
            current_idx = torch.argmax(distances).item()
            selected_indices.append(current_idx)
        
        # Extract selected cameras and remove from stack
        camlist = []
        # Sort indices in descending order to avoid index shifting issues when popping
        for idx in sorted(selected_indices, reverse=True):
            camlist.append(my_viewpoint_stack.pop(idx))
        
        # Reverse to maintain original order
        camlist.reverse()
        
        return camlist
        
    
    else:
        raise ValueError(f"Unknown sampling mode: {mode}. Use 'random' or 'fps'.")




def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_projected_axes_subset(means2D, depths, scales, rotations, viewpoint_camera):
    """
    Computes projected 2D axes for a SUBSET of Gaussians using pre-computed means2D/depths.
    """
    with torch.no_grad():
        fx = viewpoint_camera.focal_x
        fy = viewpoint_camera.focal_y
        
        # 1. Back-project means2D to Camera Space (x, y)
        # u = (x/z)*fx + cx  =>  x = (u - cx) * z / fx
        # We need this to build the Jacobian at the correct location
        W = viewpoint_camera.image_width
        H = viewpoint_camera.image_height
        
        # Normalize means2D to pixels relative to center (assuming cx=W/2, cy=H/2 for simplicity 
        # or use actual projection matrix if needed, but standard pinhole approx is usually fine here)
        # Note: means2D from rasterizer is in Pixel Coordinates [0, W], [0, H]
        
        vec_x = (means2D[:, 0] - W * 0.5) * (depths / fx)
        vec_y = (means2D[:, 1] - H * 0.5) * (depths / fy)
        
        # 2. Build Jacobian J per point
        # J = [ fx/z   0     -fx*x/z^2 ]
        #     [ 0      fy/z  -fy*y/z^2 ]
        
        inv_z = 1.0 / (depths + 1e-7)
        inv_z2 = inv_z * inv_z
        
        J_00 = fx * inv_z
        J_02 = -fx * vec_x * inv_z2
        J_11 = fy * inv_z
        J_12 = -fy * vec_y * inv_z2
        
        # 3. Rotate and Scale Axes in Camera Space
        # Get World->View rotation
        W_view = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
        R_view = W_view[:3, :3] # [3, 3]
        
        # Convert quaternions to rotation matrices [N, 3, 3]
        R_local = build_rotation(rotations) 
        
        # R_total = R_view @ R_local
        # Expand R_view to match batch size
        R_view_batch = R_view.unsqueeze(0).expand(scales.shape[0], -1, -1)
        R_total = torch.bmm(R_view_batch, R_local) # [N, 3, 3]
        
        # Scale axes: Axis_vectors = R_total * Scales
        # scales is [N, 3]. We broadcast multiply columns.
        # This gives us the 3 axes (columns) in Camera coordinates
        axes_cam = R_total * scales.unsqueeze(1) # [N, 3, 3]
        
        # 4. Project Axes to 2D
        ax_x = axes_cam[:, 0, :] # [N, 3] (x-component of axis 0, 1, 2)
        ax_y = axes_cam[:, 1, :]
        ax_z = axes_cam[:, 2, :]
        
        # u = J00*x + J02*z
        # v = J11*y + J12*z
        u_vec = J_00.unsqueeze(1) * ax_x + J_02.unsqueeze(1) * ax_z
        v_vec = J_11.unsqueeze(1) * ax_y + J_12.unsqueeze(1) * ax_z
        
        # [N, 3, 2]
        return torch.stack([u_vec, v_vec], dim=2)


def update_freq_stats_online(viewpoint_cam, gaussians, cov2D, visibility_filter, structure_tensor_cache, viewspace_point_tensor=None, grad_threshold=None, transmittance_threshold=0.0, opacity_threshold=0.05, eta_compute_mode="wavelength"):
    """
    Online accumulation of frequency statistics for a single view.
    """
    if structure_tensor_cache is None:
        return

    # 1. Get Indices of Gaussians touching the view
    if visibility_filter.dtype == torch.bool:
        global_indices = torch.nonzero(visibility_filter).flatten()
    else:
        global_indices = visibility_filter
    
    if global_indices.shape[0] == 0:
        return

    # 2. Extract Data from Cov2D for visible points
    # Use integer indices for extraction
    current_cov2D = cov2D[global_indices] # [N_vis, 7]
    
    means2D_computed = current_cov2D[:, 3:5] # [N_vis, 2]
    depths = current_cov2D[:, 5]             # [N_vis]
    max_transmittance = current_cov2D[:, 6]  # [N_vis]

    # 3. Transmittance AND Opacity-based Active Mask
    current_opacities = gaussians.get_opacity[global_indices].squeeze(-1)
    is_active_vis = (max_transmittance > transmittance_threshold) & (current_opacities > opacity_threshold)
    
    # [NEW] Gradient-Based Masking
    if viewspace_point_tensor is not None and grad_threshold is not None:
        if viewspace_point_tensor.grad is not None:
            # viewspace_point_tensor is [N_total, 3]
            # Extract grads for visible points
            # Standard 3DGS accumulates norm of first 2 dimensions (x, y)
            current_grads = viewspace_point_tensor.grad[global_indices] # [N_vis, 3]
            grad_norms = torch.norm(current_grads[:, :2], dim=-1)
            
            is_grad_high = grad_norms > grad_threshold
            is_active_vis = is_active_vis & is_grad_high

    # [NEW] Densify Count Filtering - Skip GS that have been densified too many times
    # This prevents over-densification in already densified regions
    max_densify_count = 3  
    # if hasattr(gaussians, 'densify_count') and gaussians.densify_count.numel() > 0:
    #     current_densify_counts = gaussians.densify_count[global_indices]
    #     is_not_overdensified = current_densify_counts <= max_densify_count
    #     is_active_vis = is_active_vis & is_not_overdensified

    # Filter to get "Active & Visible" indices in the global array
    valid_indices_global = global_indices[is_active_vis] # [N_valid]
    
    if valid_indices_global.shape[0] > 0:
        st_map = structure_tensor_cache[viewpoint_cam.image_name].cuda()
        h, w = viewpoint_cam.image_height, viewpoint_cam.image_width

        # [NEW] Extract Transmittance Weights for the valid subset
        # We use this to scale the frequency violation
        weights_valid = torch.ones_like(max_transmittance)[is_active_vis] # [N_valid]

        # 4. Get Subsets of Data for Valid Points
        means2D_valid = means2D_computed[is_active_vis]
        depths_valid = depths[is_active_vis]
        
        scales_valid = gaussians.get_scaling[valid_indices_global]
        rots_valid = gaussians.get_rotation[valid_indices_global]

        # --- [MODIFICATION START] STOCHASTIC JITTER SAMPLING ---
        # Instead of sampling only at the center, we sample a random point 
        # within the 1-sigma ellipsoid of the Gaussian.
        
        # 1. Extract 2D Covariance Components
        # current_cov2D format: [cov_xx, cov_xy, cov_yy, mean_x, mean_y, ...]
        cov_xx = current_cov2D[is_active_vis, 0]
        cov_xy = current_cov2D[is_active_vis, 1]
        cov_yy = current_cov2D[is_active_vis, 2]
        
        # 2. Generate Random Jitter using Cholesky Decomposition
        # We sample from a 2D multivariate normal N(0, Sigma).
        # Sigma = L @ L.T, where L is the lower triangular Cholesky factor.
        # L = [[sqrt(xx), 0],
        #      [xy/sqrt(xx), sqrt(yy - xy^2/xx)]]
        
        eps1 = torch.randn_like(cov_xx)
        eps2 = torch.randn_like(cov_xx)
        
        # L11 = sqrt(cov_xx)
        L11 = torch.sqrt(torch.clamp(cov_xx, min=1e-6))
        # L21 = cov_xy / L11
        L21 = cov_xy / L11
        # L22 = sqrt(cov_yy - L21^2)
        L22 = torch.sqrt(torch.clamp(cov_yy - L21**2, min=1e-6))
        
        jitter_x = L11 * eps1
        jitter_y = L21 * eps1 + L22 * eps2
        
        # 4. Apply Jitter to Sampling Coordinates
        sample_x = means2D_valid[:, 0] + jitter_x
        sample_y = means2D_valid[:, 1] + jitter_y
        
        # Mirror Jitter if outside boundary
        # out_x = (sample_x < 0) | (sample_x >= w)
        # jitter_x[out_x] *= -1
        # sample_x = means2D_valid[:, 0] + jitter_x

        # out_y = (sample_y < 0) | (sample_y >= h)
        # jitter_y[out_y] *= -1
        # sample_y = means2D_valid[:, 1] + jitter_y

        # 5. Compute Grid using JITTERED coordinates
        norm_x = (sample_x / (w - 1)) * 2 - 1
        norm_y = (sample_y / (h - 1)) * 2 - 1
        grid_valid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0) 
        # --- [MODIFICATION END] ---
        
        # 6. Sample ST [3, N_valid] -> (Sxx, Sxy, Syy)
        sampled_S = F.grid_sample(st_map, grid_valid, align_corners=True).squeeze() #, padding_mode='border'
        if len(sampled_S.shape) == 1: sampled_S = sampled_S.unsqueeze(1)
        sampled_S = sampled_S.permute(1, 0) # [N_valid, 3]

        # 7. Compute 3-Channel Eta
        axes_2d = compute_projected_axes_subset(means2D_valid, depths_valid, scales_valid, rots_valid, viewpoint_cam)
        
        # Sxx = sampled_S[:, 0].unsqueeze(1)
        # Sxy = sampled_S[:, 1].unsqueeze(1)
        # Syy = sampled_S[:, 2].unsqueeze(1)
        
        # u = axes_2d[:, :, 0] 
        # v = axes_2d[:, :, 1] 
        
        # # Projection Frequency Violation
        # eta_3ch = Sxx * u**2 + 2 * Sxy * u * v + Syy * v**2 # [N_valid, 3]

        if eta_compute_mode == "wavelength":
            # --- [MODE 1] TRUE SIZE / WAVELENGTH ETA ---

            # 1. Extract Structure Tensor Components
            # sampled_S is [N, 3] -> (Sxx, Sxy, Syy)
            raw_Sxx = sampled_S[:, 0]
            raw_Sxy = sampled_S[:, 1]
            raw_Syy = sampled_S[:, 2]

            # 2. Compute Local Texture Wavelength (The "Speed Limit" for Size)
            # We compute the principal eigenvalue (Lambda1) of the Structure Tensor.
            # Lambda1 represents the maximum frequency energy (squared gradient).
            trace = raw_Sxx + raw_Syy
            det = raw_Sxx * raw_Syy - raw_Sxy**2
            delta = torch.sqrt(torch.clamp((trace/2)**2 - det, min=0.0))

            lambda1 = (trace / 2) + delta  # Max Eigenvalue (High Frequency Energy)

            # Convert Energy to Wavelength (Pixels)
            # Frequency f ~ sqrt(lambda). Wavelength w ~ 1/f.
            # We add epsilon to prevent division by zero in flat regions.
            wavelength_min = 1.0 / (torch.sqrt(lambda1) + 1e-5) 

            # 3. Compute Projected Gaussian Axis Lengths
            # axes_2d is [N, 3, 2] -> (N Gaussians, 3 Axes, u/v coordinates)
            u_vec = axes_2d[:, :, 0] 
            v_vec = axes_2d[:, :, 1] 

            # The physical length of the Gaussian's axis on the screen in pixels
            axis_lengths = torch.sqrt(u_vec**2 + v_vec**2 + 1e-8) # [N, 3]

            # 4. Compute Eta as a Ratio
            # Eta = (Gaussian Size) / (Texture Feature Size)
            # If Eta > 1.0, the Gaussian is larger than the feature -> Aliasing -> Split.
            # We expand wavelength_min to match the 3 axes [N, 1]
            eta_3ch = axis_lengths / (wavelength_min.unsqueeze(1))
        
        elif eta_compute_mode == "projection":
            # --- [MODE 2] DIRECTIONAL PROJECTION ETA ---
            # This formula properly projects the structure tensor onto each Gaussian axis direction.
            # For horizontal lines: Syy is high, Sxx ≈ 0
            # - Axes aligned with X (u component) → low eta
            # - Axes aligned with Y (v component) → high eta
            
            # Extract Structure Tensor Components
            # sampled_S is [N, 3] -> (Sxx, Sxy, Syy)
            Sxx = sampled_S[:, 0].unsqueeze(1)  # [N, 1]
            Sxy = sampled_S[:, 1].unsqueeze(1)  # [N, 1]
            Syy = sampled_S[:, 2].unsqueeze(1)  # [N, 1]
            
            # Get projected axis directions
            # axes_2d is [N, 3, 2] -> (N Gaussians, 3 Axes, u/v coordinates)
            u = axes_2d[:, :, 0]  # [N, 3] - X component of each axis
            v = axes_2d[:, :, 1]  # [N, 3] - Y component of each axis
            
            # Projection Frequency Violation:
            # eta = axis^T @ S @ axis = Sxx*u² + 2*Sxy*u*v + Syy*v²
            # This is the quadratic form that measures frequency energy along each axis direction.
            eta_3ch = Sxx * u**2 + 2 * Sxy * u * v + Syy * v**2  # [N, 3]

            eta_3ch = torch.sqrt(eta_3ch)
            
        else:
            raise ValueError(f"Unknown eta_compute_mode: {eta_compute_mode}")
        
        # [NEW] Weight by Transmittance (Importance Sampling)
        # If the Gaussian is transparent or occluded, we care less about its aliasing.
        eta_3ch = eta_3ch * weights_valid.unsqueeze(1)

        eta_total = eta_3ch.sum(dim=1)
        
        gaussians.accum_eta[valid_indices_global] += eta_total
        gaussians.accum_view_count[valid_indices_global] += 1.0
        gaussians.accum_weights_valid[valid_indices_global] += weights_valid
        
        # Accumulate 3-Channel Eta (Sum, for Leaky Average)
        # Note: We are now accumulating "Weighted" Eta.
        # gaussians.max_eta_3ch[valid_indices_global] += eta_3ch
        current_max = gaussians.max_eta_3ch[valid_indices_global]
        gaussians.max_eta_3ch[valid_indices_global] = torch.max(current_max, eta_3ch)
        
        # [NEW] Multiview Consistency: Track high/mid/low eta per GS
        TAU_HIGH = 1.0  # Frequency violation threshold
        TAU_LOW = 0.1   # Background/smooth threshold
        
        # Get max eta across 3 axes for classification
        eta_max_scalar = eta_3ch.max(dim=1).values  # [N_valid]
        
        # Classify each observation
        is_high = eta_max_scalar > TAU_HIGH
        is_low = eta_max_scalar <= TAU_LOW
        is_mid = ~is_high & ~is_low
        
        # Update counts
        gaussians.eta_high_count[valid_indices_global[is_high]] += 1.0
        gaussians.eta_mid_count[valid_indices_global[is_mid]] += 1.0
        gaussians.eta_low_count[valid_indices_global[is_low]] += 1.0
        
        # Accumulate sums for high and mid (for computing average later)
        if is_high.any():
            gaussians.eta_high_sum_3ch[valid_indices_global[is_high]] += eta_3ch[is_high]
        if is_mid.any():
            gaussians.eta_mid_sum_3ch[valid_indices_global[is_mid]] += eta_3ch[is_mid]


