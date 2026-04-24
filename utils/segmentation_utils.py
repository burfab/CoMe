import torch


def features_to_rgb_pca(features: torch.Tensor) -> torch.Tensor:
    """
    Convert feature map (C, H, W) to RGB image using PCA in PyTorch.

    Args:
        features: torch.Tensor of shape (C, H, W)

    Returns:
        rgb_array: uint8 tensor of shape (H, W, 3)
    """
    assert features.ndim == 3, f"Expected (C, H, W), got {features.shape}"

    C, H, W = features.shape

    # Normalize each feature vector per pixel
    # (C, H, W) -> norm over C
    features = features / (torch.norm(features, dim=0, keepdim=True) + 1e-9)

    # Reshape to (H*W, C)
    X = features.view(C, -1).T  # (N, C), N = H*W

    # Center data
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    # PCA via SVD
    # X_centered = U S Vh
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    # Take first 3 principal directions
    components = Vh[:3]  # (3, C)

    # Project to PCA space
    X_pca = X_centered @ components.T  # (N, 3)

    # Reshape back to image
    pca_result = X_pca.view(H, W, 3).permute(2,0,1)

    # Normalize to [0, 255]
    pca_min = pca_result.min()
    pca_max = pca_result.max()
    pca_normalized = (pca_result - pca_min) / (pca_max - pca_min + 1e-9)
    return pca_normalized

#currently bg is 0, i want bg to be 1, and 0 to be at border of currently != 0 segmentations
def set_bg_to_one_and_class_borders_to_zero(gt_segmask: torch.Tensor) -> torch.Tensor:
    """
    Keeps nonzero labels unchanged.
    Sets background (==0) to 1.
    Sets pixels on boundaries between different nonzero classes to 0.
    """
    x = gt_segmask.clone()

    # Start by turning background into 1
    out = x.clone()
    out[x == 0] = 1

    # Compare with 4-neighbors
    up    = torch.zeros_like(x, dtype=torch.bool)
    down  = torch.zeros_like(x, dtype=torch.bool)
    left  = torch.zeros_like(x, dtype=torch.bool)
    right = torch.zeros_like(x, dtype=torch.bool)

    # Boundary only if both pixels are nonzero and labels differ
    up[1:]    = (x[1:]    != x[:-1]) & (x[1:]    != 0) & (x[:-1] != 0)
    down[:-1] = (x[:-1]   != x[1:])  & (x[:-1]   != 0) & (x[1:]  != 0)
    left[:,1:]  = (x[:,1:]  != x[:,:-1]) & (x[:,1:]  != 0) & (x[:,:-1] != 0)
    right[:,:-1]= (x[:,:-1] != x[:,1:])  & (x[:,:-1] != 0) & (x[:,1:]  != 0)

    border = up | down | left | right

    # Set class-border pixels to 0
    out[border] = 0

    return out 

    
def contrastive_2d_loss(segmask, features, id_unique_list, n_i_list, dim_features=4, lambda_val=1e-4):
    """
    Compute the contrastive clustering loss for a 2D feature map.

    :param segmask: Tensor of shape (H, W).
    :param features: Tensor of shape (H, W, D), where (H, W) is the resolution and D is the dimensionality of these features.
    :param id_unique_list: Tensor of shape (n_p).
    :n_i_list: Tensor of shape (n_p).
    :dim_features: is the dimensionality of the features (equal to D).
    :lambda_val: Weighting factor for the loss.

    :return: loss value.
    """
    n_p = id_unique_list.shape[0] # Number of ids

    f_mean_per_cluster = torch.zeros((n_p, dim_features)).cuda()
    phi_per_cluster = torch.zeros((n_p, 1)).cuda()
            
    for i in range(n_p):
        mask = (segmask == id_unique_list[i]).squeeze()
        f_mean_per_cluster[i, ...] = torch.mean(features[mask,:], dim=0, keepdim=True)
        phi_per_cluster[i] = torch.norm(features[mask, :] - f_mean_per_cluster[i], dim=1, keepdim=True).sum() / (n_i_list[i] * torch.log(n_i_list[i] + 100))
            
    phi_per_cluster = torch.clip(phi_per_cluster * 10, min=0.1, max=1.0)
    phi_per_cluster = phi_per_cluster.detach()
    loss_per_cluster = torch.zeros(n_p).cuda()
            
    for i in range(n_p):
        f_mean = f_mean_per_cluster[i]
        phi = phi_per_cluster[i]
        mask = (segmask == id_unique_list[i]).squeeze()
        f_ij = features[mask, :] # shape (ni, 16)
        num = torch.exp(torch.matmul(f_ij, f_mean) / (phi + 1e-6)) # dim (ni)
        den = torch.sum(torch.exp(torch.matmul(f_ij, f_mean_per_cluster.transpose(-1, -2)) / (phi_per_cluster.transpose(-1, -2) + 1e-6)), dim=1) # dim (n_i)
        loss_per_cluster[i] = torch.sum(torch.log(num / (den + 1e-6)))
            
    loss_obj = - lambda_val * torch.mean(loss_per_cluster)
    return loss_obj

def spatial_loss(xyz, features, k_pull=2, k_push=5, lambda_pull=0.05, lambda_push=0.15, max_points=200000, sample_size=800):
    """
    Compute the spatial-similarity regularization loss for a 3D point cloud using Top-k neighbors and Top-k distant elements
    
    :param xyz: Tensor of shape (N, D), where N is the number of points and D is the dimensionality.
    :param features: Tensor of shape (N, C), where C is the dimensionality of these features.
    :param k_pull: Number of neighbors to consider.
    :param k_push: Number of remote elements to consider.
    :param lambda_pull: Weighting factor for the loss.
    :param lambda_push: Weighting factor for the loss.
    :param max_points: Maximum number of points for downsampling. If the number of points exceeds this, they are randomly downsampled.
    :param sample_size: Number of points to randomly sample for computing the loss.
    
    :return: Computed loss value.
    """
    # Conditionally downsample if points exceed max_points
    if xyz.size(0) > max_points:
        indices = torch.randperm(xyz.size(0))[:max_points]
        xyz = xyz[indices]
        features = features[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(xyz.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_preds = features[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_xyz, xyz)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k_pull, largest=False)  # Get top-k smallest distances

    # Compute top-k farest gaussians
    _, faraway_indices_tensor = dists.topk(k_push, largest=True)  # Get top-k bigest distances 

    # Fetch neighbor features using indexing
    neighbor_preds = features[neighbor_indices_tensor]

    # Fetch remote features using indexing
    faraway_preds = features[faraway_indices_tensor]

    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    
    pull_loss = cos(sample_preds.unsqueeze(1).expand(-1, k_pull, -1), neighbor_preds) #more similar of they are close
    
    push_loss =  cos(sample_preds.unsqueeze(1).expand(-1, k_push, -1), faraway_preds) #less similar if they are far away
    
    # Total loss

    loss = (
        lambda_pull * torch.sigmoid(1.0 - pull_loss.mean()) +
        lambda_push * torch.sigmoid(push_loss.mean())
    )
    
    return loss

def no_similarity_loss(xyz, features, k=5, lambda_val=0.15, max_points=200000, sample_size=800):
    # Conditionally downsample if points exceed max_points
    if xyz.size(0) > max_points:
        indices = torch.randperm(xyz.size(0))[:max_points]
        xyz = xyz[indices]
        features = features[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(xyz.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_preds = features[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_xyz, xyz)  # Compute pairwise distances

    # Compute top-k farest gaussians
    _, faraway_indices_tensor = dists.topk(k, largest=True)  # Get top-k bigest distances 

    # Fetch remote features using indexing
    faraway_preds = features[faraway_indices_tensor]

    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    
    push_loss =  cos(sample_preds.unsqueeze(1).expand(-1, k, -1, -1), faraway_preds) # dissimilar if they are far away
    
    # Total loss
    loss = lambda_val * torch.sigmoid(torch.mean(push_loss[..., 0].reshape(-1), dim=-1))
    
    return loss

def no_dissimilarity_loss(xyz, features, k=2, lambda_val=0.05, max_points=200000, sample_size=800):
    # Conditionally downsample if points exceed max_points
    if xyz.size(0) > max_points:
        indices = torch.randperm(xyz.size(0))[:max_points]
        xyz = xyz[indices]
        features = features[indices]

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(xyz.size(0))[:sample_size]
    sample_xyz = xyz[indices]
    sample_preds = features[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_xyz, xyz)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor features using indexing
    neighbor_preds = features[neighbor_indices_tensor]

    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-10)
    
    pull_loss = cos(sample_preds.unsqueeze(1).expand(-1, k, -1, -1), neighbor_preds) # more similar of they are close
    
    # Total loss
    loss = lambda_val * torch.sigmoid(1.0 - torch.mean(pull_loss[..., 0].reshape(-1), dim=-1))
    
    return loss

def variance_in_feature_clusters(segmask, features, id_unique_list, dim_features=16):
    n_p = id_unique_list.shape[0] # Number of ids
        
    f_mean_per_cluster = torch.zeros((n_p, dim_features)).cuda()
            
    for i in range(n_p):
        mask = segmask == id_unique_list[i]
        f_mean_per_cluster[i, ...] = torch.mean(features[mask, :], dim=0, keepdim=True)
    
    variance_per_cluster = torch.zeros(n_p, dim_features).cuda()

    for i in range(n_p):
        f_mean = f_mean_per_cluster[i]
        mask = segmask == id_unique_list[i]
        f_ij = features[mask, :] # shape (ni, 16)
        variance_per_cluster[i] = torch.mean(f_ij * f_ij, dim=0) - (f_mean * f_mean)
    
    return torch.mean(variance_per_cluster) 
    
    
def get_unique_id_list(segmask, min_pixnum):
    id_unique_list, n_i_list_ = torch.unique(segmask, return_counts=True)

    # Remove id 0 (related to borders)        
    if id_unique_list[0] == 0:
        id_unique_list = id_unique_list[1:]
        n_i_list_ = n_i_list_[1:]

    # Remove small clusters
    id_unique_list = id_unique_list[n_i_list_ > min_pixnum]
    n_i_list = n_i_list_[n_i_list_ > min_pixnum]

    return id_unique_list, n_i_list 