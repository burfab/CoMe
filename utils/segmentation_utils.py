import torch

FEATURES_RANDOM_ORTH_PROJ = None

def features_to_rgb_random_orth_proj(features: torch.Tensor) -> torch.Tensor:
    global FEATURES_RANDOM_ORTH_PROJ

    C, M, N = features.shape

    if FEATURES_RANDOM_ORTH_PROJ is None:
        device = features.device

        Q, _ = torch.linalg.qr(torch.randn(C, C, device=device))
        FEATURES_RANDOM_ORTH_PROJ = Q[:, :3]  # (C, 3)

    x = features.permute(1, 2, 0).reshape(-1, C)
    rgb = x @ FEATURES_RANDOM_ORTH_PROJ
    rgb = rgb.view(M, N, 3).permute(2, 0, 1)

    return rgb

class Segmenter(torch.nn.Module):
    def __init__(self, feature_dim, n_classes):
        super().__init__()
        segmentation_prototypes = torch.nn.Parameter(
            torch.nn.functional.normalize(
            torch.randn(n_classes, feature_dim, device="cuda"),
                dim=-1)
        )
        self.segmentation_prototypes = segmentation_prototypes
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        
        
    def get_prototypes(self):
        #return torch.nn.functional.normalize(torch.cat([torch.zeros(1,self.feature_dim,device=self.segmentation_prototypes.device),self.segmentation_prototypes],dim=0), dim=-1)
        return torch.nn.functional.normalize(self.segmentation_prototypes, dim=-1)
        
    def forward_with_reg_loss(self, feature_map, compute_reg_loss=False):
        W = self.get_prototypes()
        features_normalized = torch.nn.functional.normalize(feature_map, dim=1)

        # cosine similarity logits
        scale = 10
        sim = scale * torch.einsum('bchw,kc->bkhw', features_normalized, W)

        orth_constraint = torch.tensor(0.0, device=feature_map.device)
        if compute_reg_loss:
            pass
            #orth_constraint = ((W @ W.T - torch.eye(self.n_classes, device=W.device))**2).sum()

        return sim, orth_constraint
    def forward(self, feature_map):
        return self.forward_with_reg_loss(feature_map, False)[0]
    
    @classmethod
    def from_checkpoint(cls, d, device="cuda"):
        model = cls(d["feature_dim"], d["n_classes"]).to(device)
        model.load_state_dict(d["state_dict"])
        return model
        
    def capture(self):
        return {
            "state_dict": {k: v.cpu() for k, v in self.state_dict().items()},
            "feature_dim": self.feature_dim,
            "n_classes": self.n_classes,
        }
        
def structured_hinge_segmentation(
    image,        # (B, 3, H, W)
    logits,       # (B, C, H, W)
    target,       # (B, H, W)
    margin=1.0,
    lambda_smooth=1.0,
    beta=None
):
    assert image.dtype == torch.float32
    assert logits.dtype == torch.float32
    B, C, H, W = logits.shape
    image = image.detach()

    # -----------------------
    # 1. Pixel-wise hinge (unary term)
    # -----------------------
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
    target_flat = target.view(-1)

    idx = torch.arange(logits_flat.shape[0], device=logits.device)

    correct = logits_flat[idx, target_flat][:, None]

    loss = torch.clamp(logits_flat - correct + margin, min=0)
    loss[idx, target_flat] = 0
    unary_loss = loss.max(dim=1)[0].mean()

    #edge aware smoothness
    diff_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    diff_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    if beta == None: beta = 1.0 / ((diff_x.detach().pow(2).mean() + diff_y.detach().pow(2).mean()) + 1e-8)
    
    weight_x = torch.exp(-beta * (diff_x ** 2).sum(1))
    weight_y = torch.exp(-beta * (diff_y ** 2).sum(1))

    dx = ((logits[:,:,:, 1:] - logits[:,:,:,:-1]) ** 2).sum(1)
    dy = ((logits[:,:,1:,:] - logits[:,:,:-1,:]) ** 2).sum(1)

    smoothness_loss = (
        (weight_x.squeeze(1) * dx).mean() +
        (weight_y.squeeze(1) * dy).mean()
    )

    # -----------------------
    # 3. Combine
    # -----------------------
    return unary_loss + lambda_smooth * smoothness_loss

def classify_gaussians(gaussians, segmentation_network):
    #(1, N, FDIM) -> (FDIM, N, 1) -> (1, FDIM, N, 1)
    features_gaussians = gaussians.get_segmentation.expand(1,-1,-1).permute(2,1,0).unsqueeze(0)
    segmentation_gaussians = segmentation_network(torch.clamp_min(features_gaussians.detach(), 1e-6).log()).squeeze(0).squeeze(-1)
    segmentation_gaussians = torch.softmax(segmentation_gaussians, dim=-1)
    return segmentation_gaussians.permute(1,0)


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
    out = x.clone()+1

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

    
def contrastive_2d_loss(segmask, log_features, id_unique_list, n_i_list, prototypes,lambda_val=1e-4, lambda_prototypes=0.9):
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
    n_p = len(id_unique_list) # Number of ids
    
    non_zero_clusters = 0
    for i in range(n_p): 
        if n_i_list[i] > 0: non_zero_clusters+=1
        

    f_mean_per_cluster = torch.zeros((non_zero_clusters, log_features.shape[-1])).cuda()
    non_zero_prototypes = torch.zeros((non_zero_clusters, log_features.shape[-1])).cuda()
    t = torch.ones(non_zero_clusters, device=f_mean_per_cluster.device, dtype=torch.float32) * 0.1
    
    j = 0
    for i in range(n_p):
        if n_i_list[i] == 0: continue
        mask = (segmask == id_unique_list[i]).squeeze()
        #f_mean_per_cluster[j, ...] = torch.mean(features[mask,:], dim=0, keepdim=True)
        non_zero_prototypes[j,...] = prototypes[i,...]
        j+=1
            
    #f_mean_norm_per_cluster = torch.functional.norm(f_mean_per_cluster, dim=-1, keepdim=True)
    f_mean_per_cluster = non_zero_prototypes#torch.nn.functional.normalize(non_zero_prototypes, dim=-1)
    features_normalized = torch.nn.functional.normalize(log_features, dim=-1) 
    sim_per_cluster = torch.zeros(n_p).cuda()
    
    def similarity(fs, mu):
        return ((fs * mu)**2).sum(-1)
            
    j = 0
    for i in range(n_p):
        if n_i_list[i] == 0: continue
        fs_normalized = features_normalized[segmask == id_unique_list[i]]
        #similarity((N,1,8) , (1, K, 8)) = (N,K)
        sim_all = (similarity(fs_normalized.unsqueeze(1), f_mean_per_cluster)/t.unsqueeze(0))
        sim_pos = sim_all[:, j]
        sim_per_cluster[i] = (sim_pos - torch.logsumexp(sim_all, dim=-1)).mean()
        j+=1
            
    loss_obj = - lambda_val * torch.mean(sim_per_cluster)
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

    features = torch.nn.functional.normalize(features, dim=-1) 
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
    def loss(x, y):
        return ((x-y)**2).mean(dim=-1)
    
    pull_loss = loss(sample_preds.unsqueeze(1).expand(-1, k_pull, -1), neighbor_preds) #more similar of they are close
    push_loss =  2-loss(sample_preds.unsqueeze(1).expand(-1, k_push, -1), faraway_preds) #less similar if they are far away
    
    # Total loss

    loss = (lambda_pull * pull_loss.mean() + lambda_push * push_loss.mean())
    
    
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
    
    
def get_unique_id_list(segmask, min_pixnum, set_of_ids):
    id_unique_list, n_i_list_ = torch.unique(segmask, return_counts=True)
    
    labels = []
    cnt_per_label = []
    for i in range(len(id_unique_list)):
        labels.append(id_unique_list[i].item())
        if n_i_list_[i] < min_pixnum: cnt_per_label.append(0)
        else: cnt_per_label.append(n_i_list_[i].item())
    
    not_included = set_of_ids.difference(set(labels))
    labels = labels + list(not_included)
    cnt_per_label = cnt_per_label + [0]*len(not_included)
    
        

    return labels, cnt_per_label