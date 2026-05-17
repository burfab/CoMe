from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


import torch
import torch.nn as nn
from copy import deepcopy
import torch

def compute_bounds_for_deform(xyz):
    QH = xyz.quantile(0.9,0) 
    QL = xyz.quantile(0.1,0) 
    QM = xyz.quantile(0.5,0) 
    MUL = 1.5
    MIN = QM - (QH - QL)*MUL * 0.5
    MAX = QM + (QH - QL)*MUL * 0.5
    return torch.cat((MIN,MAX)).cpu().detach().numpy().tolist()
    
    
def make_deform_config(gaussians, scene, opt, num_cameras):
    deform_cfg = {
        "bound": compute_bounds_for_deform(gaussians.get_xyz),
        "hidden_dim": 64,
        "control_point_rate": 4,
        "num_degree": 3,
        "number_of_weights": 64,
        "n_levels": 16,
        "n_features_per_level": 4,
        "log2_hashmap_size": 19,
        "base_resolution": 8,
        "per_level_scale": 2,
        "num_layer": 4,
        "deform_lr_iter": opt.iterations,
        "first_iter": opt.deform_first_step,
        "weight_decay": 1e-7,
        "weight_lr": 0.001,"rotation_lr": 0.001, "scaling_lr": 0.005, "curve_lr": 0.001 ,
        "warmup": opt.deform_warmup,
        "lr_lambda": 0.99,

        "camera_extent": scene.cameras_extent.item(),
        "frame": num_cameras,
    }
    return deform_cfg

class B_Spline(nn.Module):
    def __init__(self, n_cp, p, clamp_start=False, clamp_end=False):
        assert n_cp > p
        super().__init__()
        self.n_cp = n_cp
        self.p = p
        self.clamp_start = clamp_start
        self.clamp_end = clamp_end
        self.knots = self.create_knots().cuda()

    def create_knots(self):
        #calculate knots from logits
        knots_digits = torch.zeros((self.n_cp + self.p),  dtype=torch.float, device="cuda").contiguous()
        knots_in_between = torch.softmax(knots_digits, dim=0)
        knots_before_pad = knots_in_between.cumsum(dim=0)
        pad_start = knots_before_pad.new_zeros(self.p + 1 if self.clamp_start else 1)
        pad_end = knots_before_pad.new_ones(self.p if self.clamp_end else 0)
        knots = torch.cat((pad_start, knots_before_pad, pad_end), dim=0)
        return knots

    @staticmethod
    def _robust_fraction(num, den):
        return num / (den + 1e-12)

    def forward(self, t, knots =None,sparse_output=False):
        assert 0 <= t <= 1

        # calculate knots
        if knots is not None:
            t_knots = torch.softmax(knots,dim=-1).cumsum(dim=-1)
            t_knots = torch.cat((torch.tensor([0.]).cuda(), t_knots),dim=-1)
        else:
            t_knots =self.knots

        # rescale t if either side is not clamped
        if not (self.clamp_start and self.clamp_end):
            t_start = 0 if self.clamp_start else t_knots[[self.p]]
            t_end = 1 if self.clamp_end else t_knots[[-self.p - 1]]
            t = t_start + t * (t_end - t_start)

        # set base indices and retrieve relevant knots
        idx_cp = (t_knots[1:] <= t).sum(dim=0, keepdim=True).clamp(min=self.p, max=self.n_cp - 1)
        idx_cp = idx_cp + torch.arange(-self.p, self.p + 2, dtype=idx_cp.dtype, device=idx_cp.device)
        # idx_blob = torch.arange(self.n_blob, dtype=idx_cp.dtype, device=idx_cp.device)[:, None]
        t_knots_ret = t_knots[idx_cp]

        # initialize B-spline weights
        _B = t_knots.new_ones(1)

        _dB = t_knots.new_zeros(1)

        for k in range(1, self.p + 1):
            # expand B-spline weights
            _B = nn.functional.pad(_B, (1, 1))
            _dB = nn.functional.pad(_dB, (1, 1))

            # retrieve barriers
            t_i = t_knots_ret[ (self.p - k):(self.p + 1)]
            t_i1 = t_knots_ret[ (self.p - k + 1):(self.p + 2)]
            t_ik = t_knots_ret[ self.p:(self.p + k + 1)]
            t_ik1 = t_knots_ret[ (self.p + 1):(self.p + k + 2)]

            # calculate B-spline weights
            w1 = self._robust_fraction(t - t_i, t_ik - t_i)
            w2 = self._robust_fraction(t_ik1 - t, t_ik1 - t_i1)
            w3 = self._robust_fraction(1, t_ik - t_i)
            w4 = self._robust_fraction(1, t_ik1 - t_i1)
            _B_new = _B[ :-1] * w1 + _B[ 1:] * w2
            _dB_new = _dB[ :-1] * w1 +_B[:-1]*w3 + _dB[1:]*w2 + _B[1:]*w4

            _B = _B_new
            _dB = _dB_new

        _idx_cp = idx_cp[ :(self.p + 1)]
        if sparse_output:
            return _idx_cp, _B, _dB
        else:
            B = _B.new_zeros(self.n_blob, self.n_cp)
            B[ _idx_cp] = _B
            diff_B = _dB.new_zeros(self.n_blob, self.n_cp)
            diff_B[_idx_cp] = _dB
            return B, diff_B*self.p

class simple_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=4, negative_slope=0.01):
        super().__init__()

        layers = []

        if num_layer <= 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))

            for _ in range(num_layer - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))

            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layer1 = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer1(x)


class simple_network_tcnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=4):
        super().__init__()
        self.layer1 = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layer,
            },
        )

    def forward(self, x):
        return self.layer1(x)
    
from utils.general_utils import quaternion_multiply
    
class Deformation:
    def __init__(self, d_xyz=None, d_rot=None, d_scale=None):
        self.d_xyz = d_xyz
        self.d_rot = d_rot
        self.d_scale = d_scale
    
    def apply(self, xyz, rotations, scaling, scaling_activation):
        apply_translation = lambda xyz: xyz if self.d_xyz is None else xyz + self.d_xyz
        apply_rotation = lambda rotations: rotations if self.d_rot is None else quaternion_multiply(rotations, self.d_rot)
        apply_scaling = lambda scaling_logits: scaling_logits if self.d_scale is None else scaling_logits * scaling_activation(self.d_scale)
        return apply_translation(xyz), apply_rotation(rotations), apply_scaling(scaling)

class DeformModel:
    def __init__(self, cfg):
        self.iteration =- 1
        self.optimizer = None
        self.cfg = cfg
        if 'bound' in cfg and cfg['bound'] is not None:
            self.bound = torch.tensor(cfg['bound'], device='cuda')
        else:
            self.bound = torch.tensor([-1., -1., -1., 1., 1., 1.], device='cuda')
        self.spatial_lr_scale = cfg["camera_extent"]
        self.hidden_dim = cfg["hidden_dim"]

        self.n_cp = cfg["frame"]// cfg["control_point_rate"]
        self.degree = cfg["num_degree"]
        self.n_w = cfg["number_of_weights"]
        self.b_spliner = B_Spline(self.n_cp, self.degree)

        self.deform_HASH = tcnn.Encoding(
                 n_input_dims=3,
                 dtype=torch.float32,
                 encoding_config={
                    "otype": "HashGrid",
                    "n_levels": cfg["n_levels"],
                    "n_features_per_level": cfg["n_features_per_level"],
                    "log2_hashmap_size": cfg["log2_hashmap_size"],
                    "base_resolution": cfg["base_resolution"],
                    "per_level_scale": cfg["per_level_scale"],
                },

        )
        
        simple_network_class = simple_network if True else simple_network_tcnn

        curve = (torch.randn(self.n_w, self.n_cp,6)*1e-5).contiguous().cuda()
        self.curve = torch.nn.Parameter(curve.requires_grad_(True))
        knots = torch.ones(self.n_w,self.n_cp).contiguous().cuda()
        self.knots = torch.nn.Parameter(knots.requires_grad_(True))
        self.mlp_head = simple_network_class(self.deform_HASH.n_output_dims,self.hidden_dim, self.n_w , cfg["num_layer"]).cuda()
        
    def optimizer_step(self, iteration):
        if iteration < self.cfg["first_iter"]: return
        self.optimizer.step()
        self.knots.data = torch.clamp(self.knots.data, min = 1e-6)
        if iteration < self.cfg["deform_lr_iter"]:
            self.scheduler_net.step()
    
    def optimizer_zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def capture(self):
        checkpoint = {
            "cfg": self.cfg,
            "deform_HASH": self.deform_HASH.state_dict(),
            "mlp_head": self.mlp_head.state_dict(),
            "curve": self.curve.detach().cpu(),
            "knots": self.knots.detach().cpu(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_net": self.scheduler_net.state_dict() if self.scheduler_net is not None else None,
        }
        return checkpoint
            
    @classmethod
    def restore(cls, checkpoint, iteration):
        model = cls(checkpoint["cfg"])
        model.cfg = checkpoint["cfg"]
        model.iteration = checkpoint.get("iteration", -1)

        model.deform_HASH.load_state_dict(checkpoint["deform_HASH"])
        model.mlp_head.load_state_dict(checkpoint["mlp_head"])

        with torch.no_grad():
            model.curve.copy_(checkpoint["curve"].to(model.curve.device, dtype=model.curve.dtype))
            model.knots.copy_(checkpoint["knots"].to(model.knots.device, dtype=model.knots.dtype))

        model.training_setting()
        if model.optimizer is None:
            raise RuntimeError("Optimizer is None. Call training_setting() before load(...).")
        model.optimizer.load_state_dict(checkpoint["optimizer"])

        if model.scheduler_net is None:
            raise RuntimeError("Scheduler is None. Call training_setting() before load(...).")
        model.scheduler_net.load_state_dict(checkpoint["scheduler_net"])

        return model

    @classmethod
    def load_from_checkpoint(cls, model_path, load_optimizer=True, load_scheduler=True, map_location="cuda"):
        checkpoint = torch.load(model_path, map_location=map_location)

        cfg = checkpoint["cfg"]
        model = cls(cfg)

        model.iteration = checkpoint.get("iteration", -1)
        model.deform_HASH.load_state_dict(checkpoint["deform_HASH"])
        model.mlp_head.load_state_dict(checkpoint["mlp_head"])

        with torch.no_grad():
            model.curve.copy_(checkpoint["curve"].to(model.curve.device, dtype=model.curve.dtype))
            model.knots.copy_(checkpoint["knots"].to(model.knots.device, dtype=model.knots.dtype))

        if load_optimizer or load_scheduler:
            model.training_setting()

        if load_optimizer and checkpoint.get("optimizer", None) is not None:
            model.optimizer.load_state_dict(checkpoint["optimizer"])

        if load_scheduler and checkpoint.get("scheduler_net", None) is not None:
            model.scheduler_net.load_state_dict(checkpoint["scheduler_net"])

        return model 


    def deformation(self, pc, t):
        #from pytorch3d
        
        def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
            """
            Convert rotations given as axis/angle to quaternions.

            Args:
                axis_angle: Rotations given as a vector in axis angle form,
                    as a tensor of shape (..., 3), where the magnitude is
                    the angle turned anticlockwise in radians around the
                    vector's direction.

            Returns:
                quaternions with real part first, as tensor of shape (..., 4).
            """
            angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
            sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
            return torch.cat(
                [torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1
            ) 
        

        idx_cp, weights, _ = self.b_spliner(t, sparse_output=True)
        
        #mask = pc.get_extra_feature("fg") > 0.2
        #key, _ = self.contract_to_unisphere(pc.get_xyz[mask].clone().detach(),self.bound)
        key, _ = self.contract_to_unisphere(pc.get_xyz.clone().detach(),self.bound)

        deform_feature = self.deform_HASH(key)
        deform_weight = self.mlp_head(deform_feature).float()

        b_weights = weights.unsqueeze(0)
        cp_weights = self.knots[:, idx_cp]
        weight_ = (b_weights*cp_weights)
        weight_ = weight_ / (weight_.sum(dim=-1,keepdim=True)+1e-6)

        deform_weight = torch.tanh(deform_weight)

        repre_curve = torch.einsum('wcf,wc-> wf', self.curve[:, idx_cp], weight_)
        deform_ = torch.einsum('nw,wf->nf', deform_weight, repre_curve[...,:6])

        d_xyz = deform_[...,:3]
        d_scaling = None
        d_rotation = axis_angle_to_quaternion(deform_[...,3:6])
        
        #return d_xyz and d_rotation and deform_weight as if it was for all points not just those with mask == True
        #whats the best way? it requires grad

        deform_weight = deform_weight.abs().sum(-1)
        return Deformation(d_xyz, torch.nn.functional.normalize(d_rotation), d_scaling)

    def training_setting(self):


        l = [
            {'params': self.deform_HASH.parameters(), 'lr': 0.01* self.spatial_lr_scale, "name": "deform_HASH", "weight_decay": self.cfg["weight_decay"]},
            {'params': self.mlp_head.parameters(), 'lr': 0.0001, "name": "mlp_head", "weight_decay": self.cfg["weight_decay"]},


            {'params': [self.curve], 'lr': self.cfg["curve_lr"], "name": "curve",},
            {'params': [self.knots], 'lr': self.cfg["weight_lr"], "name": "knots"},

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01/(self.cfg["warmup"]/100), total_iters=self.cfg["warmup"]
                ),

                torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=100,
                    gamma=self.cfg["lr_lambda"],
                ),
            ]
        )

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x,mask