from typing import Dict, Any, Tuple, Optional, List, Callable
import numpy as np
import torch
from argparse import Namespace
from arguments import PipelineParams
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.regularization.normal_field import (
    get_pivots_from_normals,
    get_signed_distance_to_depthmap,
    get_gaussian_std_in_direction,
)
from utils.densification.normal_error import compute_normal_error
from utils.geometry_utils import depth_to_normal, depth_to_normal_with_mask
from utils.general_utils import build_rotation

