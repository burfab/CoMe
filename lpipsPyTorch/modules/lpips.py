import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]

        if mask is not None:
            mask = mask.float()

        res = []
        for d, l in zip(diff, self.lin):
            v = l(d)  # [B,1,H,W]

            if mask is not None:
                # resize mask to feature resolution
                m = torch.nn.functional.interpolate(
                    mask,
                    size=v.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

                v = (v * m).sum((2, 3), True) / (m.sum((2, 3), True) + 1e-8)
            else:
                v = v.mean((2, 3), True)

            res.append(v)

        return torch.sum(torch.cat(res, 0), 0, True)