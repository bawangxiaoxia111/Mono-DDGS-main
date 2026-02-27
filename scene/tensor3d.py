import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0

def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0

def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:

    grid_dim = coords.shape[-1] # 2

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

def init_grid_param(out_dim: int,reso: Sequence[int],a: float = 0.1,b: float = 0.5):
    matMode = [[1,2], [0,2], [0,1]]

    grid_coefs = nn.ParameterList()
    for idx in range(3):
        new_grid_coef = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in matMode[idx]]))
        nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


class Tensor3D(nn.Module):
    def __init__(
        self,
        bounds,
        planeconfig,
        multires
    ) -> None:
        super().__init__()

        aabb = torch.tensor([[bounds,bounds,bounds], [-bounds,-bounds,-bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [planeconfig]
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        self.matMode = torch.BoolTensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).cuda()

        # 1. Init planes
        self.denses = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [r * res for r in config["resolution"][:3]]
            gp = init_grid_param(out_dim=config["output_coordinate_dim"], reso=config["resolution"], )
            self.denses.append(gp)
            self.feat_dim += config["output_coordinate_dim"] * 3

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def set_aabb(self,xyz_max, xyz_min):
        aabb = torch.tensor([xyz_max, xyz_min],dtype=torch.float32)
        self.aabb = nn.Parameter(aabb,requires_grad=False)
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        # breakpoint()

        pts = normalize_aabb(pts, self.aabb)

        matMode = self.matMode
        coordinate_plane = torch.stack((pts[..., matMode[0]], pts[..., matMode[1]], pts[..., matMode[2]]))
        features_list = []
        feature_dim = self.denses[0][0].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
        for level in range(len(self.multiscale_res_multipliers)):
            for idx in range(3):
                sample_points = coordinate_plane[idx]
                feature = grid_sample_wrapper(self.denses[level][idx], sample_points).view(-1, feature_dim)
                features_list.append(feature)
        multi_scale_interp = torch.cat(features_list, dim=-1)
        return multi_scale_interp

    def forward(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):

        features = self.get_density(pts, timestamps)

        return features
