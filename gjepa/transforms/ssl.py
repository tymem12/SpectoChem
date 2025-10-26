import torch
from torch import Tensor


def drop_features(x: Tensor, p: float) -> Tensor:
    """Credit: https://github.com/nerdslab/bgrl/blob/main/bgrl/transforms.py"""
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def drop_features_two_views(x: Tensor, z: Tensor, p: float) -> (Tensor, Tensor):
    assert x.size(0) == z.size(0), "x and z must have the same batch size"
    drop_mask = torch.rand(x.size(0), device=x.device) < p
    x_masked = x.clone()
    z_masked = z.clone()
    x_masked[drop_mask] = 0
    z_masked[drop_mask] = 0
    return x_masked, z_masked

def add_noise_positions(pos: Tensor, p: float, std: float) -> Tensor:
    mask = (torch.rand_like(pos) < p).float()
    noise = torch.randn_like(pos) * std
    return pos + mask * noise

def add_noise_positional_encoding(positional_encoding: Tensor, p: float, rel_std: float) -> Tensor:
    mask = (torch.rand_like(positional_encoding) < p).float()
    local_std = rel_std * positional_encoding.abs()
    noise = torch.randn_like(positional_encoding) * local_std
    return positional_encoding + mask * noise

def mask_nodes(x: Tensor, p: float) -> tuple[Tensor, Tensor]:
    """Credit: https://github.com/sycny/GiGaMAE/blob/main/data_aug.py"""
    mask = torch.empty((x.size(0),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[mask, :] = 0

    return x, mask
