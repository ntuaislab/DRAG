"""
stylegan.py

This module contains the code to load a NVLabs StyleGAN model from pickle.
"""

import os
import pickle

import torch
from torch import nn, Tensor


def load_model(fname: os.PathLike) -> nn.Module:
    """ Load a pickled StyleGAN series model from a file. """
    # Need torch_utils and dnnlib from NVLabs/stylegan2-ada-pytorch
    with open(fname, 'rb') as f: # pylint: disable=invalid-name
        model = pickle.load(f)['G_ema']

    return model

def _mapping_forward(
    self,
    z : Tensor,
    c : Tensor, # => as `embed` in GAN-Inversion
    truncation_psi=1,
    truncation_cutoff=None,
    update_emas=False,
):
    if truncation_cutoff is None:
        truncation_cutoff = self.num_ws

    # Embed, normalize, and concatenate inputs.
    x = z.to(torch.float32)
    x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
    if self.c_dim > 0:
        y = self.embed_proj(c)
        y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        x = torch.cat([x, y], dim=1) if x is not None else y

    # Execute layers.
    for idx in range(self.num_layers):
        x = getattr(self, f'fc{idx}')(x)

    # Update moving average of W.
    if update_emas:
        self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

    # Broadcast and apply truncation.
    x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
    if truncation_psi != 1:
        x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
    return x

def redefine_mapping_network(mapping_network):
    mapping_network.forward = _mapping_forward.__get__(mapping_network, type(mapping_network))
    return mapping_network
