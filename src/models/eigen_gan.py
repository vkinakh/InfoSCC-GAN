from typing import Tuple
import math

import torch
import torch.nn as nn

from .blocks import EigenBlock, ConvLayer, SubspaceLayer


class EigenGANGenerator(nn.Module):

    """Original EigenGAN"""

    def __init__(self,
                 out_channels: int,
                 size: int,
                 n_basis: int = 6,
                 noise_dim: int = 512,
                 base_channels: int = 16,
                 max_channels: int = 512):

        super(EigenGANGenerator, self).__init__()

        self.noise_dim = noise_dim
        self.n_basis = n_basis
        self.n_blocks = int(math.log(size, 2)) - 2

        def get_channels(i_block):
            return min(max_channels, base_channels * (2 ** (self.n_blocks - i_block)))

        self.fc = nn.Linear(self.noise_dim, 4 * 4 * get_channels(0))

        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            self.blocks.append(
                EigenBlock(
                    width=4 * (2 ** i),
                    height=4 * (2 ** i),
                    in_channels=get_channels(i),
                    out_channels=get_channels(i + 1),
                    n_basis=self.n_basis,
                )
            )

        self.out = nn.Sequential(
            ConvLayer(base_channels, out_channels, kernel_size=7, stride=1, padding=3, pre_activate=True),
            nn.Tanh(),
        )

    def sample_random_latent(self, batch: int, truncation: float = 1.0):
        device = self.get_device()
        es = torch.randn(batch, self.noise_dim, device=device)
        zs = torch.randn(batch, self.n_blocks, self.n_basis, device=device)

        if truncation < 1.0:
            es = torch.zeros_like(es) * (1 - truncation) + es * truncation
            zs = torch.zeros_like(zs) * (1 - truncation) + zs * truncation
        return es, zs

    def sample(self, batch: int, truncation: float = 1.0):
        latents = self.sample_random_latent(batch, truncation)
        return self.forward(latents)

    def forward(self, latents: Tuple):
        eps, zs = latents

        out = self.fc(eps).view(len(eps), -1, 4, 4)
        for block, z in zip(self.blocks, zs.permute(1, 0, 2)):
            out = block(z, out)

        return self.out(out)

    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).mean()
                )
        return sum(reg) / len(reg)

    def get_device(self):
        return self.fc.weight.device