from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import EigenBlock, ConvLayer, SubspaceLayer


class ConditionalGenerator(nn.Module):

    """Conditional generator
    It generates images from one hot label + noise sampled from N(0, 1) with explorable z injection space
    Based on EigenGAN
    """

    def __init__(self,
                 size: int,
                 y_size: int,
                 z_size: int,
                 out_channels: int = 3,
                 n_basis: int = 6,
                 noise_dim: int = 512,
                 base_channels: int = 16,
                 max_channels: int = 512,
                 y_type: str = 'one_hot'):

        if y_type not in ['one_hot', 'multi_label', 'mixed', 'real']:
            raise ValueError('Unsupported `y_type`')

        super(ConditionalGenerator, self).__init__()

        assert (size & (size - 1) == 0) and size != 0, "img size should be a power of 2"

        self.y_type = y_type
        self.y_size = y_size
        self.eps_size = z_size

        self.noise_dim = noise_dim
        self.n_basis = n_basis
        self.n_blocks = int(math.log(size, 2)) - 2

        def get_channels(i_block):
            return min(max_channels, base_channels * (2 ** (self.n_blocks - i_block)))

        self.y_fc = nn.Linear(self.y_size, self.y_size)
        self.concat_fc = nn.Linear(self.y_size + self.eps_size, self.noise_dim)

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

    def forward(self,
                y: torch.Tensor,
                eps: Optional[torch.Tensor] = None,
                zs: Optional[torch.Tensor] = None,
                return_eps: bool = False):

        bs = y.size(0)

        if eps is None:
            eps = self.sample_eps(bs)

        if zs is None:
            zs = self.sample_zs(bs)

        y_out = self.y_fc(y)
        concat = torch.cat((y_out, eps), dim=1)
        concat = self.concat_fc(concat)

        out = self.fc(concat).view(len(eps), -1, 4, 4)
        for block, z in zip(self.blocks, zs.permute(1, 0, 2)):
            out = block(z, out)
        out = self.out(out)

        if return_eps:
            return out, concat

        return out

    def sample_zs(self, batch: int, truncation: float = 1.):
        device = self.get_device()
        zs = torch.randn(batch, self.n_blocks, self.n_basis, device=device)

        if truncation < 1.:
            zs = torch.zeros_like(zs) * (1 - truncation) + zs * truncation
        return zs

    def sample_eps(self, batch: int, truncation: float = 1.):
        device = self.get_device()
        eps = torch.randn(batch, self.eps_size, device=device)

        if truncation < 1.:
            eps = torch.zeros_like(eps) * (1 - truncation) + eps * truncation
        return eps

    def sample(self, batch: int):
        device = self.get_device()

        if self.y_type == 'one_hot':
            y = torch.randint(self.y_size, (batch,))
            y = F.one_hot(y, num_classes=self.y_size).float().to(device)
        elif self.y_type in ['multi_label', 'real']:
            y = torch.randint(2, (batch, self.y_size)).float().to(device)
        elif self.y_type == 'mixed':
            # first half of samples are one-hot encoded, other are multilabel
            k = self.y_size // 2
            y_one_hot = torch.randint(k, (batch,))
            y_one_hot = F.one_hot(y_one_hot, num_classes=k)
            y_mult = torch.randint(2, (batch, k))

            y = torch.cat((y_one_hot, y_mult), dim=1).float().to(device)

        return self.forward(y)

    def get_device(self):
        return self.fc.weight.device

    def orthogonal_regularizer(self):
        reg = []
        for layer in self.modules():
            if isinstance(layer, SubspaceLayer):
                UUT = layer.U @ layer.U.t()
                reg.append(
                    ((UUT - torch.eye(UUT.shape[0], device=UUT.device)) ** 2).mean()
                )
        return sum(reg) / len(reg)
