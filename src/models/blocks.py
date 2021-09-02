from typing import Optional
from functools import partial
import math

import torch
import torch.nn as nn


def get_activation(activation: str = "lrelu"):
    actv_layers = {
        "relu":  nn.ReLU,
        "lrelu": partial(nn.LeakyReLU, 0.2),
    }
    assert activation in actv_layers, f"activation [{activation}] not implemented"
    return actv_layers[activation]


def get_normalization(normalization: str = "batch_norm"):
    norm_layers = {
        "instance_norm": nn.InstanceNorm2d,
        "batch_norm":    nn.BatchNorm2d,
        "group_norm":    partial(nn.GroupNorm, num_groups=8),
        "layer_norm":    partial(nn.GroupNorm, num_groups=1),
    }
    assert normalization in norm_layers, f"normalization [{normalization}] not implemented"
    return norm_layers[normalization]


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = 1,
        padding_mode: str = "zeros",
        groups: int = 1,
        bias: bool = True,
        transposed: bool = False,
        normalization: Optional[str] = None,
        activation: Optional[str] = "lrelu",
        pre_activate: bool = False,
    ):
        if transposed:
            conv = partial(nn.ConvTranspose2d, output_padding=stride-1)
            padding_mode = "zeros"
        else:
            conv = nn.Conv2d
        layers = [
            conv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                groups=groups,
                bias=bias,
            )
        ]

        norm_actv = []
        if normalization is not None:
            norm_actv.append(
                get_normalization(normalization)(
                    num_channels=in_channels if pre_activate else out_channels
                )
            )
        if activation is not None:
            norm_actv.append(
                get_activation(activation)(inplace=True)
            )

        if pre_activate:
            layers = norm_actv + layers
        else:
            layers = layers + norm_actv

        super().__init__(
            *layers
        )


class SubspaceLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_basis: int,
    ):
        super().__init__()

        self.U = nn.Parameter(torch.empty(n_basis, dim))
        nn.init.orthogonal_(self.U)
        self.L = nn.Parameter(torch.FloatTensor([3 * i for i in range(n_basis, 0, -1)]))
        self.mu = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return (self.L * z) @ self.U + self.mu


class EigenBlock(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        in_channels: int,
        out_channels: int,
        n_basis: int,
    ):
        super().__init__()

        self.projection = SubspaceLayer(dim=width*height*in_channels, n_basis=n_basis)
        self.subspace_conv1 = ConvLayer(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            transposed=True,
            activation=None,
            normalization=None,
        )
        self.subspace_conv2 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            transposed=True,
            activation=None,
            normalization=None,
        )

        self.feature_conv1 = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            transposed=True,
            pre_activate=True,
        )
        self.feature_conv2 = ConvLayer(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            transposed=True,
            pre_activate=True,
        )

    def forward(self, z, h):
        phi = self.projection(z).view(h.shape)
        h = self.feature_conv1(h + self.subspace_conv1(phi))
        h = self.feature_conv2(h + self.subspace_conv2(phi))
        return h
