import math

import torch
import torch.nn as nn

from .blocks import ConvLayer


class MulticlassDiscriminator(nn.Module):

    def __init__(self,
                 in_channels: int,
                 size: int,
                 n_classes: int,
                 base_channels: int = 16,
                 max_channels: int = 512):

        super().__init__()

        blocks = [
            ConvLayer(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
        ]

        channels = base_channels
        for _ in range(int(math.log(size, 2)) - 2):
            next_channels = min(max_channels, channels * 2)
            blocks += [
                ConvLayer(channels, channels, kernel_size=3, stride=1),
                ConvLayer(channels, next_channels, kernel_size=3, stride=2),
            ]
            channels = next_channels

        blocks.append(
            ConvLayer(channels, channels, kernel_size=3, stride=1)
        )
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * channels, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels, n_classes)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        out = self.blocks(x)
        out = self.classifier(out.reshape(len(out), -1))
        index = torch.LongTensor(range(out.size(0))).cuda()
        out = out[index, y]
        return out
