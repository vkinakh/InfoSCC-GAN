from math import log2
from functools import partial

import torch
import torch.nn as nn

from linear_attention_transformer import ImageLinearAttention


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)


def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),
        leaky_relu(),
        nn.Conv2d(chan_out, chan_out, 3, padding=1),
        leaky_relu()
    )


class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x.flatten(self.index)


class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride=2)
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size=(h * 2, w * 2))
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x


attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries=True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])


class UnetDiscriminator(nn.Module):
    def __init__(self, image_size, n_channels: int, network_capacity=16, fmap_max=512):
        super().__init__()
        num_layers = int(log2(image_size) - 3)
        num_init_filters = n_channels

        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample=is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        b, *_ = x.shape

        residuals = []

        for (down_block, attn_block) in zip(self.down_blocks, self.attn_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return enc_out.squeeze(), dec_out


if __name__ == '__main__':

    from src.loss import get_adversarial_losses

    img_size = 256
    n_channels = 3
    img = torch.randn((16, n_channels, img_size, img_size))

    disc = UnetDiscriminator(img_size, n_channels)

    enc_out, dec_out = disc(img)
    print(enc_out.shape, dec_out.shape)

    d_loss, _ = get_adversarial_losses()

    enc_loss = d_loss(enc_out, enc_out)
    dec_loss = d_loss(dec_out, dec_out)
    print(enc_loss.item(), dec_loss.item())
