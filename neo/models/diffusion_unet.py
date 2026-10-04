"""Noise-conditioned U-Net denoiser for conditional diffusion super-resolution.

Fully convolutional (no attention), so it trains on crops and samples full cutouts. Each residual
block is modulated by an embedding of the noise level (Fourier features, as in EDM's NCSN++).
Input channels: the noisy HR image and the upsampled LR conditioning image.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels):
    groups = 32 if channels % 32 == 0 else 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class FourierEmbedding(nn.Module):
    def __init__(self, channels, scale=16.0):
        super().__init__()
        self.register_buffer("freqs", torch.randn(channels // 2) * scale)

    def forward(self, x):
        x = x.ger(2 * math.pi * self.freqs)
        return torch.cat([x.cos(), x.sin()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb_dim, dropout):
        super().__init__()
        self.norm1 = group_norm(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * cout)
        self.norm2 = group_norm(cout)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return (self.skip(x) + h) / math.sqrt(2)


class DiffusionUNet(nn.Module):
    def __init__(
        self, in_channels=2, out_channels=1, base=64, mults=(1, 1, 2, 3, 4), blocks=2, dropout=0.0
    ):
        super().__init__()
        emb_dim = 4 * base
        self.factor = 2 ** (len(mults) - 1)
        self.map_noise = nn.Sequential(
            FourierEmbedding(base),
            nn.Linear(base, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, base, 3, padding=1)
        self.down = nn.ModuleList()
        skips, ch = [base], base
        for level, mult in enumerate(mults):
            for _ in range(blocks):
                self.down.append(ResBlock(ch, base * mult, emb_dim, dropout))
                ch = base * mult
                skips.append(ch)
            if level < len(mults) - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skips.append(ch)
        self.mid = nn.ModuleList(
            [ResBlock(ch, ch, emb_dim, dropout), ResBlock(ch, ch, emb_dim, dropout)]
        )
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(mults))):
            for _ in range(blocks + 1):
                self.up.append(ResBlock(ch + skips.pop(), base * mult, emb_dim, dropout))
                ch = base * mult
            if level > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.up.append(nn.Conv2d(ch, ch, 3, padding=1))
        self.norm_out = group_norm(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, noise_label):
        if x.shape[-1] % self.factor or x.shape[-2] % self.factor:
            raise ValueError(
                f"spatial size {tuple(x.shape[-2:])} must be a multiple of {self.factor}"
            )
        emb = self.map_noise(noise_label)
        h = self.conv_in(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, emb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, hs.pop()], dim=1), emb)
            else:
                h = layer(h)
        return self.conv_out(F.silu(self.norm_out(h)))
