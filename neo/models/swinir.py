"""SwinIR (Liang et al. 2021, arXiv:2108.10257) as a drop-in NEO generator: 128 px LR -> 768 px HR.

Shallow 3x3 conv -> residual Swin Transformer blocks (RSTB: shifted-window self-attention with
relative position bias, then a 3x3 conv) at LR resolution -> global residual -> pixel-shuffle
upsampling x3 then x2 (NEO's 6x) -> 3x3 conv -> tanh. The tanh keeps the output in the same
log-scaled range as the NEO U-Net generator, so losses, discriminator and data are unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def window_partition(x, ws):
    """(B, H, W, C) -> (B * num_windows, ws, ws, C)."""
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, c)


def window_reverse(windows, ws, h, w):
    """(B * num_windows, ws, ws, C) -> (B, H, W, C)."""
    b = windows.shape[0] // ((h // ws) * (w // ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


def shift_mask(h, w, ws, shift, device):
    """Attention mask keeping shifted windows from mixing pixels that were not adjacent."""
    img = torch.zeros((1, h, w, 1), device=device)
    region = 0
    for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
        for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            img[:, hs, wsl, :] = region
            region += 1
    labels = window_partition(img, ws).view(-1, ws * ws)
    diff = labels.unsqueeze(1) - labels.unsqueeze(2)
    return diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij"))
        coords = coords.flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (ws - 1)
        self.register_buffer(
            "relative_position_index", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False
        )
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        bn, n, c = x.shape
        qkv = self.qkv(x).reshape(bn, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        attn = attn + bias.view(n, n, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(bn // nw, nw, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, n, n)
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(bn, n, c))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, mlp_ratio):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, size, mask):
        h, w = size
        b, length, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(x, self.ws).view(-1, self.ws * self.ws, c)
        windows = self.attn(windows, mask if self.shift else None)
        x = window_reverse(windows.view(-1, self.ws, self.ws, c), self.ws, h, w)
        if self.shift:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))
        x = shortcut + x.reshape(b, length, c)
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):
    """Residual Swin Transformer Block: `depth` Swin layers, a 3x3 conv, and a skip connection."""

    def __init__(self, dim, depth, heads, ws, mlp_ratio, use_checkpoint):
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinTransformerBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2, mlp_ratio)
            for i in range(depth)
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, size, mask):
        h, w = size
        shortcut = x
        for blk in self.blocks:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint(blk, x, size, mask, use_reentrant=False)
            else:
                x = blk(x, size, mask)
        b, _, c = x.shape
        x = self.conv(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
        return x + shortcut


class SwinIR(nn.Module):
    """Defaults are SwinIR's classical-SR configuration (embed 180, 6 RSTB x 6 layers, window 8)."""

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        embed_dim=180,
        depths=(6, 6, 6, 6, 6, 6),
        num_heads=(6, 6, 6, 6, 6, 6),
        window_size=8,
        mlp_ratio=2.0,
        num_feat=64,
        upscale_factors=(3, 2),
        use_checkpoint=False,
    ):
        super().__init__()
        self.ws = window_size
        self.scale = math.prod(upscale_factors)
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            RSTB(embed_dim, d, h, window_size, mlp_ratio, use_checkpoint)
            for d, h in zip(depths, num_heads, strict=True)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
        )
        upsample = []
        for f in upscale_factors:
            upsample += [nn.Conv2d(num_feat, num_feat * f * f, 3, 1, 1), nn.PixelShuffle(f)]
        self.upsample = nn.Sequential(*upsample)
        self.conv_last = nn.Conv2d(num_feat, out_chans, 3, 1, 1)
        self.apply(self._init_weights)
        self._masks = {}

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _mask(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._masks:
            self._masks[key] = shift_mask(h, w, self.ws, self.ws // 2, device)
        return self._masks[key]

    def forward_features(self, x):
        b, c, h, w = x.shape
        mask = self._mask(h, w, x.device)
        tokens = self.patch_norm(x.flatten(2).transpose(1, 2))
        for layer in self.layers:
            tokens = layer(tokens, (h, w), mask)
        return self.norm(tokens).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x, identity_map=False):
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % self.ws, (-w) % self.ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_last(self.upsample(self.conv_before_upsample(x)))
        return torch.tanh(x[..., : h * self.scale, : w * self.scale])
