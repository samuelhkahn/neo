import torch.nn as nn


class ConvPixelShuffle(nn.Module):
    """Sub-pixel convolution: a 3x3 conv producing r^2 x out_channels maps, then PixelShuffle(r)."""

    def __init__(self, in_channels, out_channels, upscale_factor, kernel_size=3):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels * upscale_factor**2, kernel_size, padding="same"
        )
        self.upsample = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.upsample(self.convolution(x))
