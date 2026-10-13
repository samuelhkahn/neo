"""Two ablations of the NEO generator (neo/models/generator.py), chosen by [MODEL] generator.

neo_relu (Arm A, NeoReLUGenerator)
    The paper's expansion operator is E_f(X) = ReLU(BN(T(X))), but UpSampleConv builds its ReLU
    (self.act) and its forward never applies it, so NEO's seven-block decoder (26.5 M parameters)
    is linear in practice. Arm A applies each decoder block's ReLU after batch norm and dropout,
    as the paper describes. Nothing else changes: the same modules, state_dict keys and
    parameter count as Pix2PixGenerator.

neo_lrtrunk (Arm B, NeoLRTrunkGenerator)
    Arm A's U-Net up to its last decoder block (32 channels at 128 px), plus a super-resolution
    trunk on the full 128 px LR grid: the input itself enters through a 3x3 conv, so per-LR-pixel
    detail no longer has to survive the stride-2 first encoder layer (NEO's finest skip is at
    64 px). The two are merged into 64 channels, refined by 6 residual channel-attention blocks
    (RCAN, Zhang et al. 2018) with a conv and a long skip around them, and upsampled by
    ICNR-initialised sub-pixel convs (3x, PReLU, one residual block at 384 px, 2x, PReLU) and a
    3x3 output conv with tanh. The parent's upsampler and 1x1 output conv are removed.

Both can record, for each decoder block, the fraction of its batch-norm output that is <= 0, i.e.
what the ReLU zeroes apart from dropout: it is measured before Dropout2d (d1-d3, p = 0.5), which in
train mode zeroes half the channels on top (there about 0.5 + 0.5 f of the ReLU outputs are zero).
Only while record_activations is set (train.py sets it for the validation pass at display steps,
logged as "Decoder ReLU zero fraction/dN"), so training steps pay nothing. activation_stats()
returns them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neo.models.conv_pixel_shuffle import ConvPixelShuffle
from neo.models.generator import Pix2PixGenerator


def icnr_(conv: nn.Conv2d, r: int, init=nn.init.kaiming_normal_) -> nn.Conv2d:
    """ICNR initialisation (Aitken et al. 2017) of a conv whose output goes to PixelShuffle(r).

    PixelShuffle(r) fills pixel (i, j) of each r x r output block of channel c from conv channel
    c * r**2 + i * r + j. One sub-filter is drawn per output channel c and copied to all r**2 of
    its conv channels (biases start at zero), so at initialisation conv + shuffle equals the conv
    followed by nearest-neighbour upsampling: training does not start from, and have to unlearn,
    the checkerboard pattern of independently initialised sub-filters. `init` draws the
    sub-filters (Kaiming normal, as in the reference implementations, since PReLU follows).
    """
    out_c, in_c, kh, kw = conv.weight.shape
    if out_c % (r * r):
        raise ValueError(f"{out_c} conv channels are not a multiple of r**2 = {r * r}")
    sub = torch.empty(
        out_c // (r * r), in_c, kh, kw, dtype=conv.weight.dtype, device=conv.weight.device
    )
    init(sub)
    with torch.no_grad():
        conv.weight.copy_(sub.repeat_interleave(r * r, dim=0))
        if conv.bias is not None:
            conv.bias.zero_()
    return conv


class ResBlock(nn.Module):
    """EDSR-style residual block without batch norm: x + res_scale * conv(PReLU(conv(x)))."""

    def __init__(self, ch, res_scale=0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.PReLU(ch), nn.Conv2d(ch, ch, 3, padding=1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class RCAB(nn.Module):
    """Residual channel-attention block (RCAN, Zhang et al. 2018).

    body = conv3x3-ReLU-conv3x3; channel attention = global average pool, a 1x1 squeeze to
    ch // reduction channels, ReLU, a 1x1 excitation back to ch, sigmoid.
    Output: x + body(x) * attention(body(x)).
    """

    def __init__(self, ch, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(ch, ch, 3, padding=1)
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.body(x)
        return x + y * self.attention(y)


class NeoReLUGenerator(Pix2PixGenerator):
    """Arm A: Pix2PixGenerator with the decoder ReLU the paper specifies and the code never applied.

    Same constructor, modules and state_dict keys as Pix2PixGenerator; only forward differs. So a
    strict load_state_dict cannot tell their checkpoints apart: train.py records the generator in
    each checkpoint and the loaders refuse a mismatch (neo/models/registry.py check_generator).
    """

    def __init__(self, in_channels=1, out_channels=1, n_ps_blocks=2, resize_conv=True):
        super().__init__(in_channels, out_channels, n_ps_blocks, resize_conv)
        self.record_activations = False
        self._zero_fraction = {}

    def _decode(self, name, block, x):
        """UpSampleConv.forward plus its ReLU: deconv -> batch norm -> dropout -> ReLU.

        F.relu, not the block's in-place nn.ReLU(True), so no tensor that another op may still
        need is overwritten. The zero fraction is measured before dropout: Dropout2d (d1-d3,
        p = 0.5 in train mode) zeroes whole channels and keeps the sign of the rest, so measuring
        after it would add ~0.5 that is dropout's doing, not the ReLU's, and d1-d3 would not
        compare with d4-d7.
        """
        x = block.deconv(x)
        if block.batchnorm:
            x = block.bn(x)
        if block.activation and self.record_activations:
            self._zero_fraction[name] = (x <= 0).float().mean().detach()
        if block.dropout:
            x = block.drop(x)
        if block.activation:
            x = F.relu(x)
        return x

    def _unet(self, x):
        """Pix2PixGenerator.forward's encoder-decoder up to the last decoder block (32 x 128 px)."""
        if self.record_activations:
            self._zero_fraction = {}
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        skips = list(reversed(skips[:-1]))
        for i, (decoder, skip) in enumerate(zip(self.decoders[:-1], skips, strict=True)):
            x = torch.cat((self._decode(f"d{i + 1}", decoder, x), skip), axis=1)
        return self._decode(f"d{len(self.decoders)}", self.decoders[-1], x)

    def forward(self, x, identity_map=False):
        x = self._unet(x)
        x = self.ps_blocks(x)
        x = self.final_conv(x)
        return self.tanh(x)

    def activation_stats(self) -> dict:
        """{decoder block: fraction of its batch-norm output <= 0, before dropout} from the last
        recorded forward pass."""
        if not self._zero_fraction:
            return {}
        values = torch.stack(list(self._zero_fraction.values())).tolist()  # one device sync
        return dict(zip(self._zero_fraction, values, strict=True))


class NeoLRTrunkGenerator(NeoReLUGenerator):
    """Arm B: Arm A's U-Net as context, plus an RCAN trunk at the LR grid and a wider upsampler.

    Channel choices follow the scratch prototype that was profiled (LRTrunkGenerator in
    neo_outputs/scratch/curves_fine_structure/architecture/variants.py): 64 trunk channels,
    32 after each shuffle as in NEO.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_ps_blocks=2,
        resize_conv=True,
        ch=64,
        n_blocks=6,
        reduction=16,
        n_res_384=1,
    ):
        super().__init__(in_channels, out_channels, n_ps_blocks, resize_conv)
        # Replaced below: dropped so they are neither counted nor handed to the optimizer
        del self.ps_blocks, self.final_conv
        unet_ch = self.decoders[-1].deconv.out_channels  # 32
        self.head = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.merge = nn.Conv2d(ch + unet_ch, ch, 3, padding=1)
        self.trunk = nn.Sequential(
            *[RCAB(ch, reduction) for _ in range(n_blocks)], nn.Conv2d(ch, ch, 3, padding=1)
        )
        up3 = ConvPixelShuffle(ch, 32, upscale_factor=3)
        up2 = ConvPixelShuffle(32, 32, upscale_factor=2)
        icnr_(up3.convolution, 3)
        icnr_(up2.convolution, 2)
        self.upsampler = nn.Sequential(
            up3, nn.PReLU(32), *[ResBlock(32) for _ in range(n_res_384)], up2, nn.PReLU(32)
        )
        self.out_conv = nn.Conv2d(32, out_channels, 3, padding=1)

    def forward(self, x, identity_map=False):
        context = self._unet(x)  # 32 x 128 x 128
        feat = self.merge(torch.cat((self.head(x), context), axis=1))
        feat = feat + self.trunk(feat)  # long skip around the trunk, as in EDSR / RCAN
        return self.tanh(self.out_conv(self.upsampler(feat)))
