"""VGG-19 perceptual loss for encouraging high-level feature similarity."""

import torch.nn as nn

from neo.models.vgg19 import Vgg19


class VGGLoss(nn.Module):
    """Weighted multi-scale perceptual loss using VGG-19 features.

    Computes MSE between VGG feature maps of generated and target images
    at 5 hierarchical levels, with configurable per-layer weights.

    Args:
        device: Torch device for the VGG model.
        weights: List of 5 floats weighting each VGG layer's contribution to the loss.
    """

    def __init__(self, device, weights):
        super().__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.MSELoss()
        self.weights = weights

    def forward(self, x, y):
        """Compute perceptual loss between generated image x and target y.

        Args:
            x: Generated image tensor.
            y: Target (ground truth) image tensor.

        Returns:
            Weighted sum of MSE losses across VGG feature layers.
        """
        # Blocks after the last nonzero weight would only add 0 * loss (and zero gradients): skip
        # them. lux.ini's weights [0, 1, 1, 0, 0] need blocks 1-3 of 5. A missing weight is a 0.
        n_slices = max((i + 1 for i, w in enumerate(self.weights[:5]) if w != 0), default=1)
        x_vgg, y_vgg = self.vgg(x, n_slices), self.vgg(y, n_slices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
