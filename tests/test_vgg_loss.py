"""VGGLoss runs VGG19 only up to its last block with a nonzero weight (neo/models/vgg19_loss.py).

lux.ini's weights [0, 1, 1, 0, 0] need blocks 1-3 of 5. The truncated loss must equal the full
five-block computation, in value and in the gradients it sends back to the generator.
"""

import pytest
import torch
import torch.nn as nn
import torchvision

import neo.models.vgg19
from neo.models.vgg19_loss import VGGLoss

SLICES = ("slice1", "slice2", "slice3", "slice4", "slice5")


@pytest.fixture
def loss_fn(monkeypatch):
    """VGGLoss on random VGG19 weights (no pretrained download), on the CPU."""
    vgg19 = torchvision.models.vgg19
    monkeypatch.setattr(neo.models.vgg19.models, "vgg19", lambda pretrained=True: vgg19())
    torch.manual_seed(0)
    return VGGLoss("cpu", weights=[0.0, 1.0, 1.0, 0.0, 0.0])


def full_loss(vgg, weights, x, y):
    """VGGLoss.forward as it was before the truncation: all five blocks, every weight applied."""

    def features(t):
        t = vgg.normalize(vgg._duplicate_channels(vgg._tensor_zero_one_transform(t)))
        out = []
        for name in SLICES:
            t = getattr(vgg, name)(t)
            out.append(t)
        return out

    fx, fy = features(x), features(y)
    loss = 0
    for i in range(5):
        loss += weights[i] * nn.functional.mse_loss(fx[i], fy[i].detach())
    return loss


def loss_and_grads(fn, seed=1):
    """The loss of a tiny generator's output against a target, and that generator's gradients."""
    torch.manual_seed(seed)
    gen = nn.Conv2d(1, 1, 3, padding=1)
    x = torch.tanh(gen(torch.randn(2, 1, 64, 64)))
    y = torch.rand(2, 1, 64, 64) * 2 - 1
    loss = fn(x, y)
    loss.backward()
    return loss.detach(), [p.grad.clone() for p in gen.parameters()]


def blocks_run(vgg, fn):
    """Names of the VGG blocks that ran during fn()."""
    ran = []
    hooks = [
        getattr(vgg, name).register_forward_hook(lambda m, i, o, name=name: ran.append(name))
        for name in SLICES
    ]
    fn()
    for h in hooks:
        h.remove()
    return sorted(set(ran))


@pytest.mark.parametrize(
    "weights",
    [
        [0.0, 1.0, 1.0, 0.0, 0.0],  # lux.ini
        [1.0, 1.0, 0.0, 0.0, 0.0],  # example.ini's default in Pix2Pix
        [0.5, 1.0, 2.0, 0.25, 3.0],  # all nonzero: nothing to skip
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
)
def test_truncated_loss_and_gradients_equal_the_full_computation(loss_fn, weights):
    loss_fn.weights = weights
    loss, grads = loss_and_grads(loss_fn)
    expected_loss, expected_grads = loss_and_grads(
        lambda x, y: full_loss(loss_fn.vgg, weights, x, y)
    )
    assert torch.equal(loss, expected_loss)
    for g, e in zip(grads, expected_grads, strict=True):
        assert torch.equal(g, e)


def test_only_blocks_up_to_the_last_nonzero_weight_run(loss_fn):
    x, y = torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64)
    cases = {
        (0.0, 1.0, 1.0, 0.0, 0.0): SLICES[:3],
        (0.0, 1.0, 1.0): SLICES[:3],  # a missing trailing weight is a zero one
        (1.0, 0.0, 0.0, 0.0, 0.0): SLICES[:1],
        (0.0, 0.0, 0.0, 0.0, 0.0): SLICES[:1],  # still a tensor loss (0), as before
        (0.0, 0.0, 0.0, 0.0, 1.0): SLICES,
    }
    for weights, expected in cases.items():
        loss_fn.weights = list(weights)
        assert blocks_run(loss_fn.vgg, lambda: loss_fn(x, y)) == sorted(expected), weights


def test_zero_or_missing_trailing_weights_never_change_the_result(loss_fn):
    results = []
    for weights in ([0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0]):
        loss_fn.weights = weights
        results.append(loss_and_grads(loss_fn))
    for loss, grads in results[1:]:
        assert torch.equal(loss, results[0][0])
        assert all(torch.equal(g, e) for g, e in zip(grads, results[0][1], strict=True))
    loss_fn.weights = [0.0, 0.0, 0.0, 0.0, 0.0]
    zero = loss_fn(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64))
    assert torch.is_tensor(zero) and zero.item() == 0  # train.py calls .item() on it


def test_vgg_returns_every_block_by_default_and_the_first_n_on_request(loss_fn):
    vgg = loss_fn.vgg
    x = torch.rand(1, 1, 64, 64)
    with torch.no_grad():
        everything = vgg(x)
        first_two = vgg(x, 2)
    assert [f.shape[1] for f in everything] == [64, 128, 256, 512, 512]
    assert len(first_two) == 2
    assert all(torch.equal(a, b) for a, b in zip(first_two, everything[:2], strict=True))
