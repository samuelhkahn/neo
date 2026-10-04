import configparser
import itertools
import random

import numpy as np
import pytest
import torch
from astropy.io import fits

from neo.data.augment import ShiftD4Dataset, shift_d4, train_dataset_class
from neo.data.dataset import SR_HST_HSC_Dataset

PAD_HR, PAD_LR = 84, 14  # the dataset's reflect padding around the 600 / 100 px crops


def write_pairs(root, n=2, seed=0):
    rng = np.random.default_rng(seed)
    for split in ("hr", "lr"):
        (root / split).mkdir(parents=True)
    for i in range(n):
        lr = rng.normal(0, 0.05, size=(142, 142)).astype(np.float32)  # background-subtracted sky
        lr[rng.integers(30, 112), rng.integers(30, 112)] = 50.0  # one unique brightest LR pixel
        hr = np.kron(lr, np.ones((6, 6), np.float32))  # HR exactly nested in LR pixels
        fits.PrimaryHDU(lr).writeto(root / "lr" / f"p{i}.fits")
        fits.PrimaryHDU(hr).writeto(root / "hr" / f"p{i}.fits")


def make(cls, root):
    return cls(
        hst_path=str(root / "hr"),
        hsc_path=str(root / "lr"),
        hr_size=[600, 600],
        lr_size=[100, 100],
        transform_type="ds9_scale",
        data_aug=False,
        experiment=None,
    )


def peak(t):
    return np.unravel_index(int(torch.argmax(t)), t.shape)


def test_shift_d4_keeps_lr_nested_in_hr():
    lr = np.random.default_rng(1).random((142, 142))
    hr = np.kron(lr, np.ones((6, 6)))
    for dy, dx, k, flip in [
        (1, 0, 0, False),
        (-21, 21, 1, True),
        (7, -3, 2, False),
        (0, 0, 3, True),
    ]:
        a, b = shift_d4(lr, dy, dx, k, flip), shift_d4(hr, dy, dx, k, flip, factor=6)
        assert np.array_equal(np.kron(a, np.ones((6, 6))), b)


def test_zero_shift_identity_matches_the_plain_dataset(tmp_path):
    write_pairs(tmp_path)
    plain, aug = make(SR_HST_HSC_Dataset, tmp_path), make(ShiftD4Dataset, tmp_path)
    assert aug.max_shift == 21
    aug.draw = lambda: (0, 0, 0, False)
    for i in range(len(plain)):
        for a, b in zip(plain[i], aug[i], strict=True):
            assert torch.equal(a, b)


def test_every_view_stays_registered_and_moves_the_crop(tmp_path):
    write_pairs(tmp_path, n=1)
    plain, aug = make(SR_HST_HSC_Dataset, tmp_path), make(ShiftD4Dataset, tmp_path)
    hst0, hsc0, _, _ = plain[0]
    y0, x0 = peak(hsc0[PAD_LR:-PAD_LR, PAD_LR:-PAD_LR])  # brightest pixel in the centre crop
    for dy, dx, k, flip in itertools.product((-5, 0, 3), (0, 4), range(4), (False, True)):
        aug.draw = lambda p=(dy, dx, k, flip): p
        hst, hsc, hsc_hr, seg = aug[0]
        lr_crop = hsc[PAD_LR:-PAD_LR, PAD_LR:-PAD_LR]
        hr_crop = hst[PAD_HR:-PAD_HR, PAD_HR:-PAD_HR]
        ly, lx = peak(lr_crop)
        hy, hx = peak(hr_crop)
        assert (hy // 6, hx // 6) == (ly, lx)  # HR peak lies in the LR peak's 6x6 block
        assert peak(hsc_hr[PAD_HR:-PAD_HR, PAD_HR:-PAD_HR]) == (ly * 6, lx * 6)
        # the peak moved exactly as the transform says
        expected = np.zeros((100, 100))
        expected[y0, x0] = 1
        expected = np.pad(expected, 21)
        expected = shift_d4(expected, dy, dx, k, flip)[21:121, 21:121]
        assert (ly, lx) == tuple(int(v) for v in np.argwhere(expected)[0])
        assert hst.shape == (768, 768) and hsc.shape == (128, 128) and seg.shape == (768, 768)


def test_draw_covers_the_range_and_follows_the_seed(tmp_path):
    write_pairs(tmp_path, n=1)
    aug = make(ShiftD4Dataset, tmp_path)
    random.seed(0)
    draws = [aug.draw() for _ in range(4000)]
    random.seed(0)
    assert draws[:50] == [aug.draw() for _ in range(50)]
    assert {d[0] for d in draws} == set(range(-21, 22)) and {d[2] for d in draws} == {0, 1, 2, 3}
    assert {d[3] for d in draws} == {False, True}


def test_train_dataset_class_from_config():
    config = configparser.ConfigParser()
    assert train_dataset_class(config) == ("none", SR_HST_HSC_Dataset)
    config.read_dict({"DATA_AUG": {"augment": "shift_d4"}})
    assert train_dataset_class(config) == ("shift_d4", ShiftD4Dataset)
    config.read_dict({"DATA_AUG": {"augment": "rot45"}})
    with pytest.raises(ValueError, match="unknown"):
        train_dataset_class(config)
