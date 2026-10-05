import ast
import configparser
import itertools
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from astropy.io import fits

from neo.data.augment import (
    Paper8BitMixin,
    ShiftD4Dataset,
    ShiftDataset,
    dataset_classes,
    hsc_hr_8bit_mode,
    paper_8bit,
    shift_d4,
    train_dataset_class,
)
from neo.data.dataset import SR_HST_HSC_Dataset

PAD_HR, PAD_LR = 84, 14  # the dataset's reflect padding around the 600 / 100 px crops
CONFIGS = Path(__file__).resolve().parents[1] / "neo" / "configs"
AUGMENTS = {"none": SR_HST_HSC_Dataset, "shift": ShiftDataset, "shift_d4": ShiftD4Dataset}


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


def roll_pairs(src, dst, dy, dx):
    """Copy the pairs in src to dst, rolled by (dy, dx) LR pixels (6x for HR)."""
    for split, factor in (("lr", 1), ("hr", 6)):
        (dst / split).mkdir(parents=True)
        for path in (src / split).iterdir():
            rolled = np.roll(fits.getdata(path), (dy * factor, dx * factor), axis=(0, 1))
            fits.PrimaryHDU(rolled).writeto(dst / split / path.name)


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


def make_config(augment=None, mode=None):
    config = configparser.ConfigParser()
    if augment is not None:
        config.read_dict({"DATA_AUG": {"data_aug": "False", "augment": augment}})
    if mode is not None:
        config.read_dict({"DATASET": {"hsc_hr_8bit": mode}})
    return config


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
    plain = make(SR_HST_HSC_Dataset, tmp_path)
    for cls in (ShiftDataset, ShiftD4Dataset):
        aug = make(cls, tmp_path)
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


class RecordingShift(ShiftDataset):
    def draw(self):
        self.last = super().draw()
        return self.last


def test_shift_mode_never_rotates_or_flips(tmp_path):
    write_pairs(tmp_path / "pairs", n=1)
    aug = make(RecordingShift, tmp_path / "pairs")
    random.seed(0)
    draws = [aug.draw() for _ in range(4000)]
    assert {(k, flip) for _, _, k, flip in draws} == {(0, False)}
    assert {d[0] for d in draws} == {d[1] for d in draws} == set(range(-21, 22))
    # each random view is exactly the plain dataset on the pair rolled by the drawn shift
    random.seed(1)
    for i in range(4):
        views = aug[0]
        dy, dx, _, _ = aug.last
        roll_pairs(tmp_path / "pairs", tmp_path / f"rolled{i}", dy, dx)
        for a, b in zip(make(SR_HST_HSC_Dataset, tmp_path / f"rolled{i}")[0], views, strict=True):
            assert torch.equal(a, b)


def test_shift_mode_stays_registered(tmp_path):
    write_pairs(tmp_path, n=1)
    plain, aug = make(SR_HST_HSC_Dataset, tmp_path), make(ShiftDataset, tmp_path)
    y0, x0 = peak(plain[0][1][PAD_LR:-PAD_LR, PAD_LR:-PAD_LR])
    for dy, dx in itertools.product((-7, 0, 5), (-3, 0, 6)):
        aug.draw = lambda p=(dy, dx, 0, False): p
        hst, hsc, hsc_hr, _ = aug[0]
        ly, lx = peak(hsc[PAD_LR:-PAD_LR, PAD_LR:-PAD_LR])
        hy, hx = peak(hst[PAD_HR:-PAD_HR, PAD_HR:-PAD_HR])
        assert (ly, lx) == (y0 + dy, x0 + dx)  # a pure translation of the crop
        assert (hy // 6, hx // 6) == (ly, lx)
        assert peak(hsc_hr[PAD_HR:-PAD_HR, PAD_HR:-PAD_HR]) == (ly * 6, lx * 6)


X = [-1, -0.5, -0.004, 0, 0.3, 0.999, 1.2]


@pytest.mark.parametrize(
    "mode, expected",
    [("wrap", [1, 129, 255, 0, 76, 254, 50]), ("saturate", [0, 0, 0, 0, 76, 254, 255])],
)
def test_paper_8bit_values(mode, expected):
    out = paper_8bit(torch.tensor(X), mode)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor(expected, dtype=torch.float32) / 255)


@pytest.mark.parametrize("mode", ["wrap", "saturate"])
def test_paper_8bit_zero_is_positive(mode):
    # like ToTensor of a uint8 0: no -0.0 from small negative values
    out = paper_8bit(torch.tensor([-0.001, -0.0, 0.0, 0.001]), mode)
    assert torch.equal(out, torch.zeros(4)) and not torch.signbit(out).any()


def test_paper_8bit_none_and_unknown():
    x = torch.tensor(X)
    assert paper_8bit(x, "none") is x
    with pytest.raises(ValueError, match="unknown"):
        paper_8bit(x, "round")


@pytest.mark.parametrize("mode", ["wrap", "saturate"])
def test_paper_8bit_matches_torchvision_in_range(mode):
    # inside [0, 1) the cast is well defined: ToPILImage() without a mode, then ToTensor
    x = torch.rand(1, 64, 64, generator=torch.Generator().manual_seed(0))
    x[0, 0, :3] = torch.tensor([0.0, 254.99 / 255, 1 / 255])
    assert torch.equal(T.ToTensor()(T.ToPILImage()(x)), paper_8bit(x, mode))


@pytest.mark.parametrize("augment", list(AUGMENTS))
@pytest.mark.parametrize("mode", ["wrap", "saturate"])
def test_hsc_hr_8bit_applies_to_train_and_val_only_on_hsc_hr(tmp_path, augment, mode):
    write_pairs(tmp_path)
    _, train_cls, val_cls = dataset_classes(make_config(augment, mode))
    plain = make(SR_HST_HSC_Dataset, tmp_path)
    expected = [plain[i] for i in range(len(plain))]
    for cls in (train_cls, val_cls):
        ds = make(cls, tmp_path)
        if isinstance(ds, ShiftDataset):
            ds.draw = lambda: (0, 0, 0, False)  # the plain dataset's crop
        for i in range(len(ds)):
            hst, hsc, hsc_hr, seg = ds[i]
            p_hst, p_hsc, p_hsc_hr, p_seg = expected[i]
            assert torch.equal(hst, p_hst) and torch.equal(hsc, p_hsc) and torch.equal(seg, p_seg)
            assert torch.equal(hsc_hr, paper_8bit(p_hsc_hr, mode))
            assert hsc_hr.dtype == torch.float32 and not torch.equal(hsc_hr, p_hsc_hr)
            assert torch.isin(hsc_hr, torch.arange(256, dtype=torch.float32) / 255).all()


@pytest.mark.parametrize("augment", [None, *AUGMENTS])
@pytest.mark.parametrize("mode", [None, "none", "wrap", "saturate"])
def test_dataset_classes_for_every_config(augment, mode):
    config = make_config(augment, mode)
    name, train, val = dataset_classes(config)
    base = AUGMENTS[augment or "none"]
    assert name == (augment or "none") and hsc_hr_8bit_mode(config) == (mode or "none")
    assert train_dataset_class(config) == (name, train)
    assert issubclass(train, base) and issubclass(val, SR_HST_HSC_Dataset)
    assert not issubclass(val, ShiftDataset)  # validation is never augmented
    if mode in (None, "none"):
        assert (train, val) == (base, SR_HST_HSC_Dataset)
    else:
        assert issubclass(train, Paper8BitMixin) and issubclass(val, Paper8BitMixin)
        assert train.hsc_hr_8bit == val.hsc_hr_8bit == mode
    for cls in (train, val):  # importable by name, so spawned DataLoader workers can unpickle it
        assert pickle.loads(pickle.dumps(cls)) is cls


def test_unknown_config_values_raise():
    with pytest.raises(ValueError, match="augment"):
        dataset_classes(make_config("rot45"))
    with pytest.raises(ValueError, match="hsc_hr_8bit"):
        dataset_classes(make_config("shift", "round"))


def test_train_dataset_class_from_config():
    config = configparser.ConfigParser()
    assert train_dataset_class(config) == ("none", SR_HST_HSC_Dataset)
    config.read_dict({"DATA_AUG": {"augment": "shift_d4"}})
    assert train_dataset_class(config) == ("shift_d4", ShiftD4Dataset)
    config.read_dict({"DATA_AUG": {"augment": "shift"}})
    assert train_dataset_class(config) == ("shift", ShiftDataset)
    config.read_dict({"DATA_AUG": {"augment": "rot45"}})
    with pytest.raises(ValueError, match="unknown"):
        train_dataset_class(config)


def test_old_configs_without_dataset_section_behave_as_before():
    # before [DATASET] existed: augment picked the training class, validation used the plain one
    assert dataset_classes(make_config()) == ("none", SR_HST_HSC_Dataset, SR_HST_HSC_Dataset)
    assert dataset_classes(make_config("shift_d4")) == (
        "shift_d4",
        ShiftD4Dataset,
        SR_HST_HSC_Dataset,
    )
    example = configparser.ConfigParser()
    example.read(CONFIGS / "example.ini")
    assert not example.has_section("DATASET")
    assert dataset_classes(example) == ("none", SR_HST_HSC_Dataset, SR_HST_HSC_Dataset)


def test_train_py_builds_each_loader_from_its_class():
    # train.py has no tests of its own: check that the loaders use dataset_classes' classes
    tree = ast.parse((CONFIGS.parents[1] / "train.py").read_text())
    unpacked = [
        [t.id for t in node.targets[0].elts]
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "dataset_classes"
    ]
    assert len(unpacked) == 1
    _, train_name, val_name = unpacked[0]
    loaders = {
        node.targets[0].id: node.value.args[0].func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", None) == "DataLoader"
    }
    assert loaders == {"dataloader_train": train_name, "dataloader_val": val_name}


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.ini")), ids=lambda p: p.name)
def test_shipped_configs_resolve(path):
    config = configparser.ConfigParser()
    config.read(path)
    name, train, val = dataset_classes(config)
    assert getattr(train, "hsc_hr_8bit", "none") == getattr(val, "hsc_hr_8bit", "none")
