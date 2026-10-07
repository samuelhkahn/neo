"""Train-time augmentation that keeps LR and HR exactly registered, and the paper's 8-bit hsc_hr.

SR_HST_HSC_Dataset centre-crops every stored pair (142 px LR / 852 px HR) to 100 / 600 px.
ShiftDataset moves that crop: before the dataset's own processing it rolls both cutouts by the
same whole number of LR pixels (6x as many HR pixels, so LR pixels stay nested in HR blocks), up to
the 21 px of slack around the centre crop, giving 43 x 43 distinct training views per pair, all
inside its stored footprint (which is what the train/val leakage check covers). ShiftD4Dataset
also applies the same random 90-degree rotation and flip (the 8 lossless symmetries of the pixel
grid), for 43 x 43 x 8 views; the paper had no rotations or flips. Rot90Dataset only rotates (by
0, 90, 180 or 270 degrees; no shift, no flip): the centre crop in 4 orientations. Everything
after loading (clip, log scaling, segmentation map, crop, padding) is the dataset's unchanged
code. Validation and evaluation keep the plain centre crop.

The paper's code passed hsc_hr (the discriminator's conditioning image) through ToPILImage()
without a mode, which cast it to 8 bits; paper_8bit reproduces that on hsc_hr only, for both the
training and the validation dataset (the paper used one dataset class for both).

Config: [DATA_AUG] augment = none | shift | shift_d4 | rot90 (default: none) picks the training
class; [DATASET] hsc_hr_8bit = none | wrap | saturate (default: none) applies to training and
validation.
"""

import os
import random

import numpy as np
import torch
from astropy.io import fits

from neo.data.dataset import SR_HST_HSC_Dataset

LR_CROP = 100
FACTOR = 6
HSC_HR_8BIT = ("none", "wrap", "saturate")


def shift_d4(array, dy, dx, k, flip, factor=1):
    """Roll by (dy, dx) LR pixels (x factor), rotate by k * 90 degrees, then optionally flip."""
    array = np.roll(array, (dy * factor, dx * factor), axis=(0, 1))
    array = np.rot90(array, k)
    if flip:
        array = array[:, ::-1]
    return np.ascontiguousarray(array)


def paper_8bit(x, mode):
    """The paper code's 8-bit round trip of a float tensor (main:neo/data/dataset.py:333).

    pad_array_hr's ToPILImage() without a mode did pic.mul(255).byte() (torchvision 0.10) and
    ToTensor divided by 255. The cast of values outside [0, 255] depends on the CPU code path:
    "wrap" keeps the low 8 bits, "saturate" clamps. Elementwise, so applying it after the reflect
    padding equals the paper's order (cast, then pad).
    """
    if mode == "none":
        return x
    if mode not in HSC_HR_8BIT:
        raise ValueError(f"unknown hsc_hr_8bit mode {mode!r}; choose from {list(HSC_HR_8BIT)}")
    t = (x.to(torch.float32) * 255).long()  # truncates toward zero, like the byte cast
    t = torch.remainder(t, 256) if mode == "wrap" else t.clamp(0, 255)
    return t.to(torch.float32) / 255


class ShiftDataset(SR_HST_HSC_Dataset):
    # Paper: data_aug=False but ~1.5M random crops; random shifts of 36k cutouts stand in for them.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hst_dir = os.path.normpath(self.hst_path)
        self._params = None
        first = os.path.join(self.hsc_path, self.filenames[0]) if self.filenames else None
        size = fits.getheader(first)["NAXIS1"] if first else LR_CROP
        self.max_shift = (size - LR_CROP) // 2

    def draw(self):
        """Random (dy, dx, 0, False); `random` is reseeded per DataLoader worker by PyTorch."""
        m = self.max_shift
        return random.randint(-m, m), random.randint(-m, m), 0, False

    def __getitem__(self, idx):
        self._params = self.draw()
        try:
            return super().__getitem__(idx)
        finally:
            self._params = None

    def load_fits(self, file_path):
        array = super().load_fits(file_path)
        if self._params is None:
            return array
        hr = os.path.normpath(os.path.dirname(file_path)) == self._hst_dir
        factor = FACTOR if hr else 1
        if array.shape[0] != array.shape[1] or (array.shape[0] // factor - LR_CROP) // 2 < max(
            abs(self._params[0]), abs(self._params[1])
        ):
            raise ValueError(f"{file_path}: {array.shape} leaves no room for the shift")
        return shift_d4(array, *self._params, factor=factor)


class ShiftD4Dataset(ShiftDataset):
    def draw(self):
        """Random (dy, dx, k, flip); `random` is reseeded per DataLoader worker by PyTorch."""
        m = self.max_shift
        return (
            random.randint(-m, m),
            random.randint(-m, m),
            random.randrange(4),
            random.random() < 0.5,
        )


class Rot90Dataset(ShiftDataset):
    def draw(self):
        """No shift or flip; rotation by 0, 90, 180 or 270 degrees."""
        return 0, 0, random.randrange(4), False


class Paper8BitMixin:
    """Passes hsc_hr through paper_8bit(hsc_hr, self.hsc_hr_8bit); hst, hsc, segmap untouched."""

    hsc_hr_8bit = "none"

    def __getitem__(self, idx):
        hst, hsc, hsc_hr, seg = super().__getitem__(idx)
        if hsc_hr is None:
            return hst, hsc, hsc_hr, seg
        return hst, hsc, paper_8bit(hsc_hr, self.hsc_hr_8bit), seg


AUGMENTATIONS = {
    "none": SR_HST_HSC_Dataset,
    "shift": ShiftDataset,
    "shift_d4": ShiftD4Dataset,
    "rot90": Rot90Dataset,
}


def _with_8bit(base, mode):
    if mode == "none":
        return base
    name = f"{base.__name__}_8bit_{mode}"
    cls = type(name, (Paper8BitMixin, base), {"hsc_hr_8bit": mode, "__module__": __name__})
    globals()[name] = cls  # a module attribute, so spawned DataLoader workers can unpickle it
    return cls


DATASET_CLASSES = {
    (name, mode): _with_8bit(base, mode)
    for name, base in AUGMENTATIONS.items()
    for mode in HSC_HR_8BIT
}


def hsc_hr_8bit_mode(config):
    """[DATASET] hsc_hr_8bit (default: none)."""
    mode = config.get("DATASET", "hsc_hr_8bit", fallback="none")
    if mode not in HSC_HR_8BIT:
        raise ValueError(f"unknown [DATASET] hsc_hr_8bit {mode!r}; choose from {list(HSC_HR_8BIT)}")
    return mode


def dataset_classes(config):
    """(augment name, training class, validation class) from [DATA_AUG] augment and [DATASET].

    Validation is never augmented but gets the same hsc_hr_8bit as training.
    """
    name = config.get("DATA_AUG", "augment", fallback="none")
    if name not in AUGMENTATIONS:
        raise ValueError(f"unknown [DATA_AUG] augment {name!r}; choose from {list(AUGMENTATIONS)}")
    mode = hsc_hr_8bit_mode(config)
    return name, DATASET_CLASSES[name, mode], DATASET_CLASSES["none", mode]


def train_dataset_class(config):
    """(augment name, training class); see dataset_classes."""
    name, train, _ = dataset_classes(config)
    return name, train
