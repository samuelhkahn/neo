"""Train-time augmentation that keeps LR and HR exactly registered.

SR_HST_HSC_Dataset centre-crops every stored pair (142 px LR / 852 px HR) to 100 / 600 px.
ShiftD4Dataset moves that crop: before the dataset's own processing it rolls both cutouts by the
same whole number of LR pixels (6x as many HR pixels, so LR pixels stay nested in HR blocks), up to
the 21 px of slack around the centre crop, and applies the same random 90-degree rotation and
flip (the 8 lossless symmetries of the pixel grid). Each pair thus yields 43 x 43 x 8 distinct
training views, all inside its stored footprint (which is what the train/val leakage check
covers). Everything after loading (clip, log scaling, segmentation map, crop, padding) is the
dataset's unchanged code. Validation and evaluation keep the plain centre crop.

Select it in a training config with [DATA_AUG] augment = shift_d4 (default: none).
"""

import os
import random

import numpy as np
from astropy.io import fits

from neo.data.dataset import SR_HST_HSC_Dataset

LR_CROP = 100
FACTOR = 6


def shift_d4(array, dy, dx, k, flip, factor=1):
    """Roll by (dy, dx) LR pixels (x factor), rotate by k * 90 degrees, then optionally flip."""
    array = np.roll(array, (dy * factor, dx * factor), axis=(0, 1))
    array = np.rot90(array, k)
    if flip:
        array = array[:, ::-1]
    return np.ascontiguousarray(array)


class ShiftD4Dataset(SR_HST_HSC_Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hst_dir = os.path.normpath(self.hst_path)
        self._params = None
        first = os.path.join(self.hsc_path, self.filenames[0]) if self.filenames else None
        size = fits.getheader(first)["NAXIS1"] if first else LR_CROP
        self.max_shift = (size - LR_CROP) // 2

    def draw(self):
        """Random (dy, dx, k, flip); `random` is reseeded per DataLoader worker by PyTorch."""
        m = self.max_shift
        return (
            random.randint(-m, m),
            random.randint(-m, m),
            random.randrange(4),
            random.random() < 0.5,
        )

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


AUGMENTATIONS = {"none": SR_HST_HSC_Dataset, "shift_d4": ShiftD4Dataset}


def train_dataset_class(config):
    """Dataset class for the training split, from [DATA_AUG] augment (default: none)."""
    name = config.get("DATA_AUG", "augment", fallback="none")
    if name not in AUGMENTATIONS:
        raise ValueError(f"unknown [DATA_AUG] augment {name!r}; choose from {list(AUGMENTATIONS)}")
    return name, AUGMENTATIONS[name]
