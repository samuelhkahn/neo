import numpy as np
import pytest
from astropy.wcs.utils import proj_plane_pixel_scales
from conftest import make_tan_wcs

from neo.preprocess.grid import lr_window_mask, upsampled_wcs


@pytest.mark.parametrize("cd", [False, True])
def test_upsampled_wcs_nests_hr_pixels_inside_lr_pixels(cd):
    lr, f = make_tan_wcs(0.2, (20, 30), cd=cd), 6
    hr = upsampled_wcs(lr, f)
    assert hr.pixel_shape == (180, 120)
    assert np.allclose(proj_plane_pixel_scales(hr) * 3600, 0.2 / f)

    x, y = np.array([0, 7, 29]), np.array([0, 3, 19])
    lr_centers = lr.pixel_to_world(x, y)
    hr_centers = hr.pixel_to_world(f * x + (f - 1) / 2, f * y + (f - 1) / 2)
    assert np.all(lr_centers.separation(hr_centers).arcsec < 1e-6)

    lr_corner = lr.pixel_to_world(-0.5, -0.5)
    hr_corner = hr.pixel_to_world(-0.5, -0.5)
    assert lr_corner.separation(hr_corner).arcsec < 1e-6


def test_upsampled_wcs_commutes_with_slicing():
    lr, f = make_tan_wcs(0.2, (20, 30)), 6
    sliced_then_up = upsampled_wcs(lr[5:15, 3:20], f)
    up_then_sliced = upsampled_wcs(lr, f)[30:90, 18:120]
    assert np.allclose(sliced_then_up.wcs.crpix, up_then_sliced.wcs.crpix)


def test_lr_window_mask_requires_every_hr_pixel():
    hr = np.ones((12, 12), bool)
    hr[0, 0] = False
    assert lr_window_mask(hr, 6).tolist() == [[False, True], [True, True]]
