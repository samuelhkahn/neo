import os
import numpy as np
from astropy.io import fits
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs

def main():

    print("Opening data")
    hst = fits.open("../data/hlsp_candels_hst_acs_cos-tot_f814w_v1.0_drz.fits")[0]
    hsc = fits.open("../data/cutout-HSC-I-9813-pdr2_dud-210317-161628.fits")[1]

    print("HST -> HSC")
    hst_on_hsc_array, hst_on_hsc_footprint = reproject_interp(hst, hsc.header)
    valid_footprint = np.logical_and(hst_on_hsc_footprint, hst_on_hsc_array!=0)
    fits.PrimaryHDU(
        data=valid_footprint.astype(np.uint8),
        header=hsc.header
    ).writeto("../data/hst_to_hsc_footprint.fits", overwrite=True)


if __name__=="__main__":
    main()
