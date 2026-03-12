import numpy as np
import os
from astropy.io import fits
from shutil import copyfile
import os
from os.path import isfile,join

def load_sample(file_path:str) -> np.ndarray :
    """Loads fits file   

    args:
        file_path (str): file to load

    returns
            numpy ndarray with pixel values in each cell
    """

    cutout = fits.open(file_path)
    pixels = cutout[0].data

    return pixels

def pixel_filter(pixels:np.ndarray, low:float, high:float) -> bool:
    """Checks if sum of array in log space are in interval
    args:
        pixels (np.ndarray): 2d array of pixel
        low (float): low log(sum) threshold for filtering 
        high (float): high log(sum) threshold for filtering 

    returns
            True if log(sum of pixels) is in low/high interval
    """

    pix_sum = np.log(np.sum(pixels)) # pixel sums 

    if pix_sum>low and pix_sum<high: # filter interval 
        return True
    else:
        return False

def copy_to_folders(fname:str, old_paths:list, new_paths:list):
    """Copy files to respective folders
    args:
        f_name (str): filename to laod
        old_paths (list): old paths to copy from
        new_paths (list): new paths to copy to
    """

    copyfile(old_paths[0],new_paths[0])
    copyfile(old_paths[1],new_paths[1])

    # copyfile()

def main():

    data_path = "./data/samples"
    hsc_path = join(data_path,"hsc")
    hst_path = join(data_path,"hst")

    data_dirs = [
        "./data/samples/hsc/filtered",
        "./data/samples/hst/filtered"]

    for dd in data_dirs:
        if not os.path.exists(dd):
            os.makedirs(dd)
    
    files = [f for f in os.listdir(hst_path) if isfile(join(hst_path, f))]

    low = 2
    high = 7

    for f in files:
        hst_file_path = join(hst_path,f)
        pixels = load_sample(hst_file_path)
        pixel_accept = pixel_filter(pixels,low,high)

        if pixel_accept:
            hsc_file_path = join(hsc_path,f)
            old_paths = [hsc_file_path,hst_file_path]
            new_paths = [join(path,f) for path in data_dirs]
            copy_to_folders(f,old_paths,new_paths)







if __name__=="__main__":
    main()
