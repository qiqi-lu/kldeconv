"""
Normalize 3D real data to make the raw and gt image to have a save sum of intensity.
"""

import numpy as np
import skimage.io as io
import os, tqdm
from utils.data import read_txt, win2linux
from scipy.ndimage import gaussian_filter
import torch

# ------------------------------------------------------------------------------
# filtering_raw = True
filtering_raw = False
average_pooling = True
path_dataset = "I:\Datasets\RCAN3D\Confocal_2_STED\SirDNA\\raw_live\cell2"
path_raw_txt = "I:\Datasets\RCAN3D\Confocal_2_STED\SirDNA\\raw_live\\test_cell2.txt"
path_dataset = win2linux(path_dataset)
path_raw_txt = win2linux(path_raw_txt)
filenames_raw = read_txt(path_raw_txt)

# ------------------------------------------------------------------------------
path_save_to_raw = path_dataset + "_rescale"
if filtering_raw:
    path_save_to_raw += "_filter"

os.makedirs(path_save_to_raw, exist_ok=True)

print(f"[INFo] load data from : {path_dataset}")
print(f"[INFO] save data to   : {path_save_to_raw}")


# ------------------------------------------------------------------------------
# preprocess
# ------------------------------------------------------------------------------
def preprocess(path, name_raw):
    data_raw = io.imread(os.path.join(path, name_raw)).astype(np.float32)

    # positive constriant (2, new version) -------------------------------------
    data_raw = np.clip(data_raw, 0.0, None)
    data_raw = data_raw - data_raw.min()

    # average polling ----------------------------------------------------------
    # only the last two dimensions are pooled
    if average_pooling:
        data_raw = torch.tensor(data_raw)
        data_raw = torch.nn.functional.avg_pool2d(
            data_raw, kernel_size=2, stride=2, padding=0
        )
        data_raw = data_raw.numpy()

    # gaussian filtering the raw image -----------------------------------------
    if filtering_raw:
        data_raw = gaussian_filter(data_raw, sigma=1.0)

    # normalization ------------------------------------------------------------
    ave_intensity = 100.0
    intensity_sum = ave_intensity * np.prod(data_raw.shape)
    data_raw = data_raw / data_raw.sum() * intensity_sum

    return data_raw


# ------------------------------------------------------------------------------
# process all image
# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=len(filenames_raw), desc="process data", ncols=100)
for i in range(len(filenames_raw)):
    data_raw = preprocess(path_dataset, filenames_raw[i])
    io.imsave(
        os.path.join(path_save_to_raw, filenames_raw[i]),
        arr=data_raw,
        check_contrast=False,
    )
    pbar.update(1)
pbar.close()
