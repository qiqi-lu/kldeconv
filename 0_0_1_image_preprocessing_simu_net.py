"""
Preprocessing of simulated images for deep learning network training, which need
normalization and clipping.
"""

import numpy as np
import skimage.io as io
import os, tqdm
from utils.data import win2linux, read_txt, NormalizePercentile

# ------------------------------------------------------------------------------
path_root = "E:\qiqilu\datasets_2\RLN\\unzip\kldeconv\SimuMix3D_128"

folder_names = (
    "raw_psf_31_gauss_0.5_poiss_1_sf_1_ratio_0.1",
    "raw_psf_31_gauss_0.5_poiss_1_sf_1_ratio_0.3",
    "raw_psf_31_gauss_0.5_poiss_1_sf_1_ratio_1",
    "raw_psf_31_gauss_0_poiss_0_sf_1_ratio_1",
    "gt",
)

# ------------------------------------------------------------------------------
path_root = win2linux(path_root)
path_txt = os.path.join(path_root, "all.txt")
filenames = read_txt(path_txt)

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=3)
# ------------------------------------------------------------------------------
for folder_name in folder_names:
    path = os.path.join(path_root, folder_name)
    path_save_to = os.path.join(path_root, f"{path}_norm")
    os.makedirs(path_save_to, exist_ok=True)

    pbar = tqdm.tqdm(total=len(filenames), desc=f"{folder_name}", ncols=80)
    for filename in filenames:
        img = io.imread(os.path.join(path, filename)).astype(np.float32)
        img = np.clip(img, a_min=0.0, a_max=None)
        img = normalizer(img)
        io.imsave(
            os.path.join(path_save_to, filename),
            img.astype(np.float32),
            check_contrast=False,
        )
        pbar.update(1)
    pbar.close()
