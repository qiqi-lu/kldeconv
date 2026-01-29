"""
Normalize 3D real data to make the raw and gt image to have a save sum of intensity.
Used for training and evaluation the KLdeconv algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os, tqdm
from utils.data import read_txt, win2linux
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------------------
filtering = False
ave_intensity = 100

# ------------------------------------------------------------------------------
# path_dataset= "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\Microtubule2\gt"
# path_dataset= "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\Microtubule2\\raw"
# path_txt = "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\Microtubule2\\all.txt"
# path_dataset = "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\Nuclear_Pore_complex2\gt"
# path_dataset = "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\Nuclear_Pore_complex2\\raw"
# path_txt = "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\Nuclear_Pore_complex2\\all.txt"
path_dataset = "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\SirDNA\gt"
# path_dataset = "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\SirDNA\\raw"
path_txt = "E:\qiqilu\datasets_2\RCAN3D\Confocal_2_STED\SirDNA\\all.txt"


# ------------------------------------------------------------------------------
path_dataset = win2linux(path_dataset)
path_fig = path_dataset
path_txt = win2linux(path_txt)

path_save_to = path_dataset + f"_rescale_{ave_intensity}"

if filtering:
    path_save_to += "_filter"

for path in path_save_to:
    os.makedirs(path, exist_ok=True)

filenames = read_txt(path_txt)
num_samples = len(filenames)

print(f"[INFo] load data from : {path_dataset}")
print(f"[INFO] number of data : {num_samples}")
print(f"[INFO] save data to : {path_save_to}")


# ------------------------------------------------------------------------------
# preprocess
# ------------------------------------------------------------------------------
def preprocess(path_data, filtering=False, sigma=1.0, pad_size=None):
    data = io.imread(path_data).astype(np.float32)

    # gaussian filtering -------------------------------------------------------
    if filtering:
        data = gaussian_filter(data, sigma=sigma)

    # pad the image into a shape of (1024, 1024) -------------------------------
    if pad_size is not None:
        n_pad = pad_size
        dict_pad = dict(
            pad_width=(
                (0, 0),
                (0, n_pad - data.shape[1]),
                (0, n_pad - data.shape[2]),
            ),
            mode="edge",
        )
        data = np.pad(data, **dict_pad)

    # positive constriant (2, new version) -------------------------------------
    data = np.clip(data, 0.0, None)

    # normalization ------------------------------------------------------------
    intensity_sum = ave_intensity * np.prod(data.shape)
    data = data / data.sum() * intensity_sum
    return data


# ------------------------------------------------------------------------------
# process all image
# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=num_samples, desc="Data Preprocess", ncols=80)
for i in range(num_samples):
    filename = filenames[i]
    path_file = os.path.join(path_dataset, filename)
    data = preprocess(path_file, filtering=filtering, pad_size=None)

    io.imsave(
        os.path.join(path_save_to, filename),
        arr=data,
        check_contrast=False,
    )
    pbar.update(1)
pbar.close()
