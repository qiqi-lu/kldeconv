"""
Rescale image to make the image has a custom defined average intensity.
"""

import numpy as np
import skimage.io as io
import os, tqdm
from utils.data import read_txt, win2linux
from scipy.ndimage import gaussian_filter
import torch

# ------------------------------------------------------------------------------
filtering_raw = False

average_pooling, scale_factor = False, 2
# average_pooling, scale_factor = True, 2

padding, padding_size = False, 1024

bkg_sub, bkg_value = True, None
# bkg_sub, bkg_value = True, 100.0
# bkg_sub, bkg_value = False, None

# outlier_clip, clip_value = True, None
# outlier_clip, clip_value = True, 10000.0
outlier_clip, clip_value = False, None

# path_dataset = "I:\Datasets\BioTISR\\transformed\F-actin\SIM_2d-dn-rb"
# path_dataset = "I:\Datasets\BioTISR\\transformed\F-actin\SIM_2d"
# path_dataset = "I:\Datasets\BioTISR\\transformed\F-actin\WF_noise_level_1_2d"
# path_dataset = "I:\Datasets\BioTISR\\transformed\F-actin\WF_noise_level_2_2d"
path_dataset = "I:\Datasets\BioTISR\\transformed\F-actin\WF_noise_level_3_2d"
path_raw_txt = "I:\Datasets\BioTISR\\transformed\F-actin\\all.txt"

ave_intensity = 100.0
path_dataset = win2linux(path_dataset)
path_raw_txt = win2linux(path_raw_txt)
filenames_raw = read_txt(path_raw_txt)

# ------------------------------------------------------------------------------
path_save_to_raw = path_dataset + f"_rescale_{ave_intensity}"
if filtering_raw:
    path_save_to_raw += "_filter"
if average_pooling:
    path_save_to_raw += f"_avepool_{scale_factor}"
if padding:
    path_save_to_raw += f"_pad_{padding_size}"
if bkg_sub:
    if bkg_value is not None:
        path_save_to_raw += f"_bkgsub_{bkg_value}"
    else:
        path_save_to_raw += "_bkgsub_auto"
if outlier_clip:
    if clip_value is not None:
        path_save_to_raw += f"_maxclip_{clip_value}"
    else:
        path_save_to_raw += "_maxclip_auto"

os.makedirs(path_save_to_raw, exist_ok=True)

print(f"[INFo] load data from : {path_dataset}")
print(f"[INFO] save data to   : {path_save_to_raw}")


# ------------------------------------------------------------------------------
# preprocess
# ------------------------------------------------------------------------------
def preprocess(path, name_raw):
    data_raw = io.imread(os.path.join(path, name_raw)).astype(np.float32)
    data_raw = np.squeeze(data_raw)
    ndim = data_raw.ndim

    # padding ------------------------------------------------------------------
    if padding:
        assert (
            data_raw.shape[-1] <= padding_size and data_raw.shape[-2] <= padding_size
        ), f"[ERROR] the original shape of the image is {data_raw.shape}, which has exceed the padding size {padding_size} in the last two dimensions."

        if ndim == 3:
            # only pad the last tow dimesions
            data_raw = np.pad(
                data_raw,
                pad_width=(
                    (0, 0),
                    (0, padding_size - data_raw.shape[1]),
                    (0, padding_size - data_raw.shape[2]),
                ),
                mode="edge",
            )
        elif ndim == 2:
            # only pad the last tow dimesions
            data_raw = np.pad(
                data_raw,
                pad_width=(
                    (0, padding_size - data_raw.shape[0]),
                    (0, padding_size - data_raw.shape[1]),
                ),
                mode="edge",
            )
        else:
            raise ValueError(f"[ERROR] the dimension of the image is {ndim}.")

    # background subtraction ---------------------------------------------------
    if bkg_sub:
        global bkg_value
        if bkg_value is None:
            bkg_value = np.percentile(data_raw, 2)
        data_raw = data_raw - bkg_value
    else:
        pass

    # positive constriant (2, new version) -------------------------------------
    data_raw = np.clip(data_raw, 0.0, None)

    # outlier clipping ---------------------------------------------------------
    if outlier_clip:
        global clip_value
        if clip_value is None:
            clip_value = np.percentile(data_raw, 99.9)
        data_raw = np.clip(data_raw, 0.0, clip_value)

    # average polling ----------------------------------------------------------
    # only the last two dimensions are pooled
    if average_pooling:
        if ndim == 2:
            data_raw = torch.tensor(data_raw)[None, None]
        elif ndim == 3:
            data_raw = torch.tensor(data_raw)[None]
        else:
            raise ValueError(
                f"[ERROR] the dimension of the image is {ndim}, should be 2 or 3."
            )

        data_raw = torch.nn.functional.avg_pool2d(
            data_raw, kernel_size=scale_factor, stride=scale_factor, padding=0
        )

        if ndim == 2:
            data_raw = data_raw.numpy()[0, 0]
        if ndim == 3:
            data_raw = data_raw.numpy()[0]

    # gaussian filtering the raw image -----------------------------------------
    # the number of photons in the image may be too small,
    # the zero photon pixel need to be filled with a estimated value.
    if filtering_raw:
        data_raw = gaussian_filter(data_raw, sigma=1.0)

    # normalization ------------------------------------------------------------
    intensity_sum = ave_intensity * np.prod(data_raw.shape)
    data_raw = data_raw / data_raw.sum() * intensity_sum

    return data_raw


# ------------------------------------------------------------------------------
# process all image
# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=len(filenames_raw), desc="[INFO] PREPROCESSING", ncols=100)
for i in range(len(filenames_raw)):
    data_raw = preprocess(path_dataset, filenames_raw[i])
    io.imsave(
        os.path.join(path_save_to_raw, filenames_raw[i]),
        arr=data_raw,
        check_contrast=False,
    )
    pbar.update(1)
pbar.close()
