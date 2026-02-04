"""
For image preprocessing kldeconv algorithm.
Rescale image to make the image has a custom defined average intensity.
"""

import numpy as np
import skimage.io as io
import os, tqdm, cupy
from utils.data import read_txt, win2linux
from scipy.ndimage import gaussian_filter
import torch
from cupyx.scipy.ndimage import median_filter

# ------------------------------------------------------------------------------
# filtering
# gaussian_filtering = True
gaussian_filtering = False
# median_filtering = True
median_filtering = False

# average pooling
average_pooling, scale_factor = False, 1
# average_pooling, scale_factor = True, 2
# average_pooling, scale_factor = True, 3

# padding
padding, padding_size = False, 1024

# background subtraction
# bkg_sub, bkg_value = True, 1100
bkg_sub, bkg_value = True, None
# bkg_sub, bkg_value = True, 100.0
# bkg_sub, bkg_value = False, None

# outlier clipping
# outlier_clip, clip_value = True, None
# outlier_clip, clip_value = True, 10000.0
outlier_clip, clip_value = False, None

# ------------------------------------------------------------------------------
# average intensity of the image
ave_intensity = 100.0

# ------------------------------------------------------------------------------
path_root = "E:\qiqilu\datasets_2"
# path_dataset = "BioSR\\transformed\CCPs\\noise_1_sf_1\\raw"
# path_dataset = "BioSR\\transformed\CCPs\\noise_2_sf_1\\raw"
# path_dataset = "BioSR\\transformed\CCPs\\noise_3_sf_1\\raw"
# path_dataset = "BioSR\\transformed\CCPs\\noise_4_sf_1\\raw"
# path_dataset = "BioSR\\transformed\CCPs\\noise_5_sf_1\\raw"
# path_dataset = "BioSR\\transformed\CCPs\\noise_6_sf_1\\raw"
# path_dataset = "BioSR\\transformed\CCPs\\noise_7_sf_1\\raw"
# path_dataset = "BioSR\\transformed\CCPs\\noise_8_sf_1\\raw"
# path_dataset = "BioSR\\transformed\CCPs\\noise_9_sf_1\\raw"
# path_raw_txt = "BioSR\\transformed\CCPs\\all.txt"

# path_dataset = "BioSR\\transformed\ER\\noise_1_sf_1\\raw"
# path_dataset = "BioSR\\transformed\ER\\noise_2_sf_1\\raw"
# path_dataset = "BioSR\\transformed\ER\\noise_3_sf_1\\raw"
# path_dataset = "BioSR\\transformed\ER\\noise_4_sf_1\\raw"
# path_dataset = "BioSR\\transformed\ER\\noise_5_sf_1\\raw"
# path_dataset = "BioSR\\transformed\ER\\noise_6_sf_1\\raw"
# path_raw_txt = "BioSR\\transformed\ER\\all.txt"

# path_dataset = "BioSR\\transformed\F-actin\\noise_1_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_2_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_3_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_4_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_5_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_6_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_7_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_8_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_9_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_10_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_11_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin\\noise_12_sf_1\\raw"
# path_raw_txt = "BioSR\\transformed\F-actin\\all.txt"

path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_1_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_2_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_3_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_4_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_5_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_6_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_7_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_8_sf_1\\raw"
# path_dataset = "BioSR\\transformed\F-actin_Nonlinear\\noise_9_sf_1\\raw"
path_raw_txt = "BioSR\\transformed\F-actin_Nonlinear\\all.txt"

# path_dataset = "BioSR\\transformed\Microtubules2\\noise_1_sf_1\\raw"
# path_dataset = "BioSR\\transformed\Microtubules2\\noise_2_sf_1\\raw"
# path_dataset = "BioSR\\transformed\Microtubules2\\noise_3_sf_1\\raw"
# path_dataset = "BioSR\\transformed\Microtubules2\\noise_4_sf_1\\raw"
# path_dataset = "BioSR\\transformed\Microtubules2\\noise_5_sf_1\\raw"
# path_dataset = "BioSR\\transformed\Microtubules2\\noise_6_sf_1\\raw"
# path_dataset = "BioSR\\transformed\Microtubules2\\noise_7_sf_1\\raw"
# path_dataset = "BioSR\\transformed\Microtubules2\\noise_8_sf_1\\raw"
# path_dataset = "BioSR\\transformed\Microtubules2\\noise_9_sf_1\\raw"
# path_raw_txt = "BioSR\\transformed\Microtubules2\\all.txt"

# ------------------------------------------------------------------------------
# path_dataset = "BioTISR\\transformed\F-actin\SIM_2d"
# path_dataset = "BioTISR\\transformed\F-actin\WF_noise_level_1_2d"
# path_dataset = "BioTISR\\transformed\F-actin\WF_noise_level_2_2d"
# path_dataset = "BioTISR\\transformed\F-actin\WF_noise_level_3_2d"
# path_raw_txt = "BioTISR\\transformed\F-actin\\all.txt"

# path_dataset = "BioTISR\\transformed\F-actin_nonlinear\SIM_2d"
# path_dataset = "BioTISR\\transformed\F-actin_nonlinear\WF_noise_level_1_2d"
# path_dataset = "BioTISR\\transformed\F-actin_nonlinear\WF_noise_level_2_2d"
# path_dataset = "BioTISR\\transformed\F-actin_nonlinear\WF_noise_level_3_2d"
# path_raw_txt = "BioTISR\\transformed\F-actin_nonlinear\\all.txt"

# path_dataset = "BioTISR\\transformed\Microtubules\SIM_2d"
# path_dataset = "BioTISR\\transformed\Microtubules\WF_noise_level_1_2d"
# path_dataset = "BioTISR\\transformed\Microtubules\WF_noise_level_2_2d"
# path_dataset = "BioTISR\\transformed\Microtubules\WF_noise_level_3_2d"
# path_raw_txt = "BioTISR\\transformed\Microtubules\\all.txt"

# path_dataset = "BioTISR\\transformed\Mitochondria\SIM_2d"
# path_dataset = "BioTISR\\transformed\Mitochondria\WF_noise_level_1_2d"
# path_dataset = "BioTISR\\transformed\Mitochondria\WF_noise_level_2_2d"
# path_dataset = "BioTISR\\transformed\Mitochondria\WF_noise_level_3_2d"
# path_raw_txt = "BioTISR\\transformed\Mitochondria\\all.txt"

# path_dataset = "BioTISR\\transformed\Lysosomes\SIM_2d"
# path_dataset = "BioTISR\\transformed\Lysosomes\WF_noise_level_1_2d"
# path_dataset = "BioTISR\\transformed\Lysosomes\WF_noise_level_2_2d"
# path_dataset = "BioTISR\\transformed\Lysosomes\WF_noise_level_3_2d"
# path_raw_txt = "BioTISR\\transformed\Lysosomes\\all.txt"

# path_dataset = "DeepBacs\Saureus\WF"
# path_dataset = "DeepBacs\Saureus\SIM"
# path_raw_txt = "DeepBacs\Saureus\\all.txt"

# path_dataset = "DeepBacs\Ecoli\WF"
# path_dataset = "DeepBacs\Ecoli\SIM"
# path_raw_txt = "DeepBacs\Ecoli\\all.txt"

# path_dataset = "W2S\\transformed\channle_2\SIM"
# path_dataset = "W2S\\transformed\channle_2\SIM_ave"
# path_dataset = "W2S\\transformed\channle_0\WF_ave_10"
# path_dataset = "W2S\\transformed\channle_0\WF_ave_50"
# path_dataset = "W2S\\transformed\channle_0\WF_ave_100"
# path_dataset = "W2S\\transformed\channle_0\WF_ave_200"
# path_dataset = "W2S\\transformed\channle_0\WF_ave_300"
# path_dataset = "W2S\\transformed\channle_2\WF_ave_400"
# path_raw_txt = "W2S\\transformed\\all.txt"

# path_dataset = "RCAN3D\Confocal_2_STED\SirDNA\gt"
# path_dataset = "RCAN3D\Confocal_2_STED\SirDNA\\raw"
# path_raw_txt = "RCAN3D\Confocal_2_STED\SirDNA\\all.txt"

# path_dataset = "RCAN3D\Confocal_2_STED\Microtubule2\gt"
# path_dataset = "RCAN3D\Confocal_2_STED\Microtubule2\\raw"
# path_raw_txt = "RCAN3D\Confocal_2_STED\Microtubule2\\all.txt"

# path_dataset = "RCAN3D\Confocal_2_STED\Nuclear_Pore_complex2\gt"
# path_dataset = "RCAN3D\Confocal_2_STED\Nuclear_Pore_complex2\\raw"
# path_raw_txt = "RCAN3D\Confocal_2_STED\Nuclear_Pore_complex2\\all.txt"

# path_dataset = "BioTISR\\transformed\Mitochondria-3D\SIM_remove_last_t0"
# path_dataset = "BioTISR\\transformed\Mitochondria-3D\WF_noise_level_1_remove_last_t0"
# path_dataset = "BioTISR\\transformed\Mitochondria-3D\WF_noise_level_2_remove_last_t0"
# path_raw_txt = "BioTISR\\transformed\Mitochondria-3D\\all.txt"

# path_dataset = "BioTISR\\transformed\F-actin-3D\SIM_remove_last_t0"
# path_dataset = "BioTISR\\transformed\F-actin-3D\WF_noise_level_1_remove_last_t0"
# path_dataset = "BioTISR\\transformed\F-actin-3D\WF_noise_level_2_remove_last_t0"
# path_raw_txt = "BioTISR\\transformed\F-actin-3D\\all.txt"

# path_dataset = "BioTISR\\transformed\Microtubules-3D\SIM_remove_last_t0"
# path_dataset = "BioTISR\\transformed\Microtubules-3D\WF_noise_level_1_remove_last_t0"
# path_dataset = "BioTISR\\transformed\Microtubules-3D\WF_noise_level_2_remove_last_t0"
# path_raw_txt = "BioTISR\\transformed\Microtubules-3D\\all.txt"


# ------------------------------------------------------------------------------
path_root = win2linux(path_root)
path_dataset = win2linux(path_dataset)
path_raw_txt = win2linux(path_raw_txt)
path_dataset = os.path.join(path_root, path_dataset)
path_raw_txt = os.path.join(path_root, path_raw_txt)
filenames_raw = read_txt(path_raw_txt)

# ------------------------------------------------------------------------------
path_save_to_raw = path_dataset + f"_rescale_{ave_intensity}"
if gaussian_filtering:
    path_save_to_raw += "_gaufilter"
if median_filtering:
    path_save_to_raw += "_medfilter"
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
def preprocess(path_img):
    data_raw = io.imread(path_img).astype(np.float32)
    data_raw = np.squeeze(data_raw)
    ndim = data_raw.ndim
    assert (
        ndim == 2 or ndim == 3
    ), f"[ERROR] the dimension of the image is {ndim}, should be 2 or 3."

    # padding ------------------------------------------------------------------
    # if the image is smaller than the padding size in the last two dimensions,
    # then pad the image.
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
            bv = np.percentile(data_raw, 2)
        else:
            bv = bkg_value
        print(f"[INFO] background value: {bv}")
        data_raw = data_raw - bv
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

    # average pooling ----------------------------------------------------------
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
    if gaussian_filtering:
        data_raw = gaussian_filter(data_raw, sigma=1.0, axes=(-2, -1))
    if median_filtering:
        data_raw = cupy.asarray(data_raw)
        data_raw = median_filter(data_raw, size=3)
        data_raw = cupy.asnumpy(data_raw)

    # normalization ------------------------------------------------------------
    intensity_sum = ave_intensity * np.prod(data_raw.shape)
    data_raw = data_raw / data_raw.sum() * intensity_sum

    return data_raw


# ------------------------------------------------------------------------------
# process all image
# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=len(filenames_raw), desc="[INFO] PREPROCESSING", ncols=80)
for i in range(len(filenames_raw)):
    data_raw = preprocess(os.path.join(path_dataset, filenames_raw[i]))
    io.imsave(
        os.path.join(path_save_to_raw, filenames_raw[i]),
        arr=data_raw,
        check_contrast=False,
    )
    pbar.update(1)
pbar.close()
