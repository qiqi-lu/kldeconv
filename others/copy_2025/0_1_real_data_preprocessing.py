"""
Real data preprocessing.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os, tqdm
from utils.data import win2linux, read_txt, pad_img_xy


path_root = win2linux("I:\Datasets\RCAN3D\Confocal_2_STED")
# ------------------------------------------------------------------------------
dataset_name = "Microtubule2"
# dataset_name = 'Nuclear_Pore_complex2'

# patch_enable = True
patch_enable = False

# ------------------------------------------------------------------------------
path_fig = os.path.join("outputs", "figures", dataset_name.lower())
os.makedirs(path_fig, exist_ok=True)

# ------------------------------------------------------------------------------
# load data
path_dataset = os.path.join(path_root, dataset_name)
filenames = read_txt(os.path.join(path_dataset, "gt.txt"))

if patch_enable:
    # step, patch_size, N_step = 125, 128, 8
    step, patch_size, N_step = 500, 512, 2
    save_to_gt = os.path.join(path_dataset, "gt_512x512")
    save_to_raw = os.path.join(path_dataset, "raw_512x512")
else:
    save_to_gt = os.path.join(path_dataset, "gt_1024x1024")
    save_to_raw = os.path.join(path_dataset, "raw_1024x1024")

for path in [save_to_raw, save_to_gt]:
    os.makedirs(path, exist_ok=True)

num_sample = len(filenames)
print(f"[INFO] Load data from : {path_dataset}")
print(f"[INFO] number of data : {num_sample}")
print(f"[INFO] Save data to : {save_to_gt} | {save_to_raw}")

ave_intensity = 100.0

for i in range(num_sample):
    sample_name = filenames[i]
    img_gt = io.imread(os.path.join(path, "gt", sample_name)).astype(np.float32)
    img_raw = io.imread(os.path.join(path, "raw", sample_name)).astype(np.float32)

    print("-" * 50)
    print(f"[INFO] Sample: {sample_name}")
    print(f"[INFO] Image size: (GT) {img_gt.shape}, (Input) {img_raw.shape}")
    print(f"[INFO] Mean (GT) {img_gt.mean()}, (RAW) {img_raw.mean()}")
    print(f"[INFO] Ratio: {img_gt.sum() / img_raw.sum()}")

    # preprocess ---------------------------------------------------------------
    n_pad = 1024
    img_gt = pad_img_xy(img_gt, n_pad)
    img_raw = pad_img_xy(img_raw, n_pad)

    # positive constriant (2, new version)
    img_gt = np.maximum(img_gt, 0.0)
    img_raw = np.maximum(img_raw, 0.0)

    # normalization
    intensity_sum = ave_intensity * np.prod(img_raw.shape)
    img_gt = img_gt / img_gt.sum() * intensity_sum
    img_raw = img_raw / img_raw.sum() * intensity_sum

    # patching -----------------------------------------------------------------
    if patch_enable == False:
        io.imsave(
            os.path.join(save_to_gt, sample_name), arr=img_gt, check_contrast=False
        )
        io.imsave(
            os.path.join(save_to_raw, sample_name), arr=img_raw, check_contrast=False
        )

    if patch_enable == True:
        for m in range(N_step):
            for n in range(N_step):
                patch_gt = img_gt[
                    :,
                    (0 + step * m) : (patch_size + step * m),
                    (0 + step * n) : (patch_size + step * n),
                ]
                patch_input = img_raw[
                    :,
                    (0 + step * m) : (patch_size + step * m),
                    (0 + step * n) : (patch_size + step * n),
                ]
                io.imsave(
                    os.path.join(save_to_gt, f"{sample_name}_{m}_{n}.tif"),
                    arr=patch_gt,
                    check_contrast=False,
                )
                io.imsave(
                    os.path.join(save_to_raw, f"{sample_name}_{m}_{n}.tif"),
                    arr=patch_input,
                    check_contrast=False,
                )
