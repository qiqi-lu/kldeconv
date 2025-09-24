"""
Normalize 3D real data to make the raw and gt image to have a save sum of intensity.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os, tqdm
from utils.data import read_txt, win2linux

# ------------------------------------------------------------------------------
# dataset_name = "Microtubule2"
# dataset_name = 'Nuclear_Pore_complex2'
dataset_name = "SirDNA"

# ------------------------------------------------------------------------------
path_dataset = os.path.join("I:\Datasets\RCAN3D\Confocal_2_STED", dataset_name)
path_dataset = win2linux(path_dataset)
path_fig = path_dataset
path_gt_txt = os.path.join(path_dataset, "gt.txt")
path_raw_txt = os.path.join(path_dataset, "raw.txt")

# ------------------------------------------------------------------------------
filenames_gt = read_txt(path_gt_txt)
filenames_raw = read_txt(path_raw_txt)

# patch_enable = True
patch_enable = False

if patch_enable:
    # step, patch_size, N_step = 125, 128, 8
    step, patch_size, N_step = 500, 512, 2
    path_save_to_gt = os.path.join(path_dataset, "gt_512x512")
    path_save_to_raw = os.path.join(path_dataset, "raw_512x512")
else:
    path_save_to_gt = os.path.join(path_dataset, "gt_1024x1024")
    path_save_to_raw = os.path.join(path_dataset, "raw_1024x1024")

for path in [path_save_to_raw, path_save_to_gt]:
    os.makedirs(path, exist_ok=True)

print(f"[INFo] load data from : {path_dataset}")
print(f"[INFO] number of data : {len(filenames_gt)}")
print(f"[INFO] save data to : {path_save_to_gt}")
print(f"[INFO] save data to : {path_save_to_raw}")


# ------------------------------------------------------------------------------
# preprocess
# ------------------------------------------------------------------------------
def preprocess(path, name_gt, name_raw):
    data_gt = io.imread(os.path.join(path, "gt", name_gt)).astype(np.float32)
    data_raw = io.imread(os.path.join(path, "raw", name_raw)).astype(np.float32)

    # pad the image into a shape of (1024, 1024) -------------------------------
    n_pad = 1024
    dict_pad = dict(
        pad_width=(
            (0, 0),
            (0, n_pad - data_gt.shape[1]),
            (0, n_pad - data_gt.shape[2]),
        ),
        mode="edge",
    )
    data_gt = np.pad(data_gt, **dict_pad)
    data_raw = np.pad(data_raw, **dict_pad)

    # positive constriant (2, new version) -------------------------------------
    data_gt = np.clip(data_gt, 0.0, None)
    data_raw = np.clip(data_raw, 0.0, None)

    # normalization ------------------------------------------------------------
    ave_intensity = 100.0
    intensity_sum = ave_intensity * np.prod(data_raw.shape)
    data_gt = data_gt / data_gt.sum() * intensity_sum
    data_raw = data_raw / data_raw.sum() * intensity_sum

    return data_gt, data_raw


# ------------------------------------------------------------------------------
# show example
# ------------------------------------------------------------------------------
id_data_show = 1
data_gt, data_raw = preprocess(
    path_dataset, filenames_gt[id_data_show], filenames_raw[id_data_show]
)

Nz, Ny, Nx = data_gt.shape
# ------------------------------------------------------------------------------
nr, nc = 2, 3
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, dpi=300, figsize=(3 * nc, 3 * nr), constrained_layout=True
)
[ax.set_axis_off() for ax in axes[0:2, 0:2].ravel()]

dict_img = {"cmap": "gray", "vmax": data_gt.max() * 0.6, "vmin": 0}

axes[0, 0].set_title("GT (max={:.2f})".format(data_gt.max()))
axes[0, 1].set_title("RAW (max={:.2f})".format(data_raw.max()))

axes[0, 0].imshow(data_gt[Nz // 2], **dict_img)
axes[0, 1].imshow(data_raw[Nz // 2], **dict_img)
axes[0, 2].plot(data_gt[Nz // 2, 100, 50:500], "red")
axes[0, 2].plot(data_raw[Nz // 2, 100, 50:500], "green")

axes[1, 0].imshow(data_gt[Nz // 2 + 1], **dict_img)
axes[1, 1].imshow(data_raw[Nz // 2 + 1], **dict_img)
axes[1, 2].plot(data_gt[Nz // 2 + 1, 100, 50:500], "red")
axes[1, 2].plot(data_raw[Nz // 2 + 1, 100, 50:500], "green")
plt.savefig(os.path.join(path_fig, "data_check.png"))

# ------------------------------------------------------------------------------
# process all image
# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=len(filenames_gt), desc="process data", ncols=100)
if patch_enable == False:
    for i in range(len(filenames_gt)):
        data_gt, data_raw = preprocess(
            path_dataset,
            filenames_gt[i],
            filenames_raw[i],
        )
        io.imsave(
            os.path.join(path_save_to_gt, filenames_gt[i]),
            arr=data_gt,
            check_contrast=False,
        )
        io.imsave(
            os.path.join(path_save_to_raw, filenames_gt[i]),
            arr=data_raw,
            check_contrast=False,
        )
        pbar.update(1)

if patch_enable == True:
    for i in range(len(filenames_gt)):
        data_gt, data_raw = preprocess(
            path_dataset,
            filenames_gt[i],
            filenames_raw[i],
        )
        for m in range(N_step):
            for n in range(N_step):
                patch_gt = data_gt[
                    :,
                    (0 + step * m) : (patch_size + step * m),
                    (0 + step * n) : (patch_size + step * n),
                ]
                patch_input = data_raw[
                    :,
                    (0 + step * m) : (patch_size + step * m),
                    (0 + step * n) : (patch_size + step * n),
                ]
                io.imsave(
                    os.path.join(
                        path_save_to_gt, f"{filenames_gt[i].split('.')[0]}_{m}_{n}.tif"
                    ),
                    arr=patch_gt,
                    check_contrast=False,
                )
                io.imsave(
                    os.path.join(
                        path_save_to_raw, f"{filenames_gt[i].split('.')[0]}_{m}_{n}.tif"
                    ),
                    arr=patch_input,
                    check_contrast=False,
                )
        pbar.update(1)
pbar.close()
