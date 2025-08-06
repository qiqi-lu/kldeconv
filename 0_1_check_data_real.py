"""
Show example of real dataset.
Plot specific slice of the image.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva
from utils.data import win2linux, read_txt

path_root = win2linux("I:\Datasets\RCAN3D\Confocal_2_STED")
# ------------------------------------------------------------------------------
dataset_name = "Microtubule"
# dataset_name = "Nuclear_Pore_complex"
id_img = 1

fig_path = os.path.join("outputs", "figures", dataset_name.lower())
os.makedirs(fig_path, exist_ok=True)

# ------------------------------------------------------------------------------
# load data
dataset_path = os.path.join(path_root, dataset_name)

# get all the filenames from 'all.txt' file
filenames = read_txt(os.path.join(dataset_path, "all.txt"))

img_gt = io.imread(os.path.join(dataset_path, "gt", filenames[id_img]))
img_raw = io.imread(os.path.join(dataset_path, "raw", filenames[id_img]))

img_gt = img_gt.astype(np.float32)
img_raw = img_raw.astype(np.float32)
print("[INFO] GT: {}, RAW: {}".format(img_gt.shape, img_raw.shape))

Nz, Ny, Nx = img_gt.shape
nr, nc = 2, 3
rescale = 100.0 * 6 * 1021 * 1024
img_gt = img_gt / img_gt.sum() * rescale
img_raw = img_raw / img_raw.sum() * rescale

# ------------------------------------------------------------------------------
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, dpi=300, figsize=(3 * nc, 3 * nr), constrained_layout=True
)
# [ax.set_axis_off() for ax in axes[0:2,0:2].ravel()]
dict_img = dict(cmap="gray", vmin=0, vmax=img_gt.max() * 0.6)
z_idx = Nz // 2
y_idx = 100
x_range = slice(50, 150)

axes[0, 0].imshow(img_gt[z_idx], **dict_img)
axes[0, 1].imshow(img_raw[z_idx], **dict_img)

axes[0, 2].plot(img_gt[z_idx, y_idx, x_range], "red")
axes[0, 2].plot(img_raw[z_idx, y_idx, x_range], "green")

axes[0, 0].set_title(f"GT (sum={img_gt.max():.2f}) (slice={z_idx})")
axes[0, 1].set_title(f"RAW (sum={img_raw.max():.2f})")

axes[1, 0].imshow(img_gt[z_idx + 1], **dict_img)
axes[1, 1].imshow(img_raw[z_idx + 1], **dict_img)
axes[1, 2].plot(img_gt[z_idx + 1, y_idx, x_range], "red")
axes[1, 2].plot(img_raw[z_idx + 1, y_idx, x_range], "green")

axes[1, 0].set_title(f"GT (sum={img_gt.max():.2f}) (slice={z_idx+1})")
axes[1, 1].set_title(f"RAW (sum={img_raw.max():.2f})")

path_save_to = os.path.join(fig_path, "examples")
os.makedirs(path_save_to, exist_ok=True)
plt.savefig(os.path.join(path_save_to, f"{filenames[id_img].split('.')[0]}.png"))
