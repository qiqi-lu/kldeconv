"""
Show example of real dataset.
Plot specific slice of the image.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os, pandas
from utils.data import win2linux, read_txt, NormalizePercentile

# dataset_id = "Nuclear-pore-complex2-1024"
# dataset_id = "Microtubule2-3d-1024"
# dataset_id = "F-actin-nonlinear-1"
# dataset_id = "F-actin-nonlinear-3"
# dataset_id = "F-actin-nonlinear-9"
# dataset_id = "SirDNA-1024"
# dataset_id = "biotisr-3d-factin-1"
# dataset_id = "biotisr-3d-factin-2"
# dataset_id = "biotisr-3d-mt-1"
# dataset_id = "biotisr-3d-mt-2"
# dataset_id = "biotisr-3d-mito-1"
dataset_id = "biotisr-3d-mito-2"

id_image_show = 0

# ------------------------------------------------------------------------------
info_df = pandas.read_excel("datasets_train.xlsx")
info = info_df[info_df["id"] == dataset_id].iloc[0]

path_raw = win2linux(info["path_lr"]) + "_dark"
path_gt = win2linux(info["path_hr"])
path_txt = win2linux(info["path_txt"])
filenames = read_txt(path_txt.replace("train.txt", "all.txt"))
ndim = info["ndim"]

filename = filenames[id_image_show]

path_figures = os.path.join("outputs", "figures", dataset_id, "examples")
os.makedirs(path_figures, exist_ok=True)

img_gt = io.imread(os.path.join(path_gt, filename)).astype(np.float32)
img_raw = io.imread(os.path.join(path_raw, filename)).astype(np.float32)

print("-" * 80)
print(f"[INFO] {dataset_id}: {filename}")
print(f"[INFO] GT: {img_gt.shape}, RAW: {img_raw.shape}")

# ------------------------------------------------------------------------------
if img_gt.ndim == 2:
    img_gt = img_gt[np.newaxis, ...]
if img_raw.ndim == 2:
    img_raw = img_raw[np.newaxis, ...]

Nz, Ny, Nx = img_gt.shape

# normalization v1
# rescale = 100.0 * Nz * Ny * Nx
# img_gt = img_gt / img_gt.mean() * rescale
# img_raw = img_raw / img_raw.mean() * rescale

# normalization v2
# normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=ndim)
# img_raw = normalizer(img_raw)
# img_gt = normalizer(img_gt)

# ------------------------------------------------------------------------------
# show the image and profile
# ------------------------------------------------------------------------------
nr, nc = 2, 3
dict_fig = dict(dpi=300, constrained_layout=True)
dict_img = dict(cmap="hot", vmin=0, vmax=img_gt.max() * 0.5)
x_range = slice(200, 300)

# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
# [ax.set_axis_off() for ax in axes[0:2,0:2].ravel()]

z_idx = Nz // 2  # center slice

# get the position of max value in the image[z_idx]
y_idx, x_idx = np.unravel_index(img_gt[z_idx].argmax(), img_gt[z_idx].shape)

# for 2D image
axes[0, 0].imshow(img_gt[z_idx], **dict_img)
axes[0, 1].imshow(img_raw[z_idx], **dict_img)

axes[0, 2].plot(img_gt[z_idx, y_idx, x_range], "red", label="GT")
axes[0, 2].plot(img_raw[z_idx, y_idx, x_range], "green", label="RAW")
axes[0, 2].legend()

# plot the line in the image
axes[0, 0].plot([x_range.start, x_range.stop], [y_idx, y_idx], "blue", linewidth=1)
axes[0, 1].plot([x_range.start, x_range.stop], [y_idx, y_idx], "blue", linewidth=1)

axes[0, 0].set_title(f"GT (mean={img_gt.mean():.2f}) (slice={z_idx})")
axes[0, 1].set_title(f"RAW (mean={img_raw.mean():.2f})")

if ndim == 3:
    axes[1, 0].imshow(img_gt[z_idx + 1], **dict_img)
    axes[1, 1].imshow(img_raw[z_idx + 1], **dict_img)

    axes[1, 2].plot(img_gt[z_idx + 1, y_idx, x_range], "red", label="GT")
    axes[1, 2].plot(img_raw[z_idx + 1, y_idx, x_range], "green", label="RAW")
    axes[1, 2].legend()

    axes[1, 0].set_title(f"GT (mean={img_gt.mean():.2f}) (slice={z_idx+1})")
    axes[1, 1].set_title(f"RAW (mean={img_raw.mean():.2f})")

plt.savefig(os.path.join(path_figures, f"{filename.split('.')[0]}.png"))

# ------------------------------------------------------------------------------
# print information of each image in the dataset
# ------------------------------------------------------------------------------
# print("-" * 80)
# for filename in filenames:
#     img_gt = io.imread(os.path.join(path_gt, filename)).astype(np.float32)
#     img_raw = io.imread(os.path.join(path_raw, filename)).astype(np.float32)
#     print(f"[INFO] {filename}: GT: {img_gt.shape}, RAW: {img_raw.shape}")
