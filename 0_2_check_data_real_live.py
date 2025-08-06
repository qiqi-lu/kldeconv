import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import methods.deconvolution as dcv
import utils.evaluation as eva
from utils.data import win2linux, read_txt

path_root = win2linux("I:\Datasets")
# ------------------------------------------------------------------------------
dataset_name = "ZeroShotDeconvNet"
path_dataset = os.path.join(
    path_root, dataset_name, "3D time-lapsing data_LLSM_Mitosis_H2B", "642"
)
id_img = 0

# ------------------------------------------------------------------------------
# load example image
fig_path = os.path.join("outputs", "figures", dataset_name.lower())
os.makedirs(fig_path, exist_ok=True)
path_txt = os.path.join(path_dataset, "all.txt")
filenames = read_txt(path_txt)

path_raw = os.path.join(path_dataset, "raw", filenames[id_img])
img_raw = io.imread(path_raw).astype(np.float32)

Nz, Ny, Nx = img_raw.shape
rescale = 100.0 * 6 * 1021 * 1024
img_raw = img_raw / img_raw.sum() * rescale

# ------------------------------------------------------------------------------
# show image
vmax_gt = img_raw.max() * 0.6
nr, nc = 2, 2
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, dpi=300, figsize=(3 * nc, 3 * nr), constrained_layout=True
)
# [ax.set_axis_off() for ax in axes[0:2,0:2].ravel()]
dict_img = dict(cmap="gray", vmin=0, vmax=vmax_gt)
z_idx = Nz // 2
y_idx = 200
x_range = slice(0, 500)

axes[0, 0].imshow(img_raw[z_idx], **dict_img)
axes[0, 1].plot(img_raw[z_idx, y_idx, x_range], "green")
axes[1, 0].imshow(img_raw[z_idx + 1], **dict_img)
axes[1, 1].plot(img_raw[z_idx + 1, y_idx, x_range], "green")

axes[0, 0].set_title(f"RAW (sum={img_raw.max():.2f}) (slice={z_idx})")

path_save_to = os.path.join(fig_path, "examples")
os.makedirs(path_save_to, exist_ok=True)
plt.savefig(os.path.join(path_save_to, f"{filenames[id_img].split('.')[0]}.png"))
