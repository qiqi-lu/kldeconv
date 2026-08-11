"""
Test the learned forward kernel.
"""

import os
import numpy as np
import skimage.io as io
from utils.data import win2linux
import matplotlib.pyplot as plt
from methods.deconvolution import convolution

path_root_results = "outputs\predictions\\biotisr-3d-mito-2\kernelnet\\biotisr-3d-mito-2\\fp_n1_r1_bp_n1_r1"
path_figures = "outputs/figures/test"

path_figures = win2linux(path_figures)
path_root_results = win2linux(path_root_results)
path_img = os.path.join(path_root_results, "train_iter_5", "Cell_001_0")

path_kernel_fp = os.path.join(path_root_results, "kernel_iter_5", "kernel_fp.tif")

path_x0 = os.path.join(path_img, "x0.tif")
path_y = os.path.join(path_img, "y.tif")
path_y_fp = os.path.join(path_img, "y_fp.tif")
path_y_pred = os.path.join(path_img, "y_pred_all.tif")


# load kernel and image
kernel_fp = io.imread(path_kernel_fp)
x0 = io.imread(path_x0)
y = io.imread(path_y)
y_fp = io.imread(path_y_fp)
y_pred = io.imread(path_y_pred)[-1]
y_pred_fp = convolution(y_pred, kernel_fp, domain="fft")

print(f"Kernel : {kernel_fp.shape}")
print(f"Image (x0) : {x0.shape}")
print(f"Image (y)  : {y.shape}")
print(f"Image (y-fp)  : {y_fp.shape}")
print(f"Image (y-pred)  : {y_pred.shape}")

num_slice = y.shape[0]
num_slice_k = kernel_fp.shape[0]
Ny_k, Nx_k = kernel_fp.shape[1], kernel_fp.shape[2]

# remove background
# print("[INFO] Remove background...")
# bkg = restoration.rolling_ball(x0, radius=100)
# x0_sub_bkg = x0 - bkg
# print("[INFO] Done!")

# ------------------------------------------------------------------------------
nr, nc = 3, 6
dict_fig = dict(dpi=300, constrained_layout=True)
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)


# show images ------------------------------------------------------------------
id_slice = num_slice // 2

dict_img_raw = dict(cmap="hot", vmin=0, vmax=700)
dict_img_y = dict(cmap="hot", vmin=0, vmax=1500)
# dict_diff = dict(cmap="seismic", vmin=-500, vmax=500)
dict_diff = dict(cmap="hot", vmin=0, vmax=700)

axes[0, 0].imshow(x0[id_slice], **dict_img_raw)
axes[0, 0].set_title("x0")
axes[0, 1].imshow(y[id_slice], **dict_img_y)
axes[0, 1].set_title("y")
axes[0, 2].imshow(y_fp[id_slice], **dict_img_raw)
axes[0, 2].set_title("y-fp")
axes[0, 3].imshow(y_pred_fp[id_slice], **dict_img_raw)
axes[0, 3].set_title("y-pred-fp")
axes[0, 4].imshow(y_pred[id_slice], **dict_img_y)
axes[0, 4].set_title("y-pred")
# axes[0, 5].imshow(x0_sub_bkg[id_slice], **dict_img_raw)


diff_yfp_x0 = y_fp[id_slice] - x0[id_slice]
diff_yfp_x0 = np.abs(diff_yfp_x0)
axes[1, 2].imshow(diff_yfp_x0, **dict_diff)
axes[1, 2].text(10, 30, f"MAE: {np.mean(diff_yfp_x0):.2f}", color="white")

diff_ypredfp_x0 = y_pred_fp[id_slice] - x0[id_slice]
diff_ypredfp_x0 = np.abs(diff_ypredfp_x0)
axes[1, 3].imshow(diff_ypredfp_x0, **dict_diff)
axes[1, 3].text(10, 30, f"MAE: {np.mean(diff_ypredfp_x0):.2f}", color="white")

# image ptrofile
pos_y = 400
axes[2, 2].plot(x0[id_slice, pos_y, :], color="black")
# axes[2, 2].plot(y[id_slice, pos_y, :], color="red")
axes[2, 2].plot(y_fp[id_slice, pos_y, :], color="blue", linestyle="--")

axes[2, 3].plot(x0[id_slice, pos_y, :], color="black")
# axes[2, 3].plot(y[id_slice, pos_y, :], color="red")
axes[2, 3].plot(y_pred_fp[id_slice, pos_y, :], color="blue", linestyle="--")

axes[2, 4].plot(y[id_slice, pos_y, :], color="black")
axes[2, 4].plot(y_pred[id_slice, pos_y, :], color="red", linestyle="--")

# show kernel ------------------------------------------------------------------
id_slice_k = num_slice_k // 2
axes[1, 0].imshow(kernel_fp[id_slice_k], cmap="hot")
# profile along the middle
axes[2, 0].plot(kernel_fp[id_slice_k, Ny_k // 2, :], color="black")

axes[1, 1].imshow(kernel_fp[:, Ny_k // 2], cmap="hot")
axes[2, 1].plot(kernel_fp[:, Ny_k // 2, Nx_k // 2], color="black")

plt.savefig(os.path.join(path_figures, "test_forward.png"))
