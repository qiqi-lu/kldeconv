"""
Display the iteration process of a specified sample in simulation datasets.
"""

import matplotlib.pyplot as plt
from utils.data import read_txt, win2linux
import skimage.io as io
import numpy as np
import os, pandas

plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------------------
#                    dataset name | (num_data, id_repeat) | id_sample
# ------------------------------------------------------------------------------
data_info = ("SimuMix3D-128-31-0-0-1", "fp_knonw_bp_n3_r1", 6)

# ------------------------------------------------------------------------------
dataset_name_test, id_experiment, id_sample = data_info
name_net = "kernelnet"
num_iter_train = 2
eps = 0.000001

info_df = pandas.read_excel("datasets_test.xlsx")
info = info_df[info_df["id"] == dataset_name_test].iloc[0]

path_txt = win2linux(info["path_txt"])
path_lr = win2linux(info["path_lr"])
path_hr = win2linux(info["path_hr"])
pixel_size = info["pixel_size"]

filenames = read_txt(path_txt)

path_predictions = os.path.join(
    "outputs",
    "predictions",
    dataset_name_test,
    name_net,
    dataset_name_test,
    id_experiment,
)
path_kernel = os.path.join(path_predictions, "kernel")
path_sample = os.path.join(path_predictions, filenames[id_sample].split(".")[0])
path_fig = os.path.join(
    "outputs", "figures", dataset_name_test, filenames[id_sample].split(".")[0]
)
os.makedirs(path_fig, exist_ok=True)

print("-" * 80)
print("[INFO] load results from :", path_sample)
print("[INFO] load kernels from :", path_kernel)
print("[INFO] Save figures to :", path_fig)

# ------------------------------------------------------------------------------
# load kernels and results of KLD
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] load kernels ...")
ker_init = io.imread(os.path.join(path_kernel, "kernel_init.tif"))
ker_true = io.imread(os.path.join(path_kernel, "kernel_true.tif"))
ker_FP = io.imread(os.path.join(path_kernel, "kernel_fp.tif"))
ker_BP = io.imread(os.path.join(path_kernel, "kernel_bp.tif"))

print("[INFO] load results of KLD ...")
img_y = io.imread(os.path.join(path_sample, "y.tif"))
img_x = io.imread(os.path.join(path_sample, "x.tif"))
img_x0 = io.imread(os.path.join(path_sample, "x0.tif"))
img_y_fp = io.imread(os.path.join(path_sample, "y_fp.tif"))
img_x0_fp = io.imread(os.path.join(path_sample, "x0_fp.tif"))
img_bp = io.imread(os.path.join(path_sample, "bp.tif"))
y_pred_all = io.imread(os.path.join(path_sample, "y_pred_all.tif"))
# ------------------------------------------------------------------------------
y_pred = y_pred_all[num_iter_train]

Nz_kt, Ny_kt, Nx_kt = ker_true.shape
Nz_kf, Ny_kf, Nx_kf = ker_FP.shape
Nz_kb, Ny_kb, Nx_kb = ker_BP.shape
Nz, Ny, Nx = img_y.shape

# ------------------------------------------------------------------------------
dict_fig = {"dpi": 300, "constrained_layout": True}
dict_ker_fp = {"cmap": "hot", "vmin": 0.0, "vmax": ker_true.max()}
dict_ker_bp = {"cmap": "hot", "vmin": 0.0, "vmax": np.max(ker_BP)}
dict_img = {"cmap": "gray", "vmin": 0.0, "vmax": img_y.max() * 0.8}
dict_dv = {"cmap": "seismic", "vmin": 0.5, "vmax": 1.5}
dict_bp = {"cmap": "seismic", "vmin": -0.5, "vmax": 2.5}
dict_text = {"color": "white", "fontsize": 20, "ha": "left", "va": "bottom"}

# ------------------------------------------------------------------------------
# show iteration process
# ------------------------------------------------------------------------------
print("[INFO] plot iteration process...")
nr, nc = 2, 9
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]


def show_xy_zx(ax, img, dict_params):
    Nz, Ny, Nx = img.shape
    ax[0].imshow(img[Nz // 2], **dict_params)
    ax[1].imshow(img[:, Ny // 2, :], **dict_params)


show_xy_zx(axes[:, 0], img_x, dict_img)
show_xy_zx(axes[:, 1], img_x0, dict_img)
show_xy_zx(axes[:, 2], ker_FP, dict_ker_fp)
show_xy_zx(axes[:, 3], img_x0_fp, dict_img)
show_xy_zx(axes[:, 4], img_x / img_x0_fp, dict_dv)
show_xy_zx(axes[:, 5], ker_BP, dict_ker_bp)
show_xy_zx(axes[:, 6], img_bp, dict_bp)
show_xy_zx(axes[:, 7], y_pred, dict_img)
show_xy_zx(axes[:, 8], img_y, dict_img)

pos_x, pos_y = Nx * 0.05, Ny * 0.95
axes[0, 0].text(pos_x, pos_y, "xy", **dict_text)
axes[1, 0].text(pos_x, pos_y, "xz", **dict_text)

# save
plt.savefig(os.path.join(path_fig, "iteration_process.png"))
plt.savefig(os.path.join(path_fig, "iteration_process.svg"))
