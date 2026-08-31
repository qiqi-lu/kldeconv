"""
Display the intermediate reuslts during iteration process.
- simulation datasets
"""

import matplotlib.pyplot as plt
from utils.data import read_txt, win2linux
from utils.plot import add_scale_bar
import skimage.io as io
import numpy as np
import os, pandas

plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------------------
#            dataset name | (num_data, id_repeat) | id_sample
# ------------------------------------------------------------------------------
data_info = ("SimuMix3D-128-31-0-0-1", "fp_knonw_bp_n3_r1", 0)
# data_info = ("SirDNA-1024", "fp_n1_r1_bp_n1_r1", 0)

# ------------------------------------------------------------------------------
dataset_name_test, id_experiment, id_sample = data_info
name_net = "kernelnet"
num_iter_train = 2
eps = 0.000001

# ------------------------------------------------------------------------------
info_df = pandas.read_excel("datasets_test.xlsx")
info = info_df[info_df["id"] == dataset_name_test].iloc[0]

path_txt = win2linux(info["path_txt"])
path_lr = win2linux(info["path_lr"])
path_hr = win2linux(info["path_hr"])
pixel_size = info["pixel_size"] / 1000
slice_space = info["slice_space"] / 1000

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

file_id = filenames[id_sample].split(".")[0]
path_sample = os.path.join(path_predictions, file_id)
path_figure = os.path.join(
    "outputs", "figures", "analysis_image", dataset_name_test, file_id, "intermediate"
)
os.makedirs(path_figure, exist_ok=True)

print("-" * 80)
print(f"[INFO] Load results from : {path_sample}")
print(f"[INFO] Load kernels from : {path_kernel}")
print(f"[INFO] Save figures to   : {path_figure}")

# ------------------------------------------------------------------------------
# load kernels and results of KLD
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] load kernels ...")
ker_init = io.imread(os.path.join(path_kernel, "kernel_init.tif"))
ker_true = io.imread(os.path.join(path_kernel, "kernel_true.tif"))
ker_FP = io.imread(os.path.join(path_kernel, "kernel_fp.tif"))
ker_BP = io.imread(os.path.join(path_kernel, "kernel_bp.tif"))

print("[INFO] load results ...")
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
# show iteration process
# ------------------------------------------------------------------------------
dict_fig = {"dpi": 300, "constrained_layout": True}
dict_ker_fp = {"cmap": "hot", "vmin": 0.0, "vmax": ker_true.max()}
dict_ker_bp = {"cmap": "hot", "vmin": 0.0, "vmax": np.max(ker_BP)}
dict_img = {"cmap": "gray", "vmin": 0.0, "vmax": img_y.max() * 0.6}
dict_dv = {"cmap": "bwr", "vmin": 0.5, "vmax": 1.5}
dict_bp = {"cmap": "bwr", "vmin": -0.5, "vmax": 2.5}
dict_text_lb = {
    "color": "white",
    "fontsize": 20,
    "ha": "left",
    "va": "bottom",
    "x": 0.05,
    "y": 0.05,
}
dict_text_rt = {
    "color": "white",
    "fontsize": 20,
    "ha": "right",
    "va": "top",
    "x": 0.95,
    "y": 0.95,
}

# ------------------------------------------------------------------------------
print("[INFO] plot iteration process...")
nr, nc = 3, 9
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

pos_x, pos_y = 0.05, 0.95

axes[0, 0].text(s="xy", transform=axes[0, 0].transAxes, **dict_text_lb)
axes[1, 0].text(s="xz", transform=axes[1, 0].transAxes, **dict_text_lb)
axes[0, 0].text(s="input", transform=axes[0, 0].transAxes, **dict_text_rt)
axes[0, 1].text(s="x$_0$", transform=axes[0, 1].transAxes, **dict_text_rt)
axes[0, 8].text(s="GT", transform=axes[0, 8].transAxes, **dict_text_rt)

# add scale bar to image
img_shape = img_x.shape
print(f"[INFO] img_shape = {img_shape}")
tp = 0.05
dict_scale_bar = {
    "pixel_size": pixel_size,
    "bar_length": 5,  # um
    "bar_height": 0.01,
    "bar_color": "white",
    "pos": (int(img_shape[1] * (1 - 6 * tp)), int(img_shape[0] * (1 - tp))),
}
add_scale_bar(axes[0, 0], image=img_x, **dict_scale_bar)

# add scale bar to kernel
ker_shape = ker_FP.shape
print(f"[INFO] ker_shape = {ker_shape}")
dict_scale_bar = {
    "pixel_size": pixel_size,
    "bar_length": 1,  # um
    "bar_height": 0.01,
    "bar_color": "white",
    "pos": (int(ker_shape[1] * (1 - 5 * tp)), int(ker_shape[0] * (1 - tp))),
}
add_scale_bar(axes[0, 2], image=ker_FP, **dict_scale_bar)

# get the colormap of axes[0, 4] and plot it in axes[2, 4]
aximage = axes[0, 4].get_images()[0]
cbar = fig.colorbar(aximage, cax=axes[2, 4], orientation="horizontal")
# set the aspect of axes
axes[2, 4].set_aspect(0.1)
axes[2, 4].set_axis_on()


# save
plt.savefig(os.path.join(path_figure, "iteration_process.png"))
plt.savefig(os.path.join(path_figure, "iteration_process.svg"))

# ------------------------------------------------------------------------------
# compare the x0 and y_fp
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] plot compare the x0 and y_fp...")

nr, nc = 2, 4
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)

Nz, Ny, Nx = img_x.shape

# xy plane ---------------------------------------------------------------------
i_slice = Nz // 2
axes[0, 0].imshow(img_x0[i_slice], **dict_img)
axes[0, 1].imshow(img_y_fp[i_slice], **dict_img)
axes[1, 0].imshow(img_y[i_slice], **dict_img)

# get the position of max value in the image[i_slice]
xy_plane = img_x0[i_slice]
y_idx, x_idx = np.unravel_index(xy_plane.argmax(), xy_plane.shape)

axes[1, 1].plot(img_x0[i_slice, y_idx], "black", label="x0")
axes[1, 1].plot(img_y[i_slice, y_idx], "red", label="y")
axes[1, 1].plot(img_y_fp[i_slice, y_idx], "green", label="y_fp", linestyle="--")
# remove box of legend
axes[1, 1].legend(frameon=False)

# plot the line in the image
for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
    ax.plot([0, Nx - 1], [y_idx, y_idx], "red", linewidth=1)
axes[0, 0].set_title("x0")
axes[0, 1].set_title("y_fp")
axes[1, 0].set_title("y")

# xz plane ---------------------------------------------------------------------
i_slice = Ny // 2
axes[0, 2].imshow(img_x0[:, i_slice], **dict_img)
axes[0, 3].imshow(img_y_fp[:, i_slice], **dict_img)
axes[1, 2].imshow(img_y[:, i_slice], **dict_img)

# get the position of max value in the image[i_slice]
xz_plane = img_x0[:, i_slice]
y_idx, x_idx = np.unravel_index(xz_plane.argmax(), xz_plane.shape)

axes[1, 3].plot(img_x0[:, i_slice, x_idx], "black", label="x0")
axes[1, 3].plot(img_y[:, i_slice, x_idx], "red", label="y")
axes[1, 3].plot(img_y_fp[:, i_slice, x_idx], "green", label="y_fp", linestyle="--")
# remove box of legend

axes[1, 3].legend(frameon=False)
# plot the line in the image
for ax in [axes[0, 2], axes[0, 3], axes[1, 2]]:
    ax.plot([x_idx, x_idx], [0, Nz - 1], "red", linewidth=1)
axes[0, 2].set_title("x$_0$")
axes[0, 3].set_title("FP(y)")
axes[1, 2].set_title("y")

# save
plt.savefig(os.path.join(path_figure, "compare_x0_y_fp.png"))
