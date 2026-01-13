"""
Show specified sample in simulation datasets.
"""

import matplotlib.pyplot as plt
import utils.evaluation as eva
import utils.data as utils_data
from utils.data import read_txt, win2linux
import skimage.io as io
import skimage.exposure as exposure
import numpy as np
import os, pandas
from utils import evaluation as eva
from skimage.measure import profile_line

from tabulate import tabulate as tabu


def tabulate(arr, floatfmt=".8f"):
    return tabu(arr, floatfmt=floatfmt, tablefmt="plain")


# ------------------------------------------------------------------------------
def cal_ssim(x, y):
    # need 3D input
    if y.shape[0] >= 7:  # the size of the filter in SSIM is at least 7
        return eva.SSIM(
            img_true=y,
            img_test=x,
            data_range=y.max() - y.min(),
            multichannel=False,
            channle_axis=None,
            version_wang=False,
        )
    else:
        return eva.SSIM(
            img_true=y,
            img_test=x,
            data_range=y.max() - y.min(),
            multichannel=True,
            channle_axis=0,
            version_wang=False,
        )


def cal_psnr(x, y):
    # need 3D input
    return eva.PSNR(img_true=y, img_test=x, data_range=y.max() - y.min())


def cal_ncc(x, y):
    return eva.NCC(img_true=y, img_test=x)


# ------------------------------------------------------------------------------
#                    dataset name | (num_data, id_repeat) | id_sample
# ------------------------------------------------------------------------------
data_info = ("SimuMix3D-128-31-0-0-1", "fp_knonw_bp_n3_r1", 0)

# methods_iter = [100, 100, 100, 30]
# methods_iter = [30, 30, 30, 30]
methods_iter = [2, 2, 2, 2]

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
    "outputs", "predictions", dataset_name_test, name_net, dataset_name_test
)
path_kernel = os.path.join(path_predictions, "kernel")
path_sample = os.path.join(
    path_predictions, id_experiment, filenames[id_sample].split(".")[0]
)

print("-" * 80)
print("[INFO] load results from :", path_sample)
print("[INFO] load kernels from :", path_kernel)

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
vmax_ker, color_map_ker = ker_true.max(), "hot"
vmax_img, color_map_img = img_y.max(), "gray"
vamx_img_diff = vmax_img

dict_ker = {"cmap": color_map_ker, "vmin": 0.0, "vmax": vmax_ker}
dict_ker_profile = {"linewidth": 0.5}
dict_img = {"cmap": color_map_img, "vmin": 0.0, "vmax": vmax_img}
dict_img_diff = {"cmap": "gray", "vmin": 0.0, "vmax": vamx_img_diff}
dict_fig = {"dpi": 300, "constrained_layout": True}

# ------------------------------------------------------------------------------
# load reaults of conventional methods
# ------------------------------------------------------------------------------
print("load results of conventional methods ...")
methods_name = ["traditional", "gaussian", "butterworth", "wiener_butterworth"]
methods_color = ["#D04848", "#007F73", "#4CCD99", "#FFC700", "#FFF455"]


def load_result(path, name, iter):
    out = []
    out.append(io.imread(os.path.join(path, name, f"deconv_{iter}.tif")))
    out.append(io.imread(os.path.join(path, name, "deconv_bp.tif")))
    out.append(np.load(os.path.join(path, name, f"deconv_metrics_{iter}.npy")))
    return out


# load results from conventional methods
out_trad = load_result(path_sample, methods_name[0], methods_iter[0])
out_gaus = load_result(path_sample, methods_name[1], methods_iter[1])
out_butt = load_result(path_sample, methods_name[2], methods_iter[2])
out_wien = load_result(path_sample, methods_name[3], methods_iter[3])

# ------------------------------------------------------------------------------
# show the restored images
# ------------------------------------------------------------------------------
print("-" * 80)
print("plot restored images ...")
pos_text_x, pos_text_y = 5, 10
line_start_xy, line_end_xy = (38, 34), (56, 19)
line_start_xz, line_end_xz = (86, 44), (86, 64)
id_slice = Nz // 2
# ------------------------------------------------------------------------------
nr, nc = 4, 7
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, dpi=300, figsize=(3 * nc, 3 * nr), constrained_layout=True
)
[ax.set_axis_off() for ax in axes.ravel()]

dict_text = {"color": "white", "fontsize": "x-large"}
dict_line = {"color": "white", "linewidth": 1}
dict_image = {"cmap": "gray", "vmin": 0, "vmax": vmax_img}
dict_image_diff = {"cmap": "gray", "vmin": 0, "vmax": vamx_img_diff}

axes[0, 0].text(
    pos_text_x,
    pos_text_y,
    "RAW (xy) ({:>.2f}, {:>.4f})".format(
        cal_psnr(img_x, img_y), cal_ssim(img_x, img_y)
    ),
    **dict_text,
)
axes[0, 6].text(pos_text_x, pos_text_y, "GT (xy)", **dict_text)
axes[2, 0].text(pos_text_x, pos_text_y, "RAW (xz)", **dict_text)
axes[2, 6].text(pos_text_x, pos_text_y, "GT (xz)", **dict_text)

# RAW
axes[0, 0].imshow(img_x[id_slice], **dict_image)
axes[2, 0].imshow(img_x[:, Ny // 2, :], **dict_image)
axes[0, 0].plot(
    (line_start_xy[0], line_end_xy[0]), (line_start_xy[1], line_end_xy[1]), **dict_line
)
axes[2, 0].plot(
    (line_start_xz[0], line_end_xz[0]), (line_start_xz[1], line_end_xz[1]), **dict_line
)
print("RAW", "PSNR:", cal_psnr(img_x, img_y))

axes[0, 6].imshow(img_y[id_slice], **dict_image)
axes[2, 6].imshow(img_y[:, Ny // 2, :], **dict_image)


# Traditional, Gaussian, Butterworth, WB
def show_result(out, axes, name):
    diff = np.abs(out[0] - img_y)
    num_iter = out[-1].shape[0] - 1
    axes[0].text(
        pos_text_x, pos_text_y, "{} {:d} it".format(name, num_iter), **dict_text
    )
    axes[1].text(
        pos_text_x,
        pos_text_y,
        "({:>.2f}, {:>.4f})".format(out[-1][-1, 0], out[-1][-1, 1]),
        **dict_text,
    )
    axes[0].imshow(out[0][id_slice], **dict_image)
    axes[1].imshow(diff[id_slice], **dict_image_diff)
    axes[2].imshow(out[0][:, Ny // 2, :], **dict_image)
    axes[3].imshow(diff[:, Ny // 2, :], **dict_image_diff)
    print(name, "PSNR:", cal_psnr(out[0], img_y))


show_result(out_trad, axes[0:4, 1], name="Traditional")
show_result(out_gaus, axes[0:4, 2], name="Gaussian")
show_result(out_butt, axes[0:4, 3], name="Butterworth")
show_result(out_wien, axes[0:4, 4], name="WB")

# KLD
axes[0, 5].text(
    pos_text_x, pos_text_y, "{} {:d} it".format("KLD", num_iter_train), **dict_text
)
axes[0, 5].imshow(y_pred[id_slice], **dict_image)
axes[1, 5].text(
    pos_text_x,
    pos_text_y,
    "({:>.2f}, {:>.4f})".format(cal_psnr(y_pred, img_y), cal_ssim(y_pred, img_y)),
    **dict_text,
)
axes[1, 5].imshow(np.abs(y_pred - img_y)[id_slice], **dict_image_diff)
diff_ypred_y = np.abs(y_pred - img_y)
axes[2, 5].imshow(y_pred[:, Ny // 2, :], **dict_image)
axes[3, 5].imshow(diff_ypred_y[:, Ny // 2, :], **dict_image_diff)
print("KLD", "PSNR:", cal_psnr(y_pred, img_y))

io.imsave(
    fname=os.path.join(path_sample, "xz.tif"),
    arr=y_pred[:, Ny // 2, :],
    check_contrast=False,
)
io.imsave(
    fname=os.path.join(path_sample, "xy.tif"),
    arr=y_pred[id_slice],
    check_contrast=False,
)

plt.savefig(os.path.join(path_sample, "img_restored.png"))
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(os.path.join(path_sample, "img_restored.svg"))
print("-" * 80)

# ------------------------------------------------------------------------------
# profile line
# ------------------------------------------------------------------------------
print("-" * 80)
print("plot profiel lines ...")
methods_name = ["Traditional", "Gaussian", "Butterworth", "WB"]
colors = ["#B3BE9D", "#FDE767", "#F3B95F", "#E28154"]
# ['KernelNet', 'WB', 'Butterworth', 'Gaussian', 'Traditional', 'RAW']
# ['#D04848', '#E28154', '#F3B95F', '#FDE767', '#B3BE9D','#6895D2']

# ------------------------------------------------------------------------------
nr, nc = 1, 2
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, dpi=300, figsize=(3 * nc, 3 * nr), constrained_layout=True
)

dict_profile = {"linewidth": 0.5}

profiles_xy, profiles_xz = [], []
for i, ax in enumerate(axes.ravel()):
    # line in xy plane
    if i == 0:
        line_start = (line_start_xy[1], line_start_xy[0])
        line_end = (line_end_xy[1], line_end_xy[0])
        # RAW
        profile = profile_line(img_x[id_slice], line_start, line_end, linewidth=1)
        ax.plot(profile, label="RAW", color="#2A629A", **dict_profile)
        profiles_xy.append(profile.tolist())
        # Traditional, Gaussian, Butterworth, WB
        for out, name, color in zip(
            [out_trad, out_gaus, out_butt, out_wien], methods_name, colors
        ):
            profile = profile_line(out[0][id_slice], line_start, line_end, linewidth=1)
            ax.plot(profile, label=name, color=color, **dict_profile)
            profiles_xy.append(profile.tolist())
        # KLD
        profile = profile_line(y_pred[id_slice], line_start, line_end, linewidth=1)
        ax.plot(profile, label="KLD", color="red", **dict_profile)
        profiles_xy.append(profile.tolist())
        # GT
        profile = profile_line(img_y[id_slice], line_start, line_end, linewidth=1)
        ax.plot(profile, label="GT", color="black", linestyle="--", **dict_profile)
        profiles_xy.append(profile.tolist())

    # line in xz plane
    if i == 1:
        line_start = (line_start_xz[1], line_start_xz[0])
        line_end = (line_end_xz[1], line_end_xz[0])
        # RAW
        profile = profile_line(img_x[:, Ny // 2, :], line_start, line_end, linewidth=1)
        ax.plot(profile, label="RAW", color="#2A629A", **dict_profile)
        profiles_xz.append(profile.tolist())
        # Traditional, Gaussian, Butterworth, WB
        for out, name, color in zip(
            [out_trad, out_gaus, out_butt, out_wien], methods_name, colors
        ):
            profile = profile_line(
                out[0][:, Ny // 2, :], line_start, line_end, linewidth=1
            )
            ax.plot(profile, label=name, color=color, **dict_profile)
            profiles_xz.append(profile.tolist())
        # KLD
        profile = profile_line(y_pred[:, Ny // 2, :], line_start, line_end, linewidth=1)
        ax.plot(profile, label="KLD", color="red", **dict_profile)
        profiles_xz.append(profile.tolist())
        # GT
        profile = profile_line(img_y[:, Ny // 2, :], line_start, line_end, linewidth=1)
        ax.plot(profile, label="GT", color="black", linestyle="--", **dict_profile)
        profiles_xz.append(profile.tolist())

    for pos in ["top", "bottom", "left", "right"]:
        ax.spines[pos].set_linewidth(0.5)
        ax.tick_params(width=0.5)
    ax.tick_params(direction="in")
    ax.set_xlim((0, None))
    ax.set_ylim((0, None))
    ax.set_ylabel("Intensity (AU)")
    ax.set_xlabel("Distance (pixel)")

print("-" * 80)
print("Vallue of line profiels:")
print("-" * 80)
print(tabulate(profiles_xy))
print("-" * 80)
print(tabulate(profiles_xz))
print("-" * 80)

plt.legend(fontsize="xx-small")
plt.savefig(os.path.join(path_sample, "img_restored_profile.png"))
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(os.path.join(path_sample, "img_restored_profile.svg"))

# ------------------------------------------------------------------------------
# show metrics curve
# ------------------------------------------------------------------------------
print("plot metrics ...")
psnrs, ssims, nccs = [], [], []

for i in range(len(y_pred_all)):
    psnrs.append(cal_psnr(y_pred_all[i], img_y))
    ssims.append(cal_ssim(y_pred_all[i], img_y))
    nccs.append(cal_ncc(y_pred_all[i], img_y))
mtrics_kld = np.stack([psnrs, ssims, nccs]).transpose()
# ------------------------------------------------------------------------------
nr, nc = 1, 3
fig, axes = plt.subplots(
    nrows=nr, ncols=nc, dpi=300, figsize=(3 * nc, 3 * nr), constrained_layout=True
)

dict_line_metrics = {"linestyle": "-", "marker": ".", "markersize": 2, "linewidth": 0.5}
dict_line_axh = {"linestyle": "--", "linewidth": 0.5}

# methods_color = ['red', '#E28154', '#F3B95F', '#FDE767', '#B3BE9D',\
#     '#6895D2']

for i in range(3):  # (PSNR, SSIM, NCC)
    axes[i].plot(
        out_trad[-1][:, i],
        color=methods_color[4],
        label="Traditional",
        **dict_line_metrics,
    )
    axes[i].plot(
        out_gaus[-1][:, i],
        color=methods_color[3],
        label="Gaussian",
        **dict_line_metrics,
    )
    axes[i].plot(
        out_butt[-1][:, i],
        color=methods_color[2],
        label="Butterworth",
        **dict_line_metrics,
    )
    axes[i].plot(
        out_wien[-1][:, i], color=methods_color[1], label="WB", **dict_line_metrics
    )
    axes[i].plot(
        mtrics_kld[:, i], color=methods_color[0], label="KLD", **dict_line_metrics
    )

    axes[i].axhline(
        y=mtrics_kld[num_iter_train, i], color=methods_color[0], **dict_line_axh
    )
    axes[i].axhline(
        y=out_wien[-1][num_iter_train, i], color=methods_color[1], **dict_line_axh
    )

    print("-" * 80)
    print(
        tabulate(
            [
                out_trad[-1][:, i],
                out_gaus[-1][:, i],
                out_butt[-1][:, i],
                out_wien[-1][:, i],
                mtrics_kld[:, i],
            ]
        )
    )
    print("-" * 80)

    axes[i].spines[["right", "top"]].set_visible(False)
    axes[i].set_xlabel("Iteration Number")
    axes[i].set_xlim([0, None])
    axes[i].legend(edgecolor="white")

axes[0].set_ylabel("PSNR")
axes[1].set_ylabel("SSIM")
axes[2].set_ylabel("NCC")

plt.savefig(os.path.join(path_sample, "curve_metrics.png"))
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(os.path.join(path_sample, "curve_metrics.svg"))
# ------------------------------------------------------------------------------
