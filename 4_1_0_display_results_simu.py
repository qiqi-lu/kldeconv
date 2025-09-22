"""
Show specified sample in simulation datasets.
"""

import matplotlib.pyplot as plt
import utils.evaluation as eva
from utils.data import read_txt, win2linux, NormalizePercentile
from utils.plot import add_scale_bar, add_significant_star
import skimage.io as io
import numpy as np
import os, pandas
from utils import evaluation as eva
from skimage.measure import profile_line
from scipy.stats import wilcoxon

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
#                    dataset name | (num_data, id_repeat) | id_sample
# ------------------------------------------------------------------------------
# data_info = ("SimuMix3D-128-31-0-0-1", "fp_knonw_bp_n3_r1", 0)
data_info = (
    "SimuMix3D-128-31-05-1-01",
    "fp_knonw_bp_n3_r1",
    0,
    64,  # id_slice_xy
    64,  # id_slice_xz
    ((38, 34), (56, 19)),
    ((86, 44), (86, 64)),
)

methods_name = (
    ("raw", "RAW", 2, "#1B3E22"),
    ("traditional", "Traditional@2", 2, "#2F67AC"),
    ("traditional", "Traditional@30", 30, "#3C6DA8"),
    # ("gaussian", "Gaussian", 2, "#D2E6F0"),
    # ("butterworth", "Butterworth", 2, "#FADCC8"),
    ("wiener-butterworth", "WB@2", 2, "#EC8860"),
    ("rln", "RLN", 2, "#FADCC8"),
    ("kernelnet", "KLD@2", 2, "#B21F2B"),
    # ("kernelnet_ss", "KLD-ss", 2, "#F3B95F"),
    ("gt", "GT", 2, "#212C3E"),
)

id_sample_statitstic = list(range(20))
num_samples_statistic = len(id_sample_statitstic)

# ------------------------------------------------------------------------------
(
    dataset_name_test,
    id_experiment,
    id_sample,
    id_slice_xy,
    id_slice_zx,
    line_xy,
    line_xz,
) = data_info

eps = 0.000001

path_predictions = os.path.join("outputs", "predictions")

info_df = pandas.read_excel("datasets_test.xlsx")
info = info_df[info_df["id"] == dataset_name_test].iloc[0]

path_txt = win2linux(info["path_txt"])
path_lr = win2linux(info["path_lr"])
path_hr = win2linux(info["path_hr"])
pixel_size = info["pixel_size"] / 1000  # um

filenames = read_txt(path_txt)
ratio = info["ratio"]

path_figure = os.path.join(
    "outputs",
    "figures",
    dataset_name_test,
    id_experiment,
    filenames[id_sample].split(".")[0],
)
os.makedirs(path_figure, exist_ok=True)

# ------------------------------------------------------------------------------


num_methods = len(methods_name)
print("-" * 80)
print(f"[INFO] Show methods : {[name[0] for name in methods_name]}")

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=3)

# ------------------------------------------------------------------------------
# load images
# ------------------------------------------------------------------------------
imgs_all = []
for i_meth in range(num_methods):
    name_meth, _, iter, _ = methods_name[i_meth]
    if name_meth == "raw":
        path_sample = os.path.join(path_lr, filenames[id_sample])
        img = io.imread(path_sample).astype(np.float32)
        imgs_all.append(img)

    elif name_meth == "gt":
        path_sample = os.path.join(path_hr, filenames[id_sample])
        img = io.imread(path_sample).astype(np.float32) * ratio
        imgs_all.append(img)

    elif name_meth in ["kernelnet", "kernelnet_ss"]:
        path_sample = os.path.join(
            path_predictions,
            dataset_name_test,
            name_meth,
            dataset_name_test,
            id_experiment,
            filenames[id_sample].split(".")[0],
        )
        y_pred_all = io.imread(os.path.join(path_sample, "y_pred_all.tif"))
        y_pred = y_pred_all[iter]
        imgs_all.append(y_pred)
    elif name_meth in ["rln"]:
        path_sample = os.path.join(
            path_predictions,
            dataset_name_test,
            name_meth,
            dataset_name_test,
            "n1_r1",
            filenames[id_sample].split(".")[0],
            f"y_pred.tif",
        )
        img = io.imread(path_sample).astype(np.float32)
        imgs_all.append(img)
    else:
        path_sample = os.path.join(
            path_predictions,
            dataset_name_test,
            name_meth,
            filenames[id_sample].split(".")[0],
            f"deconv_iter_{iter}.tif",
        )
        img = io.imread(path_sample).astype(np.float32)
        imgs_all.append(img)

imgs_all = np.array(imgs_all)
print("[INFO] results shape : ", imgs_all.shape)

# image normalization ----------------------------------------------------------
imgs_all_norm = np.zeros_like(imgs_all)
for i in range(imgs_all.shape[0]):
    imgs_all_norm[i] = np.clip(normalizer(imgs_all[i]), a_min=0, a_max=2.5)
data_range = 2.5

# ------------------------------------------------------------------------------
# show the restored images
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] plot restored images ...")
Nz, Ny, Nx = imgs_all.shape[1:]

pos_name = Nx * 0.95, Ny * 0.05
pos_metric = Nx * 0.95, Ny * 0.95
pos_direction = Nx * 0.05, Ny * 0.95

line_start_xy, line_end_xy = line_xy[0], line_xy[1]
line_start_xz, line_end_xz = line_xz[0], line_xz[1]

dict_fig = {"dpi": 300, "constrained_layout": True}
dict_text_name = {"color": "white", "fontsize": 18, "ha": "right", "va": "top"}
dict_text_direction = {"color": "white", "fontsize": 18, "ha": "left", "va": "bottom"}
dict_text_metric = {"color": "white", "fontsize": 18, "ha": "right", "va": "bottom"}
dict_line = {"color": "deeppink", "linewidth": 1.5}
dict_image = {"cmap": "gray", "vmin": 0, "vmax": 0.9}
dict_scale_bar = {
    "pixel_size": pixel_size,
    "bar_length": 5,  # um
    "bar_height": 0.01,
    "bar_color": "white",
    "pos": (int(Ny * 0.05), int(Nx * (1 - 0.05))),
}

# ------------------------------------------------------------------------------
nr, nc = 2, num_methods
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

img_gt = imgs_all_norm[-1]
for i_img in range(num_methods):
    ax = axes[:, i_img]
    name_meth, name, iter, color = methods_name[i_img]
    img = imgs_all_norm[i_img]
    ax[0].imshow(img[id_slice_xy], **dict_image)
    ax[1].imshow(img[:, id_slice_zx, :], **dict_image)

    if i_img == 0 or i_img == num_methods - 1:
        ax[0].text(pos_name[0], pos_name[1], name, **dict_text_name)
    else:
        ax[0].text(pos_name[0], pos_name[1], name, **dict_text_name)

    if i_img == 0:
        ax[0].text(pos_direction[0], pos_direction[1], "xy", **dict_text_direction)
        ax[1].text(pos_direction[0], pos_direction[1], "xz", **dict_text_direction)
        # add lines
        ax[0].plot(
            (line_start_xy[0], line_end_xy[0]),
            (line_start_xy[1], line_end_xy[1]),
            **dict_line,
        )
        ax[1].plot(
            (line_start_xz[0], line_end_xz[0]),
            (line_start_xz[1], line_end_xz[1]),
            **dict_line,
        )

    # add metrics
    if i_img != num_methods - 1:
        psnr = eva.PSNR(img_true=img_gt, img_test=img, data_range=data_range)
        # ssim = eva.SSIM(img_true=img_gt, img_test=img, data_range=data_range)
        ssim = eva.MSSSIM(
            img_true=img_gt, img_test=img, data_range=data_range, interp_sf=2
        )

        ax[0].text(
            pos_metric[0],
            pos_metric[1],
            f"{psnr:.2f} | {ssim*100:.2f}",
            **dict_text_metric,
        )

    # add scale bar
    if i_img == num_methods - 1:
        add_scale_bar(ax[0], image=img, **dict_scale_bar)

plt.savefig(os.path.join(path_figure, "img_restored.png"))
plt.savefig(os.path.join(path_figure, "img_restored.svg"))

# ------------------------------------------------------------------------------
# profile line
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] plot profiel lines ...")
dict_profile = {"linewidth": 1.0, "linestyle": "-"}
dict_profile_gt = {"linewidth": 1.0, "linestyle": "--"}
dict_profile_text = {"color": "black", "fontsize": 12, "ha": "left", "va": "top"}
profiles_xy, profiles_xz = [], []

# ------------------------------------------------------------------------------
nr, nc = 1, 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
for i_line in range(2):
    ax = axes[i_line]

    ax.tick_params(direction="in")
    ax.set_xlabel("Distance (pixel)")
    # del top and right axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ticks_y = np.linspace(0, 2, 5)
    ax.set_yticks(ticks_y)
    ax.set_yticklabels(ticks_y)

    ticks_x = [int(x) for x in np.linspace(0, 100, 21)]
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(ticks_x)

    # --------------------------------------------------------------------------
    if i_line == 0:
        print("[INFO] plot line in xy plane")
        line_start = (line_start_xy[1], line_start_xy[0])
        line_end = (line_end_xy[1], line_end_xy[0])

        for i_meth in range(num_methods):
            name_meth, name, iter, color = methods_name[i_meth]
            img = imgs_all_norm[i_meth]
            profile = profile_line(img[id_slice_xy], line_start, line_end, linewidth=1)
            if i_meth == num_methods - 1:
                ax.plot(profile, label=name, color=color, **dict_profile_gt)
            else:
                ax.plot(profile, label=name, color=color, **dict_profile)
            profiles_xy.append(profile.tolist())
        profiles_xy = np.array(profiles_xy)
        ax.set_ylim((0, profiles_xy.max() + 0.1))
        ax.set_ylabel("Normalized intensity")

        # add text -------------------------------------------------------------
        pos_x = profiles_xy.shape[1] * 0.05
        pos_y = (profiles_xy.max() + 0.1) * 0.95
        ax.text(pos_x, pos_y, "xy", **dict_profile_text)

        ax.legend(loc="best", fontsize=6, frameon=False)

    # --------------------------------------------------------------------------
    if i_line == 1:
        print("[INFO] plot line in zx plane")
        line_start = (line_start_xz[1], line_start_xz[0])
        line_end = (line_end_xz[1], line_end_xz[0])
        for i_meth in range(num_methods):
            name_meth, name, iter, color = methods_name[i_meth]
            img = imgs_all_norm[i_meth]
            profile = profile_line(
                img[:, id_slice_zx, :], line_start, line_end, linewidth=1
            )
            if i_meth == num_methods - 1:
                ax.plot(profile, label=name, color=color, **dict_profile_gt)
            else:
                ax.plot(profile, label=name, color=color, **dict_profile)
            profiles_xz.append(profile.tolist())
        profiles_xz = np.array(profiles_xz)
        ax.set_ylim((0, profiles_xz.max() + 0.1))

        # add text -------------------------------------------------------------
        pos_x = profiles_xz.shape[1] * 0.05
        pos_y = (profiles_xz.max() + 0.1) * 0.95
        ax.text(pos_x, pos_y, "xz", **dict_profile_text)
    # --------------------------------------------------------------------------

    ax.set_xlim((0, None))
    # set the axes to square
    ax.set_box_aspect(1)

plt.savefig(os.path.join(path_figure, "img_restored_profile.png"))
plt.savefig(os.path.join(path_figure, "img_restored_profile.svg"))

# ------------------------------------------------------------------------------
# plot metrics of all samples
# ------------------------------------------------------------------------------
print("-" * 80)
metrics_names = ["PSNR", "SSIM", "ZNCC"]
num_metrics = len(metrics_names)

# load all samples -------------------------------------------------------------
print("[INFO] load all samples...")
imgs_all_samples = []
for i_meth in range(num_methods):
    name_meth, name, iter, color = methods_name[i_meth]
    print(f"[INFO] {name}")
    imgs_meth = []
    for i_sample in id_sample_statitstic:
        if name_meth == "raw":
            path_sample = os.path.join(path_lr, filenames[i_sample])
            img = io.imread(path_sample).astype(np.float32)
            imgs_meth.append(img)

        elif name_meth == "gt":
            path_sample = os.path.join(path_hr, filenames[i_sample])
            img = io.imread(path_sample).astype(np.float32) * ratio
            imgs_meth.append(img)

        elif name_meth in ["kernelnet", "kernelnet_ss"]:
            path_sample = os.path.join(
                path_predictions,
                dataset_name_test,
                name_meth,
                dataset_name_test,
                id_experiment,
                filenames[i_sample].split(".")[0],
            )
            y_pred_all = io.imread(os.path.join(path_sample, "y_pred_all.tif"))
            y_pred = y_pred_all[iter]
            imgs_meth.append(y_pred)
        elif name_meth in ["rln"]:
            path_sample = os.path.join(
                path_predictions,
                dataset_name_test,
                name_meth,
                dataset_name_test,
                "n1_r1",
                filenames[i_sample].split(".")[0],
                f"y_pred.tif",
            )
            img = io.imread(path_sample).astype(np.float32)
            imgs_meth.append(img)
        else:
            path_sample = os.path.join(
                path_predictions,
                dataset_name_test,
                name_meth,
                filenames[i_sample].split(".")[0],
                f"deconv_iter_{iter}.tif",
            )
            img = io.imread(path_sample).astype(np.float32)
            imgs_meth.append(img)
    imgs_all_samples.append(imgs_meth)
imgs_all_samples = np.array(imgs_all_samples)
print("[INFO] results shape : ", imgs_all_samples.shape)

# image normalization ----------------------------------------------------------
imgs_all_samples_norm = np.zeros_like(imgs_all_samples)
for i in range(imgs_all_samples.shape[0]):
    for j in range(imgs_all_samples.shape[1]):
        imgs_all_samples_norm[i, j] = np.clip(
            normalizer(imgs_all_samples[i, j]), a_min=0, a_max=2.5
        )
data_range = 2.5

# calculate metrics ------------------------------------------------------------
metrics_all_samples = np.zeros((num_methods - 1, num_samples_statistic, num_metrics))
for i_meth in range(num_methods - 1):
    for i_sample in range(num_samples_statistic):
        img_gt = imgs_all_samples_norm[-1, i_sample]
        img_pred = imgs_all_samples_norm[i_meth, i_sample]
        psnr = eva.PSNR(img_true=img_gt, img_test=img_pred, data_range=data_range)
        # ssim = eva.SSIM(img_true=img_gt, img_test=img_pred, data_range=data_range)
        ssim = eva.MSSSIM(
            img_true=img_gt, img_test=img_pred, data_range=data_range, interp_sf=2
        )
        zncc = eva.NCC(img_true=img_gt, img_test=img_pred)
        metrics_all_samples[i_meth, i_sample, :] = [psnr, ssim, zncc]
print("[INFO] metrics shape : ", metrics_all_samples.shape)

# ------------------------------------------------------------------------------
# plot metrics of all samples
# ------------------------------------------------------------------------------
print("[INFO] plot metrics of all samples ...")
nr, nc = 1, num_metrics
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)

ticks = list(np.linspace(0, 35, 15))
axes[0].set_yticks(ticks)
axes[0].set_yticklabels([f"{x:.1f}" for x in ticks])

ticks = list(np.linspace(0, 1, 11))
axes[1].set_yticks(ticks)
axes[1].set_yticklabels([f"{x:.1f}" for x in ticks])

ticks = list(np.linspace(0, 1, 21))
axes[2].set_yticks(ticks)
axes[2].set_yticklabels([f"{x:.1f}" for x in ticks])

metrics_all_samples_mean = metrics_all_samples.mean(axis=1)
metrics_all_samples_std = metrics_all_samples.std(axis=1)
# calculate pvalue
test_pairs = [(i, num_methods - 2) for i in range(num_methods - 2)]
pvalues = np.zeros((num_metrics, len(test_pairs)))
for i_metric in range(num_metrics):
    for i_pair, pair in enumerate(test_pairs):
        pvalues[i_metric, i_pair] = wilcoxon(
            metrics_all_samples[pair[0], :, i_metric],
            metrics_all_samples[pair[1], :, i_metric],
            alternative="two-sided",
        )[1]
print("[INFO] pvalues shape : ", pvalues.shape)

for i_metric in range(num_metrics):
    ax = axes[i_metric]
    data = metrics_all_samples[:, :, i_metric]
    data_mean = metrics_all_samples_mean[:, i_metric]
    data_std = metrics_all_samples_std[:, i_metric]
    data_max, data_min = data.max(), data.min()
    pvs = pvalues[i_metric, :]
    # --------------------------------------------------------------------------

    x_pos = np.arange(num_methods - 1)
    ax.bar(
        x_pos,
        data_mean,
        yerr=data_std,
        capsize=5,
        width=0.8,
        color=[color for _, _, _, color in methods_name[:-1]],
        label=[name for _, name, _, _ in methods_name[:-1]],
    )

    # --------------------------------------------------------------------------
    if i_metric == 0:
        ax.legend(loc="best", fontsize=6, frameon=False)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel(metrics_names[i_metric])
    y_lim = (
        data_min - (data_max - data_min) * 0.1,
        data_max + (data_max - data_min) * 0.1,
    )
    ax.set_ylim(y_lim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)

    # add pvalue marker --------------------------------------------------------
    for i_pair, pair in enumerate(test_pairs):
        pv = pvs[i_pair]
        star_x = x_pos[pair[0]]
        star_y = data_mean[pair[0]] + data_std[pair[0]] + 0.02 * (y_lim[1] - y_lim[0])
        add_significant_star(ax, star_x, star_y, pv)

plt.savefig(os.path.join(os.path.dirname(path_figure), "img_restored_metrics.png"))
plt.savefig(os.path.join(os.path.dirname(path_figure), "img_restored_metrics.svg"))
