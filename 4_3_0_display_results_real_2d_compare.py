"""
Display the image restored by different methods.
"""

import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
from skimage.measure import profile_line
import numpy as np
import os, pandas, tqdm
from utils import evaluation as eva
import matplotlib.patches as patches
import matplotlib.colors as colors
from utils.data import NormalizePercentile, win2linux, read_txt
from utils.plot import (
    colorize,
    add_scale_bar,
    add_patch,
    add_significant_bars,
    add_significant_star,
)
from scipy.stats import wilcoxon

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
# figure_name = "2d_real"
figure_name = "3d_real"
num_samples = 6

# ------------------------------------------------------------------------------
# dataset name, id sample, pos, size
dataset_show_info_list = {
    "2d_real": (
        ("F-actin-nonlinear-9", 5, (200, 200), 100, (255, 0, 0)),
        ("Microtubules2-9", 0, (200, 200), 100, (3, 174, 210)),
    ),
    "3d_real": (
        ("Microtubule2-3d-1024", 0, 1, (200, 200), 100, (3, 174, 210)),
        ("Nuclear-pore-complex2-1024", 0, 1, (200, 200), 100, (0, 255, 0)),
    ),
}

methods_show_info_list = {
    "2d_real": (
        ("DeconvBlind", "deconvblind", "deconv.tif", "#42B4B5"),
        ("DFCAN", "dfcan", "y_pred.tif", "#57AA3E"),
        ("RLD@100", "traditional", "deconv_iter_100.tif", "#4D8FCB"),
        ("KLD", "kernelnet", "y_pred_all.tif", "#D95D5B"),
    ),
    "3d_real": (
        ("DeconvBlind", "deconvblind", "deconv.tif", "#42B4B5"),
        ("RLN", "rln", "y_pred.tif", "#B78E72"),
        ("RLD@15", "traditional", "deconv_15.tif", "#4D8FCB"),
        ("KLD", "kernelnet", "y_pred_all.tif", "#D95D5B"),
    ),
}

# ------------------------------------------------------------------------------
dataset_show_info = dataset_show_info_list[figure_name]
methods_show_info = methods_show_info_list[figure_name]

if "2d" in figure_name:
    ndim = 2
elif "3d" in figure_name:
    ndim = 3
else:
    raise ValueError(f"[ERROR] Cannot find the dimension of the figure.")

show_patch = True
path_fig = os.path.join("outputs", "figures")
info_df = pandas.read_excel("datasets_test.xlsx")
normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=ndim)
dict_clip = {"a_min": 0, "a_max": 2.5}
data_range = dict_clip["a_max"] - dict_clip["a_min"]
dict_fig = dict(constrained_layout=True, dpi=300)

# ------------------------------------------------------------------------------
num_datasets = len(dataset_show_info)
num_methods = len(methods_show_info)

methods_names = ["RAW"]
for i_meth in range(len(methods_show_info)):
    methods_names.append(methods_show_info[i_meth][0])
methods_names.append("GT")

metrics_names = ["PSNR", "SSIM", "NCC"]
num_metrics = len(metrics_names)

dataset_names = []
for i_dataset in range(num_datasets):
    dataset_names.append(dataset_show_info[i_dataset][0])

print("-" * 80)
print(f"[INFO] methods titles: {methods_names}")

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
results_all = []
for i_dataset in range(num_datasets):
    # load results
    dataset_name_test = dataset_show_info[i_dataset][0]
    path_result = os.path.join("outputs", "predictions", dataset_name_test)
    print("[INFO] Load results from :", path_result)

    info = info_df[info_df["id"] == dataset_name_test].iloc[0]

    # --------------------------------------------------------------------------
    results = []
    pbar = tqdm.tqdm(total=range(num_samples), desc="Load results", ncols=80)
    for i_sample in range(num_samples):
        pbar.update(1)
        results_ss = []
        # load raw and gt images
        path_raw, path_gt = win2linux(info["path_lr"]), win2linux(info["path_hr"])
        filenames = read_txt(win2linux(info["path_txt"]))
        x = io.imread(os.path.join(path_raw, filenames[i_sample]))
        y = io.imread(os.path.join(path_gt, filenames[i_sample]))

        results_ss.append(x.astype(np.float32))

        for i_meth in range(num_methods):
            meth_title, meth_tag, meth_file = methods_show_info[i_meth][:3]

            # load restoed image from KLDeconv method --------------------------
            if meth_title == "KLD":
                path_tmp = os.path.join(
                    path_result,
                    meth_tag,
                    dataset_name_test,
                    "fp_n1_r1_bp_n1_r1",
                    f"sample_{i_sample}",
                )
                y_pred = io.imread(os.path.join(path_tmp, "y_pred_all.tif"))

                # the imread funciton will automaticly reshape the results
                # when having 3 channels.
                if ndim == 2:
                    if y_pred.shape[-1] in [3, 4]:
                        y_pred = np.transpose(y_pred, axes=(-1, 0, 1))
                y_pred = y_pred[-1]
            elif meth_title in ["DFCAN", "RLN"]:
                path_tmp = os.path.join(
                    path_result,
                    meth_tag,
                    dataset_name_test,
                    "n1_r1",
                    f"sample_{i_sample}",
                    meth_file,
                )
                y_pred = io.imread(path_tmp)
            else:
                path_tmp = os.path.join(
                    path_result, meth_tag, f"sample_{i_sample}", meth_file
                )
                y_pred = io.imread(path_tmp)

            results_ss.append(y_pred.astype(np.float32))
        results_ss.append(y.astype(np.float32))
        results.append(results_ss)
    pbar.close()
    results_all.append(results)


print("-" * 80)
print(f"[INFO] Num of datasets: {len(results_all)}")
print(f"[INFO] Num of samples : {len(results_all[0])}")
print(f"[INFO] Num of methods (+ raw and gt): {len(results_all[0][0])}")
print(f"[INFO] Shape of image : {results_all[0][0][0].shape}")
print("-" * 80)

# ------------------------------------------------------------------------------
# show results
# ------------------------------------------------------------------------------
nr, nc = num_datasets, num_methods + 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

for i_dataset in range(num_datasets):
    axes_ds = axes[i_dataset]

    if ndim == 2:
        dataset_name_test, id_sample, pos, size, color_img = dataset_show_info[
            i_dataset
        ]
    if ndim == 3:
        dataset_name_test, id_sample, id_slice, pos, size, color_img = (
            dataset_show_info[i_dataset]
        )

    info = info_df[info_df["id"] == dataset_name_test].iloc[0]
    pixel_size = float(info["pixel_size"])
    results = results_all[i_dataset][id_sample]
    N_meth = len(results)  # number of methods (include raw and gt)

    # get input and gt image ---------------------------------------------------
    img_raw, img_gt = results[0], results[-1]
    # normalize image
    img_raw = np.clip(normalizer(img_raw), **dict_clip)
    img_gt = np.clip(normalizer(img_gt), **dict_clip)

    # show restored image from different methods -------------------------------
    for i_meth in range(N_meth):
        img = np.clip(normalizer(results[i_meth]), **dict_clip)

        # plot color image
        img_color = colorize(img, vmin=0.0, vmax=0.9, color=color_img)
        axes_ds[i_meth].imshow(img_color)

        # add scale bar --------------------------------------------------------
        if i_meth == len(results) - 1:
            img_shape = img.shape
            tp = 0.05
            dict_scale_bar = {
                "pixel_size": pixel_size,
                "bar_length": 5,  # um
                "bar_height": 0.01,
                "bar_color": "white",
                "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
            }
            add_scale_bar(axes_ds[i_meth], image=img, **dict_scale_bar)

        # add metrics value ----------------------------------------------------
        pos_text = (int(img.shape[0] * 0.04), int(img.shape[1] * 0.04))
        dict_text_metric = {
            "fontsize": 14,
            "color": "white",
            "ha": "left",
            "va": "bottom",
        }
        if i_meth != len(results) - 1:
            dict_eva = {"img_true": img_gt, "img_test": img}
            psnr = eva.PSNR(**dict_eva, data_range=data_range)
            ssim = eva.SSIM(**dict_eva, data_range=data_range)
            axes_ds[i_meth].text(
                pos_text[1],
                img.shape[0] - pos_text[0],
                f"{psnr:.2f} | {ssim*100:.2f}",
                **dict_text_metric,
            )

        # add zoom patch -------------------------------------------------------
        if show_patch:
            show_box = True if i_meth == len(results) - 1 else False
            add_patch(
                ax=axes_ds[i_meth],
                image=img_color,
                pos=pos,
                size=size,
                show_box=show_box,
                axes_lw=1,
                box_lw=0.5,
                box_color="white",
            )

        # add title ------------------------------------------------------------
        dict_text_meth = {"fontsize": 14, "color": "white", "ha": "right", "va": "top"}
        pos_text = (int(img.shape[0] * 0.04), int(img.shape[1] * 0.04))
        if i_dataset == 0:
            axes_ds[i_meth].text(
                img.shape[1] - pos_text[1],
                pos_text[0],
                methods_names[i_meth],
                **dict_text_meth,
            )

        dict_text_struc = {"fontsize": 14, "color": "white", "ha": "left", "va": "top"}
        if i_meth == 0:
            axes_ds[i_meth].text(
                pos_text[1], pos_text[0], dataset_name_test, **dict_text_struc
            )

    plt.savefig(os.path.join(path_fig, f"image_restored_compare_{figure_name}.png"))
    plt.savefig(os.path.join(path_fig, f"image_restored_compare_{figure_name}.svg"))

# ------------------------------------------------------------------------------
# statistics analysis
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] Statistics analysis ...")
# calculate the metrics value of each method
metrics_dataset = []
for i_dataset in range(num_datasets):
    metrics_sample = []
    for i_sample in range(num_samples):
        results = results_all[i_dataset][i_sample]

        img_gt = results[-1]
        img_gt = np.clip(normalizer(img_gt), **dict_clip)

        metrics_meth = []
        for i_meth in range(num_methods + 1):
            img = results[i_meth]
            img = np.clip(normalizer(img), **dict_clip)

            dict_eva = {"img_true": img_gt, "img_test": img}
            psnr = eva.PSNR(**dict_eva, data_range=data_range)
            ssim = eva.SSIM(**dict_eva, data_range=data_range)
            ncc = eva.NCC(**dict_eva)
            metrics_meth.append([psnr, ssim, ncc])
        metrics_sample.append(metrics_meth)
    metrics_dataset.append(metrics_sample)

# calculate the mean and std of each method
metrics_dataset = np.array(metrics_dataset)  # (N_dataset, N_sample, N_meth, N_metrics)
metrics_mean = metrics_dataset.mean(axis=1)  # (N_dataset, N_meth, N_metrics)
metrics_std = metrics_dataset.std(axis=1)  # (N_dataset, N_meth, N_metrics)

# add calculation of p-value
test_pairs = ((0, 4), (1, 4), (2, 4), (3, 4))

pvalues_data = []  # (N_dataset, N_metrics, N_pairs)
for i_dataset in range(num_datasets):
    pvalues_metrics = []
    for i_metric in range(num_metrics):
        pvalues = []
        for i_pair in range(len(test_pairs)):
            pair = test_pairs[i_pair]
            test_result = wilcoxon(
                metrics_dataset[i_dataset, :, pair[0], i_metric],
                metrics_dataset[i_dataset, :, pair[1], i_metric],
                alternative="two-sided",
            )
            pvalues.append(test_result[1])
        pvalues_metrics.append(pvalues)
    pvalues_data.append(pvalues_metrics)
pvalues_data = np.array(pvalues_data)  # (N_dataset, N_metrics, N_pairs)

print(f"[INFO] metrics shape: {metrics_dataset.shape}")
print(f"[INFO] metrics mean : {metrics_mean.shape}")
print(f"[INFO] metrics std  : {metrics_std.shape}")
print(f"[INFO] pvalues shape : {pvalues_data.shape}")


# ------------------------------------------------------------------------------
# show statistics analysis
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] Show statistics analysis...")
y_lim_metrics = ((20, 36), (0.4, 1), (0.75, 1))

method_colors = ["#8E99AB"]
for i_meth in range(num_methods):
    method_colors.append(methods_show_info[i_meth][3])

font_size = 14

# ------------------------------------------------------------------------------
nr, nc = 1, num_metrics
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(6 * nc, 3 * nr), **dict_fig)

width = 0.9 / (num_methods + 1)  # the width of the bars
width_bar = width * 0.85
x = np.arange(num_datasets)  # the label locations

axes[0].set_yticks([20, 25, 30, 35, 40])
axes[0].set_yticklabels([20, 25, 30, 35, 40], fontsize=font_size)
axes[1].set_yticks([0.4, 0.6, 0.8, 1.0])
axes[1].set_yticklabels([0.4, 0.6, 0.8, 1.0], fontsize=font_size)
axes[2].set_yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
axes[2].set_yticklabels([0.75, 0.8, 0.85, 0.9, 0.95, 1.0], fontsize=font_size)

for i_metric in range(num_metrics):
    ax = axes[i_metric]
    metrics_mean_value = metrics_mean[:, :, i_metric]
    metrics_std_value = metrics_std[:, :, i_metric]

    # --------------------------------------------------------------------------
    multiplier = 0
    for i_meth in range(num_methods + 1):
        offset = width * multiplier
        ax.bar(
            x=x + offset,
            height=metrics_mean_value[:, i_meth],
            width=width_bar,
            label=methods_names[i_meth],
            yerr=metrics_std_value[:, i_meth],
            color=method_colors[i_meth],
            capsize=2,
        )
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc. ------
    ax.set_ylabel(metrics_names[i_metric], fontsize=font_size)
    ax.set_xticks(x + width * 2, dataset_names, fontsize=font_size)
    ax.set_ylim(y_lim_metrics[i_metric])
    # del right and top axis
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # add p-value markers ------------------------------------------------------
    for i_dataset in range(num_datasets):
        pvalues_tmp = np.array(pvalues_data)[i_dataset, i_metric, :]
        for i_pair in range(len(test_pairs)):
            star_x = width * test_pairs[i_pair][0] + x[i_dataset]
            star_y = (metrics_mean_value + metrics_std_value)[
                i_dataset, test_pairs[i_pair][0]
            ].max()
            star_y += (y_lim_metrics[i_metric][1] - y_lim_metrics[i_metric][0]) * 0.02
            add_significant_star(ax=ax, x=star_x, y=star_y, p_value=pvalues_tmp[i_pair])


axes[0].legend(
    loc="upper left",
    ncols=2,
    facecolor="none",
    edgecolor="none",
    fontsize=font_size - 2,
)

plt.savefig(os.path.join(path_fig, f"image_restored_compare_{figure_name}_metrics.png"))
plt.savefig(os.path.join(path_fig, f"image_restored_compare_{figure_name}_metrics.svg"))
