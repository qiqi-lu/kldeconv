"""
Display the image restored by different methods. (2D / 3D)
"""

import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
from skimage.measure import profile_line
import numpy as np
import os, pandas, tqdm
from utils import evaluation as eva
from utils.data import NormalizePercentile, win2linux, read_txt
from utils.plot import (
    colorize,
    add_scale_bar,
    add_patch,
    add_significant_star,
    interp_iso_z,
)
from scipy.stats import wilcoxon

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
# figure_name = "2d_real"
figure_name = "3d_real"
num_samples = 20

# ------------------------------------------------------------------------------
# dataset name, id sample, pos, size
dataset_show_info_list = {
    "2d_real": (
        # ("F-actin-nonlinear-9", 5, (200, 200), 100, (255, 0, 0)),
        ("CCPs-9", 0, (200, 200), 100, (0, 255, 0)),
        ("Microtubules2-9", 0, (200, 200), 100, (3, 174, 210)),
        ("ER-6", 0, (200, 200), 100, (255, 0, 255)),
        ("F-actin-9", 0, (200, 200), 100, (255, 165, 0)),
    ),
    "3d_real": (
        ("Microtubule2-3d-1024", 0, (1, 512, 100, 200), (200, 200), 100, (3, 174, 210)),
        (
            "Nuclear-pore-complex2-1024",
            0,
            (1, 512, 100, 200),
            (200, 200),
            100,
            (0, 255, 0),
        ),
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
        ("RLD@20", "traditional", "deconv_iter_20.tif", "#4D8FCB"),
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
    pbar = tqdm.tqdm(total=num_samples, desc="Load results", ncols=80)
    for i_sample in range(num_samples):
        pbar.update(1)
        results_ss = []
        # load raw and gt images
        path_raw, path_gt = win2linux(info["path_lr"]), win2linux(info["path_hr"])
        filenames = read_txt(win2linux(info["path_txt"]))
        x = io.imread(os.path.join(path_raw, filenames[i_sample]))
        y = io.imread(os.path.join(path_gt, filenames[i_sample]))
        results_ss.append(x.astype(np.float32))

        filename_wo_ext = filenames[i_sample].split(".")[0]
        for i_meth in range(num_methods):
            meth_title, meth_tag, meth_file = methods_show_info[i_meth][:3]

            # load restoed image from KLDeconv method --------------------------
            if meth_title == "KLD":
                path_tmp = os.path.join(
                    path_result,
                    meth_tag,
                    dataset_name_test,
                    "fp_n1_r1_bp_n1_r1",
                    filename_wo_ext,
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
                    filename_wo_ext,
                    meth_file,
                )
                y_pred = io.imread(path_tmp)
            else:
                path_tmp = os.path.join(
                    path_result, meth_tag, filename_wo_ext, meth_file
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
#                               show results
# ------------------------------------------------------------------------------
print("[INFO] Show results...")
nr, nc = num_datasets, num_methods + 2

if ndim == 3:
    nr = nr * 2

fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

for i_dataset in range(num_datasets):
    if ndim == 2:
        axes_ds = axes[i_dataset]
        dataset_name_test, id_sample, pos, size, color_img = dataset_show_info[
            i_dataset
        ]
    if ndim == 3:
        axes_ds = axes[i_dataset * 2 : i_dataset * 2 + 2]
        dataset_name_test, id_sample, id_slice, pos, size, color_img = (
            dataset_show_info[i_dataset]
        )

    info = info_df[info_df["id"] == dataset_name_test].iloc[0]
    pixel_size = float(info["pixel_size"]) / 1000  # pixel size (um)

    if ndim == 3:
        id_slice_xy, id_slice_zx, x_start, x_stop = (
            id_slice  # slice index to show in xy and xz plane
        )
        slice_space = float(info["slice_space"]) / 1000  # slice spacing (um)
        # recalculate the slice index
        id_slice_xy = round((id_slice_xy + 1) * slice_space / pixel_size) - 1

    results = results_all[i_dataset][id_sample]

    # get gt image -------------------------------------------------------------
    img_gt = np.clip(normalizer(results[-1]), **dict_clip)

    # show restored image from different methods -------------------------------
    for i_meth in range(len(results)):
        img = np.clip(normalizer(results[i_meth]), **dict_clip)
        if ndim == 2:
            img_color = colorize(img, vmin=0.0, vmax=0.9, color=color_img)
        elif ndim == 3:
            # interpolate the image to have a isotropic voxel size
            img_interp = interp_iso_z(
                img, ps_xy=pixel_size, ps_z=slice_space
            )  # (D, H, W)
            # plot color image
            img_color = colorize(img_interp, vmin=0.0, vmax=0.9, color=color_img)
        else:
            raise ValueError(f"[ERROR] Cannot find the dimension of the figure.")

        if ndim == 2:
            axes_ds[i_meth].imshow(img_color)
            img_shape = img.shape
        elif ndim == 3:
            img_shape = img[0].shape
            # show image in xy plane
            axes_ds[0, i_meth].imshow(img_color[id_slice_xy, :, :])
            # show image in xz plane
            axes_ds[1, i_meth].imshow(img_color[:, id_slice_zx, x_start:x_stop])

        # set which ax to show info
        if ndim == 2:
            axes_tag = axes_ds[i_meth]
        elif ndim == 3:
            axes_tag = axes_ds[0, i_meth]
            axes_tag.plot(
                [x_start, x_stop],
                [id_slice_zx, id_slice_zx],
                "-",
                linewidth=1,
                color="white",
            )

        # add scale bar --------------------------------------------------------
        if i_meth == len(results) - 1:
            tp = 0.05
            dict_scale_bar = {
                "pixel_size": pixel_size,
                "bar_length": 5,  # um
                "bar_height": 0.01,
                "bar_color": "white",
                "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
            }
            if ndim == 2:
                add_scale_bar(axes_tag, image=img, **dict_scale_bar)
            elif ndim == 3:
                add_scale_bar(axes_tag, image=img[0], **dict_scale_bar)

                # image_crop = img[:, id_slice_zx, x_start:x_stop]
                # dict_scale_bar["pos"] = (
                #     int(image_crop.shape[1] * tp),
                #     int(image_crop.shape[0] * (1 - tp)),
                # )
                # add_scale_bar(axes_ds[1, i_meth], image=image_crop, **dict_scale_bar)

        # add metrics value ----------------------------------------------------
        if ndim == 2:
            pos_text = (
                img.shape[0] - int(img.shape[0] * 0.04),
                int(img.shape[1] * 0.04),
            )
        elif ndim == 3:
            pos_text = (
                img[0].shape[0] - int(img[0].shape[0] * 0.04),
                int(img[0].shape[1] * 0.04),
            )

        dict_text_metric = {
            "fontsize": 14,
            "color": "white",
            "ha": "left",
            "va": "bottom",
        }
        if i_meth != len(results) - 1:
            dict_eva = {"img_true": img_gt, "img_test": img}
            psnr = eva.PSNR(**dict_eva, data_range=data_range)
            # ssim = eva.SSIM(**dict_eva, data_range=data_range)
            ssim = eva.SSIM_tb(
                img_true=img_gt[None, None],
                img_test=img[None, None],
                data_range=data_range,
            )
            axes_tag.text(
                pos_text[1],
                pos_text[0],
                f"{psnr:.2f} | {ssim*100:.2f}",
                **dict_text_metric,
            )

        # add zoom patch -------------------------------------------------------
        if show_patch:
            show_box = True if i_meth == len(results) - 1 else False

            if ndim == 2:
                img_tmp = img_color
            elif ndim == 3:
                img_tmp = img_color[id_slice_xy]

            add_patch(
                ax=axes_tag,
                image=img_tmp,
                pos=pos,
                size=size,
                show_box=show_box,
                axes_lw=1,
                box_lw=0.5,
                box_color="white",
            )

        # add title ------------------------------------------------------------
        dict_text_meth = {"fontsize": 14, "color": "white", "ha": "right", "va": "top"}

        if ndim == 2:
            pos_text = (
                int(img.shape[0] * 0.04),
                img.shape[1] - int(img.shape[1] * 0.04),
            )
        elif ndim == 3:
            pos_text = (
                int(img[0].shape[0] * 0.04),
                img.shape[1] - int(img[0].shape[1] * 0.04),
            )

        if i_dataset == 0:
            axes_tag.text(
                pos_text[1], pos_text[0], methods_names[i_meth], **dict_text_meth
            )

        # add title (structure) ------------------------------------------------
        dict_text_struc = {"fontsize": 14, "color": "white", "ha": "left", "va": "top"}

        if ndim == 2:
            pos_text = (int(img.shape[0] * 0.04), int(img.shape[1] * 0.04))
        elif ndim == 3:
            pos_text = (int(img[0].shape[0] * 0.04), int(img[0].shape[1] * 0.04))

        if i_meth == 0:
            axes_tag.text(
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
            # ssim = eva.SSIM(**dict_eva, data_range=data_range)
            ssim = eva.SSIM_tb(
                img_true=img_gt[None, None],
                img_test=img[None, None],
                data_range=data_range,
            )
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
if figure_name == "2d_real":
    y_lim_metrics = ((20, 36), (0.4, 1), (0.75, 1))
elif figure_name == "3d_real":
    y_lim_metrics = ((20, 30), (0.3, 0.65), (0.6, 0.85))

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

ticks = np.round(np.linspace(0, 60, 13), decimals=1)
axes[0].set_yticks(ticks)
axes[0].set_yticklabels(ticks, fontsize=font_size)
ticks = np.round(np.linspace(0.0, 1.0, 11), decimals=2)
axes[1].set_yticks(ticks)
axes[1].set_yticklabels(ticks, fontsize=font_size)
ticks = np.round(np.linspace(0.0, 1.0, 21), decimals=2)
axes[2].set_yticks(ticks)
axes[2].set_yticklabels(ticks, fontsize=font_size)

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
