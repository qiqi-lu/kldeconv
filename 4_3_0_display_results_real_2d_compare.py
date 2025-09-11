"""
Display the image restored by different methods.
"""

import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
from skimage.measure import profile_line
import numpy as np
import os, pandas
from utils import evaluation as eva
import matplotlib.patches as patches
import matplotlib.colors as colors
from utils.data import NormalizePercentile, win2linux, read_txt
from utils.plot import colorize, add_scale_bar, add_patch

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
figure_name = "2d_real"
# figure_name = '3d_real'
num_samples = 6

# ------------------------------------------------------------------------------
# dataset name, id sample, pos, size
dataset_show_info_list = {
    "2d_real": (
        ("F-actin-nonlinear-9", 5, (200, 200), 100, (255, 0, 0)),
        ("Microtubules2-9", 0, (200, 200), 100, (3, 174, 210)),
    ),
    "3d_real": (
        ("F-actin-nonlinear-1", 5, (200, 200), 100, (255, 0, 0)),
        ("Microtubules2-1", 0, (200, 200), 100, (3, 174, 210)),
    ),
}

methods_show_info_list = {
    "2d_real": (
        ("DeconvBlind", "deconvblind", "deconv.tif", (0, 255, 0)),
        ("RLD@100", "traditional", "deconv_iter_100.tif", (0, 0, 255)),
        ("KLD", "kernelnet", "y_pred_all.tif", (255, 0, 0)),
    ),
    "3d_real": (
        ("DeconvBlind", "deconvblind", "deconv.tif", (0, 255, 0)),
        ("RLD@100", "traditional", "deconv_iter_100.tif", (0, 0, 255)),
        ("KLD", "kernelnet", "y_pred_all.tif", (255, 0, 0)),
    ),
}

# ------------------------------------------------------------------------------
dataset_show_info = dataset_show_info_list[figure_name]
methods_show_info = methods_show_info_list[figure_name]

show_patch = True
path_fig = os.path.join("outputs", "figures")
info_df = pandas.read_excel("datasets_test.xlsx")
num_datasets = len(dataset_show_info)
num_methods = len(methods_show_info)
normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=2)
dict_clip = {"a_min": 0, "a_max": 2.5}
data_range = dict_clip["a_max"] - dict_clip["a_min"]

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
    for i_sample in range(num_samples):
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
                if "2d" in figure_name:
                    if y_pred.shape[-1] in [3, 4]:
                        y_pred = np.transpose(y_pred, axes=(-1, 0, 1))
                y_pred = y_pred[2]
            else:
                path_tmp = os.path.join(
                    path_result, meth_tag, f"sample_{i_sample}", meth_file
                )
                y_pred = io.imread(path_tmp)

            results_ss.append(y_pred.astype(np.float32))
        results_ss.append(y.astype(np.float32))
        results.append(results_ss)
    results_all.append(results)

print("-" * 80)
print(f"[INFO] Num of datasets: {len(results_all)}")
print(f"[INFO] Num of samples : {len(results_all[0])}")
print(f"[INFO] Num of methods : {len(results_all[0][0])}")
print(f"[INFO] Shape of image : {results_all[0][0][0].shape}")
print("-" * 80)

# ------------------------------------------------------------------------------
# show results
# ------------------------------------------------------------------------------
nr, nc = num_datasets, num_methods + 2
dict_fig = dict(constrained_layout=True, dpi=300)
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

titles = ["RAW"]
for i_meth in range(len(methods_show_info)):
    titles.append(methods_show_info[i_meth][0])
titles.append("GT")

for i_dataset in range(num_datasets):
    axes_ds = axes[i_dataset]
    dataset_name_test, id_sample, pos, size, color_img = dataset_show_info[i_dataset]
    pixel_size = float(info["pixel_size"])
    results = results_all[i_dataset][id_sample]

    # --------------------------------------------------------------------------
    N_meth = len(results)  # number of methods (include raw and gt)
    vmax_img = results[-1].max() * 0.6

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
                titles[i_meth],
                **dict_text_meth,
            )
        dict_text_struc = {"fontsize": 14, "color": "white", "ha": "left", "va": "top"}
        if i_meth == 0:
            axes_ds[i_meth].text(
                pos_text[1], pos_text[0], dataset_name_test, **dict_text_struc
            )

    plt.savefig(os.path.join(path_fig, f"image_restored_compare_{figure_name}.png"))
    plt.savefig(os.path.join(path_fig, f"image_restored_compare_{figure_name}.svg"))

    # # ------------------------------------------------------------------------
    # # profile line
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 3), **dict_fig)

    # for i in range(N_meth):
    #     start, end = (35, 0), (35, 100)
    #     profile = profile_line(results[i], start, end, linewidth=2)
    #     axes.plot(profile, label=methods_name[i])

    # plt.legend()
    # plt.savefig(os.path.join(path_fig, "image_restored_profile.png"))
    # plt.savefig(os.path.join(path_fig, "image_restored_profile.svg"))

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
        metrics_meth = []
        for i_meth in range(N_meth):
            img = results[i_meth]
            dict_eva = {"img_true": img_gt, "img_test": img}
            psnr = eva.PSNR(**dict_eva, data_range=data_range)
            ssim = eva.SSIM(**dict_eva, data_range=data_range)
            metrics_meth.append((psnr, ssim))
        metrics_sample.append(metrics_meth)
    metrics_dataset.append(metrics_sample)

# calculate the mean and std of each method
metrics_dataset = np.array(metrics_dataset)
metrics_mean = metrics_dataset.mean(axis=1)
metrics_std = metrics_dataset.std(axis=1)

print(f"[INFO] metrics shape: {metrics_dataset.shape}")
print(f"[INFO] metrics mean : {metrics_mean}")
print(f"[INFO] metrics std  : {metrics_std}")


# ------------------------------------------------------------------------------
# show statistics analysis
# ------------------------------------------------------------------------------
