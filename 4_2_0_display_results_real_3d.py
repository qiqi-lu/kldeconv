"""
Display the image restored by different methods. (2D / 3D)
"""

import matplotlib.pyplot as plt
import utils.evaluation as eva
import skimage.io as io
import numpy as np
import os, pandas, tqdm
import seaborn as sns
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
# subgroup, show_image, show_statistic = "2d_real", True, True
# subgroup, show_image, show_statistic = "2d_real", False, True
# subgroup, show_image, show_statistic = "3d_real", True, False
subgroup, show_image, show_statistic = "3d_real", True, True
# subgroup, show_image, show_statistic = "3d_real", False, True
# subgroup, show_image, show_statistic = "2d_real_many", False, True
# num_samples_max = 3
num_samples_max = 10
show_patch = True

# ------------------------------------------------------------------------------
settings = {
    "2d_real": {
        "datasets": (
            # ------------------------------------------------------------------
            # dataset_name | dataset_id | id sample | roi pos (y0,x0,y1,x1)
            # ------------------------------------------------------------------
            # ('F-actin',"F-actin-nonlinear-9", 5, (200, 100, 250, 150)),
            # ("CCP", "CCPs-9", 0, (200, 100, 250, 150)),
            # ("MT", "Microtubules2-9", 0, (100, 200, 200, 300)),
            # ("ER", "ER-6", 0, (100, 200, 200, 300)),
            # ("F-actin", "F-actin-9", 0, (100, 200, 200, 300)),
            # ("CCP (BioTISR)-1", "biotisr-ccps-1", 0, (200, 100, 250, 150)),
            # ("CCP (BioTISR)-2", "biotisr-ccps-2", 0, (200, 100, 250, 150)),
            ("CCP (BioTISR)-3", "biotisr-ccps-3", 0, (200, 100, 250, 150)),
            # ("F-actin (BioTISR)-1", "biotisr-factin-1", 0, (200, 100, 250, 150)),
            # ("F-actin (BioTISR)-2", "biotisr-factin-2", 0, (200, 100, 250, 150)),
            ("F-actin (BioTISR)-3", "biotisr-factin-3", 0, (200, 100, 250, 150)),
            # ("F-actin-nl (BioTISR)-1", "biotisr-factin-nonlinear-1", 0, (200, 100, 250, 150)),
            # ("F-actin-nl (BioTISR)-2", "biotisr-factin-nonlinear-2", 0, (200, 100, 250, 150)),
            (
                "F-actin-nl (BioTISR)-3",
                "biotisr-factin-nonlinear-3",
                0,
                (200, 100, 250, 150),
            ),
            # ("lysosomes (BioTISR)-1", "biotisr-lysosomes-1", 0, (200, 100, 250, 150)),
            # ("lysosomes (BioTISR)-2", "biotisr-lysosomes-2", 0, (200, 100, 250, 150)),
            ("lysosomes (BioTISR)-3", "biotisr-lysosomes-3", 0, (200, 100, 250, 150)),
            # ("Mito (BioTISR)-1", "biotisr-mito-1", 0, (200, 100, 250, 150)),
            # ("Mito (BioTISR)-2", "biotisr-mito-2", 0, (200, 100, 250, 150)),
            # ("Mito (BioTISR)-3", "biotisr-mito-3", 0, (200, 100, 250, 150)),
            # ("MT (BioTISR)-1", "biotisr-mt-1", 0, (200, 100, 250, 150)),
            # ("MT (BioTISR)-2", "biotisr-mt-2", 0, (200, 100, 250, 150)),
            # ("MT (BioTISR)-3", "biotisr-mt-3", 0, (200, 100, 250, 150)),
            # ("E.coli", "deepbacs-ecoli-ave2", 0, (200, 100, 250, 150)),
            # ("S.aureus", "deepbacs-saureus-ave2", 0, (200, 100, 250, 150)),
            # ("W2S-0", "w2s-0-sim-ave", 0, (200, 100, 250, 150)),
            # ("W2S-1", "w2s-1-sim-ave", 0, (200, 100, 250, 150)),
            # ("W2S-2", "w2s-2-sim-ave", 0, (200, 100, 250, 150)),
        ),
        "methods": (
            # ------------------------------------------------------------------
            # method name | method id | file name | iter_train | color
            # ------------------------------------------------------------------
            ("DeconvBlind", "deconvblind", "deconv.tif", 2, "#42B4B5"),
            ("DFCAN", "dfcan", "y_pred.tif", 2, "#57AA3E"),
            ("RLD@100", "traditional", "deconv_iter_100.tif", 2, "#4D8FCB"),
            ("KLD", "kernelnet", "y_pred_all.tif", 2, "#D95D5B"),
        ),
        "ndim": 2,
        "ticks_boxplot": {
            "PSNR": (np.round(np.linspace(0, 60, 25), decimals=1), (25, None)),
            "MS-SSIM": (np.round(np.linspace(0.0, 1.0, 21), decimals=2), (0.8, 1.0)),
            "ZNCC": (np.round(np.linspace(0.0, 1.0, 21), decimals=2), (0.7, 1.0)),
        },
    },
    "3d_real": {
        "datasets": (
            # ------------------------------------------------------------------
            # dataset name | dataset id | sample id | slice id | roi pos | xz plane (y,x0,x1)
            # (
            #     "MT",
            #     "Microtubule2-3d-1024",
            #     0,
            #     1,
            #     (122, 360, 322, 560),
            #     (512, 100, 200),
            # ),
            # (
            #     "NPC",
            #     "Nuclear-pore-complex2-1024",
            #     0,
            #     1,
            #     (200, 200, 300, 300),
            #     (512, 100, 200),
            # ),
            # (
            #     "MT (BioTISR)-1",
            #     "biotisr-3d-mt-1",
            #     0,
            #     1,
            #     (122, 360, 272, 510),
            #     (128, 100, 200),
            # ),
            # (
            #     "MT (BioTISR)-2",
            #     "biotisr-3d-mt-2",
            #     0,
            #     1,
            #     (122, 360, 272, 510),
            #     (128, 100, 200),
            # ),
            (
                "mito (BioTISR)-1",
                "biotisr-3d-mito-1",
                0,
                1,
                (122, 360, 272, 510),
                (128, 100, 200),
            ),
            (
                "mito (BioTISR)-2",
                "biotisr-3d-mito-2",
                0,
                1,
                (122, 360, 272, 510),
                (128, 100, 200),
            ),
            # (
            #     "F-actin (BioTISR)-1",
            #     "biotisr-3d-factin-1",
            #     0,
            #     1,
            #     (122, 360, 272, 510),
            #     (128, 100, 200),
            # ),
            # (
            #     "F-actin (BioTISR)-2",
            #     "biotisr-3d-factin-2",
            #     0,
            #     1,
            #     (122, 360, 272, 510),
            #     (128, 100, 200),
            # ),
        ),
        "methods": (
            ("DeconvBlind", "deconvblind", "deconv.tif", 2, "#42B4B5"),
            ("RLN", "rln", "y_pred.tif", 2, "#B78E72"),
            ("RLD@20", "traditional", "deconv_iter_20.tif", 2, "#4D8FCB"),
            # ("KLD", "kernelnet", "y_pred_all.tif", 2, "#D95D5B"),
            ("KLD", "kernelnet", "y_pred_all.tif", 5, "#D95D5B"),
        ),
        "ndim": 3,
        "ticks_boxplot": {
            "PSNR": (np.round(np.linspace(0, 60, 25), decimals=1), (12.5, None)),
            "MS-SSIM": (np.round(np.linspace(0.0, 1.0, 11), decimals=2), (0.45, 1.0)),
            "ZNCC": (np.round(np.linspace(0.0, 1.0, 11), decimals=2), (0.25, 1.0)),
        },
    },
}


# ------------------------------------------------------------------------------
datasets_info = settings[subgroup]["datasets"]
methods_info = settings[subgroup]["methods"]
ndim = settings[subgroup]["ndim"]

# ------------------------------------------------------------------------------
path_root_figure = os.path.join("outputs", "figures", "analysis_image")
path_root_prediction = os.path.join("outputs", "predictions")

info_df = pandas.read_excel("datasets_test.xlsx")

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=ndim)
dict_clip = {"a_min": 0, "a_max": 2.5}
data_range = dict_clip["a_max"] - dict_clip["a_min"]
dict_fig = dict(constrained_layout=True, dpi=300)

# ------------------------------------------------------------------------------
num_datasets = len(datasets_info)
num_methods = len(methods_info)

methods_names = ["RAW"]
for i_meth in range(len(methods_info)):
    methods_names.append(methods_info[i_meth][0])
methods_names.append("GT")

metrics_names = ["PSNR", "MS-SSIM", "ZNCC"]
num_metrics = len(metrics_names)

dataset_names = []
for i_dataset in range(num_datasets):
    dataset_names.append(datasets_info[i_dataset][0])

print("-" * 80)
print(f"[INFO] methods titles: {methods_names}")

# ------------------------------------------------------------------------------
# load one sample from each dataset and method
# ------------------------------------------------------------------------------
results_one = []
for i_dataset in range(num_datasets):
    # load results
    _, dataset_id, id_sample_show, _, _, _ = datasets_info[i_dataset]
    path_result = os.path.join(path_root_prediction, dataset_id)

    print("-" * 80)
    print("[INFO] Load results from :", path_result)

    info = info_df[info_df["id"] == dataset_id].iloc[0]
    path_raw, path_gt = win2linux(info["path_lr"]), win2linux(info["path_hr"])
    filenames = read_txt(win2linux(info["path_txt"]))
    # --------------------------------------------------------------------------
    results_meth = []
    # load raw and gt images
    x = io.imread(os.path.join(path_raw, filenames[id_sample_show]))
    y = io.imread(os.path.join(path_gt, filenames[id_sample_show]))
    results_meth.append(x.astype(np.float32))

    filename_wo_ext = filenames[id_sample_show].split(".")[0]
    for i_meth in range(num_methods):
        meth_name, meth_id, meth_filename, num_iter_train = methods_info[i_meth][:4]

        # load restoed image from KLDeconv method --------------------------
        if meth_name == "KLD":
            path_tmp = os.path.join(
                path_result,
                meth_id,
                dataset_id,
                "fp_n1_r1_bp_n1_r1",
                f"train_iter_{num_iter_train}",
                filename_wo_ext,
            )
            y_pred = io.imread(os.path.join(path_tmp, "y_pred_all.tif"))

            # the imread funciton will automaticly reshape the results
            # when having 3 channels.
            if (ndim == 2) and (y_pred.shape[-1] in [3, 4]):
                y_pred = np.transpose(y_pred, axes=(-1, 0, 1))
            y_pred = y_pred[-1]
        elif meth_name in ["DFCAN", "RLN"]:
            path_tmp = os.path.join(
                path_result,
                meth_id,
                dataset_id,
                "n1_r1",
                filename_wo_ext,
                meth_filename,
            )
            y_pred = io.imread(path_tmp)
        else:
            path_tmp = os.path.join(
                path_result, meth_id, filename_wo_ext, meth_filename
            )
            y_pred = io.imread(path_tmp)

        results_meth.append(y_pred.astype(np.float32))
    results_meth.append(y.astype(np.float32))
    results_one.append(results_meth)

# ------------------------------------------------------------------------------
# show one sample of each dataset
# ------------------------------------------------------------------------------
if show_image:
    print("[INFO] Show image ...")
    nr, nc = num_datasets, num_methods + 2
    nr = nr * 2 if ndim == 3 else nr

    dict_colorize = dict(vmin=0.0, vmax=0.9, color=(0, 255, 0))
    dict_text_lb = dict(
        fontsize=14, color="white", ha="left", va="bottom", x=0.05, y=0.05
    )
    dict_text_rt = dict(
        fontsize=14, color="white", ha="right", va="top", x=0.95, y=0.95
    )
    dict_text_lt = dict(fontsize=14, color="white", ha="left", va="top", x=0.05, y=0.95)
    dict_img_res = dict(cmap="hot", vmin=0.0, vmax=0.5)

    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    [ax.set_axis_off() for ax in axes.ravel()]

    fig_res, axes_res = plt.subplots(
        nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig
    )
    [ax.set_axis_off() for ax in axes_res.ravel()]

    for i_dataset in range(num_datasets):
        if ndim == 2:
            axes_ds = axes[i_dataset]
            axes_ds_res = axes_res[i_dataset]
            dataset_name, dataset_id, id_sample, pos_roi = datasets_info[i_dataset]
        if ndim == 3:
            axes_ds = axes[i_dataset * 2 : i_dataset * 2 + 2]
            axes_ds_res = axes_res[i_dataset * 2 : i_dataset * 2 + 2]
            (
                dataset_name,
                dataset_id,
                id_sample,
                id_slice_xy_ori,
                pos_roi,
                pos_plane_xz,
            ) = datasets_info[i_dataset]

        info = info_df[info_df["id"] == dataset_id].iloc[0]
        pixel_size = float(info["pixel_size"]) / 1000  # pixel size (um)

        if ndim == 3:
            id_slice_zx, x_start, x_stop = pos_plane_xz
            slice_space = float(info["slice_space"]) / 1000  # slice spacing (um)
            # recalculate the slice index
            id_slice_xy = round((id_slice_xy_ori + 1) * slice_space / pixel_size) - 1

        # ----------------------------------------------------------------------
        results = results_one[i_dataset]

        # gt image
        img_gt = results[-1]
        img_gt = np.clip(normalizer(img_gt), **dict_clip)

        # restored image from different methods
        for i_meth in range(len(results)):
            img = results[i_meth]
            img = np.clip(normalizer(img), **dict_clip)
            img_res = np.abs(img - img_gt)
            mse = np.mean(np.square(img_gt - img))

            # colorize image ---------------------------------------------------
            if ndim == 2:
                img_color = colorize(img, **dict_colorize)
            if ndim == 3:
                # interpolate the image to have a isotropic voxel size
                img_interp = interp_iso_z(img, ps_xy=pixel_size, ps_z=slice_space)
                img_color = colorize(img_interp, **dict_colorize)

            # show image -------------------------------------------------------
            if ndim == 2:
                axes_ds[i_meth].imshow(img_color)

                axes_ds_res[i_meth].imshow(img_res, **dict_img_res)
                axes_ds_res[i_meth].text(
                    s=f"{mse:.4f}",
                    transform=axes_ds_res[i_meth].transAxes,
                    **dict_text_rt,
                )

                img_shape = img.shape
            if ndim == 3:
                img_shape = img[0].shape
                axes_ds[0, i_meth].imshow(img_color[id_slice_xy, :, :])
                axes_ds[1, i_meth].imshow(img_color[:, id_slice_zx, x_start:x_stop])

                axes_ds_res[0, i_meth].imshow(
                    img_res[id_slice_xy_ori, :, :], **dict_img_res
                )
                axes_ds_res[1, i_meth].imshow(
                    img_res[:, id_slice_zx, x_start:x_stop], **dict_img_res
                )
                axes_ds_res[0, i_meth].text(
                    s=f"{mse:.4f}",
                    transform=axes_ds_res[0, i_meth].transAxes,
                    **dict_text_rt,
                )

            # set which ax to show info ----------------------------------------
            if ndim == 2:
                ax_t = axes_ds[i_meth]
            elif ndim == 3:
                ax_t = axes_ds[0, i_meth]
                ax_t.plot(
                    [x_start, x_stop],
                    [id_slice_zx, id_slice_zx],
                    "-",
                    linewidth=1,
                    color="magenta",
                )

            # add scale bar ----------------------------------------------------
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
                    add_scale_bar(ax_t, image=img, **dict_scale_bar)
                elif ndim == 3:
                    add_scale_bar(ax_t, image=img[0], **dict_scale_bar)

            # add metrics value ------------------------------------------------
            if i_meth != len(results) - 1:
                dict_eva = dict(img_true=img_gt, img_test=img)
                psnr = eva.PSNR(data_range=data_range, **dict_eva)
                ssim = eva.MSSSIM(data_range=data_range, **dict_eva)
                ax_t.text(
                    s=f"{psnr:.2f} | {ssim:.4f}",
                    transform=ax_t.transAxes,
                    **dict_text_lb,
                )

            # add zoom patch -------------------------------------------------------
            if show_patch:
                y0, x0, y1, x1 = pos_roi
                if ndim == 2:
                    patch = img_color[y0:y1, x0:x1]
                if ndim == 3:
                    patch = img_color[id_slice_xy, y0:y1, x0:x1]

                # add box in the image
                ax_t.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        linewidth=1,
                        edgecolor="magenta",
                        facecolor="none",
                    )
                )
                ax_patch = ax_t.inset_axes(
                    [0.6, 0.0, 0.4, 0.4], transform=ax_t.transAxes, zorder=10
                )

                ax_patch.imshow(patch)

                # remove all the ticks and tick labels
                ax_patch.set_xticks([])
                ax_patch.set_yticks([])
                ax_patch.set_xticklabels([])
                ax_patch.set_yticklabels([])

                ax_patch.spines["top"].set_color("white")
                ax_patch.spines["left"].set_color("white")
                ax_patch.spines["top"].set_linewidth(1)
                ax_patch.spines["left"].set_linewidth(1)
                ax_patch.spines["right"].set_visible(False)
                ax_patch.spines["bottom"].set_visible(False)

            # add title ------------------------------------------------------------

            if i_dataset == 0:
                ax_t.text(
                    s=methods_names[i_meth], transform=ax_t.transAxes, **dict_text_rt
                )
            if i_meth == 0:
                ax_t.text(s=dataset_name, transform=ax_t.transAxes, **dict_text_lt)

    fig.savefig(
        os.path.join(path_root_figure, f"image_restored_compare_{subgroup}.png")
    )
    fig.savefig(
        os.path.join(path_root_figure, f"image_restored_compare_{subgroup}.svg")
    )
    fig_res.savefig(
        os.path.join(path_root_figure, f"image_restored_compare_{subgroup}_res.png")
    )
    # fig_res.savefig(
    #     os.path.join(path_root_figure, f"image_restored_compare_{subgroup}.svg")
    # )

os._exit(0)

# ------------------------------------------------------------------------------
# load all the results from different methods and datasets
# ------------------------------------------------------------------------------
results_all = []
for i_dataset in range(num_datasets):
    # load results
    dataset_id = datasets_info[i_dataset][1]
    path_result = os.path.join(path_root_prediction, dataset_id)

    print("-" * 80)
    print("[INFO] Load results from :", path_result)

    info = info_df[info_df["id"] == dataset_id].iloc[0]
    path_raw, path_gt = win2linux(info["path_lr"]), win2linux(info["path_hr"])
    filenames = read_txt(win2linux(info["path_txt"]))

    num_samples = min(num_samples_max, len(filenames))
    # --------------------------------------------------------------------------
    results = []
    pbar = tqdm.tqdm(total=num_samples, desc="[INFO] Load results", ncols=80)
    for i_sample in range(num_samples):
        pbar.update(1)
        results_ss = []
        # load raw and gt images
        x = io.imread(os.path.join(path_raw, filenames[i_sample]))
        y = io.imread(os.path.join(path_gt, filenames[i_sample]))
        results_ss.append(x.astype(np.float32))

        filename_wo_ext = filenames[i_sample].split(".")[0]
        for i_meth in range(num_methods):
            meth_name, meth_id, meth_filename, num_iter_train = methods_info[i_meth][:4]

            # load restoed image from KLDeconv method --------------------------
            if meth_name == "KLD":
                path_tmp = os.path.join(
                    path_result,
                    meth_id,
                    dataset_id,
                    "fp_n1_r1_bp_n1_r1",
                    f"train_iter_{num_iter_train}",
                    filename_wo_ext,
                )
                y_pred = io.imread(os.path.join(path_tmp, "y_pred_all.tif"))

                # the imread funciton will automaticly reshape the results
                # when having 3 channels.
                if (ndim == 2) and (y_pred.shape[-1] in [3, 4]):
                    y_pred = np.transpose(y_pred, axes=(-1, 0, 1))
                y_pred = y_pred[-1]
            elif meth_name in ["DFCAN", "RLN"]:
                path_tmp = os.path.join(
                    path_result,
                    meth_id,
                    dataset_id,
                    "n1_r1",
                    filename_wo_ext,
                    meth_filename,
                )
                y_pred = io.imread(path_tmp)
            else:
                path_tmp = os.path.join(
                    path_result, meth_id, filename_wo_ext, meth_filename
                )
                y_pred = io.imread(path_tmp)

            results_ss.append(y_pred.astype(np.float32))
        results_ss.append(y.astype(np.float32))
        results.append(results_ss)
    pbar.close()
    results_all.append(results)


print("-" * 80)
print(f"[INFO] Num of datasets: {len(results_all)}")
for res in results_all:
    print(f"[INFO] num of samples: {len(res)}, shape of image: {res[0][0].shape}")
print("-" * 80)

# ------------------------------------------------------------------------------
# statistics analysis
# ------------------------------------------------------------------------------
if show_statistic:
    print("-" * 80)
    print("[INFO] Statistics analysis ...")
    # --------------------------------------------------------------------------
    # calculate the metrics value of each method
    metrics_dataset = []
    pbar_ana = tqdm.tqdm(total=num_datasets, desc="[INFO] Analysis", ncols=80)
    for i_dataset in range(num_datasets):
        pbar_ana.update(1)
        metrics_sample = []
        res_samples = results_all[i_dataset]
        for results in res_samples:
            img_gt = results[-1]
            img_gt = np.clip(normalizer(img_gt), **dict_clip)

            metrics_meth = []
            for i_meth in range(num_methods + 1):
                img = results[i_meth]
                img = np.clip(normalizer(img), **dict_clip)

                dict_eva = {"img_true": img_gt, "img_test": img}
                psnr = eva.PSNR(**dict_eva, data_range=data_range)
                ssim = eva.MSSSIM(**dict_eva, data_range=data_range, ndim=2)
                zncc = eva.NCC(**dict_eva)
                metrics_meth.append([psnr, ssim, zncc])
            metrics_sample.append(metrics_meth)
        metrics_dataset.append(metrics_sample)
    pbar_ana.close()
    # (N_dataset, N_sample, N_meth, N_metrics)
    # metrics_dataset = np.array(metrics_dataset)

    # --------------------------------------------------------------------------
    # calculate p-value
    test_pairs = ((0, 4), (1, 4), (2, 4), (3, 4))

    pvalues_dataset = []  # (N_dataset, N_metrics, N_pairs)
    for i_dataset in range(num_datasets):
        pvalues_metrics = []
        met = metrics_dataset[i_dataset]
        met = np.array(met)  # (N_sample, N_meth, N_metrics)
        for i_metric in range(num_metrics):
            pvalues = []
            for i_pair in range(len(test_pairs)):
                pair = test_pairs[i_pair]
                test_result = wilcoxon(
                    met[:, pair[0], i_metric],
                    met[:, pair[1], i_metric],
                    alternative="two-sided",
                )
                pvalues.append(test_result[1])
            pvalues_metrics.append(pvalues)
        pvalues_dataset.append(pvalues_metrics)
    pvalues_dataset = np.array(pvalues_dataset)  # (N_dataset, N_metrics, N_pairs)

    print(f"[INFO] pvalues shape : {pvalues_dataset.shape}")
    # --------------------------------------------------------------------------
    # transform the metric matrix into dataframe for seaborn
    df_metrics = pandas.DataFrame(
        columns=("dataset", "method", "metric", "id_sample", "value")
    )
    for i_dataset in range(num_datasets):
        for i_meth in range(len(metrics_dataset[i_dataset][0])):
            for i_metric in range(num_metrics):
                for i_sample in range(len(metrics_dataset[i_dataset])):
                    value = metrics_dataset[i_dataset][i_sample][i_meth][i_metric]
                    df_metrics.loc[len(df_metrics)] = [
                        dataset_names[i_dataset],
                        methods_names[i_meth],
                        metrics_names[i_metric],
                        i_sample,
                        value,
                    ]
    # transform the pvalue matrix into dataframe
    df_pvalue = pandas.DataFrame(columns=("dataset", "metric", "pair", "pvalue"))
    for i_dataset in range(num_datasets):
        for i_metric in range(num_metrics):
            for i_pair in range(len(test_pairs)):
                pair = test_pairs[i_pair]
                pvalue = pvalues_dataset[i_dataset, i_metric, i_pair]
                df_pvalue.loc[len(df_pvalue)] = [
                    dataset_names[i_dataset],
                    metrics_names[i_metric],
                    f"({methods_names[pair[0]]} vs. {methods_names[pair[1]]})",
                    pvalue,
                ]

    # --------------------------------------------------------------------------
    # show statistics analysis
    # --------------------------------------------------------------------------
    print("-" * 80)
    print("[INFO] Show statistics analysis...")
    # --------------------------------------------------------------------------
    methods_color = ["#8E99AB"]
    for i_meth in range(num_methods):
        methods_color.append(methods_info[i_meth][4])

    font_size = 10
    dict_ticks = settings[subgroup]["ticks_boxplot"]
    # --------------------------------------------------------------------------
    nr, nc = num_metrics, 1
    # nr, nc = 1, num_metrics
    fac = num_datasets / 2.0
    fig, axes = plt.subplots(
        nrows=nr, ncols=nc, figsize=(3 * nc * fac, 3 * nr), **dict_fig
    )

    for i_metric in range(num_metrics):
        ax = axes[i_metric]
        metric_name = metrics_names[i_metric]

        ax.set_yticks(dict_ticks[metric_name][0])
        ax.set_yticklabels(dict_ticks[metric_name][0], fontsize=font_size)

        # grouped boxplot
        df = df_metrics[df_metrics["metric"] == metric_name]
        sns.boxplot(
            x="dataset",
            y="value",
            hue="method",
            data=df,
            ax=ax,
            palette=methods_color,
            gap=0.2,
            fliersize=0.5,
            linecolor="black",
        )

        # add vline
        for i_dataset in range(num_datasets - 1):
            ax.axvline(x=i_dataset + 0.5, color="black", linestyle="--", linewidth=0.5)
        # disable the legend
        if i_metric != num_metrics - 1:
            ax.legend().set_visible(False)
        ax.set_ylabel(metric_name, fontsize=font_size)
        ax.set_xlabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(dict_ticks[metric_name][1])
        ax.tick_params(axis="both", which="major", labelsize=font_size)

        # # add p-value markers --------------------------------------------------
        for i_dataset in range(num_datasets):
            pvalues_tmp = np.array(pvalues_dataset)[i_dataset, i_metric, :]
            for i_pair in range(len(test_pairs)):
                i_meth = test_pairs[i_pair][0]

                boxes = ax.artists
                box_positions = [box.get_x() + box.get_width() / 2 for box in boxes]

                step = 1.0 / (num_methods + 2)
                mid = num_methods / 2
                star_x = (i_meth - mid) * step + i_dataset

                # get the y limit range of the boxplot
                ylim = ax.get_ylim()
                yrange = ylim[1] - ylim[0]
                star_y = ylim[0] + yrange * 0.97
                add_significant_star(
                    ax=ax, x=star_x, y=star_y, p_value=pvalues_tmp[i_pair]
                )

    plt.savefig(
        os.path.join(path_root_figure, f"image_restored_compare_{subgroup}_metrics.png")
    )
    plt.savefig(
        os.path.join(path_root_figure, f"image_restored_compare_{subgroup}_metrics.svg")
    )
    # save source data into excel file -----------------------------------------
    writer = pandas.ExcelWriter(
        os.path.join(
            path_root_figure, f"image_restored_compare_{subgroup}_metrics.xlsx"
        ),
        engine="xlsxwriter",
    )
    df_metrics.to_excel(writer, index=False, sheet_name="metrics")
    df_pvalue.to_excel(writer, index=False, sheet_name="pvalue")
    writer.close()
