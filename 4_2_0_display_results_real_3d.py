"""
Display the image restored by different methods.
3D real images.
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
show_image, show_statistic = True, True
# num_samples_max = 3
num_samples_max = 10
show_patch = True

# ------------------------------------------------------------------------------
settings = {
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
        (
            "MT (BioTISR)-2",
            "biotisr-3d-mt-2",
            2,
            1,
            (122, 360, 272, 510),
            (200, 100, 200),
        ),
        # (
        #     "mito (BioTISR)-1",
        #     "biotisr-3d-mito-1",
        #     0,
        #     1,
        #     (122, 360, 272, 510),
        #     (120, 200, 300),
        # ),
        (
            "mito (BioTISR)-2",
            "biotisr-3d-mito-2",
            0,
            1,
            (122, 360, 272, 510),
            (120, 200, 300),
        ),
        # (
        #     "F-actin (BioTISR)-1",
        #     "biotisr-3d-factin-1",
        #     0,
        #     1,
        #     (122, 360, 272, 510),
        #     (128, 100, 200),
        # ),
        (
            "F-actin (BioTISR)-2",
            "biotisr-3d-factin-2",
            0,
            1,
            (122, 360, 272, 510),
            (128, 100, 200),
        ),
    ),
    "methods": (
        ("DeconvBlind", "deconvblind", "deconv.tif", 2, "#42B4B5"),
        ("RLN", "rln", "y_pred.tif", 2, "#B78E72"),
        ("RLD@20", "traditional", "deconv_iter_20.tif", 2, "#4D8FCB"),
        # ("KLD", "kernelnet", "y_pred_all.tif", 2, "#D95D5B"),
        ("KLD", "kernelnet", "y_pred_all.tif", 5, "#D95D5B"),
    ),
}

dict_ticks = {
    "PSNR": ((17, 25, 2.5), (17, 27.5, 2.5), (17, 27.5, 2.5)),
    "MS-SSIM": ((0.6, 0.85, 0.1), (0.65, 0.9, 0.1), (0.65, 0.9, 0.1)),
    "ZNCC": ((0.35, 0.7, 0.1), (0.4, 0.8, 0.1), (0.4, 0.8, 0.1)),
}


# ------------------------------------------------------------------------------
datasets_info = settings["datasets"]
methods_info = settings["methods"]

# ------------------------------------------------------------------------------
path_figure = os.path.join("outputs", "figures", "analysis_image", "real_3d")
path_prediction = os.path.join("outputs", "predictions")

info_df = pandas.read_excel("datasets_test.xlsx")

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=3)
dict_clip = {"a_min": 0, "a_max": 2.5}


def preprocess(img):
    img = np.clip(img, 0, None)
    img = normalizer(img)
    img = np.clip(img, **dict_clip)
    return img


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
results_one = []  # store results for each dataset, one sample
for i_dataset in range(num_datasets):
    # load results
    _, dataset_id, id_sample_show, _, _, _ = datasets_info[i_dataset]
    path_result = os.path.join(path_prediction, dataset_id)

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
                meth_filename,
            )
            y_pred = io.imread(path_tmp)
            y_pred = y_pred[-1]

        elif meth_name in ["RLN"]:
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
    dict_colorize = dict(vmin=0.0, vmax=0.9, color=(0, 255, 0))
    dict_text = dict(fontsize=15, color="white")
    dict_text_lb = dict(ha="left", va="bottom", x=0.03, y=0.03, **dict_text)
    dict_text_rt = dict(ha="right", va="top", x=0.97, y=0.97, **dict_text)
    dict_text_lt = dict(ha="left", va="top", x=0.03, y=0.97, **dict_text)
    dict_img_res = dict(cmap="hot", vmin=0.0, vmax=data_range)

    # --------------------------------------------------------------------------
    nr, nc = num_datasets * 2, num_methods + 2
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    fig_res, axes_res = plt.subplots(
        nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig
    )
    [ax.set_axis_off() for ax in axes.ravel()]
    [ax.set_axis_off() for ax in axes_res.ravel()]

    for i_dataset in range(num_datasets):
        axes_ds = axes[i_dataset * 2 : i_dataset * 2 + 2]  # two rows
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
        slice_space = float(info["slice_space"]) / 1000  # slice spacing (um)

        id_slice_zx, x_start, x_stop = pos_plane_xz
        # recalculate the slice index
        id_slice_xy = round((id_slice_xy_ori + 1) * slice_space / pixel_size) - 1

        # ----------------------------------------------------------------------
        results = results_one[i_dataset]

        # gt image
        img_gt = results[-1]
        img_gt = preprocess(img_gt)

        # restored image from different methods
        for i_meth in range(len(results)):
            img = results[i_meth]
            img = preprocess(img)
            img_res = np.abs(img - img_gt)
            mse = np.mean(np.square(img_gt - img))

            ax_xy, ax_zx = axes_ds[0, i_meth], axes_ds[1, i_meth]
            ax_xy_res, ax_zx_res = axes_ds_res[0, i_meth], axes_ds_res[1, i_meth]

            # colorize image ---------------------------------------------------
            # interpolate the image to have a isotropic voxel size
            img_interp = interp_iso_z(img, ps_xy=pixel_size, ps_z=slice_space)
            img_color = colorize(img_interp, **dict_colorize)
            img_res_interp = interp_iso_z(img_res, ps_xy=pixel_size, ps_z=slice_space)

            # show image -------------------------------------------------------
            img_xy = img_color[id_slice_xy, :, :]
            img_zx = img_color[:, id_slice_zx, x_start:x_stop]
            ax_xy.imshow(img_xy)
            ax_zx.imshow(img_zx)

            img_xy_res = img_res_interp[id_slice_xy, :, :]
            img_zx_res = img_res_interp[:, id_slice_zx, x_start:x_stop]
            ax_xy_res.imshow(img_xy_res, **dict_img_res)
            ax_zx_res.imshow(img_zx_res, **dict_img_res)
            ax_xy_res.text(
                s=f"{mse:.4f}", transform=ax_xy_res.transAxes, **dict_text_rt
            )

            # set which ax to show info ----------------------------------------
            ax_xy.plot(
                [x_start, x_stop],
                [id_slice_zx, id_slice_zx],
                "-",
                linewidth=1,
                color="red",
            )

            # add scale bar ----------------------------------------------------
            if i_meth == len(results) - 1:
                tp = 0.05
                # (x,y)
                img_shape = img_xy.shape
                dict_scale_bar = {
                    "pixel_size": pixel_size,
                    "bar_length": 5,  # um
                    "bar_height": 0.01,
                    "bar_color": "white",
                    "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
                }
                add_scale_bar(ax_xy, image=img_xy, **dict_scale_bar)

                # (zx)
                img_zx_shape = img_zx.shape
                dict_scale_bar_zx = {
                    "pixel_size": slice_space,
                    "bar_length": 1,  # um
                    "bar_height": 0.01,
                    "bar_color": "white",
                    "pos": (
                        int(img_zx_shape[1] * tp),
                        int(
                            img_zx_shape[0]
                            * (1 - tp * img_zx_shape[1] / img_zx_shape[0])
                        ),
                    ),
                }
                add_scale_bar(ax_zx, image=img_zx, **dict_scale_bar_zx)

            # add metrics value ------------------------------------------------
            if i_meth != len(results) - 1:
                dict_eva = dict(img_true=img_gt, img_test=img)
                psnr = eva.PSNR(data_range=data_range, **dict_eva)
                ssim = eva.MSSSIM(data_range=data_range, **dict_eva, ndim=3)
                ax_xy.text(
                    s=f"{psnr:.2f} | {ssim:.4f}",
                    transform=ax_xy.transAxes,
                    **dict_text_lb,
                )

            # add zoom patch ---------------------------------------------------
            if show_patch:
                y0, x0, y1, x1 = pos_roi
                patch = img_color[id_slice_xy, y0:y1, x0:x1]

                # add box in the image
                ax_xy.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        linewidth=1,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
                ax_patch = ax_xy.inset_axes(
                    [0.6, 0.0, 0.4, 0.4], transform=ax_xy.transAxes, zorder=10
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
                ax_xy.text(
                    s=methods_names[i_meth], transform=ax_xy.transAxes, **dict_text_rt
                )
            if i_meth == 0:
                ax_xy.text(s=dataset_name, transform=ax_xy.transAxes, **dict_text_lt)

    fig.savefig(os.path.join(path_figure, f"image_restored_compare.png"))
    fig.savefig(os.path.join(path_figure, f"image_restored_compare.svg"))
    fig_res.savefig(os.path.join(path_figure, f"image_restored_compare_res.png"))
    fig_res.savefig(os.path.join(path_figure, f"image_restored_compare_res.svg"))

os._exit(0)

# ------------------------------------------------------------------------------
# load all the results from different methods and datasets
# ------------------------------------------------------------------------------
results_all = []
for i_dataset in range(num_datasets):
    # load results
    dataset_id = datasets_info[i_dataset][1]
    path_result = os.path.join(path_prediction, dataset_id)

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
                    meth_filename,
                )
                y_pred = io.imread(os.path.join(path_tmp))
                y_pred = y_pred[-1]

            elif meth_name in ["RLN"]:
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
    # different datasets may have different number of samples, so do not used
    # numpy.array to store the results_all


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
    # loop over each dataset
    for i_dataset in range(num_datasets):
        pbar_ana.update(1)
        metrics_sample = []
        res_samples = results_all[i_dataset]
        # loop over each sample
        for results in res_samples:
            img_gt = results[-1]
            img_gt = preprocess(img_gt)

            metrics_meth = []
            # loop over each method
            for i_meth in range(num_methods + 1):
                img = results[i_meth]
                img = preprocess(img)
                # --------------------------------------------------------------
                dict_eva = {"img_true": img_gt, "img_test": img}
                psnr = eva.PSNR(**dict_eva, data_range=data_range)
                ssim = eva.MSSSIM(**dict_eva, data_range=data_range, ndim=3)
                zncc = eva.NCC(**dict_eva)
                # --------------------------------------------------------------
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
    # loop over each dataset
    for i_dataset in range(num_datasets):
        pvalues_metrics = []
        met = metrics_dataset[i_dataset]
        met = np.array(met)  # (N_sample, N_meth, N_metrics)
        # loop over each metric
        for i_metric in range(num_metrics):
            pvalues = []
            # loop over each pair of methods
            for i_pair in range(len(test_pairs)):
                pair = test_pairs[i_pair]
                # --------------------------------------------------------------
                test_result = wilcoxon(
                    met[:, pair[0], i_metric],
                    met[:, pair[1], i_metric],
                    alternative="two-sided",
                )
                # --------------------------------------------------------------
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

    font_size = 15
    aspect = 1.25
    # --------------------------------------------------------------------------
    nr, nc = num_metrics, num_datasets
    fig, axes = plt.subplots(
        nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr * aspect), **dict_fig
    )

    for i_metric in range(num_metrics):
        for i_dataset in range(num_datasets):
            ax = axes[i_metric, i_dataset]
            metric_name = metrics_names[i_metric]

            tick_para = dict_ticks[metric_name][i_dataset]
            yticks = np.round(
                np.arange(tick_para[0], tick_para[1] + tick_para[2] / 2, tick_para[2]),
                decimals=2,
            )

            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks, fontsize=font_size)
            ax.set_box_aspect(aspect)

            # grouped boxplot
            df = df_metrics[
                (df_metrics["metric"] == metric_name)
                & (df_metrics["dataset"] == dataset_names[i_dataset])
            ]
            sns.boxplot(
                data=df,
                x="method",
                y="value",
                hue="method",
                ax=ax,
                palette=methods_color,
                gap=0.2,
                fliersize=0.5,
                linecolor="black",
                legend="brief",
            )

            # disable the legend
            if i_metric == 0 and i_dataset == 0:
                ax.legend(fontsize=12, frameon=False, title=None)
            else:
                ax.legend().set_visible(False)

            if i_dataset == 0:
                ax.set_ylabel(metric_name, fontsize=font_size)
            else:
                ax.set_ylabel("")

            if i_metric == 0:
                ax.set_title(dataset_names[i_dataset], fontsize=font_size, ha="center")

            ax.set_xlabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(tick_para[0], tick_para[1])
            ax.tick_params(axis="both", which="major", labelsize=font_size)

            ax.set_xticks([])
            ax.set_xticklabels([])

            # add p-value markers ----------------------------------------------
            pvalues_tmp = np.array(pvalues_dataset)[i_dataset, i_metric, :]
            for i_pair in range(len(test_pairs)):
                i_meth = test_pairs[i_pair][0]

                star_x = i_meth
                star_y = ax.get_ylim()[1]
                add_significant_star(
                    ax=ax,
                    x=star_x,
                    y=star_y,
                    p_value=pvalues_tmp[i_pair],
                    fontsize=font_size,
                )

    fig.savefig(os.path.join(path_figure, f"image_restored_compare_metrics.png"))
    fig.savefig(os.path.join(path_figure, f"image_restored_compare_metrics.svg"))

    # --------------------------------------------------------------------------
    # save source data into excel file
    # --------------------------------------------------------------------------
    writer = pandas.ExcelWriter(
        os.path.join(path_figure, f"image_restored_compare_metrics.xlsx"),
        engine="xlsxwriter",
    )
    df_metrics.to_excel(writer, index=False, sheet_name="metrics")
    df_pvalue.to_excel(writer, index=False, sheet_name="pvalue")
    writer.close()

    # pivot table --------------------------------------------------------------
    writer = pandas.ExcelWriter(
        os.path.join(path_figure, f"image_restored_compare_metrics_pivot.xlsx"),
        engine="xlsxwriter",
    )
    for i_dataset in range(num_datasets):
        for i_metric in range(num_metrics):
            df = df_metrics[
                (df_metrics["metric"] == metrics_names[i_metric])
                & (df_metrics["dataset"] == dataset_names[i_dataset])
            ]

            # pivot data
            df_pivot = df.pivot(index="id_sample", columns="method", values="value")[
                methods_names[:-1]
            ]
            df_pivot.to_excel(
                writer,
                sheet_name=f"{metrics_names[i_metric]} ({dataset_names[i_dataset]})",
                index=False,
            )
    writer.close()
