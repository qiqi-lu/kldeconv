"""
Display the images restored by different methods.
2D images.
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
# show_image, show_statistic =  False, True
# num_samples_max = 3
num_samples_max = 20
show_patch = True

# ------------------------------------------------------------------------------
settings = {
    "datasets": (
        # ------------------------------------------------------------------
        # dataset_name | dataset_id | id sample | roi pos (y0,x0,y1,x1)
        # ------------------------------------------------------------------
        # ('F-actin',"F-actin-nonlinear-9", 5, (200, 100, 250, 150)),
        ("CCP", "CCPs-9", 0, (200, 100, 250, 150)),
        ("MT", "Microtubules2-9", 0, (100, 200, 200, 300)),
        ("ER", "ER-6", 0, (100, 200, 200, 300)),
        ("F-actin", "F-actin-9", 0, (100, 200, 200, 300)),
        # ("CCP (BioTISR)-1", "biotisr-ccps-1", 0, (200, 100, 250, 150)),
        # ("CCP (BioTISR)-2", "biotisr-ccps-2", 0, (200, 100, 250, 150)),
        # ("CCP (BioTISR)-3", "biotisr-ccps-3", 0, (200, 100, 250, 150)),
        # ("F-actin (BioTISR)-1", "biotisr-factin-1", 0, (200, 100, 250, 150)),
        # ("F-actin (BioTISR)-2", "biotisr-factin-2", 0, (200, 100, 250, 150)),
        # ("F-actin (BioTISR)-3", "biotisr-factin-3", 0, (200, 100, 250, 150)),
        # ("F-actin-nl (BioTISR)-1", "biotisr-factin-nonlinear-1", 0, (200, 100, 250, 150)),
        # ("F-actin-nl (BioTISR)-2", "biotisr-factin-nonlinear-2", 0, (200, 100, 250, 150)),
        # (
        #     "F-actin-nl (BioTISR)-3",
        #     "biotisr-factin-nonlinear-3",
        #     0,
        #     (200, 100, 250, 150),
        # ),
        # ("lysosomes (BioTISR)-1", "biotisr-lysosomes-1", 0, (200, 100, 250, 150)),
        # ("lysosomes (BioTISR)-2", "biotisr-lysosomes-2", 0, (200, 100, 250, 150)),
        # ("lysosomes (BioTISR)-3", "biotisr-lysosomes-3", 0, (200, 100, 250, 150)),
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
}

dict_ticks = {
    # metrics name, yticks, ylim
    "PSNR": ((20, 35, 5), (25, 37.5, 2.5), (25, 40, 2.5), (25, 37.5, 2.5)),
    "MS-SSIM": (
        (0.5, 1.0, 0.1),
        (0.92, 1.0, 0.02),
        (0.94, 1.0, 0.02),
        (0.92, 1.0, 0.02),
    ),
    "ZNCC": ((0.7, 0.95, 0.1), (0.85, 1.0, 0.05), (0.9, 1.0, 0.02), (0.9, 1.0, 0.02)),
}

# ------------------------------------------------------------------------------
datasets_info = settings["datasets"]
methods_info = settings["methods"]

# ------------------------------------------------------------------------------
path_figure = os.path.join("outputs", "figures", "analysis_image", "real_2d")
path_prediction = os.path.join("outputs", "predictions")

info_df = pandas.read_excel("datasets_test.xlsx")

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=2)
dict_clip = {"a_min": 0, "a_max": 2.5}


def preprocess(img):
    # img = np.clip(img, 0, None)
    img = normalizer(img)
    img = np.clip(img, **dict_clip)
    return img


data_range = dict_clip["a_max"] - dict_clip["a_min"]
dict_fig = dict(constrained_layout=True, dpi=300)

# ------------------------------------------------------------------------------
num_datasets = len(datasets_info)
num_methods = len(methods_info)

# get all the method names (shown in legend)
methods_names = ["RAW"]
for i_meth in range(len(methods_info)):
    methods_names.append(methods_info[i_meth][0])
methods_names.append("GT")

# set the metrics names to calculate (in the statistic analysis)
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
results_one = []  # results of one sample from each dataset and method
for i_dataset in range(num_datasets):
    # load results
    # get the id of dataset and the id of sample to show
    _, dataset_id, id_sample_show, _ = datasets_info[i_dataset]
    path_result = os.path.join(path_prediction, dataset_id)

    print("-" * 80)
    print("[INFO] Load results from :", path_result)

    # get the info of dataset from the excel file
    info = info_df[info_df["id"] == dataset_id].iloc[0]
    path_raw, path_gt = win2linux(info["path_lr"]), win2linux(info["path_hr"])
    filenames = read_txt(win2linux(info["path_txt"]))

    # --------------------------------------------------------------------------
    results_meth = []
    # load raw and gt images
    filename = filenames[id_sample_show]
    x = io.imread(os.path.join(path_raw, filename))
    y = io.imread(os.path.join(path_gt, filename))
    results_meth.append(x.astype(np.float32))

    # load restoed images from each method
    filename_wo_ext = filename.split(".")[0]
    for i_meth in range(num_methods):
        meth_name, meth_id, meth_filename, num_iter_train = methods_info[i_meth][:4]

        # KLDeconv
        if meth_name == "KLD":
            path_tmp = os.path.join(
                path_result,
                meth_id,
                dataset_id,  # dataset test
                "fp_n1_r1_bp_n1_r1",
                f"train_iter_{num_iter_train}",
                filename_wo_ext,
                meth_filename,
            )
            y_pred = io.imread(path_tmp)

            # the imread funciton will automaticly reshape the results
            # when having 3 or 4 channels.
            if y_pred.shape[-1] in [3, 4]:
                y_pred = np.transpose(y_pred, axes=(-1, 0, 1))

            # get the result of last iteration
            y_pred = y_pred[-1]

        # conventional deep learning methods
        elif meth_name in ["DFCAN"]:
            path_tmp = os.path.join(
                path_result,
                meth_id,
                dataset_id,  # dataset test
                "n1_r1",
                filename_wo_ext,
                meth_filename,
            )
            y_pred = io.imread(path_tmp)

        # traditional deconvolution methods
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

    dict_colorize = dict(vmin=0.0, vmax=0.9, color=(0, 255, 0))
    dict_text = dict(fontsize=15, color="white")
    dict_text_lb = dict(ha="left", va="bottom", x=0.03, y=0.03, **dict_text)
    dict_text_rb = dict(ha="right", va="bottom", x=0.97, y=0.03, **dict_text)
    dict_text_rt = dict(ha="right", va="top", x=0.97, y=0.97, **dict_text)
    dict_text_lt = dict(ha="left", va="top", x=0.03, y=0.97, **dict_text)
    dict_img_res = dict(cmap="hot", vmin=0.0, vmax=1.0)

    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    [ax.set_axis_off() for ax in axes.ravel()]

    fig_res, axes_res = plt.subplots(
        nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig
    )
    [ax.set_axis_off() for ax in axes_res.ravel()]

    # plot results in each dataset one by one
    for i_dataset in range(num_datasets):
        axes_ds = axes[i_dataset]
        axes_ds_res = axes_res[i_dataset]
        dataset_name, dataset_id, id_sample, pos_roi = datasets_info[i_dataset]

        info = info_df[info_df["id"] == dataset_id].iloc[0]
        # get pixel size used for scale bar
        pixel_size = float(info["pixel_size"]) / 1000  # pixel size (um)

        # ----------------------------------------------------------------------
        results = results_one[i_dataset]

        # gt image
        img_gt = results[-1]
        img_gt = preprocess(img_gt)

        # plot image from different methods one by one
        for i_meth in range(len(results)):
            img = results[i_meth]
            img = preprocess(img)
            img_res = np.abs(img - img_gt)
            mse = np.mean(np.square(img_gt - img))

            # colorize image ---------------------------------------------------
            img_color = colorize(img, **dict_colorize)

            # show image -------------------------------------------------------
            ax_img = axes_ds[i_meth]
            ax_res = axes_ds_res[i_meth]

            ax_img.imshow(img_color)
            ax_res.imshow(img_res, **dict_img_res)
            ax_res.text(s=f"{mse:.4f}", transform=ax_res.transAxes, **dict_text_rb)

            img_shape = img.shape

            # add scale bar ----------------------------------------------------
            # only add scale bar to the last method (i.e., GT)
            if i_meth == len(results) - 1:
                tp = 0.05
                dict_scale_bar = {
                    "pixel_size": pixel_size,
                    "bar_length": 5,  # um
                    "bar_height": 0.01,
                    "bar_color": "white",
                    "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
                }
                add_scale_bar(ax_img, image=img, **dict_scale_bar)

            # add metrics value ------------------------------------------------
            if i_meth != len(results) - 1:
                dict_eva = dict(img_true=img_gt, img_test=img)
                psnr = eva.PSNR(data_range=data_range, **dict_eva)
                ssim = eva.MSSSIM(data_range=data_range, **dict_eva)
                ax_img.text(
                    s=f"{psnr:.2f} | {ssim:.4f}",
                    transform=ax_img.transAxes,
                    **dict_text_lb,
                )

            # add zoom patch ---------------------------------------------------
            if show_patch:
                y0, x0, y1, x1 = pos_roi
                patch = img_color[y0:y1, x0:x1]

                # add box in the image
                ax_img.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        linewidth=1,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
                ax_patch = ax_img.inset_axes(
                    [0.6, 0.0, 0.4, 0.4], transform=ax_img.transAxes, zorder=10
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
                for ax in [ax_img, ax_res]:
                    ax.text(
                        s=methods_names[i_meth], transform=ax.transAxes, **dict_text_rt
                    )
            if i_meth == 0:
                for ax in [ax_img, ax_res]:
                    ax.text(s=dataset_name, transform=ax.transAxes, **dict_text_lt)

    fig.savefig(os.path.join(path_figure, f"image_restored_compare.png"))
    fig.savefig(os.path.join(path_figure, f"image_restored_compare.svg"))
    fig_res.savefig(os.path.join(path_figure, f"image_restored_compare_res.png"))
    fig_res.savefig(os.path.join(path_figure, f"image_restored_compare_res.svg"))

# os._exit(0)

# ------------------------------------------------------------------------------
# load all the results from different methods and datasets
# ------------------------------------------------------------------------------
results_all = []  # results of all the samples from all the datasets
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
        results_ss = []  # to store results from different methods

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
                y_pred = io.imread(path_tmp)

                # the imread funciton will automaticly reshape the results
                # when having 3 channels.
                if y_pred.shape[-1] in [3, 4]:
                    y_pred = np.transpose(y_pred, axes=(-1, 0, 1))

                y_pred = y_pred[-1]

            elif meth_name in ["DFCAN"]:
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

    # print number of samples
    print(
        f"[INFO] Dataset {dataset_id}: num of samples: {len(results)}, shape of image: {results[0][0].shape}"
    )

    results_all.append(results)

# ------------------------------------------------------------------------------
# statistics analysis on all the datasets
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
            img_gt = preprocess(img_gt)

            metrics_meth = []
            for i_meth in range(num_methods + 1):
                img = results[i_meth]
                img = preprocess(img)

                # --------------------------------------------------------------
                dict_eva = {"img_true": img_gt, "img_test": img}
                psnr = eva.PSNR(**dict_eva, data_range=data_range)
                ssim = eva.MSSSIM(**dict_eva, data_range=data_range, ndim=2)
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
    for i_dataset in range(num_datasets):
        pvalues_metrics = []
        met = metrics_dataset[i_dataset]
        met = np.array(met)  # (N_sample, N_meth, N_metrics)

        for i_metric in range(num_metrics):
            pvalues = []
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
    methods_color = ["#8E99AB"]  # the first one is for raw
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
            ax.set_ylim((tick_para[0], tick_para[1]))
            ax.tick_params(axis="both", which="major", labelsize=font_size)

            # disable xticklabels
            ax.set_xticklabels([])
            ax.set_xticks([])

            # # add p-value markers --------------------------------------------------
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

    plt.savefig(os.path.join(path_figure, f"image_restored_compare_metrics.png"))
    plt.savefig(os.path.join(path_figure, f"image_restored_compare_metrics.svg"))
    # save source data into excel file -----------------------------------------
    writer = pandas.ExcelWriter(
        os.path.join(path_figure, f"image_restored_compare_metrics.xlsx"),
        engine="xlsxwriter",
    )
    df_metrics.to_excel(writer, index=False, sheet_name="metrics")
    df_pvalue.to_excel(writer, index=False, sheet_name="pvalue")
    writer.close()

    # save metrics value to each sheet (dataset-metric)
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

            # convert to a matrix form
            df_pivot = df.pivot(index="id_sample", columns="method", values="value")[
                methods_names[:-1]
            ]
            df_pivot.to_excel(
                writer,
                sheet_name=f"{dataset_names[i_dataset]} ({metrics_names[i_metric]})",
                index=False,
            )
    writer.close()
