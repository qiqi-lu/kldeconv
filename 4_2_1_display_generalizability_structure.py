"""
Display the generalizability of the KLDeconv model on structures.
The model was trained on one structure and tested on another same or different structure.
--------------------------------------------------------------------------------
             Raw      | Train 1 | Train 2 | Train 3 | ... | GT
--------------------------------------------------------------------------------
data_test_1 |   img   |   img   |   img   |...      |     | img
data_test_2 |   img   |   img   |   img   |...      |     | img
data_test_3 |   img   |   img   |   img   |...      |     | img
...
--------------------------------------------------------------------------------
"""

import os, pandas, tqdm, colorcet
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from utils.data import win2linux, read_txt, NormalizePercentile
import utils.evaluation as eva
import seaborn as sns
from utils.plot import colorize, add_scale_bar

plt.rcParams["svg.fonttype"] = "none"
# subgroup = "noise_level_9"
subgroup = "noise_level_6"
# subgroup = "noise_level_3"
# ------------------------------------------------------------------------------
dict_settings = {
    "noise_level_9": {
        "id_sample_show_each_dataset": [0, 0, 0, 0],
        "rois": (
            # (y0,x0,y1,x1)
            (128, 350, 178, 400),
            (100, 200, 150, 250),
            (100, 200, 150, 250),
            (100, 200, 150, 250),
        ),
        "id_sample_analysis": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        # "id_sample_analysis": [0, 1, 2],
        "dataset_test": (
            ("Microtubules2-9", "MT"),
            ("CCPs-9", "CCP"),
            ("ER-6", "ER"),
            ("F-actin-9", "F-actin"),
        ),
        "dataset_train": (
            ("Microtubules2-9", "MT"),
            ("CCPs-9", "CCP"),
            ("ER-6", "ER"),
            ("F-actin-9", "F-actin"),
        ),
        # "num_iter": 2,
        "num_iter": 5,
    },
    "noise_level_6": {
        "id_sample_show_each_dataset": [0, 0, 0, 0],
        "id_sample_analysis": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        # "id_sample_analysis": [],
        "rois": (
            # (y0,x0,y1,x1)
            (128, 350, 178, 400),
            (120, 50, 170, 100),
            (280, 30, 330, 80),
            (185, 306, 235, 356),
        ),
        "dataset_test": (
            ("Microtubules2-6", "MT"),
            ("CCPs-6", "CCP"),
            ("ER-6", "ER"),
            ("F-actin-6", "F-actin"),
        ),
        "dataset_train": (
            ("Microtubules2-6", "MT"),
            ("CCPs-6", "CCP"),
            ("ER-6", "ER"),
            ("F-actin-6", "F-actin"),
        ),
        # "num_iter": 2,
        "num_iter": 5,
    },
    "noise_level_3": {
        "id_sample_show_each_dataset": [0, 0, 0, 0],
        "id_sample_analysis": [0, 1, 2, 3, 4, 5, 6],
        # "id_sample_analysis": [],
        "rois": (
            # (y0,x0,y1,x1)
            (128, 350, 178, 400),
            (100, 200, 150, 250),
            (100, 200, 150, 250),
            (100, 200, 150, 250),
        ),
        "dataset_test": (
            ("Microtubules2-3", "MT"),
            ("CCPs-3", "CCP"),
            ("ER-3", "ER"),
            ("F-actin-3", "F-actin"),
        ),
        "dataset_train": (
            ("Microtubules2-3", "MT"),
            ("CCPs-3", "CCP"),
            ("ER-3", "ER"),
            ("F-actin-3", "F-actin"),
        ),
        # "num_iter": 2,
        "num_iter": 5,
    },
}

settings = dict_settings[subgroup]

# ------------------------------------------------------------------------------
id_sample_show_each_dataset = settings["id_sample_show_each_dataset"]
datasets_id_test = [ds[0] for ds in settings["dataset_test"]]
datasets_label_test = [ds[1] for ds in settings["dataset_test"]]
datasets_id_train = [ds[0] for ds in settings["dataset_train"]]
datasets_label_train = [ds[1] for ds in settings["dataset_train"]]
id_sample_analysis = settings["id_sample_analysis"]
num_iter_train = settings["num_iter"]
rois_pos = settings["rois"]

model_info = ("kernelnet", "KLD", "fp_n1_r1_bp_n1_r1")
# model_info = ("dfcan", "DFCAN", "n1_r1")

# ------------------------------------------------------------------------------
model_id, model_name, experiment = model_info

path_prediction = os.path.join("outputs", "predictions")
path_figure = os.path.join(
    "outputs",
    "figures",
    "analysis_image",
    "generalization",
    f"{subgroup}_iter_{num_iter_train}",
)
os.makedirs(path_figure, exist_ok=True)

num_dataset_test = len(datasets_id_test)
num_dataset_train = len(datasets_id_train)
num_sample = len(id_sample_analysis)

print(f"-" * 80)
print(f"[INFO] Number of datasets (train) : {num_dataset_train}")
print(f"[INFO] Number of datasets (test) : {num_dataset_test}")
print(f"[INFO] Number of samples : {num_sample}")

# ------------------------------------------------------------------------------
# Load the results
# ------------------------------------------------------------------------------
info_df = pandas.read_excel("datasets_test.xlsx")

results = []
pixel_size_list = []
pbar = tqdm.tqdm(datasets_id_test, desc="Loading results", ncols=80)
for ds_test in datasets_id_test:
    results_train = []

    info = info_df[info_df["id"] == ds_test].iloc[0]
    path_lr = win2linux(info["path_lr"])
    path_hr = win2linux(info["path_hr"])
    path_txt = win2linux(info["path_txt"])
    filenames = read_txt(path_txt)
    pixel_size = info["pixel_size"] / 1000  # um
    pixel_size_list.append(pixel_size)

    for ds_train in datasets_id_train:
        path_results = os.path.join(
            path_prediction,
            ds_test,
            model_id,
            ds_train,
            experiment,
            f"train_iter_{num_iter_train}",
        )
        imgs = []
        for id in id_sample_analysis:
            # load the raw and ground truth images
            img_raw = io.imread(os.path.join(path_lr, filenames[id])).astype(np.float32)
            img_gt = io.imread(os.path.join(path_hr, filenames[id])).astype(np.float32)
            # Load the results
            fname = filenames[id].split(".")[0]
            if model_id == "kernelnet":
                img_pred = io.imread(
                    os.path.join(path_results, fname, "y_pred_all.tif")
                ).astype(np.float32)
                if min(img_pred.shape) == 3:
                    img_pred = img_pred[..., -1]
                else:
                    img_pred = img_pred[-1]
            else:
                img_pred = io.imread(
                    os.path.join(path_results, fname, "y_pred.tif")
                ).astype(np.float32)
            imgs.append([img_raw, img_gt, img_pred])
        imgs = np.stack(imgs, axis=0)
        results_train.append(imgs)
    results.append(results_train)
    pbar.update(1)
pbar.close()

# print the shape of reults ----------------------------------------------------
print("-" * 80)
print("Shape of results:")
for i, ds_test in enumerate(datasets_id_test):
    print(f"{ds_test}:")
    for j, ds_train in enumerate(datasets_id_train):
        print(f"  {ds_train}: {results[i][j].shape}")
print("-" * 80)

# ------------------------------------------------------------------------------
normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=2)
dict_clip = {"a_min": 0.0, "a_max": 2.5}
data_range = dict_clip["a_max"] - dict_clip["a_min"]

# ------------------------------------------------------------------------------
# display the deconvoled images
# ------------------------------------------------------------------------------
print("[INFO] Display the deconvoled images ...")
dict_fig = dict(dpi=300, constrained_layout=True)
dict_img = dict(cmap="hot", vmin=0, vmax=data_range * 0.6)
dict_text_lt = dict(color="white", fontsize=16, ha="left", va="top", x=0.05, y=0.95)
dict_text_rt = dict(color="white", fontsize=16, ha="right", va="top", x=0.95, y=0.95)
dict_text_lb = dict(color="white", fontsize=14, ha="left", va="bottom", x=0.05, y=0.05)
dict_colorize = dict(vmin=0, vmax=0.8, color=(0, 255, 0))
dict_colorize_p = dict(vmin=0, vmax=0.8, color=(0, 255, 0))

# dict_img = dict(cmap="hot", vmin=0, vmax=0.8)
# dict_img = dict(cmap=colorcet.cm.fire, vmin=0, vmax=0.8)

# ------------------------------------------------------------------------------
title_columns = ("Raw",) + tuple(datasets_label_train) + ("GT",)
title_rows = datasets_label_test
# ------------------------------------------------------------------------------
nr, nc = num_dataset_test, num_dataset_train + 2
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

# display the raw images -------------------------------------------------------
for i in range(nr):
    id_sample_show = id_sample_show_each_dataset[i]
    for j in range(nc):
        ax = axes[i, j]
        if j == 0:
            # display the raw images
            img = results[i][0][id_sample_show, 0]
        elif j == nc - 1:
            # display the ground truth
            img = results[i][0][id_sample_show, 1]
        else:
            # display the deconvoled images
            img = results[i][j - 1][id_sample_show, 2]
        # show -----------------------------------------------------------------
        # normalize
        img = normalizer(img)
        # clip
        img = np.clip(img, **dict_clip)

        img_color = colorize(img, **dict_colorize)
        img_color_p = colorize(img, **dict_colorize_p)
        ax.imshow(img_color)
        # ax.imshow(img, **dict_img)

        # add scale bar --------------------------------------------------------
        pixel_size = pixel_size_list[i]
        if j == nc - 1:
            img_shape = img.shape
            tp = 0.05
            dict_scale_bar = {
                "pixel_size": pixel_size,
                "bar_length": 5,  # um
                "bar_height": 0.01,
                "bar_color": "white",
                "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
            }
            add_scale_bar(ax, image=img, **dict_scale_bar)

        # show rois ------------------------------------------------------------
        # show the zoom in region at the bottom right corner of the image
        # crop roi
        pos_roi = rois_pos[i]
        y0, x0, y1, x1 = pos_roi

        patch = img_color_p[y0:y1, x0:x1]
        # patch = img[y0:y1, x0:x1]

        # show the roi box in image
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1,
                edgecolor="magenta",
                facecolor="none",
            )
        )
        # place the roi image at the bottom right corner of the image
        ax_patch = ax.inset_axes(
            [0.6, 0.0, 0.4, 0.4], transform=ax.transAxes, zorder=10
        )

        # ax_patch.imshow(patch)
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

        # add metrics value ----------------------------------------------------
        if j != nc - 1:
            img_gt = results[i][0][id_sample_show, 1]
            if j == 0:
                img_pred = results[i][0][id_sample_show, 0]
            else:
                img_pred = results[i][j - 1][id_sample_show, 2]

            # normalize
            img_gt = normalizer(img_gt)
            img_pred = normalizer(img_pred)
            # clip
            img_gt = np.clip(img_gt, **dict_clip)
            img_pred = np.clip(img_pred, **dict_clip)

            ssim = eva.MSSSIM(img_true=img_gt, img_test=img_pred, data_range=data_range)
            psnr = eva.PSNR(img_true=img_gt, img_test=img_pred, data_range=data_range)
            ax.text(
                s=f"{psnr:.2f} | {ssim:.4f}", transform=ax.transAxes, **dict_text_lb
            )

        # add text -------------------------------------------------------------
        if i == 0:
            ax.text(s=title_columns[j], transform=ax.transAxes, **dict_text_rt)
        if j == 0:
            ax.text(s=title_rows[i], transform=ax.transAxes, **dict_text_lt)


# save the figure
plt.savefig(
    os.path.join(path_figure, f"generalizability_structures_{model_id}_image.png")
)
plt.savefig(
    os.path.join(path_figure, f"generalizability_structures_{model_id}_image.svg")
)

os._exit(0)
# ------------------------------------------------------------------------------
# calulate the metrics
# ------------------------------------------------------------------------------
metrics_test = []
for i in range(num_dataset_test):
    metrics_train = []
    for j in range(num_dataset_train + 1):
        metrics_sample = []
        for k in range(num_sample):
            # calulate the metrics
            img_gt = results[i][0][k][1]
            if j == 0:
                img_pred = results[i][0][k][0]
            else:
                img_pred = results[i][j - 1][k][2]
            # calculate the PSNR
            img_pred = normalizer(img_pred)
            img_gt = normalizer(img_gt)
            img_pred = np.clip(img_pred, **dict_clip)
            img_gt = np.clip(img_gt, **dict_clip)
            psnr = eva.PSNR(img_true=img_gt, img_test=img_pred, data_range=data_range)
            ssim = eva.MSSSIM(img_true=img_gt, img_test=img_pred, data_range=data_range)
            zncc = eva.NCC(img_true=img_gt, img_test=img_pred)
            metrics_sample.append((psnr, ssim, zncc))
        metrics_train.append(metrics_sample)
    metrics_test.append(metrics_train)
metrics = np.array(metrics_test)
print(f"[INFO] metrics: {metrics.shape}")

# ------------------------------------------------------------------------------
# display the metrics heatmap
# ------------------------------------------------------------------------------
print("[INFO] Display the metrics heatmap ...")

# calulate the mean and std of the metrics
# metrics_mean = metrics.mean(axis=2)
metrics_mean = metrics.mean(axis=2)
metrics_std = metrics.std(axis=2)

# display the metrics
dict_fig = {"dpi": 300, "constrained_layout": True}
# ------------------------------------------------------------------------------
nr, nc = 1, metrics.shape[-1]
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

x_ticks = ("Raw",) + tuple(datasets_label_train)
y_ticks = datasets_label_test
metric_names = ("PSNR", "MS-SSIM", "ZNCC")

dict_range = {
    "PSNR": (20, 40),
    "MS-SSIM": (0.8, 1.0),
    "ZNCC": (0.6, 1.0),
}

for i_metric in range(len(metric_names)):
    ax = axes[i_metric]
    metric_name = metric_names[i_metric]
    vmin, vmax = dict_range[metric_name]

    if metric_name in ["PSNR"]:
        # dict_heatmap = {"vmin": vmin, "vmax": vmax, "fmt": ".2f"}
        dict_heatmap = {"fmt": ".2f"}
    if metric_name in ["MS-SSIM", "ZNCC"]:
        # dict_heatmap = {"vmin": vmin, "vmax": vmax, "fmt": ".3f"}
        dict_heatmap = {"fmt": ".3f"}

    data = metrics_mean[:, :, i_metric]

    sns.heatmap(
        data=data,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.4},
        xticklabels=x_ticks,
        yticklabels=y_ticks,
        cmap="rocket",
        annot=True,
        annot_kws={"size": 6},
        linewidths=0.5,
        **dict_heatmap,
    )
    ax.set_title(metric_names[i_metric], fontsize=10)

plt.savefig(
    os.path.join(path_figure, f"generalizability_structures_{model_id}_heatmap.png")
)
plt.savefig(
    os.path.join(path_figure, f"generalizability_structures_{model_id}_heatmap.svg")
)
# save source data -------------------------------------------------------------


# ------------------------------------------------------------------------------
# plot boxplot
# ------------------------------------------------------------------------------
print("[INFO] Plot boxplot...")
# convert metrics matrix to dataframe
df = pandas.DataFrame(
    columns=["dataset_test", "dataset_train", "id_sample"] + list(metric_names)
)

labels = ("Raw",) + tuple(datasets_label_train)
for i_ds_test in range(num_dataset_test):
    for i_ds_train in range(num_dataset_train + 1):
        for i_sample in range(num_sample):
            dataset_test = datasets_label_test[i_ds_test]
            dataset_train = labels[i_ds_train]
            id_sample = i_sample
            psnr, ssim, zncc = metrics[i_ds_test, i_ds_train, i_sample]
            df.loc[len(df)] = [dataset_test, dataset_train, id_sample, psnr, ssim, zncc]

# plot grouped boxplot ---------------------------------------------------------
nr, nc = 1, len(metric_names)
fac = len(datasets_label_test) / 3
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3 * fac, nr * 3), **dict_fig)

dict_ticks = {
    "PSNR": [20, 24, 28, 32, 36, 40],
    "MS-SSIM": [0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
    "ZNCC": [0.6, 0.70, 0.80, 0.90, 1.0],
}

# palette = ("#C1C7D5", "#92C4E9", "#8CCCCE", "#96C36E", "#EA9A9D")
palette = ("#8E99AB", "#4D8FCB", "#42B4B5", "#57AA3E", "#D95D5B")
# palette = "rocket"
for i_metric in range(len(metric_names)):
    metric_name = metric_names[i_metric]
    ax = axes[i_metric]

    # set ticks
    ax.yaxis.set_ticks(dict_ticks[metric_name])
    ax.yaxis.set_ticklabels(dict_ticks[metric_name])

    sns.boxplot(
        x="dataset_test",
        y=metric_name,
        hue="dataset_train",
        data=df,
        ax=ax,
        palette=palette,
        gap=0.2,
        fliersize=0.5,
    )
    # disable the legend
    if i_metric != len(metric_names) - 1:
        ax.legend().set_visible(False)

    ax.set_ylabel(metric_name)
    ax.set_xlabel("")
    ax.axvline(x=0.5, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(x=1.5, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(x=2.5, color="black", linestyle="--", linewidth=0.5)

    if metric_name in ["MS-SSIM", "ZNCC"]:
        ax.set_ylim([None, 1.0])

plt.savefig(
    os.path.join(path_figure, f"generalizability_structures_{model_id}_boxplot.png")
)
plt.savefig(
    os.path.join(path_figure, f"generalizability_structures_{model_id}_boxplot.svg")
)
