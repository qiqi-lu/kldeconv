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

import os, pandas, tqdm
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from utils.data import win2linux, read_txt, NormalizePercentile
import utils.evaluation as eva
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
id_sample_show_each_dataset = [0, 0, 0, 0, 0]

# ------------noise level, PSNR range, SSIM range ------------------------------
noise_level = (9, (20, 40), (50, 95), [0, 0, 0, 0, 0])
noise_level = (2, (15, 30), (20, 75), [2, 2, 2, 2, 2])
# noise_level = (1, (15, 30), (5, 60), [1, 1, 1, 1, 1])

id_sample_show_each_dataset = noise_level[3]

suffix = f"_nl_{noise_level[0]}"

datasets_name_test = (
    # "F-actin-nonlinear-9",
    # "Microtubules2-9",
    # "CCPs-9",
    # "ER-6",
    # "F-actin-9",
    # "F-actin-nonlinear-1",
    # "Microtubules2-1",
    # "CCPs-1",
    # "ER-1",
    # "F-actin-1",
    # "F-actin-nonlinear-2",
    "Microtubules2-2",
    "CCPs-2",
    "ER-2",
    "F-actin-2",
)

datasets_name_train = (
    # "F-actin-nonlinear-9",
    # "Microtubules2-9",
    # "CCPs-9",
    # "ER-6",
    # "F-actin-9",
    # "F-actin-nonlinear-1",
    # "Microtubules2-1",
    # "CCPs-1",
    # "ER-1",
    # "F-actin-1",
    # "F-actin-nonlinear-2",
    "Microtubules2-2",
    "CCPs-2",
    "ER-2",
    "F-actin-2",
)

model_info = ("kernelnet", "KLDeconv", "fp_n1_r1_bp_n1_r1")
# model_info = ("dfcan", "DFCAN", "n1_r1")

# ------------------------------------------------------------------------------
model_id, model_name, experiment = model_info

path_prediction = os.path.join("outputs", "predictions")
path_figure = os.path.join("outputs", "figures")
id_sample = [0, 1, 2, 3, 4, 5, 6]

num_ds_test = len(datasets_name_test)
num_ds_train = len(datasets_name_train)
num_sample = len(id_sample)
print(f"[INFO] Number of datasets (train) : {num_ds_train}")
print(f"[INFO] Number of datasets (test) : {num_ds_test}")
print(f"[INFO] Number of samples : {num_sample}")

info_df = pandas.read_excel("datasets_test.xlsx")

# ------------------------------------------------------------------------------
# Load the results
# ------------------------------------------------------------------------------
results = []
pbar = tqdm.tqdm(datasets_name_test, desc="Loading results", ncols=80)
for ds_test in datasets_name_test:
    results_train = []

    info = info_df[info_df["id"] == ds_test].iloc[0]
    path_lr = win2linux(info["path_lr"])
    path_hr = win2linux(info["path_hr"])
    path_txt = win2linux(info["path_txt"])
    filenames = read_txt(path_txt)

    for ds_train in datasets_name_train:
        path_results = os.path.join(
            path_prediction, ds_test, model_id, ds_train, experiment
        )
        imgs = []
        for id in id_sample:
            tmp = []
            # load the raw and ground truth images
            img_raw = io.imread(os.path.join(path_lr, filenames[id])).astype(np.float32)
            img_gt = io.imread(os.path.join(path_hr, filenames[id])).astype(np.float32)
            # Load the results
            if model_id == "kernelnet":
                img_pred = io.imread(
                    os.path.join(path_results, f"sample_{id}", "y_pred_all.tif")
                ).astype(np.float32)[..., -1]
            else:
                img_pred = io.imread(
                    os.path.join(path_results, f"sample_{id}", "y_pred.tif")
                ).astype(np.float32)
            tmp.extend([img_raw, img_gt, img_pred])
            imgs.append(tmp)
        imgs = np.stack(imgs, axis=0)
        results_train.append(imgs)
    results.append(results_train)
    pbar.update(1)
pbar.close()

# print the shape of reults ----------------------------------------------------
print("-" * 80)
print("Shape of results:")
for i, ds_test in enumerate(datasets_name_test):
    print(f"{ds_test}:")
    for j, ds_train in enumerate(datasets_name_train):
        print(f"  {ds_train}: {results[i][j].shape}")
print("-" * 80)

# ------------------------------------------------------------------------------
# calulate the metrics
# ------------------------------------------------------------------------------
normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=2)
dict_clip = {"a_min": 0.0, "a_max": 2.5}
data_range = dict_clip["a_max"] - dict_clip["a_min"]

metrics_test = []
for i in range(num_ds_test):
    metrics_train = []
    for j in range(num_ds_train + 1):
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
            ssim = eva.SSIM(img_true=img_gt, img_test=img_pred, data_range=data_range)
            metrics_sample.append((psnr, ssim))
        metrics_train.append(metrics_sample)
    metrics_test.append(metrics_train)
metrics = np.array(metrics_test)
print(f"[INFO] metrics: {metrics.shape}")

# ------------------------------------------------------------------------------
# display the metrics heat map
# ------------------------------------------------------------------------------
print("[INFO] Display the metrics heatmap ...")

# calulate the mean and std of the metrics
metrics_mean = metrics.mean(axis=2)
metrics_std = metrics.std(axis=2)
# display the metrics
dict_fig = {"dpi": 300, "constrained_layout": True}
nc, nr = 1, metrics.shape[-1]
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
x_ticks = ("Raw",) + datasets_name_train
y_ticks = datasets_name_test
titles = ("PSNR", "SSIM")

for i_metric in range(nr):
    ax = axes[i_metric]

    if i_metric == 0:
        dict_heatmap = {
            "vmin": noise_level[1][0],
            "vmax": noise_level[1][1],
            "fmt": ".2f",
        }
        scale = 1
    if i_metric == 1:
        dict_heatmap = {
            "vmin": noise_level[2][0],
            "vmax": noise_level[2][1],
            "fmt": ".2f",
        }
        scale = 100
    data = metrics_mean[:, :, i_metric] * scale

    sns.heatmap(
        data=data,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.4},
        xticklabels=x_ticks,
        yticklabels=y_ticks,
        cmap="rocket",
        annot=True,
        annot_kws={"size": 4},
        **dict_heatmap,
    )
    ax.set_title(titles[i_metric], fontsize=10)

plt.savefig(
    os.path.join(
        path_figure, f"generalizability_structures_{model_id}_heatmap{suffix}.png"
    )
)
plt.savefig(
    os.path.join(
        path_figure, f"generalizability_structures_{model_id}_heatmap{suffix}.svg"
    )
)


# ------------------------------------------------------------------------------
# display the deconvoled images
# ------------------------------------------------------------------------------
print("[INFO] Display the deconvoled images ...")
dict_fig = {"dpi": 300, "constrained_layout": True}
dict_img = {"cmap": "hot", "vmin": 0, "vmax": data_range * 0.6}
dict_text_lt = {"color": "white", "fontsize": 12, "ha": "left", "va": "top"}
dict_text_rt = {"color": "white", "fontsize": 12, "ha": "right", "va": "top"}
dict_text_lb = {"color": "white", "fontsize": 12, "ha": "left", "va": "bottom"}
dict_text_rb = {"color": "white", "fontsize": 12, "ha": "right", "va": "bottom"}

title_columns = ("Raw",) + datasets_name_train + ("GT",)
title_rows = datasets_name_test

nr, nc = num_ds_test, num_ds_train + 2
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

# close all the axis in the subplots
[ax.set_axis_off() for ax in axes.ravel()]


# display the raw images
for i in range(nr):
    id_sample_show = id_sample_show_each_dataset[i]
    for j in range(nc):
        if j == 0:
            # display the raw images
            img = results[i][0][id_sample_show, 0]
        elif j == nc - 1:
            # display the ground truth
            img = results[i][0][id_sample_show, 1]
        else:
            # display the deconvoled images
            img = results[i][j - 1][id_sample_show, 2]
        # normalize
        img = normalizer(img)
        # clip
        img = np.clip(img, **dict_clip)
        axes[i, j].imshow(img, **dict_img)

        Ny, Nx = img.shape
        posy, pox = int(Ny * 0.05), int(Nx * 0.05)

        # add metrics value
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

            ssim = eva.SSIM(img_true=img_gt, img_test=img_pred, data_range=data_range)
            psnr = eva.PSNR(img_true=img_gt, img_test=img_pred, data_range=data_range)
            axes[i, j].text(pox, Ny - posy, f"{psnr:.2f} | {ssim:.4f}", **dict_text_lb)

        # add text

        if i == 0:
            axes[i, j].text(Nx - pox, posy, title_columns[j], **dict_text_rt)
        if j == 0:
            axes[i, j].text(pox, posy, title_rows[i], **dict_text_lt)


# save the figure
plt.savefig(
    os.path.join(
        path_figure, f"generalizability_structures_{model_id}_image{suffix}.png"
    )
)
plt.savefig(
    os.path.join(
        path_figure, f"generalizability_structures_{model_id}_image{suffix}.svg"
    )
)
