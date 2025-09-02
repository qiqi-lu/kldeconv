"""
Display the generalizability of the KLDeconv model on structures.
The model was trained on one structure and tested on another same or different structure.
"""

import os, pandas, tqdm
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from utils.data import win2linux, read_txt
import utils.evaluation as eva

plt.rcParams["svg.fonttype"] = "none"


def cal_ssim(x, y):
    return eva.SSIM(img_true=y, img_test=x, data_range=y.max() - y.min())


def cal_psnr(x, y):
    return eva.PSNR(img_true=y, img_test=x, data_range=y.max() - y.min())


# ------------------------------------------------------------------------------
datasets_name_test = (
    "CCPs-9",
    "ER-6",
    "F-actin-9",
    "F-actin-nonlinear-9",
    "Microtubules2-9",
)

datasets_name_train = (
    "F-actin-nonlinear-9",
    "Microtubules2-9",
)

model_id = "kernelnet"
model_name = "KLDeconv"

path_prediction = os.path.join("outputs", "predictions")
path_figure = os.path.join("outputs", "figures")
id_sample = [0, 1, 2, 3, 4, 5, 6]

num_ds_train = len(datasets_name_train)
num_ds_test = len(datasets_name_test)
num_sample = len(id_sample)
print(f"[INFO] Number of datasets (train) : {num_ds_train}")
print(f"[INFO] Number of datasets (test) : {num_ds_test}")
print(f"[INFO] Number of samples : {num_sample}")

# ------------------------------------------------------------------------------
# Load the results
# ------------------------------------------------------------------------------
results = []
pbar = tqdm.tqdm(datasets_name_test, desc="Loading results", ncols=80)
for ds_test in datasets_name_test:
    results_train = []
    for ds_train in datasets_name_train:
        path_results = os.path.join(
            path_prediction, ds_test, model_id, ds_train, "fp_n1_r1_bp_n1_r1"
        )
        imgs = []
        for id in id_sample:
            tmp = []
            get_img = lambda x: io.imread(
                os.path.join(path_results, f"sample_{id}", x)
            ).astype(np.float32)
            # Load the results
            img_raw = get_img("x.tif")
            img_gt = get_img("y.tif")
            img_pred = get_img("y_pred_all.tif")[..., -1]
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
metrics = []
for i, ds_test in enumerate(datasets_name_test):
    metrics_train = []
    for j, ds_train in enumerate(datasets_name_train):
        # calulate the metrics
        img_raw = results[i][j][..., 0]
        img_gt = results[i][j][..., 1]
        img_pred = results[i][j][..., 2]
        # calculate the PSNR


# ------------------------------------------------------------------------------
# display the deconvoled images
# ------------------------------------------------------------------------------
print("[INFO] Display the deconvoled images ...")
dict_fig = {"dpi": 300, "constrained_layout": True}
# dict_img = {"cmap": "hot", "vmin": 0, "vmax": 255}
dict_img = {"cmap": "hot"}
dict_text_lt = {"color": "white", "fontsize": 12, "ha": "left", "va": "top"}
dict_text_rt = {"color": "white", "fontsize": 12, "ha": "right", "va": "top"}
dict_text_lb = {"color": "white", "fontsize": 12, "ha": "left", "va": "bottom"}
dict_text_rb = {"color": "white", "fontsize": 12, "ha": "right", "va": "bottom"}

title_columns = ("Raw",) + datasets_name_train + ("GT",)
title_rows = datasets_name_test

id_sample_show = 0
nr, nc = num_ds_test, num_ds_train + 2
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

# close all the axis in the subplots
[ax.set_axis_off() for ax in axes.ravel()]


# display the raw images
for i in range(nr):
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
            ssim = cal_ssim(img_pred, img_gt)
            psnr = cal_psnr(img_pred, img_gt)
            axes[i, j].text(pox, Ny - posy, f"{psnr:.2f} | {ssim:.4f}", **dict_text_lb)

        # add text

        if i == 0:
            axes[i, j].text(Nx - pox, posy, title_columns[j], **dict_text_rt)
        if j == 0:
            axes[i, j].text(pox, posy, title_rows[i], **dict_text_lt)


# save the figure
plt.savefig(os.path.join(path_figure, "generalizability_structures.png"))
plt.savefig(os.path.join(path_figure, "generalizability_structures.svg"))
