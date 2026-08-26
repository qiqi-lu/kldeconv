"""
Display the results when using fixed cell data to train the model,
and the live cell data to test the model.
"""

import numpy as np
import skimage.io as io
import os, tqdm, pandas
from utils.data import read_txt, win2linux, NormalizePercentile, interp
import matplotlib.pyplot as plt
from utils.plot import add_scale_bar, colorize, image_combine_2d

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
#                   test data | train data | id_experiment | num_iter_train | id_slice_xy
data_info_fixed = ("SirDNA-1024-train", "SirDNA-1024", "fp_n1_r1_bp_n1_r1", 2, 7)
# data_info_live = ("SirDNA-1024-live-cell-1", "SirDNA-1024", "fp_n1_r1_bp_n1_r1", 2, 14)
data_info_live = ("SirDNA-1024-live-cell-2", "SirDNA-1024", "fp_n1_r1_bp_n1_r1", 2, 15)

id_sample_show_fixed = 0
# timepoints = [0, 5]
timepoints = [0, 5, 10, 20]
# timepoints = [0, 5, 10, 15, 20, 25]

time_interval = 1  # min
num_timepoints = len(timepoints)

# line_length_fixed = 450  # pixel, for the fixed cell data
# line_pos = ((836, 200), (1600, 70), (1600, 70))
# line_pos = (
#     (836, 200),
#     (1600, 70),
#     (1600, 70),
#     (1600, 70),
#     (1600, 70),
#     (1600, 70),
# ) # cell 1


# position of line in xy plane, for [fixed, live (different timepoints)] (y,x)
line_pos = (
    (870, 200),  # fixed
    (560, 315),  # live 0
    (530, 315),  # live 5
    (540, 315),  # live 10
    # (560, 315), # live 15
    (560, 315),  # live 20
    # (560, 315), # live 25
)  # cell 2
line_length_fixed = 300  # pixel, for the fixed cell data

# ------------------------------------------------------------------------------
assert (
    len(line_pos) == num_timepoints + 1
), "the number of lines should be equal to the number of timepoints + 1"

info_df = pandas.read_excel("datasets_test.xlsx")
path_predictions = os.path.join("outputs", "predictions")
path_figures = os.path.join("outputs", "figures", "analysis_image", "real_live")

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=3)


def preprocess(img):
    img = normalizer(img)
    img = np.clip(img, a_min=0.0, a_max=2.5)
    return img


# fixed cell data --------------------------------------------------------------
info_fixed = info_df[info_df["id"] == data_info_fixed[0]].iloc[0]
path_txt_fixed = win2linux(info_fixed["path_txt"])
path_lr_fixed = win2linux(info_fixed["path_lr"])[:-7]  # del the '_filter' suffix
path_hr_fixed = win2linux(info_fixed["path_hr"])
pixel_size_fixed = float(info_fixed["pixel_size"]) / 1000
slice_space_fixed = float(info_fixed["slice_space"]) / 1000
filenames_fixed = read_txt(path_txt_fixed)
slice_xy_fixed = data_info_fixed[4]

# live cell data ---------------------------------------------------------------
info_live = info_df[info_df["id"] == data_info_live[0]].iloc[0]
path_txt_live = win2linux(info_live["path_txt"])
path_lr_live = win2linux(info_live["path_lr"])[:-7]  # del the '_filter' suffix
pixel_size_live = float(info_live["pixel_size"]) / 1000
slice_space_live = float(info_live["slice_space"]) / 1000
filenames_live = read_txt(path_txt_live)
slice_xy_live = data_info_live[4]

# ------------------------------------------------------------------------------
# load the results of fixed cell data and live cell data
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] load fixed cell data...")

path_predictions_fixed = os.path.join(
    path_predictions,
    data_info_fixed[0],
    "kernelnet",
    data_info_fixed[1],
    data_info_fixed[2],
)

filename = filenames_fixed[id_sample_show_fixed]

imgs_fixed = []
# load the low resolution image and the high resolution image
for path in [path_lr_fixed, path_hr_fixed]:
    img = io.imread(os.path.join(path, filename))
    img = preprocess(img)
    imgs_fixed.append(img)
# load the predicted image
img = io.imread(
    os.path.join(path_predictions_fixed, filename.split(".")[0], "y_pred_all.tif")
)
num_iter = data_info_fixed[3]
img = img[num_iter]
img = preprocess(img)
imgs_fixed.append(img)
imgs_fixed = np.array(imgs_fixed)
# print the shape of the images
print(f"[INFO] shape of the images (fixed) : {imgs_fixed.shape}")

# ------------------------------------------------------------------------------
print("[INFO] load live cell data...")
path_predictions_live = os.path.join(
    path_predictions,
    data_info_live[0],
    "kernelnet",
    data_info_live[1],
    data_info_live[2],
)

num_iter = data_info_live[3]

imgs_live_lr, imgs_live_pred = [], []
pbar = tqdm.tqdm(total=len(timepoints), desc="[INFO] Load live data", ncols=80)
for id_sample in timepoints:
    filename = filenames_live[id_sample]
    # load the low resolution image
    img_lr = io.imread(os.path.join(path_lr_live, filename))
    img_lr = preprocess(img_lr)
    imgs_live_lr.append(img_lr)
    # load the predicted image
    img_pred = io.imread(
        os.path.join(path_predictions_live, filename.split(".")[0], "y_pred_all.tif")
    )
    img_pred = img_pred[num_iter]
    img_pred = preprocess(img_pred)
    imgs_live_pred.append(img_pred)
    pbar.update(1)
pbar.close()

imgs_live_lr = np.array(imgs_live_lr)
imgs_live_pred = np.array(imgs_live_pred)

# print the shape of the images
print(f"[INFO] shape of the images (live-lr) : {imgs_live_lr.shape}")
print(f"[INFO] shape of the images (live-pred) : {imgs_live_pred.shape}")

# ------------------------------------------------------------------------------
# plot the results
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] plot the results...")

nr, nc = 4, num_timepoints + 1

dict_fig = {"dpi": 300, "constrained_layout": True}
render_raw = lambda x: colorize(x, vmin=0.0, vmax=0.4, color=[0, 255, 0])
render_pred = lambda x: colorize(x, vmin=0.0, vmax=0.9, color=[0, 255, 0])
dict_line = dict(color="white", linewidth=1.0, linestyle="--")

dict_text = dict(color="white", fontsize=15)
dict_text_rt = dict(ha="right", va="top", x=0.96, y=0.96, **dict_text)
dict_text_lt = dict(ha="left", va="top", x=0.04, y=0.96, **dict_text)
dict_text_rb = dict(ha="right", va="bottom", x=0.96, y=0.04, **dict_text)
dict_text_lb = dict(ha="left", va="bottom", x=0.04, y=0.04, **dict_text)

# ------------------------------------------------------------------------------
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

# fixed cell data --------------------------------------------------------------
ax_fixed = axes[:, 0]

line_y = line_pos[0][0]
line_x = line_pos[0][1]
line_x_range = (line_x, line_x + line_length_fixed)

img_xy_gt = imgs_fixed[1][slice_xy_fixed]

img_xy_raw = imgs_fixed[0][slice_xy_fixed]
img_xy_pred = imgs_fixed[2][slice_xy_fixed]

img_xy_comb = image_combine_2d(img_xy_pred, img_xy_gt, flip=False)

img_xz_raw = imgs_fixed[0][:, line_y, slice(*line_x_range)]
img_xz_pred = imgs_fixed[2][:, line_y, slice(*line_x_range)]
dict_inter = dict(ps_xy=pixel_size_fixed, ps_z=slice_space_fixed)
img_xz_raw = interp(img_xz_raw, **dict_inter)
img_xz_pred = interp(img_xz_pred, **dict_inter)

img_xy_raw_color = render_raw(img_xy_raw)
img_xz_raw_color = render_raw(img_xz_raw)
img_xy_pred_color = render_pred(img_xy_pred)
img_xz_pred_color = render_pred(img_xz_pred)

img_xy_comb_color = render_pred(img_xy_comb)

# plot the images
ax_fixed[0].imshow(img_xy_raw_color)
ax_fixed[1].imshow(img_xz_raw_color)
ax_fixed[2].imshow(img_xy_comb_color)
ax_fixed[3].imshow(img_xz_pred_color)

# plot zx line in xy plane -----------------------------------------------------
ax_fixed[0].plot(line_x_range, (line_y, line_y), **dict_line)

# add dashed line from bottom-left to top-right --------------------------------
img_shape_xy = img_xy_raw.shape
ax_fixed[2].plot([0, img_shape_xy[1] - 1], [0, img_shape_xy[0] - 1], **dict_line)

# add the scale bar ------------------------------------------------------------
tp = 0.05
dict_scale_bar = {
    "pixel_size": pixel_size_fixed,
    "bar_length": 5,  # um
    "bar_height": 0.01,
    "bar_color": "white",
    "pos": (int(img_shape_xy[1] * tp), int(img_shape_xy[0] * (1 - tp))),
}
add_scale_bar(ax_fixed[0], image=img_xy_raw, **dict_scale_bar)

# add text ---------------------------------------------------------------------
tp = 0.05
ax_fixed[0].text(s="confocal", transform=ax_fixed[0].transAxes, **dict_text_rt)
ax_fixed[2].text(s="KLD", transform=ax_fixed[2].transAxes, **dict_text_rt)
ax_fixed[2].text(s="STED", transform=ax_fixed[2].transAxes, **dict_text_lb)
ax_fixed[0].text(s="Fixed-cell", transform=ax_fixed[0].transAxes, **dict_text_lt)
ax_fixed[0].text(s="xy", transform=ax_fixed[0].transAxes, **dict_text_rb)
ax_fixed[1].text(s="xz", transform=ax_fixed[1].transAxes, **dict_text_rb)
ax_fixed[2].text(s="xy", transform=ax_fixed[2].transAxes, **dict_text_rb)
ax_fixed[3].text(s="xz", transform=ax_fixed[3].transAxes, **dict_text_rb)


# live cell data ---------------------------------------------------------------
# calculate the length of line in the live cell data to keep the same xy ratio
# as the fixed cell data
line_length_live = int(
    line_length_fixed
    * imgs_live_lr[0].shape[0]
    / imgs_fixed[0].shape[0]
    * pixel_size_fixed
    / pixel_size_live
)
# ------------------------------------------------------------------------------
for i in range(num_timepoints):
    ax_live = axes[:, i + 1]
    line_y = line_pos[i + 1][0]
    line_x = line_pos[i + 1][1]
    line_x_range = (line_x, line_x + line_length_live)

    img_xy_raw = imgs_live_lr[i][slice_xy_live]
    img_xy_pred = imgs_live_pred[i][slice_xy_live]

    img_xz_raw = imgs_live_lr[i][:, line_y, slice(*line_x_range)]
    img_xz_pred = imgs_live_pred[i][:, line_y, slice(*line_x_range)]
    dict_inter = dict(ps_xy=pixel_size_live, ps_z=slice_space_live)
    img_xz_raw = interp(img_xz_raw, **dict_inter)
    img_xz_pred = interp(img_xz_pred, **dict_inter)

    img_xy_raw_color = render_pred(img_xy_raw)
    img_xz_raw_color = render_pred(img_xz_raw)
    img_xy_pred_color = render_pred(img_xy_pred)
    img_xz_pred_color = render_pred(img_xz_pred)

    # plot the raw images ------------------------------------------------------
    ax_live[0].imshow(img_xy_raw_color)
    ax_live[1].imshow(img_xz_raw_color)
    ax_live[2].imshow(img_xy_pred_color)
    ax_live[3].imshow(img_xz_pred_color)

    # plot zx line in xy plane -------------------------------------------------
    ax_live[0].plot(line_x_range, (line_y, line_y), **dict_line)

    # add text -----------------------------------------------------------------
    ax_live[0].text(
        s=f"{timepoints[i]*time_interval} min",
        transform=ax_live[0].transAxes,
        **dict_text_rb,
    )
    if i == 0:
        ax_live[0].text(s="Live-cell", transform=ax_live[0].transAxes, **dict_text_lt)
        ax_live[0].text(s="confocal", transform=ax_live[0].transAxes, **dict_text_rt)
        ax_live[2].text(s="KLD", transform=ax_live[2].transAxes, **dict_text_rt)

    # add the scale bar --------------------------------------------------------
    img_shape_xy = img_xy_raw.shape
    if i == num_timepoints - 1:
        tp = 0.05
        dict_scale_bar = {
            "pixel_size": pixel_size_live,
            "bar_length": 5,  # um
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(img_shape_xy[1] * tp), int(img_shape_xy[0] * (1 - tp))),
        }
        add_scale_bar(ax_live[0], image=img_xy_raw, **dict_scale_bar)


# save the figures
plt.savefig(os.path.join(path_figures, "image_restored_fixtrain_livetest.png"))
plt.savefig(os.path.join(path_figures, "image_restored_fixtrain_livetest.svg"))
