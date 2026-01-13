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
id_sample_show_live = [0, 5]
id_sample_show_live = [0, 5, 10, 15, 20, 25]
time_interval = 1  # min

line_length_fixed = 450  # pixel, for the fixed cell data
# line_pos_fixed = ((836, 200), (1600, 70), (1600, 70))
# line_pos_fixed = (
#     (836, 200),
#     (1600, 70),
#     (1600, 70),
#     (1600, 70),
#     (1600, 70),
#     (1600, 70),
# ) # cell 1
line_length_fixed = 300  # pixel, for the fixed cell data
line_pos_fixed = (
    (836, 200),
    (560, 315),
    (530, 315),
    (560, 315),
    (560, 315),
    (560, 315),
    (560, 315),
)  # cell 2

# ------------------------------------------------------------------------------
assert (
    len(line_pos_fixed) == len(id_sample_show_live) + 1
), "the number of lines should be equal to the number of id_sample_show_live + 1"

info_df = pandas.read_excel("datasets_test.xlsx")
path_predictions = os.path.join("outputs", "predictions")
path_figures = os.path.join("outputs", "figures")

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=3)
norm = lambda x: np.clip(normalizer(x), a_min=0.0, a_max=2.5)

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
filenames_live = read_txt(path_txt_live)

slice_xy_live = data_info_live[4]

pixel_size_live = float(info_live["pixel_size"]) / 1000
slice_space_live = float(info_live["slice_space"]) / 1000

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

filename_tmp = filenames_fixed[id_sample_show_fixed]
imgs_fixed = []

# load the low resolution image and the high resolution image
for path in [path_lr_fixed, path_hr_fixed]:
    img = io.imread(os.path.join(path, filename_tmp))
    img = norm(img)
    imgs_fixed.append(img)

# load the predicted image
img = io.imread(
    os.path.join(path_predictions_fixed, filename_tmp.split(".")[0], "y_pred_all.tif")
)
num_iter = data_info_fixed[3]
img = img[num_iter]
img = norm(img)
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
pbar = tqdm.tqdm(
    total=len(id_sample_show_live), desc="[INFO] load live data", ncols=100
)
for id_sample in id_sample_show_live:
    filename_tmp = filenames_live[id_sample]

    # load the low resolution image
    img_lr = io.imread(os.path.join(path_lr_live, filename_tmp))
    img_lr = norm(img_lr)
    imgs_live_lr.append(img_lr)

    # load the predicted image
    img_pred = io.imread(
        os.path.join(
            path_predictions_live, filename_tmp.split(".")[0], "y_pred_all.tif"
        )
    )
    img_pred = img_pred[num_iter]
    img_pred = norm(img_pred)
    imgs_live_pred.append(img_pred)
    pbar.update(1)
pbar.close()

imgs_live_lr = np.array(imgs_live_lr)
imgs_live_pred = np.array(imgs_live_pred)

# print the shape of the images
print(f"[INFO] shape of the images (live-lr) : {imgs_live_lr.shape}")
print(f"[INFO] shape of the images (live-pred) : {imgs_live_pred.shape}")
# ------------------------------------------------------------------------------
# calculate the length of line in the live cell data
line_length_live = int(
    line_length_fixed
    * imgs_live_lr[0].shape[0]
    / imgs_fixed[0].shape[0]
    * pixel_size_fixed
    / pixel_size_live
)

# ------------------------------------------------------------------------------
# plot the results
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] plot the results...")

num_timepoints = len(id_sample_show_live)
nr, nc = 4, num_timepoints + 1

dict_fig = {"dpi": 300, "constrained_layout": True}
make_color_raw = lambda x: colorize(x, vmin=0.0, vmax=0.4, color=[255, 0, 0])
make_color = lambda x: colorize(x, vmin=0.0, vmax=0.9, color=[255, 0, 0])
dict_line = dict(color="white", linewidth=1.0, linestyle="--")

dict_text_meth = dict(color="white", fontsize=18, ha="right", va="top")
dict_text_time = dict(color="white", fontsize=18, ha="left", va="top")
dict_text_plane = dict(color="white", fontsize=18, ha="right", va="bottom")
dict_text_lb = dict(color="white", fontsize=18, ha="left", va="bottom")

# ------------------------------------------------------------------------------
fig, axs = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
[ax.set_axis_off() for ax in axs.ravel()]

# fixed cell data --------------------------------------------------------------
xz_line = (line_pos_fixed[0][1], line_pos_fixed[0][1] + line_length_fixed)

img_xy_raw = imgs_fixed[0][slice_xy_fixed]
img_xy_gt = imgs_fixed[1][slice_xy_fixed]
img_xy_pred = imgs_fixed[2][slice_xy_fixed]

img_xy_comb = image_combine_2d(img_xy_pred, img_xy_gt, flip=False)

img_xz_raw = imgs_fixed[0][:, line_pos_fixed[0][0], slice(*xz_line)]
img_xz_raw = interp(img_xz_raw, ps_xy=pixel_size_fixed, ps_z=slice_space_fixed)

img_xz_pred = imgs_fixed[2][:, line_pos_fixed[0][0], slice(*xz_line)]
img_xz_pred = interp(img_xz_pred, ps_xy=pixel_size_fixed, ps_z=slice_space_fixed)

img_xy_raw_color = make_color_raw(img_xy_raw)
img_xz_raw_color = make_color_raw(img_xz_raw)
img_xy_pred_color = make_color(img_xy_pred)
img_xz_pred_color = make_color(img_xz_pred)

img_xy_comb_color = make_color(img_xy_comb)

img_shape_xy = img_xy_raw.shape
img_shape_xz = img_xz_raw.shape

# plot the raw images
axs[0, 0].imshow(img_xy_raw_color)
axs[1, 0].imshow(img_xz_raw_color)
axs[2, 0].imshow(img_xy_comb_color)
axs[3, 0].imshow(img_xz_pred_color)

# plot zx line in xy plane -----------------------------------------------------
axs[0, 0].plot(xz_line, (line_pos_fixed[0][0], line_pos_fixed[0][0]), **dict_line)

# add dashed line from bottom-left to top-right ----------------------------
axs[2, 0].plot(
    [0, img_shape_xy[1] - 1],
    [0, img_shape_xy[0] - 1],
    color="white",
    linestyle="--",
    linewidth=1,
)

# add the scale bar ------------------------------------------------------------
tp = 0.05
dict_scale_bar = {
    "pixel_size": pixel_size_fixed,
    "bar_length": 5,  # um
    "bar_height": 0.01,
    "bar_color": "white",
    "pos": (int(img_shape_xy[1] * tp), int(img_shape_xy[0] * (1 - tp))),
}
add_scale_bar(axs[0, 0], image=img_xy_raw, **dict_scale_bar)

# add text ---------------------------------------------------------------------
tp = 0.05
axs[0, 0].text(
    img_shape_xy[1] * (1 - tp), img_shape_xy[0] * tp, "confocal", **dict_text_meth
)
axs[2, 0].text(
    img_shape_xy[1] * (1 - tp), img_shape_xy[0] * tp, "KLD", **dict_text_meth
)
axs[2, 0].text(img_shape_xy[1] * tp, img_shape_xy[0] * (1 - tp), "STED", **dict_text_lb)

axs[0, 0].text(
    img_shape_xy[1] * tp, img_shape_xy[0] * tp, "Fixed-cell", **dict_text_time
)
axs[0, 0].text(
    img_shape_xy[1] * (1 - tp), img_shape_xy[0] * (1 - tp), "xy", **dict_text_plane
)
axs[1, 0].text(
    img_shape_xz[1] * (1 - tp), img_shape_xz[0] * (1 - tp), "xz", **dict_text_plane
)
axs[2, 0].text(
    img_shape_xy[1] * (1 - tp), img_shape_xy[0] * (1 - tp), "xy", **dict_text_plane
)
axs[3, 0].text(
    img_shape_xz[1] * (1 - tp), img_shape_xz[0] * (1 - tp), "xz", **dict_text_plane
)


# live cell data ---------------------------------------------------------------
for i in range(num_timepoints):

    xz_line = (line_pos_fixed[i + 1][1], line_pos_fixed[i + 1][1] + line_length_live)

    img_xy_raw = imgs_live_lr[i][slice_xy_live]
    img_xy_pred = imgs_live_pred[i][slice_xy_live]

    img_xz_raw = imgs_live_lr[i][:, line_pos_fixed[i + 1][0], slice(*xz_line)]
    img_xz_raw = interp(img_xz_raw, ps_xy=pixel_size_live, ps_z=slice_space_live)
    img_xz_pred = imgs_live_pred[i][:, line_pos_fixed[i + 1][0], slice(*xz_line)]
    img_xz_pred = interp(img_xz_pred, ps_xy=pixel_size_live, ps_z=slice_space_live)

    img_xy_raw_color = make_color(img_xy_raw)
    img_xz_raw_color = make_color(img_xz_raw)
    img_xy_pred_color = make_color(img_xy_pred)
    img_xz_pred_color = make_color(img_xz_pred)

    img_shape_xy = img_xy_raw.shape

    # plot the raw images ------------------------------------------------------
    axs[0, i + 1].imshow(img_xy_raw_color)
    axs[1, i + 1].imshow(img_xz_raw_color)
    axs[2, i + 1].imshow(img_xy_pred_color)
    axs[3, i + 1].imshow(img_xz_pred_color)

    # plot zx line in xy plane -------------------------------------------------
    axs[0, i + 1].plot(
        xz_line, (line_pos_fixed[i + 1][0], line_pos_fixed[i + 1][0]), **dict_line
    )

    # add the scale bar --------------------------------------------------------
    # if i == num_timepoints - 1:
    #     tp = 0.05
    #     dict_scale_bar = {
    #         "pixel_size": pixel_size_live,
    #         "bar_length": 5,  # um
    #         "bar_height": 0.01,
    #         "bar_color": "white",
    #         "pos": (int(img_shape_xy[1] * tp), int(img_shape_xy[0] * (1 - tp))),
    #     }
    #     add_scale_bar(axs[0, i + 1], image=img_xy_raw, **dict_scale_bar)
    # add text -----------------------------------------------------------------
    tp = 0.05
    axs[0, i + 1].text(
        img_shape_xy[1] * tp,
        img_shape_xy[0] * tp,
        f"{id_sample_show_live[i]*time_interval} min",
        **dict_text_time,
    )


# save the figures
plt.savefig(os.path.join(path_figures, "image_restored_live_fixtrain_livetest.png"))
plt.savefig(os.path.join(path_figures, "image_restored_live_fixtrain_livetest.svg"))
