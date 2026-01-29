"""
Display many samples.

"""

import numpy as np
import matplotlib.pyplot as plt
import os, pandas
from utils.data import read_txt, win2linux, NormalizePercentile
from utils.plot import image_combine_2d, colorize, add_scale_bar
import skimage.io as io

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
data_info = (
    # data name | dataset-id | sample-id | slice index | repeat_id | num_iter
    ("MT", "Microtubules2-9", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("ER", "ER-6", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("F-actin", "F-actin-9", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("MT-3D", "Microtubule2-3d-1024", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("NPC-3D", "Nuclear-pore-complex2-1024", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("MT", "biotisr-mt-3", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("Lysosomes", "biotisr-lysosomes-3", 1, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("E.coli", "deepbacs-ecoli-ave2", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("S.aureus", "deepbacs-saureus-ave2", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("Mito", "w2s-0-sim-ave", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("lysosome", "w2s-1-sim-ave", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("F-actin", "w2s-2-sim-ave", 0, 0, "fp_n1_r1_bp_n1_r1", 2),
    ("Mito-3D", "biotisr-3d-mito-2", 0, 1, "fp_n1_r1_bp_n1_r1", 2),
    ("MT-3D", "biotisr-3d-mt-2", 0, 1, "fp_n1_r1_bp_n1_r1", 2),
)

method_id = "kernelnet"

# load information of datasets from excel file
datasets_info = pandas.read_excel("datasets_test.xlsx")

path_prediction = os.path.join("outputs", "predictions")
path_figure = os.path.join("outputs", "figures")


num_data = len(data_info)

normalizer = NormalizePercentile(p_high=0.995, p_low=0.03, ndim=2)
dict_fig = dict(dpi=300, constrained_layout=True)
dict_clip = dict(a_min=0.0, a_max=2.5)
dict_colorize = dict(vmin=0, vmax=0.9, color=(0, 255, 0))
dict_text = dict(color="white", fontsize=16, ha="left", va="top")

# ------------------------------------------------------------------------------
# load and display images
# ------------------------------------------------------------------------------
nr, nc = 2, num_data
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
# del axis
[ax.set_axis_off() for ax in axes.ravel()]

# images from different datasets have different shapes
for i_data in range(num_data):
    data_name, dataset_id, sample_id, slice_id, repeat_id, num_iter = data_info[i_data]
    ax_data = axes[:, i_data]

    # get information of the dataset
    info = datasets_info[datasets_info["id"] == dataset_id].iloc[0]

    path_lr = win2linux(info["path_lr"])
    path_hr = win2linux(info["path_hr"])
    path_txt = win2linux(info["path_txt"])
    ndim = int(info["ndim"])
    pixel_size = float(info["pixel_size"]) / 1000

    # get filenames
    filenames = read_txt(path_txt)
    filename = filenames[sample_id]

    # path of prediction result
    path_pred = os.path.join(
        path_prediction,
        dataset_id,
        method_id,
        dataset_id,
        repeat_id,
        "train_iter_2",
        filename.split(".")[0],
        "y_pred_all.tif",
    )

    assert os.path.exists(path_pred), "Prediction result does not exist."

    # load low-resolution image
    img_lr = io.imread(os.path.join(path_lr, filename))

    # load high-resolution image
    img_hr = io.imread(os.path.join(path_hr, filename))

    # load deconvovled image
    img_pred = io.imread(path_pred)

    # select slice
    if ndim == 2:
        # the imread funciton will automaticly reshape the results
        # when having 3 channels.
        if img_pred.shape[-1] in [3, 4]:
            img_pred = np.transpose(img_pred, axes=(-1, 0, 1))
        img_pred = img_pred[num_iter]

    elif ndim == 3:
        img_lr = img_lr[slice_id]
        img_hr = img_hr[slice_id]
        img_pred = img_pred[num_iter][slice_id]

    else:
        raise ValueError("Invalid dimension of image.")

    # normalize and clip
    img_lr = normalizer(img_lr)
    img_hr = normalizer(img_hr)
    img_pred = normalizer(img_pred)

    img_lr = np.clip(img_lr, **dict_clip)
    img_hr = np.clip(img_hr, **dict_clip)
    img_pred = np.clip(img_pred, **dict_clip)

    print("-" * 80)
    print(
        f"Data name: {data_name}, Dataset ID: {dataset_id}, Sample ID: {sample_id}, Slice ID: {slice_id}, Image shape: {img_lr.shape}"
    )

    # display images -----------------------------------------------------------
    # crop to a square image
    diff = img_lr.shape[0] - img_lr.shape[1]
    if diff > 0:
        img_lr = img_lr[diff // 2 : -diff // 2, :]
        img_hr = img_hr[diff // 2 : -diff // 2, :]
        img_pred = img_pred[diff // 2 : -diff // 2, :]
    elif diff < 0:
        img_lr = img_lr[:, diff // 2 : -diff // 2]
        img_hr = img_hr[:, diff // 2 : -diff // 2]
        img_pred = img_pred[:, diff // 2 : -diff // 2]
    else:
        pass

    # combinetwo image into one
    img_lr_hr = image_combine_2d(img_lr, img_hr)

    # colorize
    img_lr_hr = colorize(img_lr_hr, **dict_colorize)
    img_pred = colorize(img_pred, **dict_colorize)

    ax_data[0].imshow(img_lr_hr)
    ax_data[1].imshow(img_pred)

    img_shape = img_lr.shape
    # add scale bar ------------------------------------------------------------
    tp = 0.05
    dict_scale_bar = {
        "pixel_size": pixel_size,
        "bar_length": 5,  # um
        "bar_height": 0.01,
        "bar_color": "white",
        "pos": (int(img_shape[1] * tp), int(img_shape[0] * (1 - tp))),
    }
    add_scale_bar(ax_data[1], image=img_pred, **dict_scale_bar)

    # add dashed line from bottom-left to top-right ----------------------------
    ax_data[0].plot(
        [0, img_shape[1] - 1],
        [img_shape[0] - 1, 0],
        color="white",
        linestyle="--",
        linewidth=1,
    )

    # add text -----------------------------------------------------------------
    ax_data[0].text(
        int(img_shape[1] * tp), int(img_shape[0] * tp), f"{data_name}", **dict_text
    )
    # add pixel size at bottom-right
    ax_data[1].text(
        int(img_shape[1] * tp),
        int(img_shape[0] * tp),
        f"{pixel_size*1000:.1f} nm",
        **dict_text,
    )

plt.savefig(os.path.join(path_figure, "multi_samples.png"))
plt.savefig(os.path.join(path_figure, "multi_samples.svg"))
