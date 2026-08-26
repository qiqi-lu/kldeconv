"""
Dispaly the restored image of live 3D dataset from different methods.
"""

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os, pandas
import matplotlib as mpl
import matplotlib.patches as patches
from utils.data import win2linux, read_txt, interp
from utils.plot import render, add_scale_bar, add_patch

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
p_raw = ((0.0, 0.0), (99.99, 99.9))
results_info = {
    "ZeroShotDeconvNet-mitosis-642": {
        "color": mpl.colormaps["gray"],
        "methods": (
            ("Traditional@30", "traditional", "deconv.tif", (0.0, 99.5)),
            ("Gaussian@30", "gaussian", "deconv.tif", (0.0, 99.5)),
            ("Butterworth@3", "butterworth", "deconv.tif", (0.0, 99.5)),
            ("WB@2", "wiener_butterworth", "deconv.tif", (0.0, 99.5)),
            (
                "KLD-SS@2",
                "kernelnet_ss\ZeroShotDeconvNet-mitosis-642\\n1_r1",
                "y_pred_all.tif",
                (0.0, 99.5),
            ),
            (
                "KLD@2",
                "kernelnet\SimuMix3D-382-101-05-1-1-642\\n1_r1",
                "y_pred_all.tif",
                (0.0, 99.5),
            ),
        ),
    },
    "ZeroShotDeconvNet-mitosis-560": {
        "color": mpl.colormaps["afmhot"],
        "methods": (
            ("Traditional@30", "traditional", "deconv.tif", (0.0, 99.5)),
            ("Gaussian@30", "gaussian", "deconv.tif", (0.0, 99.5)),
            ("Butterworth@30", "butterworth", "deconv.tif", (0.0, 99.5)),
            ("WB@2", "wiener_butterworth", "deconv.tif", (0.0, 99.5)),
            (
                "KLD-ss@2",
                "kernelnet_ss\ZeroShotDeconvNet-mitosis-560\\n1_r1",
                "y_pred_all.tif",
                (0.0, 99.5),
            ),
            (
                "KLD@2",
                "kernelnet\SimuMix3D-382-101-05-1-1-560\\n1_r1",
                "y_pred_all.tif",
                (0.0, 99.5),
            ),
        ),
    },
}

dataset_names = list(results_info.keys())
method_names = [info[0] for info in results_info[dataset_names[0]]["methods"]]
titles = ["Raw"] + method_names  # add 'Raw'

num_iter = 2
timepoint_show = 0
show_patch = True

# ------------------------------------------------------------------------------
# check the number of methods in each dataset
assert len(results_info[dataset_names[0]]["methods"]) == len(
    results_info[dataset_names[1]]["methods"]
), "The number of methods in each dataset is not consistent!"

# get all the name of methods
num_methods = len(method_names)
num_channels = len(dataset_names)

path_prediction = os.path.join("outputs", "predictions")
path_fig = os.path.join("outputs", "figures", "analysis_image", "real_live")
info_df = pandas.read_excel("datasets_test.xlsx")

info_one = info_df[info_df["id"] == dataset_names[0]].iloc[0]
pixel_size = float(info_one["pixel_size"]) / 1000  # um
slice_space = float(info_one["slice_space"]) / 1000  # um

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
imgs_deconv_mc = []
for i_channel in range(num_channels):
    ds_name = dataset_names[i_channel]
    methods = results_info[ds_name]["methods"]

    info = info_df[info_df["id"] == ds_name].iloc[0]

    path_lr = win2linux(info["path_lr"])
    path_txt = win2linux(info["path_txt"])
    filenames = read_txt(path_txt)
    # --------------------------------------------------------------------------
    imgs = []
    # load RAW image
    imgs.append(io.imread(os.path.join(path_lr, filenames[timepoint_show])))

    # load deconvolved image
    for i_method in range(num_methods):
        img = io.imread(
            os.path.join(
                path_prediction,
                ds_name,
                win2linux(methods[i_method][1]),
                f"sample_{timepoint_show}",
                methods[i_method][2],
            )
        )
        if img.ndim == 4:
            img = img[num_iter]
        imgs.append(img)
    imgs_deconv_mc.append(imgs)

imgs_deconv_mc = np.array(imgs_deconv_mc)

Nc, Nmeth, Nz, Ny, Nx = imgs_deconv_mc.shape
print(f"Num of channel: {Nc}, num of methods: {Nmeth}, image shape: {(Nz, Ny, Nx)}")

# ------------------------------------------------------------------------------
# show image restored
# ------------------------------------------------------------------------------
dict_fig = dict(dpi=600, constrained_layout=True)
dict_line = {"color": "white", "linewidth": 1}
cmaps = [results_info[ds_name]["color"] for ds_name in dataset_names]

dict_text = dict(fontsize=15, color="white")
dict_text_rt = dict(x=0.96, y=0.96, ha="right", va="top", **dict_text)
dict_text_lb = dict(x=0.04, y=0.04, ha="left", va="bottom", **dict_text)

# ------------------------------------------------------------------------------
# define color
nr, nc = 2, Nmeth
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

slice_range_xy = (50, 70)
slice_range_xz = (140, 160)

for i_meth in range(Nmeth):
    ax_xy = axes[0, i_meth]
    ax_xz = axes[1, i_meth]

    # --------------------------------------------------------------------------
    img = imgs_deconv_mc[:, i_meth]
    img = np.transpose(img, axes=(1, 2, 3, 0))

    # xy/xz plane with max intensity projection
    xy_plane = np.max(img[slice_range_xy[0] : slice_range_xy[1]], axis=0)
    xz_plane = np.max(img[:, slice_range_xz[0] : slice_range_xz[1], :], axis=1)

    # interpolate xz plane
    xz_plane_interp = [
        interp(xz_plane[..., i], ps_xy=pixel_size, ps_z=slice_space) for i in range(Nc)
    ]
    xz_plane = np.transpose(np.array(xz_plane_interp), axes=(1, 2, 0))
    # --------------------------------------------------------------------------
    if i_meth == 0:
        # raw image
        pl, ph = p_raw
    else:
        # deconvolved image
        pl, ph = [], []
        for ds_name in dataset_names:
            ran = results_info[ds_name]["methods"][i_meth - 1][3]
            pl.append(ran[0])
            ph.append(ran[1])

    dict_render = dict(cmaps=cmaps, plow=pl, phigh=ph)
    xy_plane_color = render(xy_plane, **dict_render)
    xz_plane_color = render(xz_plane, **dict_render)

    # --------------------------------------------------------------------------
    ax_xy.imshow(xy_plane_color)
    ax_xz.imshow(xz_plane_color)

    # add scale bar ------------------------------------------------------------
    if i_meth == Nmeth - 1:
        tp = 0.05
        dict_scale_bar = {
            "pixel_size": pixel_size,
            "bar_length": 5,
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(Nx * tp), int(Ny * (1 - tp))),
        }
        add_scale_bar(ax_xy, image=xy_plane, **dict_scale_bar)

    #  add zoom box ------------------------------------------------------------
    pos, size = (120, 155), 60
    if show_patch:
        show_box = True if i_meth == Nmeth - 1 else False
        add_patch(
            ax_xy,
            image=xy_plane_color,
            pos=pos,
            size=size,
            show_box=show_box,
            axes_lw=1,
            box_lw=1,
            box_color="red",
            percent=0.4,
        )

    # add title ----------------------------------------------------------------
    ax_xy.text(s=titles[i_meth], transform=ax_xy.transAxes, **dict_text_rt)

    # add plane label ----------------------------------------------------------
    if i_meth == 0:
        ax_xy.text(s="xy", transform=ax_xy.transAxes, **dict_text_lb)
        ax_xz.text(s="xz", transform=ax_xz.transAxes, **dict_text_lb)


plt.savefig(os.path.join(path_fig, f"image_restored_compare_id_{timepoint_show}.png"))
plt.savefig(os.path.join(path_fig, f"image_restored_compare_id_{timepoint_show}.svg"))
