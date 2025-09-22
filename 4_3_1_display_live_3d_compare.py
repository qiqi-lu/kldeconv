import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os, pandas
import utils.plot as utils_plot
import matplotlib as mpl
import matplotlib.patches as patches
from utils.data import win2linux, read_txt, interp
from utils.plot import render, add_scale_bar, add_patch

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
datasets_info = (
    ("ZeroShotDeconvNet-mitosis-642", mpl.colormaps["gray"]),
    ("ZeroShotDeconvNet-mitosis-560", mpl.colormaps["afmhot"]),
)

p_raw = ((0.0, 0.0), (99.99, 99.9))
methods_info = {
    "ZeroShotDeconvNet-mitosis-642": (
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
    "ZeroShotDeconvNet-mitosis-560": (
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
}

num_iter = 2
id_sample_show = 0
show_patch = True

# ------------------------------------------------------------------------------
assert len(methods_info["ZeroShotDeconvNet-mitosis-560"]) == len(
    methods_info["ZeroShotDeconvNet-mitosis-642"]
), "methods_info is not consistent!"

# get all the name of methods
methods_name = [info[0] for info in methods_info["ZeroShotDeconvNet-mitosis-642"]]
num_methods = len(methods_name)
num_channels = len(datasets_info)

path_prediction = os.path.join("outputs", "predictions")
path_fig = os.path.join("outputs", "figures")
info_df = pandas.read_excel("datasets_test.xlsx")

info_one = info_df[info_df["id"] == "ZeroShotDeconvNet-mitosis-642"].iloc[0]
pixel_size = float(info_one["pixel_size"]) / 1000  # um
slice_space = float(info_one["slice_space"]) / 1000  # um

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
imgs_deconv_mc = []
for i_channel in range(num_channels):
    ds_name = datasets_info[i_channel][0]
    methods = methods_info[ds_name]

    info = info_df[info_df["id"] == ds_name].iloc[0]

    path_lr = win2linux(info["path_lr"])
    path_txt = win2linux(info["path_txt"])
    filenames = read_txt(path_txt)
    # --------------------------------------------------------------------------
    imgs = []
    # load RAW image
    imgs.append(io.imread(os.path.join(path_lr, filenames[id_sample_show])))
    # load deconvolved image
    for i_method in range(num_methods):
        img = io.imread(
            os.path.join(
                path_prediction,
                ds_name,
                win2linux(methods[i_method][1]),
                f"sample_{id_sample_show}",
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
cmaps = [info[1] for info in datasets_info]
dict_text_meth = {"fontsize": 18, "color": "white", "ha": "right", "va": "top"}
dict_text_plane = {"fontsize": 18, "color": "white", "ha": "left", "va": "bottom"}

# ------------------------------------------------------------------------------
# define color
nr, nc = 2, Nmeth
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

slice_range_xy = (50, 70)
slice_range_xz = (140, 160)

for imeth in range(Nmeth):
    img = imgs_deconv_mc[:, imeth]
    img = np.transpose(img, axes=(1, 2, 3, 0))

    # xy plane
    xy_plane = np.max(img[slice_range_xy[0] : slice_range_xy[1]], axis=0)
    # xz plane
    xz_plane = np.max(img[:, slice_range_xz[0] : slice_range_xz[1], :], axis=1)

    xz_plane_interp = [
        interp(xz_plane[..., i], ps_xy=pixel_size, ps_z=slice_space) for i in range(Nc)
    ]
    xz_plane = np.transpose(np.array(xz_plane_interp), axes=(1, 2, 0))
    # --------------------------------------------------------------------------
    if imeth == 0:
        # raw image
        pl, ph = p_raw
    else:
        # deconvolved image
        pl, ph = [], []
        for ds_info in datasets_info:
            ds_name = ds_info[0]
            pl.append(methods_info[ds_name][imeth - 1][3][0])
            ph.append(methods_info[ds_name][imeth - 1][3][1])

    xy_plane_color = render(xy_plane, cmaps=cmaps, plow=pl, phigh=ph)
    xz_plane_color = render(xz_plane, cmaps=cmaps, plow=pl, phigh=ph)

    axes[0, imeth].imshow(xy_plane_color)
    axes[1, imeth].imshow(xz_plane_color)

    # add scale bar ------------------------------------------------------------
    if imeth == Nmeth - 1:
        tp = 0.05
        dict_scale_bar = {
            "pixel_size": pixel_size,
            "bar_length": 5,
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(Nx * tp), int(Ny * (1 - tp))),
        }
        add_scale_bar(axes[0, imeth], image=xy_plane, **dict_scale_bar)

    #  add zoom box ------------------------------------------------------------
    pos, size = (120, 155), 60
    if show_patch:
        show_box = True if imeth == Nmeth - 1 else False
        add_patch(
            axes[0, imeth],
            image=xy_plane_color,
            pos=pos,
            size=size,
            show_box=show_box,
            axes_lw=1,
            box_lw=1,
            box_color="white",
            percent=0.4,
        )

    # add title ----------------------------------------------------------------
    titles = ["Raw"] + methods_name
    pos_text = (
        xy_plane.shape[1] - int(img.shape[1] * 0.04),
        int(xy_plane.shape[0] * 0.04),
    )
    axes[0, imeth].text(pos_text[0], pos_text[1], titles[imeth], **dict_text_meth)

    # add plane label ----------------------------------------------------------
    if imeth == 0:
        pos_text = (
            int(xy_plane.shape[1] * 0.04),
            int(xy_plane.shape[0] * 0.96),
        )
        axes[0, imeth].text(pos_text[0], pos_text[1], "xy", **dict_text_plane)
        pos_text = (
            int(xz_plane.shape[1] * 0.04),
            int(xz_plane.shape[0] * 0.96),
        )
        axes[1, imeth].text(pos_text[0], pos_text[1], "xz", **dict_text_plane)


plt.savefig(
    os.path.join(path_fig, f"image_restored_compare_live3d_id_{id_sample_show}.png")
)
plt.savefig(
    os.path.join(path_fig, f"image_restored_compare_live3d_id_{id_sample_show}.svg")
)
