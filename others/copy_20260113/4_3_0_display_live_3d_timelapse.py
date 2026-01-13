"""
Only for ZeroshotDeconvNet dataset.
Display the restored image at differen time points.
Generate the video of the restored image.
"""

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os, pandas, cv2, tqdm
import matplotlib as mpl
from utils.data import win2linux, read_txt, NormalizePercentile, interp
from utils.plot import render

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
dataset_names = (
    ("ZeroShotDeconvNet-mitosis-642", mpl.colormaps["gray"]),
    ("ZeroShotDeconvNet-mitosis-560", mpl.colormaps["afmhot"]),
)

methods_info = {
    "ZeroShotDeconvNet-mitosis-642": (
        ("WB", "wiener_butterworth", "deconv.tif"),
        ("KLD", "kernelnet\SimuMix3D-382-101-05-1-1-642\\n1_r1", "y_pred_all.tif"),
        # ("KLD-SS", "kernelnet\ZeroShotDeconvNet-mitosis-642\\n1_r1", "y_pred_all.tif"),
    ),
    "ZeroShotDeconvNet-mitosis-560": (
        ("WB", "wiener_butterworth", "deconv.tif"),
        ("KLD", "kernelnet\SimuMix3D-382-101-05-1-1-560\\n1_r1", "y_pred_all.tif"),
        # ("KLD-ss", "kernelnet\ZeroShotDeconvNet-mitosis-560\\n1_r1", "y_pred_all.tif"),
    ),
}

methods_show = ["WB", "KLD"]
percent = (
    ((0, 0), (99.99, 99.9)),  # Raw
    ((0, 0), (99.5, 99.5)),  # WB
    ((0, 0), (99.5, 99.5)),  # KLD
)
enable_generate_video = False

# ------------------------------------------------------------------------------
info_df = pandas.read_excel("datasets_test.xlsx")
path_prediction = os.path.join("outputs", "predictions")
path_fig = os.path.join("outputs", "figures")
path_video = os.path.join(path_fig, "video")
os.makedirs(path_video, exist_ok=True)

id_sample_show = [0, 346, 609, 700, 770, 901]
text_rt = (
    "Prophase",
    "Metaphase",
    "Anaphase",
    "Telophase",
    "Cytokinesis",
    "Interphase (G1)",
)

id_slice_xy_show = (65, 85)
id_slice_xz_show = (181, 201)
interval = 10  # s
id_sample_video = range(0, 1000, 4)

normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=2)
dict_clip = {"a_max": 2.5, "a_min": 0.0}

# ------------------------------------------------------------------------------
num_channels = len(dataset_names)
num_methods_show = len(methods_show)
num_timepoints_show = len(id_sample_show)
num_timepoints_video = len(id_sample_video)

# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
print("-" * 80)
img_deconv_mc = []
for i_channel in range(num_channels):
    ds_name = dataset_names[i_channel][0]
    methods = methods_info[ds_name]

    info = info_df[info_df["id"] == ds_name].iloc[0]

    path_lr = win2linux(info["path_lr"])
    path_txt = win2linux(info["path_txt"])
    filenames = read_txt(path_txt)

    img_deconv = []
    for id_sample in id_sample_show:
        imgs = []
        # load RAW image
        imgs.append(io.imread(os.path.join(path_lr, filenames[id_sample])))
        for i_method in range(num_methods_show):
            # load deconvolved image
            img = io.imread(
                os.path.join(
                    path_prediction,
                    ds_name,
                    win2linux(methods[i_method][1]),
                    f"sample_{id_sample}",
                    methods[i_method][2],
                )
            )
            imgs.append(img)
        img_deconv.append(imgs)
    img_deconv_mc.append(img_deconv)

# ------------------------------------------------------------------------------
img_deconv_mc = np.array(img_deconv_mc)
Nc, Nt, Nmeth, Nz, Ny, Nx = img_deconv_mc.shape

print(f"[INFO] Num of channel: {Nc}")
print(f"[INFO] Num of time point: {Nt}")
print(f"[INFO] Num of methods: {Nmeth}")
print(f"[INFO] Image shape: {(Nz, Ny, Nx)}")

# ------------------------------------------------------------------------------
# show image restored
# ------------------------------------------------------------------------------
# define color
nr, nc = Nmeth, Nt
dict_fig = dict(dpi=300, constrained_layout=True)

fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

text_time = [f"{t*interval/60:.1f} min" for t in id_sample_show]
title_methods = ["Raw"] + methods_show

dict_text_rt = dict(fontsize=14, color="white", ha="right", va="top")
dict_text_rb = dict(fontsize=14, color="white", ha="right", va="bottom")
dict_text_lb = dict(fontsize=14, color="white", ha="left", va="bottom")


for i_meth in range(Nmeth):
    for i_t in range(Nt):
        img_color_mc, cmaps = [], []
        for i_channel in range(Nc):
            cmaps.append(dataset_names[i_channel][1])

            img = img_deconv_mc[i_channel, i_t, i_meth]
            xy_plane = np.max(img[id_slice_xy_show[0] : id_slice_xy_show[1]], axis=0)
            img_color_mc.append(xy_plane)

        p = percent[i_meth]
        img_color = render(
            np.stack(img_color_mc, axis=-1), cmaps=cmaps, plow=p[0], phigh=p[1]
        )

        axes[i_meth, i_t].imshow(img_color)

        # add text -------------------------------------------------------------
        pos_text = (int(img_color.shape[0] * 0.04), int(img_color.shape[1] * 0.04))
        if i_meth == 0:
            axes[i_meth, i_t].text(
                img_color.shape[1] - pos_text[1],
                pos_text[0],
                text_rt[i_t],
                **dict_text_rt,
            )

            axes[i_meth, i_t].text(
                img_color.shape[1] - pos_text[1],
                img_color.shape[0] - pos_text[0],
                text_time[i_t],
                **dict_text_rb,
            )
        if i_t == 0:
            axes[i_meth, i_t].text(
                pos_text[1],
                img_color.shape[0] - pos_text[0],
                title_methods[i_meth],
                **dict_text_lb,
            )


plt.savefig(os.path.join(path_fig, "image_restored_live.png"))
plt.savefig(os.path.join(path_fig, "image_restored_live.svg"))

# ------------------------------------------------------------------------------
#                                     create video
# ------------------------------------------------------------------------------
# reload_video_data = True
reload_video_data = False
if enable_generate_video:
    if reload_video_data:
        img_deconv_mc_xy, img_deconv_mc_zx = [], []
        for i_channel in range(num_channels):
            ds_name = dataset_names[i_channel][0]
            methods = methods_info[ds_name]

            info = info_df[info_df["id"] == ds_name].iloc[0]

            path_lr = win2linux(info["path_lr"])
            path_txt = win2linux(info["path_txt"])
            filenames = read_txt(path_txt)

            # ----------------------------------------------------------------------
            img_deconv_xy, img_deconv_zx = [], []
            pbar = tqdm.tqdm(
                total=num_timepoints_video, desc=f"[INFO] {ds_name}", ncols=80
            )
            for id_sample in id_sample_video:
                pbar.update(1)
                imgs_xy, imgs_zx = [], []

                # load RAW image
                img = io.imread(os.path.join(path_lr, filenames[id_sample]))
                img_xy = np.max(img[id_slice_xy_show[0] : id_slice_xy_show[1]], axis=0)
                img_zx = np.max(
                    img[:, id_slice_xz_show[0] : id_slice_xz_show[1]], axis=1
                )
                img_zx = interp(img_zx, ps_xy=92.6, ps_z=200)
                imgs_xy.append(img_xy)
                imgs_zx.append(img_zx)

                for i_method in range(num_methods_show):
                    # load deconvolved image
                    img = io.imread(
                        os.path.join(
                            path_prediction,
                            ds_name,
                            win2linux(methods[i_method][1]),
                            f"sample_{id_sample}",
                            methods[i_method][2],
                        )
                    )
                    img_xy = np.max(
                        img[id_slice_xy_show[0] : id_slice_xy_show[1]], axis=0
                    )
                    img_zx = np.max(
                        img[:, id_slice_xz_show[0] : id_slice_xz_show[1]], axis=1
                    )
                    img_zx = interp(img_zx, ps_xy=92.6, ps_z=200)
                    imgs_xy.append(img_xy)
                    imgs_zx.append(img_zx)

                img_deconv_xy.append(imgs_xy)
                img_deconv_zx.append(imgs_zx)
            # ----------------------------------------------------------------------
            img_deconv_mc_xy.append(img_deconv_xy)
            img_deconv_mc_zx.append(img_deconv_zx)
            pbar.close()
            # --------------------------------------------------------------------------
        img_deconv_mc_xy = np.array(img_deconv_mc_xy)
        img_deconv_mc_zx = np.array(img_deconv_mc_zx)

        np.savez(
            os.path.join(path_video, "image_restored_live.npz"),
            xy=img_deconv_mc_xy,
            zx=img_deconv_mc_zx,
        )
    else:
        data = np.load(os.path.join(path_video, "image_restored_live.npz"))
        img_deconv_mc_xy = data["xy"]
        img_deconv_mc_zx = data["zx"]

    print("-" * 80)
    print("[INFO] Generate video ...")
    print(f"[INFO] xy plane: {img_deconv_mc_xy.shape}")
    print(f"[INFO] xz plane: {img_deconv_mc_zx.shape}")

    # convert array to video
    tag_plane = ["xy", "zx"]
    tag_meth = ["raw"] + methods_show
    for ip, data in enumerate([img_deconv_mc_xy, img_deconv_mc_zx]):
        Nc, Nt, Nmeth, Ny, Nx = data.shape
        for im in range(Nmeth):
            video_writer = cv2.VideoWriter(
                os.path.join(
                    path_video,
                    f"image_restored_live_{tag_meth[im]}_{tag_plane[ip]}.mp4",
                ),
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=10,
                frameSize=(Nx, Ny),
            )
            p = percent[im]
            for i_t in range(Nt):
                img = data[:, i_t, im].transpose(1, 2, 0)
                img_color = render(
                    img, cmaps=cmaps, plow=p[0], phigh=p[1], rgb_type="bgr"
                )
                # write
                video_writer.write(img_color)
            video_writer.release()
