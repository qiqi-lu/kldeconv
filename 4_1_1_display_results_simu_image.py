"""
Display the restored images and profile lines.
Display the metrics of restored images in each test datastes.
"""

import matplotlib.pyplot as plt
import utils.evaluation as eva
from utils.data import read_txt, win2linux, NormalizePercentile, linear_transform
from utils.plot import add_scale_bar, add_significant_star
import skimage.io as io
import numpy as np
import os, pandas, torch, tqdm
from utils import evaluation as eva
from skimage.measure import profile_line
from scipy.stats import wilcoxon, pearsonr
from methods.deconvolution import Convolution

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
num_samples_statistic = 20  # the number of sample used for statistics evaluation
# num_samples_statistic = 10
enable_normalization = True  # default
# enable_normalization = False
enable_image_metrics = False
enable_image_metrics = True  # default
enable_profile = False
enable_profile = True  # default

enable_rsersp = True  # default
enable_rsersp = False

method_subgroup = "different_methods"
# method_subgroup = "along_iter"
# method_subgroup = "along_num_img_train"

# ------------------------------------------------------------------------------
#  dataset name | id_sample | slice (xy) | slice (xz) | line (xy) | line (xz) | patch (x, y, size)
# ------------------------------------------------------------------------------
data_info = (
    # "SimuMix3D-128-31-05-1-01",
    "SimuMix3D-512-31-05-1-01",
    "SimuMix3D-128-31-05-1-01",
    0,
    64,  # id_slice_xy
    70,  # id_slice_xz
    ((38, 34), (56, 19)),
    ((86, 44), (86, 64)),
    (165, 40, 80),  # 512
)

path_experiments = {
    "KLD@1": f"fp_knonw_bp_n1_r1/train_iter_1",
    "KLD@2": f"fp_knonw_bp_n1_r1/train_iter_2",
    "KLD@3": f"fp_knonw_bp_n1_r1/train_iter_3",
    "KLD@4": f"fp_knonw_bp_n1_r1/train_iter_4",
    "KLD@5": f"fp_knonw_bp_n1_r1/train_iter_5",
    "KLD@2 (1)": f"fp_knonw_bp_n1_r1/train_iter_2",
    "KLD@2 (2)": f"fp_knonw_bp_n2_r1/train_iter_2",
    "KLD@2 (3)": f"fp_knonw_bp_n3_r1/train_iter_2",
    "KLD@2 (4)": f"fp_knonw_bp_n4_r1/train_iter_2",
    "KLD@2 (5)": f"fp_knonw_bp_n5_r1/train_iter_2",
}
# ------------------------------------------------------------------------------
# name | title | iter | color
# ------------------------------------------------------------------------------
methods_info_dict = {
    "different_methods": (
        ("raw", "Raw", 2, "#647086"),
        ("traditional", "Traditional@2", 2, "#4D8FCB"),
        ("traditional", "Traditional@30", 30, "#0068A9"),
        # ("gaussian", "Gaussian", 2, "#D2E6F0"),
        # ("butterworth", "Butterworth", 2, "#FADCC8"),
        ("wiener-butterworth", "WB@2", 2, "#42B4B5"),
        # ("wiener-butterworth", "WB@3", 3, "#42B4B5"),
        ("rln", "RLN", 2, "#EC8860"),
        ("kernelnet", "KLD@2", 2, "#C23637"),
        # ("kernelnet", "KLD@5", 5, "#C23637"),
        # ("kernelnet_ss", "KLD-ss", 2, "#F3B95F"),
        ("gt", "GT", 2, "#212C3E"),
    ),
    "along_iter": (
        ("kernelnet", "KLD@1", 1, "#F9C9C7"),
        ("kernelnet", "KLD@2", 2, "#EA9A9D"),
        ("kernelnet", "KLD@3", 3, "#D95D5B"),
        ("kernelnet", "KLD@4", 4, "#C23637"),
        ("kernelnet", "KLD@5", 5, "#912322"),
        ("gt", "GT", 2, "#212C3E"),
    ),
    "along_num_img_train": (
        ("kernelnet", "KLD@2 (1)", 2, "#E8D0E6"),
        ("kernelnet", "KLD@2 (2)", 2, "#CDA0CB"),
        ("kernelnet", "KLD@2 (3)", 2, "#B271AB"),
        ("kernelnet", "KLD@2 (4)", 2, "#9E4589"),
        ("kernelnet", "KLD@2 (5)", 2, "#6E2769"),
        ("gt", "GT", 2, "#212C3E"),
    ),
}


# ------------------------------------------------------------------------------
methods_info = methods_info_dict[method_subgroup]
(
    dataset_name_test,
    dataset_name_train,
    id_sample,
    id_slice_xy,
    id_slice_zx,
    line_xy,
    line_xz,
    patch_xys,
) = data_info

eps = 0.000001

# ------------------------------------------------------------------------------
path_predictions = os.path.join("outputs", "predictions")

info_df = pandas.read_excel("datasets_test.xlsx")
info = info_df[info_df["id"] == dataset_name_test].iloc[0]

path_txt = win2linux(info["path_txt"])
path_lr = win2linux(info["path_lr"])
path_hr = win2linux(info["path_hr"])
path_psf = win2linux(info["path_psf"])

filenames = read_txt(path_txt)
pixel_size = info["pixel_size"] / 1000  # um
ratio = info["ratio"]

path_figure_root = os.path.join(
    "outputs", "figures", "analysis_image", dataset_name_test
)
path_figure = os.path.join(
    path_figure_root,
    filenames[id_sample].split(".")[0],  # the folder name is the id of sample
    method_subgroup,
)
os.makedirs(path_figure, exist_ok=True)

# ------------------------------------------------------------------------------
num_methods = len(methods_info)
print("-" * 80)
print(f"[INFO] Show methods : {[name[0] for name in methods_info]}")
print(f"[INFO] Enable normalization : {enable_normalization}")
normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=3)


def preprocess(img):
    img = normalizer(img)
    img = np.clip(img, a_min=0.0, a_max=2.5)
    return img


data_range = 2.5 if enable_normalization else None


# ------------------------------------------------------------------------------
# load results
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] load results ...")
# load deconvolved images ------------------------------------------------------
imgs_all = []
for i_meth in range(num_methods):
    name_meth, title_meth, iter, _ = methods_info[i_meth]
    if name_meth == "raw":
        path_sample = os.path.join(path_lr, filenames[id_sample])
        img = io.imread(path_sample).astype(np.float32)
        imgs_all.append(img)

    elif name_meth == "gt":
        path_sample = os.path.join(path_hr, filenames[id_sample])
        img = io.imread(path_sample).astype(np.float32) * ratio
        imgs_all.append(img)

    elif name_meth in ["kernelnet", "kernelnet_ss"]:
        path_exp = win2linux(path_experiments[title_meth])
        path_sample = os.path.join(
            path_predictions,
            dataset_name_test,
            name_meth,
            dataset_name_train,
            path_exp,
            filenames[id_sample].split(".")[0],
        )
        y_pred_all = io.imread(os.path.join(path_sample, "y_pred_all.tif"))
        y_pred = y_pred_all[iter]
        imgs_all.append(y_pred)
    elif name_meth in ["rln"]:
        path_sample = os.path.join(
            path_predictions,
            dataset_name_test,
            name_meth,
            dataset_name_train,
            "n1_r1",
            filenames[id_sample].split(".")[0],
            f"y_pred.tif",
        )
        img = io.imread(path_sample).astype(np.float32)
        imgs_all.append(img)
    else:
        path_sample = os.path.join(
            path_predictions,
            dataset_name_test,
            name_meth,
            filenames[id_sample].split(".")[0],
            f"deconv_iter_{iter}.tif",
        )
        img = io.imread(path_sample).astype(np.float32)
        imgs_all.append(img)

imgs_all = np.array(imgs_all)
print("[INFO] results shape : ", imgs_all.shape)

# load PSF ---------------------------------------------------------------------
# load ground truth PSF
print("-" * 80)
print("[INFO] load PSF...")
print(f"[INFO] PSF (GT) path : {path_psf}")
PSF_gt = io.imread(path_psf).astype(np.float32)
print(f"[INFO] PSF (GT) shape : {PSF_gt.shape}")
print(f"[INFO] PSF (GT) sum : {PSF_gt.sum():.2f}")
conv3d = Convolution(torch.tensor(PSF_gt), padding_mode="reflect", domain="fft")


def forward_project(img):
    return conv3d(torch.tensor(img)).cpu().detach().numpy()


# image normalization ----------------------------------------------------------
# whether normalize the image before calculating the metrics
if enable_normalization:
    print("[INFO] image normalization...")
    imgs_all_norm = np.zeros_like(imgs_all)
    for i in range(imgs_all.shape[0]):
        imgs_all_norm[i] = preprocess(imgs_all[i])
else:
    imgs_all_norm = imgs_all

# ------------------------------------------------------------------------------
# show the restored images for one sample
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] plot restored images ...")
# ------------------------------------------------------------------------------
Nz, Ny, Nx = imgs_all.shape[1:]
lxy_start, lxy_end = line_xy[0], line_xy[1]
lzx_start, lzx_end = line_xz[0], line_xz[1]

dict_fig = dict(dpi=300, constrained_layout=True)
dict_text = dict(color="white", fontsize=15)
dict_text_rt = dict(ha="right", va="top", x=0.96, y=0.96, **dict_text)
dict_text_lt = dict(ha="left", va="top", x=0.04, y=0.96, **dict_text)
dict_text_lb = dict(ha="left", va="bottom", x=0.04, y=0.04, **dict_text)
dict_text_rb = dict(ha="right", va="bottom", x=0.96, y=0.04, **dict_text)
dict_line = dict(color="red", linewidth=1.5)
dict_image = dict(cmap="gray", vmin=0, vmax=0.9)

# ------------------------------------------------------------------------------
nr, nc = 2, num_methods
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

img_gt = imgs_all_norm[-1]
pbar = tqdm.tqdm(total=num_methods, desc="[INFO] PLOT", ncols=80, leave=False)
for i_meth in range(num_methods):
    ax = axes[:, i_meth]
    name_meth, name, iter, color = methods_info[i_meth]

    img = imgs_all_norm[i_meth]

    ax[0].imshow(img[id_slice_xy], **dict_image)
    ax[1].imshow(img[:, id_slice_zx, :], **dict_image)

    # add text -----------------------------------------------------------------
    ax[0].text(s=name, transform=ax[0].transAxes, **dict_text_lt)

    if i_meth == 0:
        ax[0].text(s="xy", transform=ax[0].transAxes, **dict_text_rt)
        ax[1].text(s="xz", transform=ax[1].transAxes, **dict_text_rt)

        if enable_profile:
            # add lines
            ax[0].plot(
                (lxy_start[0], lxy_end[0]), (lxy_start[1], lxy_end[1]), **dict_line
            )
            ax[1].plot(
                (lzx_start[0], lzx_end[0]), (lzx_start[1], lzx_end[1]), **dict_line
            )

    # add metrics --------------------------------------------------------------
    if enable_image_metrics:
        if i_meth != num_methods - 1:
            dict_met_tmp = dict(img_true=img_gt, img_test=img)
            psnr = eva.PSNR(**dict_met_tmp, data_range=data_range)
            # ssim = eva.SSIM(**dict_met_tmp, data_range=data_range)
            ssim = eva.MSSSIM(**dict_met_tmp, data_range=data_range, interp_sf=2)
            ssim = ssim * 100
            zncc = eva.NCC(**dict_met_tmp)

            ax[0].text(
                s=f"{psnr:.2f} | {zncc:.2f}", transform=ax[0].transAxes, **dict_text_lb
            )

    # add scale bar
    if i_meth == num_methods - 1:
        dict_scale_bar = {
            "pixel_size": pixel_size,
            "bar_length": 5,  # um
            "bar_height": 0.01,
            "bar_color": "white",
            "pos": (int(Ny * 0.05), int(Nx * (1 - 0.05))),
        }
        add_scale_bar(ax[0], image=img, **dict_scale_bar)

    # add a patch at the right bottom corner
    px, py, ps = patch_xys
    patch = img[id_slice_xy, py : py + ps, px : px + ps]
    # add box in the image
    ax[0].add_patch(
        plt.Rectangle((px, py), ps, ps, edgecolor="red", linewidth=1, facecolor="none")
    )
    ax_patch = ax[0].inset_axes(
        [0.6, 0.0, 0.4, 0.4], transform=ax[0].transAxes, zorder=10
    )
    ax_patch.imshow(patch, **dict_image)
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

    pbar.update(1)
pbar.close()


plt.savefig(os.path.join(path_figure, f"img_restored.png"))
plt.savefig(os.path.join(path_figure, f"img_restored.svg"))

# os._exit(0)
# ------------------------------------------------------------------------------
# show fft of image
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] plot fft of image...")
# fft of images
imgs_all_fft = []
for i_meth in range(num_methods):
    img = imgs_all[i_meth]
    img_fft = np.fft.fftshift(np.fft.fftn(img))
    img_fft = np.log(np.abs(img_fft) + 1)
    img_fft = img_fft / img_fft.max()
    imgs_all_fft.append(img_fft)

imgs_all_fft = np.array(imgs_all_fft)

id_slice_center_xy = int(Nz // 2)
id_slice_center_zx = int(Ny // 2)
# dict_image_fft = dict(cmap="hot", vmin=0.4, vmax=1)
dict_image_fft = dict(cmap="hot", vmin=0.2, vmax=1)

# ------------------------------------------------------------------------------
# show center slice of xy and xz plane
nr, nc = 2, num_methods
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

for i_meth in range(num_methods):
    ax = axes[:, i_meth]
    name_meth, name, iter, color = methods_info[i_meth]
    img_fft = imgs_all_fft[i_meth]
    ax[0].imshow(img_fft[id_slice_center_xy], **dict_image_fft)
    ax[1].imshow(img_fft[:, id_slice_center_zx, :], **dict_image_fft)
    # add text -----------------------------------------------------------------
    ax[0].text(s=name, transform=ax[0].transAxes, **dict_text_rt)
    if i_meth == 0:
        ax[0].text(s="xy", transform=ax[0].transAxes, **dict_text_lb)
        ax[1].text(s="xz", transform=ax[1].transAxes, **dict_text_lb)

plt.savefig(os.path.join(path_figure, f"img_fft.png"))
plt.savefig(os.path.join(path_figure, f"img_fft.svg"))

# os._exit(0)
# ------------------------------------------------------------------------------
# show resolution-scale error map
# ------------------------------------------------------------------------------
# here we use the real psf to convole the restored image to get the
# resolution-scale error
if enable_rsersp:
    print("-" * 80)
    print("[INFO] plot resolution-scale error...")
    dict_image = dict(cmap="gray", vmin=0, vmax=25)
    dict_image_error = dict(cmap="plasma", vmin=0, vmax=12.5)

    # --------------------------------------------------------------------------
    nr, nc = 4, num_methods
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    [ax.set_axis_off() for ax in axes.ravel()]
    for i_meth in range(num_methods - 1):
        ax = axes[:, i_meth]
        name_meth, name, iter, color = methods_info[i_meth]
        img_ref = forward_project(imgs_all[-1])
        if i_meth == 0:
            ax[0].imshow(img_ref[id_slice_xy], **dict_image)
            ax[2].imshow(img_ref[:, id_slice_zx, :], **dict_image)
        else:
            # the image have be conv for resolution scale
            img_rescale = forward_project(imgs_all[i_meth])
            if "rln" in name_meth:
                img_rescale = linear_transform(img_rescale, img_ref)
            img_err = np.abs(img_rescale - img_ref)

            # calculate RSE and RSP (resolution scaled Pearson correlation)
            RSE = np.sqrt(np.mean(img_err**2))
            RSP = pearsonr(img_ref.flatten(), img_rescale.flatten())[0]

            ax[0].imshow(img_rescale[id_slice_xy], **dict_image)
            ax[1].imshow(img_err[id_slice_xy], **dict_image_error)
            ax[2].imshow(img_rescale[:, id_slice_zx, :], **dict_image)
            ax[3].imshow(img_err[:, id_slice_zx, :], **dict_image_error)

            # add metric value -----------------------------------------------------
            ax[1].text(
                s=f"RSE:{RSE:.3f}\nRSP:{RSP:.3f}",
                transform=ax[1].transAxes,
                **dict_text_rt,
            )
        # add text -----------------------------------------------------------------
        if i_meth == 0:
            ax[0].text(s="REF", transform=ax[0].transAxes, **dict_text_rt)
            ax[0].text(s="xy", transform=ax[0].transAxes, **dict_text_lb)
            ax[2].text(s="xz", transform=ax[2].transAxes, **dict_text_lb)
        else:
            ax[0].text(
                s=f"{name}\n(convolved)", transform=ax[0].transAxes, **dict_text_rt
            )

    # add scale bar
    dict_scale_bar = {
        "pixel_size": pixel_size,
        "bar_length": 5,  # um
        "bar_height": 0.01,
        "bar_color": "white",
        "pos": (int(Ny * 0.05), int(Nx * (1 - 0.05))),
    }
    add_scale_bar(axes[0, -2], image=img_ref, **dict_scale_bar)

    # show colorbar at the last column
    fig.colorbar(axes[0, -2].get_images()[0], cax=axes[0, -1], orientation="vertical")
    fig.colorbar(axes[1, -2].get_images()[0], cax=axes[1, -1], orientation="vertical")
    for ax in axes[:2, -1]:
        ax.set_axis_on()
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_yticklabels([0, 5, 10, 15, 20, 25])
        ax.tick_params(labelsize=16)
        ax.set_aspect(0.2)
    axes[0, -1].set_ylim(dict_image["vmin"], dict_image["vmax"])
    axes[1, -1].set_ylim(dict_image_error["vmin"], dict_image_error["vmax"])

    plt.savefig(os.path.join(path_figure, f"img_resolution_scale_error.png"))
    plt.savefig(os.path.join(path_figure, f"img_resolution_scale_error.svg"))

# os._exit(0)
# ------------------------------------------------------------------------------
# profile line
# ------------------------------------------------------------------------------
if enable_profile:
    print("-" * 80)
    print("[INFO] plot profiel lines ...")
    dict_profile = dict(linewidth=1.0, linestyle="-")
    dict_profile_gt = dict(linewidth=1.0, linestyle="--")
    dict_profile_text = dict(
        color="black", fontsize=12, ha="left", va="top", x=0.05, y=0.95
    )

    profiles_xy, profiles_xz = [], []

    # ------------------------------------------------------------------------------
    nr, nc = 1, 2
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    for i_line in range(2):
        ax = axes[i_line]

        ax.tick_params(direction="in")
        ax.set_xlabel("Distance (pixel)")
        # del top and right axis
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ticks_y = np.linspace(0, 2, 5)
        ax.set_yticks(ticks_y)
        ax.set_yticklabels(ticks_y)

        ticks_x = [int(x) for x in np.linspace(0, 100, 21)]
        ax.set_xticks(ticks_x)
        ax.set_xticklabels(ticks_x)

        # --------------------------------------------------------------------------
        if i_line == 0:
            print("[INFO] plot line in xy plane")
            line_start = (lxy_start[1], lxy_start[0])
            line_end = (lxy_end[1], lxy_end[0])

            for i_meth in range(num_methods):
                name_meth, name, iter, color = methods_info[i_meth]
                img = imgs_all_norm[i_meth]
                profile = profile_line(
                    img[id_slice_xy], line_start, line_end, linewidth=1
                )
                if i_meth == num_methods - 1:
                    ax.plot(profile, label=name, color=color, **dict_profile_gt)
                else:
                    ax.plot(profile, label=name, color=color, **dict_profile)
                profiles_xy.append(profile.tolist())
            profiles_xy = np.array(profiles_xy)
            ax.set_ylim((0, profiles_xy.max() + 0.1))
            ax.set_ylabel("Normalized intensity")

            # add text -------------------------------------------------------------
            ax.text(s="xy", transform=ax.transAxes, **dict_profile_text)
            ax.legend(loc="best", fontsize=6, frameon=False)

        # --------------------------------------------------------------------------
        if i_line == 1:
            print("[INFO] plot line in zx plane")
            line_start = (lzx_start[1], lzx_start[0])
            line_end = (lzx_end[1], lzx_end[0])

            for i_meth in range(num_methods):
                name_meth, name, iter, color = methods_info[i_meth]
                img = imgs_all_norm[i_meth]
                profile = profile_line(
                    img[:, id_slice_zx, :], line_start, line_end, linewidth=1
                )
                if i_meth == num_methods - 1:
                    ax.plot(profile, label=name, color=color, **dict_profile_gt)
                else:
                    ax.plot(profile, label=name, color=color, **dict_profile)
                profiles_xz.append(profile.tolist())
            profiles_xz = np.array(profiles_xz)
            ax.set_ylim((0, profiles_xz.max() + 0.1))

            # add text -------------------------------------------------------------
            ax.text(s="xz", transform=ax.transAxes, **dict_profile_text)
        # --------------------------------------------------------------------------

        ax.set_xlim((0, None))
        # set the axes to square
        ax.set_box_aspect(1)

    plt.savefig(os.path.join(path_figure, f"img_restored_profile.png"))
    plt.savefig(os.path.join(path_figure, f"img_restored_profile.svg"))

# os._exit(0)
# ------------------------------------------------------------------------------
# load all samples and calculate the statistics
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] load all samples and calculate metrics...")
metrics_names = ["PSNR", "MS-SSIM", "ZNCC"]
if enable_rsersp:
    metrics_names += ["RSE", "RSP"]
num_metrics = len(metrics_names)
metrics_all_samples = np.zeros((num_methods - 1, num_samples_statistic, num_metrics))

# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=num_samples_statistic, desc="loading samples", ncols=80)
for i_sample in range(num_samples_statistic):
    pbar.update(1)
    filename = filenames[i_sample]

    imgs_meth = []
    for i_meth in range(num_methods):
        name_meth, name, iter, color = methods_info[i_meth]
        path_root_meth = os.path.join(path_predictions, dataset_name_test, name_meth)

        # load the result of each method for each sample -----------------------
        if name_meth == "raw":
            path_sample = os.path.join(path_lr, filename)
            img = io.imread(path_sample).astype(np.float32)
            imgs_meth.append(img)

        elif name_meth == "gt":
            path_sample = os.path.join(path_hr, filename)
            img = io.imread(path_sample).astype(np.float32) * ratio
            imgs_meth.append(img)

        elif name_meth in ["kernelnet", "kernelnet_ss"]:
            path_exp = win2linux(path_experiments[name])
            path_sample = os.path.join(
                path_root_meth, dataset_name_train, path_exp, filename.split(".")[0]
            )
            y_pred_all = io.imread(os.path.join(path_sample, "y_pred_all.tif"))
            y_pred = y_pred_all[iter]
            imgs_meth.append(y_pred)
        elif name_meth in ["rln"]:
            path_sample = os.path.join(
                path_root_meth,
                dataset_name_train,
                "n1_r1",
                filename.split(".")[0],
                f"y_pred.tif",
            )
            img = io.imread(path_sample).astype(np.float32)
            imgs_meth.append(img)
        else:  # conventional methods
            path_sample = os.path.join(
                path_root_meth, filename.split(".")[0], f"deconv_iter_{iter}.tif"
            )
            img = io.imread(path_sample).astype(np.float32)
            imgs_meth.append(img)

    # calculate the metrics for each sample ------------------------------------
    for i_meth in range(num_methods - 1):
        img_gt_ori = imgs_meth[-1]
        img_pred_ori = imgs_meth[i_meth]

        # used for calculate PSNR, SSIM, and ZNCC
        if enable_normalization:
            img_gt = preprocess(img_gt_ori)
            img_pred = preprocess(img_pred_ori)
        else:
            img_gt = img_gt_ori
            img_pred = img_pred_ori

        dict_met = dict(img_true=img_gt, img_test=img_pred)
        psnr = eva.PSNR(**dict_met, data_range=data_range)
        # ssim = eva.SSIM(dict_met, data_range=data_range)
        ssim = eva.MSSSIM(**dict_met, data_range=data_range, interp_sf=2)
        zncc = eva.NCC(**dict_met)

        # ----------------------------------------------------------------------
        if enable_rsersp:
            # used for calculate RSE and RSP
            img_ref = forward_project(img_gt_ori)
            img_pred_fp = forward_project(img_pred_ori)
            if "rln" in methods_info[i_meth][0]:
                img_pred_fp = linear_transform(img_pred_fp, img_ref)

            rse = np.sqrt(np.mean((img_ref - img_pred_fp) ** 2))
            rsp = pearsonr(img_ref.flatten(), img_pred_fp.flatten())[0]

            metrics_all_samples[i_meth, i_sample, :] = [psnr, ssim, zncc, rse, rsp]
        else:
            metrics_all_samples[i_meth, i_sample, :] = [psnr, ssim, zncc]

pbar.close()
print("[INFO] metrics shape : ", metrics_all_samples.shape)

# ------------------------------------------------------------------------------
# plot metrics of all samples
# ------------------------------------------------------------------------------
print("[INFO] plot metrics of all samples ...")
yticks_metrics_dict = {
    "different_methods": {
        "PSNR": list(np.linspace(0, 35, 15)),
        "MS-SSIM": list(np.linspace(0, 1, 11)),
        "ZNCC": list(np.linspace(0, 1, 21)),
        "RSE": list(np.linspace(0, 5, 51)),
        "RSP": list(np.linspace(0, 1, 51)),
    },
    "along_iter": {
        "PSNR": list(np.linspace(0, 40, 41)),
        "MS-SSIM": list(np.linspace(0, 1, 101)),
        "ZNCC": list(np.linspace(0, 1, 101)),
        "RSE": list(np.linspace(0, 10, 51)),
        "RSP": list(np.linspace(0, 1, 101)),
    },
    "along_num_img_train": {
        "PSNR": list(np.linspace(0, 40, 81)),
        "MS-SSIM": list(np.linspace(0, 1, 101)),
        "ZNCC": list(np.linspace(0, 1, 101)),
        "RSE": list(np.linspace(0, 10, 51)),
        "RSP": list(np.linspace(0, 1, 101)),
    },
}
yticks_metrics = yticks_metrics_dict[method_subgroup]

# calculate pvalue -------------------------------------------------------------
# compare each method with the reference method (last method)
test_pairs = [(i, num_methods - 2) for i in range(num_methods - 2)]
pvalues = np.zeros((num_metrics, len(test_pairs)))
for i_metric in range(num_metrics):
    for i_pair, pair in enumerate(test_pairs):
        pvalues[i_metric, i_pair] = wilcoxon(
            metrics_all_samples[pair[0], :, i_metric],
            metrics_all_samples[pair[1], :, i_metric],
            alternative="two-sided",
        )[1]
print("[INFO] pvalues shape : ", pvalues.shape)

# plot -------------------------------------------------------------------------
nr, nc = 1, num_metrics
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
colors_all = [color for _, _, _, color in methods_info]
labels_all = [name for _, name, _, _ in methods_info]
dict_bar = dict(capsize=5, width=0.8)

for i_metric in range(num_metrics):
    ax = axes[i_metric]
    metric_name = metrics_names[i_metric]
    if metric_name in ["PSNR", "MS-SSIM", "ZNCC"]:
        data = metrics_all_samples[:, :, i_metric]
        pvs = pvalues[i_metric]
        colors = colors_all[:-1]
        labels = labels_all[:-1]
        x_pos = np.arange(num_methods - 1)
        test_pairs_show = test_pairs
    elif metric_name in ["RSE", "RSP"]:
        data = metrics_all_samples[1:, :, i_metric]
        pvs = pvalues[i_metric, 1:]
        colors = colors_all[1:-1]
        labels = labels_all[1:-1]
        x_pos = np.arange(num_methods - 2)
        test_pairs_show = test_pairs[1:]
    else:
        raise ValueError(
            f"Unknown metric name: {metric_name}. "
            "Please use one of PSNR, MS-SSIM, ZNCC, RSE, RSP."
        )

    # --------------------------------------------------------------------------
    data_std = data.std(axis=1)
    data_mean = data.mean(axis=1)
    data_max, data_min = data.max(), data.min()
    data_range = data_max - data_min

    # --------------------------------------------------------------------------
    ax.bar(x_pos, data_mean, yerr=data_std, color=colors, label=labels, **dict_bar)

    # --------------------------------------------------------------------------
    if i_metric == 0 or metric_name in ["RSE", "RSP"]:
        ax.legend(loc="best", fontsize=6, frameon=False)

    ax.set_xticks([])
    ax.set_xticklabels([])

    ticks = yticks_metrics[metric_name]
    ax.set_yticks(ticks)
    if metric_name == "RSP":
        ax.set_yticklabels([f"{x:.3f}" for x in ticks])
    else:
        ax.set_yticklabels([f"{x:.2f}" for x in ticks])

    ax.set_ylabel(metric_name)

    y_lim = (data_min - data_range * 0.1, data_max + data_range * 0.1)
    ax.set_ylim(y_lim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1)

    # add pvalue marker --------------------------------------------------------
    for i_pair, pair in enumerate(test_pairs_show):
        pv = pvs[i_pair]
        i_pos = pair[0]
        if metric_name in ["RSE", "RSP"]:
            i_pos = i_pos - 1
        star_x = x_pos[i_pos]
        star_y = data_mean[i_pos] + data_std[i_pos] + 0.02 * (y_lim[1] - y_lim[0])
        add_significant_star(ax, star_x, star_y, pv)

path_save = os.path.join(path_figure_root, method_subgroup)
os.makedirs(path_save, exist_ok=True)
plt.savefig(os.path.join(path_save, "img_restored_metrics.png"))
plt.savefig(os.path.join(path_save, "img_restored_metrics.svg"))

# save source data -------------------------------------------------------------
# save seach metric to a sheet of excel
print("-" * 80)
print("[INFO] save source data...")
path_excel = os.path.join(path_save, "img_restored_metrics.xlsx")
# if excel is exist, delete it
if os.path.exists(path_excel):
    os.remove(path_excel)
writer = pandas.ExcelWriter(path_excel, engine="xlsxwriter")
methods_name = [name for _, name, _, _ in methods_info]
for i_metric in range(num_metrics):
    data = metrics_all_samples[:, :, i_metric]
    df = pandas.DataFrame(columns=methods_name[:-1])
    for i_meth in range(num_methods - 1):
        df[methods_name[i_meth]] = data[i_meth]
    df.to_excel(writer, sheet_name=metrics_names[i_metric], index=True)
writer.close()
# ------------------------------------------------------------------------------
