"""
DIsplay the backward kernels of different methods.
Display the profile of the backward kernel and its fft.
"""

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os, pandas
from utils.data import fft_n, win2linux

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
# pixel_size = 162.5  # nm
path_kernel_learned = (
    (
        "Traditional",
        "SimuMix3D-128-31-0-0-1/traditional/kernel/ker_bp.tif",
        "black",
    ),
    (
        "WB",
        "SimuMix3D-128-31-0-0-1/wiener-butterworth/kernel/ker_bp.tif",
        "#6895D2",
    ),
    (
        "KLD (NF)",
        "SimuMix3D-128-31-0-0-1/kernelnet/SimuMix3D-128-31-0-0-1/fp_knonw_bp_n80_r_1/kernel/kernel_bp.tif",
        "#D04848",
    ),
    (
        "KLD (NF)",
        "SimuMix3D-128-31-05-1-03/kernelnet/SimuMix3D-128-31-05-1-03/fp_knonw_bp_n3_r1/kernel/kernel_bp.tif",
        "#F3B95F",
    ),
)

path_kernel_true = "SimuMix3D-128-31-05-1-03/kernelnet/SimuMix3D-128-31-05-1-03/fp_knonw_bp_n3_r1/kernel/kernel_true.tif"
path_image_y = "SimuMix3D-128-31-05-1-03/kernelnet/SimuMix3D-128-31-05-1-03/fp_knonw_bp_n3_r1/10/y.tif"

path_kernel_true = win2linux(path_kernel_true)
path_image_y = win2linux(path_image_y)

# ------------------------------------------------------------------------------
path_prediction = os.path.join("outputs", "predictions")
path_fig = os.path.join("outputs", "figures", "analysis_kernel", "backward_kernel")
os.makedirs(path_fig, exist_ok=True)

info_df = pandas.read_excel("datasets_test.xlsx")
info = info_df[info_df["id"] == "SimuMix3D-128-31-05-1-03"].iloc[0]
pixel_size = info["pixel_size"]
pixel_size_um = pixel_size / 1000

print(f"[INFO] Load data from: {path_prediction}")
print(f"[INFO] Save figures to: {path_fig}")
print(f"[INFO] Pixel size: {pixel_size_um} um")

methods_name = [path[0] for path in path_kernel_learned]
methods_color = [path[2] for path in path_kernel_learned]


# ------------------------------------------------------------------------------
# load kernels
# ------------------------------------------------------------------------------
print("[INFO] load kernels ...")
ker_BP = []
for path in path_kernel_learned:
    path = win2linux(path[1])
    ker_BP.append(io.imread(os.path.join(path_prediction, path)))

# true forward kernel
ker_true = io.imread(os.path.join(path_prediction, path_kernel_true))

# ------------------------------------------------------------------------------
# show backward kernel planes
# ------------------------------------------------------------------------------
y = io.imread(os.path.join(path_prediction, path_image_y))
s_fft = y.shape
dict_fig = {"dpi": 300, "constrained_layout": True}

# ------------------------------------------------------------------------------
print("[INFO] plot backward kernels (plane) ...")
nr, nc = 3, 8
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

dict_text = dict(color="white", fontsize=24)
dict_text_lb = dict(x=0.04, y=0.04, ha="left", va="bottom", **dict_text)
dict_text_rt = dict(x=0.96, y=0.96, ha="right", va="top", **dict_text)
dict_ker = {"cmap": "hot", "vmin": 0.0}


def show(ker_fp, ker_bp, axes, s=None):

    ker_fp_fft = fft_n(ker_fp, s=s)
    ker_bp_fft = fft_n(ker_bp, s=s)
    N_kb = ker_bp.shape
    N_kf_ft = ker_fp_fft.shape
    a = np.abs(ker_bp_fft)
    b = np.abs(ker_fp_fft * ker_bp_fft)

    axes[0, 0].imshow(ker_bp[N_kb[0] // 2], vmax=ker_bp.max(), **dict_ker)
    axes[0, 1].imshow(ker_bp[:, N_kb[1] // 2, :], vmax=ker_bp.max(), **dict_ker)
    axes[1, 0].imshow(a[N_kf_ft[0] // 2], vmax=a.max(), **dict_ker)
    axes[1, 1].imshow(a[:, N_kf_ft[1] // 2, :], vmax=a.max(), **dict_ker)
    axes[2, 0].imshow(b[N_kf_ft[0] // 2], vmax=b.max(), **dict_ker)
    axes[2, 1].imshow(b[:, N_kf_ft[1] // 2, :], vmax=b.max(), **dict_ker)


for i in range(len(methods_name)):
    show(ker_true, ker_BP[i], axes=axes[:, i * 2 : i * 2 + 2], s=s_fft)
    axes[0, i * 2].text(
        s=methods_name[i], transform=axes[0, i * 2].transAxes, **dict_text_rt
    )

axes[0, 0].text(s="$xy$", transform=axes[0, 0].transAxes, **dict_text_lb)
axes[0, 1].text(s="$xz$", transform=axes[0, 1].transAxes, **dict_text_lb)
axes[1, 0].text(s="$k_x$$k_y$", transform=axes[1, 0].transAxes, **dict_text_lb)
axes[1, 1].text(s="$k_x$$k_z$", transform=axes[1, 1].transAxes, **dict_text_lb)
axes[2, 0].text(s="$k_x$$k_y$", transform=axes[2, 0].transAxes, **dict_text_lb)
axes[2, 1].text(s="$k_x$$k_z$", transform=axes[2, 1].transAxes, **dict_text_lb)

plt.savefig(os.path.join(path_fig, "kernel_bp.png"))
plt.savefig(os.path.join(path_fig, "kernel_bp.svg"))

# ------------------------------------------------------------------------------
# plot FFT of backward kernels
# ------------------------------------------------------------------------------
print("-" * 80)
print("[INFO] plot profile of the fft of backward kernels ...")
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)


def plot_profile(axes, ker_fp, ker_bp, s=None, color=None, label=None):
    dict_ker_profile = {"color": color, "label": label, "linewidth": 1}

    ker_fp_fft = fft_n(ker_fp, s=s)
    ker_bp_fft = fft_n(ker_bp, s=s)
    N_kb = ker_bp.shape
    N_kf_ft = ker_fp_fft.shape
    a = np.abs(ker_bp_fft)
    b = np.abs(ker_fp_fft * ker_bp_fft)

    line_1 = ker_bp[N_kb[0] // 2, N_kb[1] // 2, :]
    line_2 = a[N_kf_ft[0] // 2, N_kf_ft[1] // 2, N_kf_ft[2] // 2 :]
    line_3 = b[N_kf_ft[0] // 2, N_kf_ft[1] // 2, N_kf_ft[2] // 2 :]

    axes[0, 0].plot(line_1, **dict_ker_profile)
    axes[0, 1].plot(line_2, **dict_ker_profile)
    axes[0, 2].plot(line_3, **dict_ker_profile)

    line_4 = ker_bp[:, N_kb[1] // 2, N_kb[2] // 2]
    line_5 = a[N_kf_ft[0] // 2 :, N_kf_ft[1] // 2, N_kf_ft[2] // 2]
    line_6 = b[N_kf_ft[0] // 2 :, N_kf_ft[1] // 2, N_kf_ft[2] // 2]

    axes[1, 0].plot(line_4, **dict_ker_profile)
    axes[1, 1].plot(line_5, **dict_ker_profile)
    axes[1, 2].plot(line_6, **dict_ker_profile)

    lines = [line_1, line_2, line_3, line_4, line_5, line_6]
    return lines


for ax in [axes[0, 0], axes[1, 0]]:
    ax.axhline(y=0.0, color="gray", lw=1.0, linestyle="--")


all_lines = []  # collect the value of all lines
for i in range(len(methods_name)):
    lines = plot_profile(
        axes,
        ker_fp=ker_true,
        ker_bp=ker_BP[i],
        s=s_fft,
        color=methods_color[i],
        label=methods_name[i],
    )
    all_lines.append(lines)

# ------------------------------------------------------------------------------
axes[0, 0].legend(frameon=False, fontsize="x-small")
axes[0, 0].set_xlabel("Pixel$_x$")
axes[1, 0].set_xlabel("Pixel$_z$")

ticks = [0.0, 0.05, 0.1, 0.15]
for ax in axes[:, 0].ravel():
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_ylabel("value")

ticks = [0, 2, 4, 6]
for ax in axes[:, 1].ravel():
    ax.set_xlim([0, None])
    ax.set_ylim([0, 6.5])
    ax.set_xlabel("Frequency (k$_x$, nm$^{-1}$)")
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)

axes[0, 2].set_ylim([0, 1.55])
axes[1, 2].set_ylim([0, 1.75])
ticks = [0.0, 0.5, 1.0, 1.5]
for ax in axes[:, 2].ravel():
    ax.set_xlim([0, None])
    ax.set_xlabel("Frequency (k$_z$, nm$^{-1}$)")
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)

# add frequency labels
ticks = [0, 20, 40, 60]
ticklables = [0]
Nx = y.shape[-1]
for tick in ticks[1:]:
    ticklables.append(f"1/{round(pixel_size*Nx/tick)}")
for ax in axes[:, 1:].ravel():
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklables)

for ax in axes.ravel():
    ax.tick_params(axis="both", which="major", length=3, width=1)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.spines[["right", "top"]].set_visible(False)

plt.savefig(os.path.join(path_fig, "kernel_bp_fft.png"))
plt.savefig(os.path.join(path_fig, "kernel_bp_fft.svg"))

# save source data -------------------------------------------------------------
excel_file = os.path.join(path_fig, "kernel_bp_fft.xlsx")
if os.path.exists(excel_file):
    os.remove(excel_file)
with pandas.ExcelWriter(excel_file, mode="w") as writer:
    for i, sh in enumerate(
        ["pixel_x", "k_x", "k_x (mul)", "pixel_z", "k_z", "k_z (mul)"]
    ):
        tmp = []
        for j in range(len(all_lines)):
            tmp.append(all_lines[j][i])
        tmp = np.array(tmp)
        df = pandas.DataFrame(tmp)
        # add index column
        df.index = methods_name
        df.to_excel(writer, sheet_name=sh, header=False)
