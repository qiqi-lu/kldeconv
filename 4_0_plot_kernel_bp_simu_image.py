"""
Plot the profile of the backward kernel and its fft.
"""

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os, pandas
import utils.data as utils_data

# ------------------------------------------------------------------------------
dataset_name, pixel_size = "SimuMix3D_128", 162.5  # nm

params_data_train = {
    "noise-free": {
        "std_gauss": 0,
        "poisson": 0,
        "ratio": 1,
        "num_data": 80,
        "id_repeat": 1,
    },
    "noisy": {
        "std_gauss": 0.5,
        "poisson": 1,
        "ratio": 0.3,
        "num_data": 3,
        "id_repeat": 1,
    },
}

path_prediction = os.path.join("outputs", "figures-v1", dataset_name)
path_fig = os.path.join("outputs", "figures", dataset_name, "kernels")
os.makedirs(path_fig, exist_ok=True)

print("[INFO] load data from:", path_prediction)
print("[INFO] save figures to:", path_fig)

# ------------------------------------------------------------------------------
# load kernels
# ------------------------------------------------------------------------------
print("[INFO] load kernels ...")
ker_BP = []
# conventional backward kernels
methods = ["traditional", "wiener_butterworth"]
for meth in methods:
    ptmp = params_data_train["noise-free"]
    path_ker_bp = os.path.join(
        path_prediction,
        f"scale_1_gauss_{ptmp['std_gauss']}_poiss_{ptmp['poisson']}_ratio_{ptmp['ratio']}",
        "sample_0",
        meth,
        "deconv_bp.tif",
    )
    ker_BP.append(io.imread(path_ker_bp))

# learned backeard kernels
noise_level = ["noise-free", "noisy"]
for nl in noise_level:
    ptmp = params_data_train[nl]
    path_fig_data = os.path.join(
        path_prediction,
        f"scale_1_gauss_{ptmp['std_gauss']}_poiss_{ptmp['poisson']}_ratio_{ptmp['ratio']}",
        f"kernels_bc_{ptmp['num_data']}_re_{ptmp['id_repeat']}",
    )
    ker_BP.append(io.imread(os.path.join(path_fig_data, "kernel_bp.tif")))

# true forward kernel
ker_true = io.imread(os.path.join(path_fig_data, "kernel_true.tif"))

# ------------------------------------------------------------------------------
y = io.imread(
    os.path.join(
        path_prediction,
        f"scale_1_gauss_0_poiss_0_ratio_1",
        f"sample_0",
        "kernelnet",
        "y.tif",
    )
)

s_fft = y.shape

dict_fig = {"dpi": 300, "constrained_layout": True}

# ------------------------------------------------------------------------------
# show backward kernel planes
# ------------------------------------------------------------------------------
print("[INFO] plot backward kernels (plane) ...")
nr, nc = 3, 8
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]


def show(ker_fp, ker_bp, axes, s=None, title=""):
    dict_ker = {"cmap": "hot", "vmin": 0.0}

    ker_fp_fft = utils_data.fft_n(ker_fp, s=s)
    ker_bp_fft = utils_data.fft_n(ker_bp, s=s)
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


show(ker_true, ker_BP[0], axes=axes[:, 0:2], s=s_fft, title="Traditional")
show(ker_true, ker_BP[1], axes=axes[:, 2:4], s=s_fft, title="WB")
show(ker_true, ker_BP[2], axes=axes[:, 4:6], s=s_fft, title="KLD (NF)")
show(ker_true, ker_BP[3], axes=axes[:, 6:], s=s_fft, title="KLD (N)")

dict_text_spa = {"x": 1, "y": ker_true.shape[-1] - 2, "color": "white", "fontsize": 24}
dict_text_fre = {"x": 8, "y": s_fft[-1] - 10, "color": "white", "fontsize": 24}
axes[0, 0].text(s=r"$xy$", **dict_text_spa)
axes[0, 1].text(s=r"$xz$", **dict_text_spa)
axes[1, 0].text(s=r"$k_x$$k_y$", **dict_text_fre)
axes[1, 1].text(s=r"$k_x$$k_z$", **dict_text_fre)
axes[2, 0].text(s=r"$k_x$$k_y$", **dict_text_fre)
axes[2, 1].text(s=r"$k_x$$k_z$", **dict_text_fre)

plt.savefig(os.path.join(path_fig, "kernel_bp.png"))
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(os.path.join(path_fig, "kernel_bp.svg"))

# ------------------------------------------------------------------------------
# plot FFT of backward kernels
# ------------------------------------------------------------------------------
print("[INFO] plot profile of the fft of backward kernels ...")
nr, nc = 2, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)


def plot_profile(axes, ker_fp, ker_bp, s=None, color=None, label=None):
    dict_ker_profile = {"color": color, "label": label, "linewidth": 1}

    ker_fp_fft = utils_data.fft_n(ker_fp, s=s)
    ker_bp_fft = utils_data.fft_n(ker_bp, s=s)
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


axes[0, 0].axhline(y=0.0, color="gray", lw=1.0)
axes[1, 0].axhline(y=0.0, color="gray", lw=1.0)

methods_color = ["black", "#6895D2", "#D04848", "#F3B95F"]
methods_name = ["Traditional", "WB", "KLD (NF)", "KLD (N)"]

all_lines = []
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

# save lines into excel --------------------------------------------------------
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
# ------------------------------------------------------------------------------

axes[0, 0].legend(edgecolor="white", fontsize="x-small")
axes[0, 0].set_xlabel(r"Pixel$_x$")
axes[1, 0].set_xlabel(r"Pixel$_z$")

ticks = [0.0, 0.05, 0.1, 0.15]
for ax in axes[:, 0].ravel():
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_ylabel("value")

ticks = [0, 1, 2, 3, 4, 5, 6]
for ax in axes[:, 1:].ravel():
    ax.set_xlim([0, None])
    ax.set_ylim([0, 6.5])
    ax.set_xlabel("Frequency (k$_x$, nm$^{-1}$)")
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)

axes[0, 2].set_ylim([0, 1.55])
axes[1, 2].set_ylim([0, 1.75])
ticks = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
for ax in axes[:, 2].ravel():
    ax.set_xlim([0, None])
    ax.set_xlabel("Frequency (k$_z$, nm$^{-1}$)")
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)

ticks = [0, 20, 40, 60]
ticklables = [0]
for tick in ticks[1:]:
    ticklables.append(f"1/{round(pixel_size*128/tick)}")
for ax in axes[:, 1:].ravel():
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklables)

for ax in axes.ravel():
    ax.tick_params(axis="both", which="major", length=3, width=1)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.spines[["right", "top"]].set_visible(False)

plt.savefig(os.path.join(path_fig, "kernel_bp_fft"))
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(os.path.join(path_fig, "kernel_bp_fft.svg"))
