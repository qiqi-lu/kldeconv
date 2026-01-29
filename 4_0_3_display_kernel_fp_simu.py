"""
Display the forward kernels learned from data with different noise levels.
Display the RMSE of forward kernels.
"""

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os, pandas
from utils.evaluation import RMSE
from utils.data import win2linux

plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------------------
data_info = (
    # (dataset_name, (id_num_data, id_repeat))
    ("SimuMix3D-256-31-0-0-1", (1, 1)),
    ("SimuMix3D-256-31-05-1-1", (1, 1)),
    ("SimuMix3D-256-31-05-1-03", (1, 1)),
    ("SimuMix3D-256-31-05-1-01", (1, 1)),
)
noise_level = ["NF", "20", "15", "10"]

# ------------------------------------------------------------------------------
datasets_name = [info[0] for info in data_info]

info_df = pandas.read_excel("datasets_test.xlsx")
info = info_df[info_df["id"] == datasets_name[0]].iloc[0]
path_psf = win2linux(info["path_psf"])
pixel_size = info["pixel_size"] / 1000  # um

path_prediction = os.path.join("outputs", "predictions")
path_fig = os.path.join("outputs", "figures", "analysis_kernel")
os.makedirs(path_fig, exist_ok=True)

kf_true = io.imread(path_psf).astype(np.float32)
print(f"[INFO] PSF path : {path_psf}")
print(f"[INFO] PSF shape : {kf_true.shape}")

# ------------------------------------------------------------------------------
# load estimated kernels
# ------------------------------------------------------------------------------
num_data = [1, 2, 3]
id_repeat = [1, 2, 3]

kf_est = []  # backward kernels
for dataset in datasets_name:
    path_kernel = os.path.join(path_prediction, dataset, "kernelnet", dataset)
    tmp = []
    for bc in num_data:
        tmpp = []
        for re in id_repeat:
            path_tmp = os.path.join(
                path_kernel,
                f"fp_n{bc}_r{re}_bp_n{bc}_r{re}",
                "kernel",
                "kernel_fp.tif",
            )
            tmpp.append(io.imread(path_tmp))
        tmp.append(tmpp)
    kf_est.append(tmp)
kf_est = np.array(kf_est)

print(f"[INFO] (N_noise_level, N_data_num, N_repeat) : {kf_est.shape}")

# ------------------------------------------------------------------------------
# display forward kernels
# ------------------------------------------------------------------------------
dict_fig = {"dpi": 600, "constrained_layout": True}

# ------------------------------------------------------------------------------
num_noise_level = len(noise_level)

Nz, Ny, Nx = kf_true.shape

dict_kernel = {"cmap": "hot", "vmin": 0, "vmax": kf_true.max()}
dict_text_lt = {
    "color": "white",
    "fontsize": 24,
    "ha": "left",
    "va": "top",
    "x": 0.05,
    "y": 0.95,
}
dict_text_lb = {
    "color": "white",
    "fontsize": 24,
    "ha": "left",
    "va": "bottom",
    "x": 0.05,
    "y": 0.05,
}
dict_text_rb = {
    "color": "white",
    "fontsize": 24,
    "ha": "right",
    "va": "bottom",
    "x": 0.95,
    "y": 0.05,
}

id_slice_xy = Nz // 2
id_slice_zx = Ny // 2

# ------------------------------------------------------------------------------
nr, nc = num_noise_level + 1, 2
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
[ax.set_axis_off() for ax in axes.ravel()]

axes[0, 0].imshow(kf_true[id_slice_xy], **dict_kernel)
axes[0, 1].imshow(kf_true[:, id_slice_zx, :], **dict_kernel)

axes[0, 0].text(s="GT", transform=axes[0, 0].transAxes, **dict_text_lt)
axes[0, 0].text(s="xy", transform=axes[0, 0].transAxes, **dict_text_lb)
axes[0, 0].text(s="k$_f$", transform=axes[0, 0].transAxes, **dict_text_rb)
axes[0, 1].text(s="zx", transform=axes[0, 1].transAxes, **dict_text_lb)

for i_nl in range(num_noise_level):
    id_num_data, id_repeat = data_info[i_nl][1]
    ker = kf_est[i_nl, id_num_data, id_repeat]
    axes[i_nl + 1, 0].imshow(ker[id_slice_xy], **dict_kernel)
    axes[i_nl + 1, 1].imshow(ker[:, id_slice_zx, :], **dict_kernel)

    text = noise_level[i_nl] if i_nl == 0 else (noise_level[i_nl] + " dB")
    axes[i_nl + 1, 0].text(
        s=text, transform=axes[i_nl + 1, 0].transAxes, **dict_text_lt
    )

plt.savefig(os.path.join(path_fig, "kf_noise_level.png"))
plt.savefig(os.path.join(path_fig, "kf_noise_level.svg"))

# ------------------------------------------------------------------------------
# calculate metric value
# ------------------------------------------------------------------------------
N_nl, N_data, N_rep = kf_est.shape[0:3]
rmse = np.zeros(shape=(N_nl, N_data, N_rep))

for i in range(N_nl):
    for j in range(N_data):
        for k in range(N_rep):
            rmse[i, j, k] = RMSE(kf_true, kf_est[i, j, k])

rmse_mean = rmse.mean(axis=-1)
rmse_std = rmse.std(axis=-1)

# ------------------------------------------------------------------------------
# display RMSE of forward kernels
# ------------------------------------------------------------------------------
dict_line = {"linewidth": 0.5, "capsize": 2, "elinewidth": 0.5, "capthick": 0.5}

# ------------------------------------------------------------------------------
nr, nc = 1, 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)

colors = ["black", "red", "green", "blue"]  # color for each noise level

for i in range(N_nl):
    axes.errorbar(
        x=num_data,
        y=rmse_mean[i],
        yerr=rmse_std[i],
        color=colors[0],
        **dict_line,
    )
    axes.plot(
        num_data,
        rmse_mean[i],
        ".",
        color=colors[i],
        label="SNR=" + noise_level[i],
        zorder=10,
    )
axes.legend(fontsize="xx-small", frameon=False, loc="best")
axes.set_ylabel("RMSE (%)")
axes.set_box_aspect(1)
axes.set_xticks(ticks=num_data, labels=num_data)
axes.set_xlabel("Number of samples")

plt.savefig(os.path.join(path_fig, "kf_rmse.png"))
plt.savefig(os.path.join(path_fig, "kf_rmse.svg"))

# ------------------------------------------------------------------------------
# save source data
excel_file = os.path.join(path_fig, "kf_rmse.xlsx")
if os.path.exists(excel_file):
    os.remove(excel_file)
with pandas.ExcelWriter(excel_file, mode="w") as writer:
    for i in range(N_nl):
        df = pandas.DataFrame(rmse[i], columns=num_data)
        df.to_excel(writer, sheet_name=noise_level[i])
# ------------------------------------------------------------------------------
