"""
Display the PCC of the backward kernels learned from different numbers of
training samples.
"""

import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os, pandas
from utils.evaluation import generation_combinations
from scipy import stats

plt.rcParams["svg.fonttype"] = "none"

# ------------------------------------------------------------------------------
datasets_name = [
    "SimuMix3D-128-31-0-0-1",
    "SimuMix3D-128-31-05-1-1",
    "SimuMix3D-128-31-05-1-03",
    "SimuMix3D-128-31-05-1-01",
]


path_prediction = os.path.join("outputs", "predictions")
path_fig = os.path.join("outputs", "figures")
os.makedirs(path_fig, exist_ok=True)

# ------------------------------------------------------------------------------
# rubost to training sample
num_data = [1, 2, 3]
id_repeat = [1, 2, 3]

kb = []  # backward kernels
noise_level = ["NF", "20", "15", "10"]

for dataset in datasets_name:
    path_kernel = os.path.join(path_prediction, dataset, "kernelnet", dataset)
    tmp = []
    for bc in num_data:
        tmpp = []
        for re in id_repeat:
            path_tmp = os.path.join(
                path_kernel, f"fp_knonw_bp_n{bc}_r{re}", "kernel", "kernel_bp.tif"
            )
            tmpp.append(io.imread(path_tmp))
        tmp.append(tmpp)
    kb.append(tmp)
kb = np.array(kb)
# ------------------------------------------------------------------------------
# calculate metric value
N_nl, N_data, N_rep = kb.shape[0:3]
print(
    f"[INFO] (N_noise_level, N_data_num, N_repeat) : {kb.shape}"
)  # dataset, num of train data, num of repeat

pearson = np.zeros(shape=(N_nl, N_data, N_rep))
combines = generation_combinations(N_rep, k=2)

for i in range(N_nl):
    for j in range(N_data):
        for ic, cb in enumerate(combines):
            pearson[i, j, ic] = stats.pearsonr(
                x=kb[i, j, cb[0]].flatten(), y=kb[i, j, cb[1]].flatten()
            )[0]

pearson_mean = pearson.mean(axis=-1)
pearson_std = pearson.std(axis=-1)

# ------------------------------------------------------------------------------
# display PCC between backward kernels
dict_fig = {"dpi": 300, "constrained_layout": True}
dict_line = {"linewidth": 0.5, "capsize": 2, "elinewidth": 0.5, "capthick": 0.5}

nr, nc = 1, 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)

colors = ["black", "red", "green", "blue"]  # color for each noise level

for i in range(N_nl):
    axes.errorbar(
        x=num_data, y=pearson_mean[i], yerr=pearson_std[i], color=colors[0], **dict_line
    )
    axes.plot(
        num_data, pearson_mean[i], ".", color=colors[i], label="SNR=" + noise_level[i]
    )
axes.legend(edgecolor="white", fontsize="x-small")
axes.set_ylabel("PCC")
axes.set_ylim([0.94, 1])
axes.set_box_aspect(1)
axes.set_xticks(ticks=num_data, labels=num_data)
axes.set_xlabel("Number of samples")

plt.savefig(os.path.join(path_fig, "kb_pcc.png"))
plt.savefig(os.path.join(path_fig, "kb_pcc.svg"))

# ------------------------------------------------------------------------------
# save pearson value into excel
excel_file = os.path.join(path_fig, "kb_pcc.xlsx")
if os.path.exists(excel_file):
    os.remove(excel_file)
with pandas.ExcelWriter(excel_file, mode="w") as writer:
    for i in range(N_nl):
        df = pandas.DataFrame(pearson[i], columns=num_data)
        df.to_excel(writer, sheet_name=noise_level[i])
