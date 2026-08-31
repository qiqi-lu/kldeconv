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
    ("NF", "SimuMix3D-128-31-0-0-1", "black"),
    ("20", "SimuMix3D-128-31-05-1-1", "red"),
    ("15", "SimuMix3D-128-31-05-1-03", "green"),
    ("10", "SimuMix3D-128-31-05-1-01", "blue"),
]

noise_level = [x[0] for x in datasets_name]
colors = [x[2] for x in datasets_name]

# ------------------------------------------------------------------------------
path_prediction = os.path.join("outputs", "predictions")
path_figure = os.path.join("outputs", "figures", "analysis_kernel", "backward_kernel")
os.makedirs(path_figure, exist_ok=True)

# ------------------------------------------------------------------------------
# load backward kernels
# ------------------------------------------------------------------------------
num_data = [1, 2, 3]
id_repeat = [1, 2, 3]

kb = []  # backward kernels

for dataset in datasets_name:
    dataset = dataset[1]
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
# ------------------------------------------------------------------------------
N_nl, N_data, N_rep = kb.shape[0:3]
print(
    f"[INFO] (N_noise_level, N_data_num, N_repeat) : {kb.shape}"
)  # dataset, num of train data, num of repeat

combines = generation_combinations(N_rep, k=2)
num_combines = len(combines)
pearson = np.zeros(shape=(N_nl, N_data, num_combines))

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
# ------------------------------------------------------------------------------
dict_fig = {"dpi": 300, "constrained_layout": True}
dict_line = {
    "linewidth": 0.5,
    "capsize": 2,
    "elinewidth": 0.5,
    "capthick": 0.5,
    "color": "black",
}

# ------------------------------------------------------------------------------
nr, nc = 1, 1
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)


for i in range(N_nl):
    axes.errorbar(x=num_data, y=pearson_mean[i], yerr=pearson_std[i], **dict_line)
    axes.plot(
        num_data,
        pearson_mean[i],
        ".",
        color=colors[i],
        label=f"SNR={noise_level[i]}",
        zorder=10,
    )
axes.legend(fontsize="small", frameon=False, loc="best")
axes.set_ylabel("PCC")
axes.set_ylim([0.94, 1])
axes.set_box_aspect(1)
axes.set_xticks(ticks=num_data, labels=num_data)
axes.set_xlabel("Number of samples")

plt.savefig(os.path.join(path_figure, "kb_pcc.png"))
plt.savefig(os.path.join(path_figure, "kb_pcc.svg"))

# ------------------------------------------------------------------------------
# save source data
excel_file = os.path.join(path_figure, "kb_pcc.xlsx")
if os.path.exists(excel_file):
    os.remove(excel_file)
with pandas.ExcelWriter(excel_file, mode="w") as writer:
    for i in range(N_nl):
        df = pandas.DataFrame(pearson[i], columns=num_data)
        df.to_excel(writer, sheet_name=noise_level[i])
