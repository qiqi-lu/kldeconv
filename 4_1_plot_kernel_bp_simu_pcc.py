import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import os
import utils.data as utils_data
from utils.evaluation import generation_combinations
from scipy import stats

# ------------------------------------------------------------------------------
dataset_name = "SimuMix3D_128"

path_prediction = os.path.join("outputs", "figures-v1", dataset_name)
path_fig = os.path.join("outputs", "figures", dataset_name, "kernels")
os.makedirs(path_fig, exist_ok=True)

# ------------------------------------------------------------------------------
# rubost to training sample
params_data_train = {
    "NF": {"std_gauss": 0, "poisson": 0, "ratio": 1},
    "20": {"std_gauss": 0.5, "poisson": 1, "ratio": 1},
    "15": {"std_gauss": 0.5, "poisson": 1, "ratio": 0.3},
    "10": {"std_gauss": 0.5, "poisson": 1, "ratio": 0.1},
}
para_data = [
    [0, 0, 1],
    [0.5, 1, 1],
    [0.5, 1, 0.3],
    [0.5, 1, 0.1],
]  # std_gauss, poisson, ratio
num_data = [1, 2, 3]
id_repeat = [1, 2, 3]

kb = []  # backward kernels
noise_level = ["NF", "20", "15", "10"]

for nl in noise_level:
    para = params_data_train[nl]
    path_kernel = os.path.join(
        path_prediction,
        f"scale_1_gauss_{para['std_gauss']}_poiss_{para['poisson']}_ratio_{para['ratio']}",
    )
    tmp = []
    for bc in num_data:
        tmpp = []
        for re in id_repeat:
            tmpp.append(
                io.imread(
                    os.path.join(
                        path_kernel, f"kernels_bc_{bc}_re_{re}", "kernel_bp.tif"
                    )
                )
            )
        tmp.append(tmpp)
    kb.append(tmp)
kb = np.array(kb)
# ------------------------------------------------------------------------------
# calculate metric value
N_nl, N_data, N_rep = kb.shape[0:3]
print(kb.shape)  # dataset, num of train data, num of repeat

pearson = np.zeros(shape=(N_nl, N_data, N_rep))
ratio = [1, 1, 0.3, 0.1]
combines = generation_combinations(N_rep, k=2)

for i in range(N_nl):
    for j in range(N_data):
        for ic, cb in enumerate(combines):
            pearson[i, j, ic] = stats.pearsonr(
                x=kb[i, j, cb[0]].flatten(), y=kb[i, j, cb[1]].flatten()
            )[0]
print(pearson)

pearson_mean = pearson.mean(axis=-1)
pearson_std = pearson.std(axis=-1)
print("[INFO] mean:", pearson_mean)
# ------------------------------------------------------------------------------
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

plt.savefig(os.path.join(path_fig, "kb.png"))
plt.rcParams["svg.fonttype"] = "none"
plt.savefig(os.path.join(path_fig, "kb.svg"))
