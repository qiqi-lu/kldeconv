"""
Dispaly the speed of different algorithms on simulation datasets.
"""

import os, pandas, seaborn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
path_root = "outputs/predictions/SimuMix3D-512-31-05-1-01"
path_figure = "outputs/figures/SimuMix3D-512-31-05-1-01"
os.makedirs(path_figure, exist_ok=True)

dict_methods_info = (
    # method name, path to time.xlsx
    (
        "RLN",
        "rln/SimuMix3D-128-31-05-1-01/n1_r1/time.xlsx",
        "#EC8860",
    ),
    (
        "KLD",
        "kernelnet/SimuMix3D-128-31-05-1-01/fp_knonw_bp_n1_r1/train_iter_2/time.xlsx",
        "#C23637",
    ),
)

# ------------------------------------------------------------------------------
num_methods = len(dict_methods_info)
methods_name = [info[0] for info in dict_methods_info]
methods_color = [info[2] for info in dict_methods_info]


# ------------------------------------------------------------------------------
# read data
# ------------------------------------------------------------------------------
# loop read data
df = pandas.DataFrame()
for method_name, path_time, _ in dict_methods_info:
    # read time
    df_time = pandas.read_excel(os.path.join(path_root, path_time))
    # add to df
    df[method_name] = df_time["time (s)"]

# print df
df = df.drop([0])
print(df)

# ------------------------------------------------------------------------------
# plot
# ------------------------------------------------------------------------------
nr, nc = 1, 1
dict_fig = dict(dpi=300, constrained_layout=True)
fig, ax = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)
data = df.values
data_mean = data.mean(axis=0)
data_std = data.std(axis=0)

ticks = list(np.linspace(0, 2, 21))
ax.set_yticks(ticks)
ax.set_yticklabels([f"{t:.2f}" for t in ticks])

# plot
dict_bar = dict(capsize=5, width=0.8)
ax.bar(
    methods_name,
    data_mean,
    # yerr=data_std,
    label=methods_name,
    color=methods_color,
    **dict_bar,
)

# set
ax.set_ylabel("Time (s)")
ax.set_xticks([])
ax.set_xticklabels([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_box_aspect(2)

path_save = os.path.join(path_figure, "time")
plt.savefig(path_save + ".png")
plt.savefig(path_save + ".svg")
