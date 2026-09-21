"""
Display convergence analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import os, pandas, tqdm
from utils.data import win2linux, NormalizePercentile
import utils.evaluation as eva
import skimage.io as io

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------
path_root = os.path.join("outputs", "predictions")
dataset_name_test, dataset_name_train, id_exp = (
    "SimuMix3D-512-31-05-1-01",
    "SimuMix3D-128-31-05-1-01",
    "fp_knonw_bp_n1_r1",
)
y_limits = [[25, 32], [0.7, 0.95], [0.7, 0.95]]

# dataset_name_test, dataset_name_train, id_exp = (
#     "SimuMix3D-512-31-0-0-1",
#     "SimuMix3D-128-31-0-0-1",
#     "fp_knonw_bp_n1_r1",
#     # "fp_knonw_bp_n80_r1",
# )
# y_limits = [[26, 34], [0.8, 1.0], [0.75, 1.0]]

# ------------------------------------------------------------------------------
# dataset_name_test, dataset_name_train, id_exp = (
#     "SimuMix3D-128-31-05-1-01",
#     "SimuMix3D-128-31-05-1-01",
#     "fp_knonw_bp_n3_r1",
# )
# y_limits = [[22, 31], [0.6, 0.90], [0.7, 0.90]]

# dataset_name_test, dataset_name_train, id_exp = (
#     "SimuMix3D-128-31-0-0-1",
#     "SimuMix3D-128-31-0-0-1",
#     "fp_knonw_bp_n80_r1",
# )
# y_limits = [[26, 36], [0.8, 1.0], [0.75, 1.0]]

# ------------------------------------------------------------------------------
sample_id = 0
recalculate_metrics = True
recalculate_metrics = False

num_iters = 100

results_info = (
    ("traditional", "Traditional", f"deconv_iter_{num_iters}_all.tif", "#647086"),
    ("gaussian", "Gaussian", f"deconv_iter_{num_iters}_all.tif", "#4D8FCB"),
    ("wiener-butterworth", "WB", f"deconv_iter_{num_iters}_all.tif", "#42B4B5"),
    ("kernelnet", "KLD", "y_pred_all.tif", "#C23637"),
)

# load ground truth
path_excel = "datasets_test.xlsx"
df_info = pandas.read_excel(path_excel)
info = df_info[df_info["id"] == dataset_name_test].iloc[0]
path_hr = win2linux(info["path_hr"])
path_lr = win2linux(info["path_lr"])
path_txt = win2linux(info["path_txt"])

# read filenames from path_txt
filenames = np.loadtxt(path_txt, dtype=str)
sample_name = filenames[0].split(".")[0]
# load ground truth and low-resolution images
img_hr = io.imread(os.path.join(path_hr, filenames[0]))
img_lr = io.imread(os.path.join(path_lr, filenames[0]))

# ------------------------------------------------------------------------------
path_figures = os.path.join(
    "outputs",
    "figures",
    "analysis_image",
    dataset_name_test,
    sample_name,
    "convergence_analysis",
)
os.makedirs(path_figures, exist_ok=True)

# ------------------------------------------------------------------------------
normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=3)


def preprocess(img):
    img = normalizer(img)
    img = np.clip(img, a_min=0.0, a_max=2.5)
    return img


data_range = 2.5
img_gt = preprocess(img_hr)
img_lr = preprocess(img_lr)

# load results and plot convergence curve
num_method = len(results_info)
metrics_all = []
for method_info in results_info:
    print("-" * 80)
    method_id, method_name, filename_res, _ = method_info
    print(f"[INFO] Process {method_name} ...")

    if method_id in ["traditional", "gaussian", "wiener-butterworth"]:
        path_res = os.path.join(
            path_root, dataset_name_test, method_id, sample_name, filename_res
        )
    elif method_id in ["kernelnet"]:
        path_res = os.path.join(
            path_root,
            dataset_name_test,
            method_id,
            dataset_name_train,
            id_exp,
            "train_iter_2_test_iter_100",
            sample_name,
            filename_res,
        )

    # calculate metrics
    if recalculate_metrics:
        print("[INFO] Load results ...")
        data = io.imread(path_res)
        num_iter = data.shape[0]

        if method_id in ["traditional", "gaussian", "wiener-butterworth"]:
            data = np.concatenate((img_lr[None, :, :, :], data), axis=0)
            num_iter += 1
        elif method_id in ["kernelnet"]:
            data = np.concatenate((img_lr[None, :, :, :], data[1:]), axis=0)

        metrics_meth = []
        pbar = tqdm.tqdm(
            range(num_iter), desc=f"[INFO] EVA {method_name}", leave=False, ncols=80
        )
        for i_iter in range(num_iter):
            img_pred = preprocess(data[i_iter])
            dict_met = dict(img_true=img_gt, img_test=img_pred)
            psnr = eva.PSNR(**dict_met, data_range=data_range)
            ssim = eva.MSSSIM(
                **dict_met, data_range=data_range, interp_sf=2, device="cuda:0"
            )
            zncc = eva.NCC(**dict_met)
            metrics_meth.append([psnr, ssim, zncc])
            pbar.update(1)
        pbar.close()

        print("[INFO] Save metrics to disk ...")
        np.save(
            os.path.join(path_figures, f"metrics_{method_id}.npy"),
            metrics_meth,
        )
        metrics_all.append(metrics_meth)
    else:
        print("[INFO] Load saved metrics ...")
        metrics_meth = np.load(os.path.join(path_figures, f"metrics_{method_id}.npy"))
        metrics_all.append(metrics_meth)

# ------------------------------------------------------------------------------
# plot convergence curve
nr, nc = 1, 3
dict_fig = dict(dpi=300, constrained_layout=True)
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)

metrics_name = ["PSNR", "MS-SSIM", "ZNCC"]
methods_name = [method_name for _, method_name, _, _ in results_info]
methods_color = [color for _, _, _, color in results_info]

for i_metric, metric_name in enumerate(metrics_name):
    ax = axes[i_metric]
    for i_method in range(num_method):
        data = np.array(metrics_all[i_method])[:, i_metric]
        ax.plot(
            data,
            label=methods_name[i_method],
            linewidth=1,
            color=methods_color[i_method],
            marker="o",
            markersize=1,
        )

        # add a line at second iteration
        ax.axhline(
            data[2], color=methods_color[i_method], linestyle="--", linewidth=0.5
        )

    ax.set_ylabel(metric_name)
    ax.set_xlabel("Iteration")
    ax.set_ylim(y_limits[i_metric])
axes[0].legend(fontsize=12, frameon=False, facecolor="none")

# save figures
plt.savefig(os.path.join(path_figures, "convergence_analysis.svg"))
plt.savefig(os.path.join(path_figures, "convergence_analysis.png"))
