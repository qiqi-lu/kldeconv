"""
check each sample, show each sample.
"""

import numpy as np
import skimage.io as io
import os, pandas, tqdm
import matplotlib.pyplot as plt
from utils.data import win2linux, read_txt
import colorcet as ccet

datasets_list = (
    # --------------------------------------------------------------------------
    # "Microtubule2-3d-1024",
    # "Nuclear-pore-complex2-1024",
    # "biotisr-3d-factin-1",
    # "biotisr-3d-factin-2",
    # "biotisr-3d-mt-1",
    # "biotisr-3d-mt-2",
    # "biotisr-3d-mito-1",
    # "biotisr-3d-mito-2",
    # # --------------------------------------------------------------------------
    "F-actin-nonlinear-9",
    "F-actin-nonlinear-8",
    "F-actin-nonlinear-7",
    "F-actin-nonlinear-6",
    "F-actin-nonlinear-5",
    "F-actin-nonlinear-4",
    "F-actin-nonlinear-3",
    "F-actin-nonlinear-2",
    "F-actin-nonlinear-1",
    "Microtubules2-9",
    "Microtubules2-8",
    "Microtubules2-7",
    "Microtubules2-6",
    "Microtubules2-5",
    "Microtubules2-4",
    "Microtubules2-3",
    "Microtubules2-2",
    "Microtubules2-1",
    "CCPs-9",
    "CCPs-8",
    "CCPs-7",
    "CCPs-6",
    "CCPs-5",
    "CCPs-4",
    "CCPs-3",
    "CCPs-2",
    "CCPs-1",
    "F-actin-12",
    "F-actin-11",
    "F-actin-10",
    "F-actin-9",
    "F-actin-8",
    "F-actin-7",
    "F-actin-6",
    "F-actin-5",
    "F-actin-4",
    "F-actin-3",
    "F-actin-2",
    "F-actin-1",
    "ER-6",
    "ER-5",
    "ER-4",
    "ER-3",
    "ER-2",
    "ER-1",
    # --------------------------------------------------------------------------
    "biotisr-ccps-1",
    "biotisr-ccps-2",
    "biotisr-ccps-3",
    "biotisr-factin-1",
    "biotisr-factin-2",
    "biotisr-factin-3",
    "biotisr-factin-nonlinear-1",
    "biotisr-factin-nonlinear-2",
    "biotisr-factin-nonlinear-3",
    "biotisr-lysosomes-1",
    "biotisr-lysosomes-2",
    "biotisr-lysosomes-3",
    "biotisr-mt-1",
    "biotisr-mt-2",
    "biotisr-mt-3",
    "biotisr-mito-1",
    "biotisr-mito-2",
    "biotisr-mito-3",
    "deepbacs-ecoli",
    "deepbacs-ecoli-ave2",
    "deepbacs-saureus",
    "deepbacs-saureus-ave2",
    "w2s-0-sim-ave",
    "w2s-0-wf-ave-400",
    "w2s-1-sim-ave",
    "w2s-1-wf-ave-400",
    "w2s-2-sim-ave",
    "w2s-2-wf-ave-400",
)

# ------------------------------------------------------------------------------
path_test_excel = "datasets_test.xlsx"
path_train_excel = "datasets_train.xlsx"
df_info_test = pandas.read_excel(path_test_excel)
df_info_train = pandas.read_excel(path_train_excel)

dict_fig = dict(dpi=150, constrained_layout=True)
path_root_figure = os.path.join("outputs", "figures")

groups = ["test", "train"]

for dataset_id in datasets_list:
    for i_df, df_info in enumerate([df_info_test, df_info_train]):
        path_save = os.path.join(path_root_figure, dataset_id, "lr-hr", groups[i_df])

        if not os.path.exists(path_save):
            os.makedirs(path_save)
        info = df_info[df_info["id"] == dataset_id].iloc[0]

        path_lr = win2linux(info["path_lr"])
        path_hr = win2linux(info["path_hr"])
        path_index = win2linux(info["path_txt"])

        filenames = read_txt(path_index)
        num_samples = len(filenames)
        pbar = tqdm.tqdm(
            total=num_samples, desc=f"[INFO] {dataset_id} ({groups[i_df]})", ncols=80
        )
        for filename in filenames:
            path_lr_sample = os.path.join(path_lr, filename)
            path_hr_sample = os.path.join(path_hr, filename)

            img_lr = np.squeeze(io.imread(path_lr_sample))
            img_hr = np.squeeze(io.imread(path_hr_sample))

            if img_hr.ndim == 3:
                img_lr = img_lr[1]
                img_hr = img_hr[1]

            nr, nc = 1, 2
            fig, axes = plt.subplots(
                nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), **dict_fig
            )

            axes[0].imshow(img_lr, cmap=ccet.cm.fire)
            axes[0].set_title("LR")
            axes[1].imshow(img_hr, cmap=ccet.cm.fire)
            axes[1].set_title("HR")
            axes[0].axis("off")
            axes[1].axis("off")

            plt.savefig(os.path.join(path_save, filename.split(".")[0] + ".png"))
            plt.close()
            pbar.update(1)
        pbar.close()
