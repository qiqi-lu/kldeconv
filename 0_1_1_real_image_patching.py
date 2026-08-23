"""
Pathcing the image for deep learning network training.
1. clip the value in the image to [0, None]
2. normalize the image with percentile normalization, but not clip the value to [0, 1].
3. patching

The path of the data are saved in the Excel file "datasets_test.xlsx".

KLDeconv does not use this preprocessing, and use the whole image to do training.
"""

import os, tqdm, pandas, json
import numpy as np
import skimage.io as io
from utils.data import win2linux, read_txt, normalization

# ------------------------------------------------------------------------------
# select the dataset names used for patching.
datasets_name = (
    # "F-actin-nonlinear-9",
    # "F-actin-nonlinear-8",
    # "F-actin-nonlinear-7",
    # "F-actin-nonlinear-6",
    # "F-actin-nonlinear-5",
    # "F-actin-nonlinear-4",
    # "F-actin-nonlinear-3",
    # "F-actin-nonlinear-2",
    # "F-actin-nonlinear-1",
    # "Microtubules2-9",
    # "Microtubules2-8",
    # "Microtubules2-7",
    # "Microtubules2-6",
    # "Microtubules2-5",
    # "Microtubules2-4",
    # "Microtubules2-3",
    # "Microtubules2-2",
    # "Microtubules2-1",
    # "CCPs-9",
    # "CCPs-8",
    # "CCPs-7",
    # "CCPs-6",
    # "CCPs-5",
    # "CCPs-4",
    # "CCPs-3",
    # "CCPs-2",
    # "CCPs-1",
    # "F-actin-12",
    # "F-actin-11",
    # "F-actin-10",
    # "F-actin-9",
    # "F-actin-8",
    # "F-actin-7",
    # "F-actin-6",
    # "F-actin-5",
    # "F-actin-4",
    # "F-actin-3",
    # "F-actin-2",
    # "F-actin-1",
    # "ER-6",
    # "ER-5",
    # "ER-4",
    # "ER-3",
    # "ER-2",
    # "ER-1",
    # --------------------------------------------------------------------------
    # "Microtubule2-3d-1024",
    # "Nuclear-pore-complex2-1024",
    # --------------------------------------------------------------------------
    # "biotisr-ccps-1",
    # "biotisr-ccps-2",
    # "biotisr-ccps-3",
    # "biotisr-factin-1",
    # "biotisr-factin-2",
    # "biotisr-factin-3",
    # "biotisr-factin-nonlinear-1",
    # "biotisr-factin-nonlinear-2",
    # "biotisr-factin-nonlinear-3",
    # "biotisr-lysosomes-1",
    # "biotisr-lysosomes-2",
    # "biotisr-lysosomes-3",
    # "biotisr-mt-1",
    # "biotisr-mt-2",
    # "biotisr-mt-3",
    # "biotisr-mito-1",
    # "biotisr-mito-2",
    # "biotisr-mito-3",
    # "deepbacs-ecoli",
    # "deepbacs-ecoli-ave2",
    # "deepbacs-saureus",
    # "deepbacs-saureus-ave2",
    # "w2s-0-sim-ave",
    # "w2s-0-wf-ave-400",
    # "w2s-1-sim-ave",
    # "w2s-1-wf-ave-400",
    # "w2s-2-sim-ave",
    # "w2s-2-wf-ave-400",
    "biotisr-3d-factin-1",
    "biotisr-3d-factin-2",
    "biotisr-3d-mt-1",
    "biotisr-3d-mt-2",
    "biotisr-3d-mito-1",
    "biotisr-3d-mito-2",
)

# the parameters for patching.
params = dict(
    patch_size=128,
    step_size=64,
    normalization=(0.03, 0.995),
)

# ------------------------------------------------------------------------------
info_df = pandas.read_excel("datasets_train.xlsx")

print("-" * 80)
print(f'[INFO] Patch size : {params["patch_size"]}')
print(f'[INFO] Step size : {params["step_size"]}')

normalizer = lambda x: normalization(
    x, params["normalization"][0], params["normalization"][1]
)[0]

done_list = []
num_datasets = len(datasets_name)
for ds in datasets_name:
    # get the info of the dataset
    info = info_df[info_df["id"] == ds].iloc[0]
    path_raw = win2linux(info["path_lr_net"])
    path_gt = win2linux(info["path_hr"])
    # only patch the data used for training
    path_txt = win2linux(info["path_txt"]).replace("train.txt", "all.txt")
    ndim = info["ndim"]

    for path in [path_raw, path_gt, path_txt]:
        assert os.path.exists(path), f"[ERROR] {path} does not exist."

    # read and process images --------------------------------------------------
    filenames = read_txt(path_txt)  # get all the filenames in the txt file
    num_sample = len(filenames)

    print("-" * 80)
    print(f"[INFO] Dataset : {ds}")
    print(f"[INFO] Number of samples : {num_sample}")
    print(f"[INFO] Path raw : {path_raw}")
    print(f"[INFO] Path gt : {path_gt}")
    print(f"[INFO] Path txt : {path_txt}")
    print(f"[INFO] Number of dimensions : {ndim}")

    # patching -----------------------------------------------------------------
    for path_img in [path_raw, path_gt]:
        if path_img in done_list:
            print(f"[INFO] {path_img} has been patched.")
            continue
        done_list.append(path_img)

        path_save = path_img + f"_patch"
        os.makedirs(path_save, exist_ok=True)

        # save each elements in the params into a json file
        with open(os.path.join(path_save, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
        print(f'[INFO] Patched images are saved to "{path_save}"')

        pbar = tqdm.tqdm(total=num_sample, desc="[INFO] Patching", ncols=80)
        for filename in filenames:
            img = io.imread(os.path.join(path_img, filename)).astype(np.float32)
            img = np.clip(img, a_min=0.0, a_max=None)  # clip negative values
            img = normalizer(img)  # normalization

            # 2D ---------------------------------------------------------------
            if ndim == 2:
                assert img.ndim == 2, f"[ERROR] Dimension is disagreement. {img.shape}"
                Ny, Nx = img.shape

                # get the number of patches
                num_patch_y = (Ny - params["patch_size"]) // params["step_size"] + 1
                num_patch_x = (Nx - params["patch_size"]) // params["step_size"] + 1

                # get the patches
                for i in range(num_patch_y):
                    for j in range(num_patch_x):
                        y = i * params["step_size"]
                        x = j * params["step_size"]
                        patch = img[
                            y : y + params["patch_size"], x : x + params["patch_size"]
                        ]
                        # save the patches, each to a single file
                        io.imsave(
                            os.path.join(
                                path_save, filename.replace(".tif", f"_{i}_{j}.tif")
                            ),
                            patch.astype(np.float32),
                            check_contrast=False,
                        )

            # 3D ---------------------------------------------------------------
            elif ndim == 3:
                assert img.ndim == 3, f"[ERROR] Dimension is disagreement. {img.shape}"
                Nz, Ny, Nx = img.shape

                # get the number of patches
                num_patch_z = (Nz - params["patch_size"]) // params["step_size"] + 1
                if num_patch_z < 0:
                    num_patch_z = 1
                num_patch_y = (Ny - params["patch_size"]) // params["step_size"] + 1
                num_patch_x = (Nx - params["patch_size"]) // params["step_size"] + 1

                # get the patches
                for k in range(num_patch_z):
                    for i in range(num_patch_y):
                        for j in range(num_patch_x):
                            z = k * params["step_size"]
                            y = i * params["step_size"]
                            x = j * params["step_size"]
                            patch = img[
                                z : z + params["patch_size"],
                                y : y + params["patch_size"],
                                x : x + params["patch_size"],
                            ]
                            # save the patches, each to a single file
                            io.imsave(
                                os.path.join(
                                    path_save,
                                    filename.replace(".tif", f"_{k}_{i}_{j}.tif"),
                                ),
                                patch.astype(np.float32),
                                check_contrast=False,
                            )
            else:
                raise ValueError(f"[ERROR] Dimension is not supported. {ndim}")
            pbar.update(1)
        pbar.close()
