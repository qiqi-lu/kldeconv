"""
Use conventional deconvolution method to restore simulated image, 2D/3D real image.
As the the PSF of the dataset is not known for real data,
we use the learned forward kernel using KLDeconv for deconvolution.

Requirements:
- Ground truth
- PSF (known or pre-learned)
"""

import numpy as np
import skimage.io as io
import os, pandas, tqdm, json
import methods.deconvolution as dcv
import utils.evaluation as eva
from utils.data import win2linux, read_txt

device_id = "cuda:1"
# device_id = "cpu"
# output_all = False
output_all = True

# ------------------------------------------------------------------------------
#                dataset_id_test | dataset_id_train | experiment name
# ------------------------------------------------------------------------------
dataset_list = (
    # ("SimuMix3D-128-31-05-1-01", "", ""),
    # ("SimuMix3D-128-31-05-1-03", "", ""),
    # ("SimuMix3D-128-31-05-1-1", "", ""),
    ("SimuMix3D-128-31-0-0-1", "", ""),
    # --------------------------------------------------------------------------
    # ("SimuMix3D-512-31-05-1-01", "", ""),
    # ("SimuMix3D-512-31-0-0-1", "", ""),
    # --------------------------------------------------------------------------
    # ("F-actin-nonlinear-9", "F-actin-nonlinear-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-9", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-8", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-7", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-6", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-5", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-4", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-3", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-2", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubules2-1", "Microtubules2-9", "fp_n1_r1_bp_n1_r1"),
    # ("CCPs-9", "CCPs-9", "fp_n1_r1_bp_n1_r1"),
    # ("ER-6", "ER-6", "fp_n1_r1_bp_n1_r1"),
    # ("F-actin-9", "F-actin-9", "fp_n1_r1_bp_n1_r1"),
    # --------------------------------------------------------------------------
    # ( "biotisr-ccps-1", "biotisr-ccps-1", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-ccps-2", "biotisr-ccps-2", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-ccps-3", "biotisr-ccps-3", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-factin-1", "biotisr-factin-1", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-factin-2", "biotisr-factin-2", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-factin-3", "biotisr-factin-3", "fp_n1_r1_bp_n1_r1"),
    # ("biotisr-factin-nonlinear-1", "biotisr-factin-nonlinear-1", "fp_n1_r1_bp_n1_r1"),
    # ("biotisr-factin-nonlinear-2", "biotisr-factin-nonlinear-2", "fp_n1_r1_bp_n1_r1"),
    # ("biotisr-factin-nonlinear-3", "biotisr-factin-nonlinear-3", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-lysosomes-1", "biotisr-lysosomes-1", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-lysosomes-2", "biotisr-lysosomes-2", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-lysosomes-3", "biotisr-lysosomes-3", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-mito-1", "biotisr-mito-1", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-mito-2", "biotisr-mito-2", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-mito-3", "biotisr-mito-3", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-mt-1", "biotisr-mt-1", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-mt-2", "biotisr-mt-2", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-mt-3", "biotisr-mt-3", "fp_n1_r1_bp_n1_r1"),
    # ( "deepbacs-ecoli-ave2", "deepbacs-ecoli-ave2", "fp_n1_r1_bp_n1_r1"),
    # ( "deepbacs-saureus-ave2", "deepbacs-saureus-ave2", "fp_n1_r1_bp_n1_r1"),
    # ( "w2s-0-sim-ave", "w2s-0-sim-ave", "fp_n1_r1_bp_n1_r1"),
    # ( "w2s-1-sim-ave", "w2s-1-sim-ave", "fp_n1_r1_bp_n1_r1"),
    # ( "w2s-2-sim-ave", "w2s-2-sim-ave", "fp_n1_r1_bp_n1_r1"),
    # --------------------------------------------------------------------------
    # ( "Microtubule2-3d-1024", "Microtubule2-3d-1024", "fp_n1_r1_bp_n1_r1"),
    # ("Microtubule2-3d-1024", "Microtubule2-3d-512", "fp_n1_r1_bp_n1_r1"),
    # ("Nuclear-pore-complex2-1024", "Nuclear-pore-complex2-1024", "fp_n1_r1_bp_n1_r1"),
    # ("Nuclear-pore-complex2-1024", "Nuclear-pore-complex2-512", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-3d-mt-1", "biotisr-3d-mt-1", "fp_n1_r1_bp_n1_r1"),
    # ("biotisr-3d-mt-2", "biotisr-3d-mt-2", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-3d-mito-1", "biotisr-3d-mito-1", "fp_n1_r1_bp_n1_r1"),
    # ("biotisr-3d-mito-2", "biotisr-3d-mito-2", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-3d-factin-1", "biotisr-3d-factin-1", "fp_n1_r1_bp_n1_r1"),
    # ( "biotisr-3d-factin-2", "biotisr-3d-factin-2", "fp_n1_r1_bp_n1_r1"),
    # --------------------------------------------------------------------------
    # ("real-3d-live", "ZeroShotDeconvNet-mitosis-642", "", ""),
    # ("real-3d-live", "ZeroShotDeconvNet-mitosis-560", "", ""),
)

# ------------------------------------------------------------------------------
bp_type = "traditional"
# bp_type = "gaussian"
# bp_type = "butterworth"
# bp_type = "wiener-butterworth"

# id_sample_global = [0, 1, 2, 3, 4, 5, 6]
# id_sample_global = [7, 8, 9, 10]
# id_sample_global = [6]
# id_sample_global = [0, 346]
# id_sample_global = [0, 346, 609, 700, 770, 901]
# id_sample_global = [0, 346, 609, 700, 770, 901]
# id_sample_global = list(range(1, 500, 10))
# id_sample_global = range(0, 1000, 4)
id_sample_global = [0]
# id_sample_global = []

# ------------------------------------------------------------------------------
methods_info_dict = {
    "real-2d": {
        "domain": "fft",
        "params_methods": {
            "traditional": {
                "bp_type": "traditional",
                "init": "measured",
                "padding_mode": "reflect",
                "num_iter": 100,
            },
            "gaussian": {
                "bp_type": "gaussian",
                "init": "measured",
                "num_iter": 30,
            },
            "butterworth": {
                "bp_type": "butterworth",
                "beta": 0.01,
                "n": 10,
                "res_flag": 1,
                "init": "measured",
                "num_iter": 30,
            },
            "wiener-butterworth": {
                "bp_type": "wiener-butterworth",
                "alpha": 0.005,
                "beta": 0.1,
                "n": 10,
                "res_flag": 1,
                "init": "measured",
                "num_iter": 2,
            },
        },
    },
    "real-3d": {
        "domain": "direct",
        "params_methods": {
            "traditional": {
                "bp_type": "traditional",
                "init": "measured",
                "padding_mode": "reflect",
                # "num_iter": 15,  # mt
                "num_iter": 20,  # npc
            },
            "gaussian": {
                "bp_type": "gaussian",
                "init": "measured",
                "num_iter": 30,
            },
            "butterworth": {
                "bp_type": "butterworth",
                "beta": 0.01,
                "n": 10,
                "res_flag": 1,
                "num_iter": 30,
                "init": "measured",
            },
            "wiener-butterworth": {
                "bp_type": "wiener-butterworth",
                "alpha": 0.005,
                "beta": 0.1,
                "n": 10,
                "res_flag": 1,
                "num_iter": 2,
                "init": "measured",
            },
        },
    },
    "simu-3d": {
        "domain": "fft",
        "params_methods": {
            "traditional": {
                "bp_type": "traditional",
                "init": "measured",
                "padding_mode": "reflect",
                # "num_iter": 2,
                # "num_iter": 3,
                # "num_iter": 20,
                # "num_iter": 30,
                "num_iter": 100,
            },
            "gaussian": {
                "bp_type": "gaussian",
                "init": "measured",
                # "num_iter": 2,
                "num_iter": 100,
            },
            "butterworth": {
                "bp_type": "butterworth",
                "beta": 0.01,
                "n": 10,
                "res_flag": 1,
                "num_iter": 2,
                # "num_iter": 100,
                "init": "measured",
            },
            "wiener-butterworth": {
                "bp_type": "wiener-butterworth",
                "alpha": 0.005,
                "beta": 0.1,  # ratio = 1 or 0.3
                "n": 10,
                "res_flag": 1,
                # "num_iter": 2,
                # "num_iter": 3,
                # "num_iter": 30,
                "num_iter": 100,
                "init": "measured",
            },
        },
    },
    "real-3d-live": {
        "domain": "fft",
        "params_methods": {
            "traditional": {
                "bp_type": "traditional",
                "init": "measured",
                "padding_mode": "reflect",
                "num_iter": 30,
            },
            "gaussian": {
                "bp_type": "gaussian",
                "init": "measured",
                "num_iter": 30,
            },
            "butterworth": {
                "bp_type": "butterworth",
                "beta": 0.01,
                "n": 10,
                "res_flag": 1,
                "num_iter": 30,
                "init": "measured",
            },
            "wiener-butterworth": {
                "bp_type": "wiener-butterworth",
                "alpha": 0.005,
                "beta": 0.1,
                "n": 10,
                "res_flag": 1,
                "num_iter": 2,
                "init": "measured",
            },
        },
    },
}

# ------------------------------------------------------------------------------
info_df = pandas.read_excel("datasets_test.xlsx")

for dataset_info in dataset_list:
    id_sample = id_sample_global
    dataset_name_test, dataset_name_train, experiment = dataset_info

    path_prediction = os.path.join("outputs", "predictions", dataset_name_test)

    info = info_df[info_df["id"] == dataset_name_test].iloc[0]
    ndim = info["ndim"]
    ratio = info["ratio"]
    data_type = info["data type"]
    enable_dark = info["dark"]

    domain = methods_info_dict[data_type]["domain"]
    params_methods = methods_info_dict[data_type]["params_methods"]
    if ratio == 0.1:
        params_methods["wiener-butterworth"]["beta"] = 0.001  # ratio = 0.1

    # ------------------------------------------------------------------------------
    # load data
    # load infomation from excel
    # path_data_gt = win2linux(info["path_hr"])
    path_data_raw = win2linux(info["path_lr"])
    path_txt = win2linux(info["path_txt"])
    path_psf = win2linux(info["path_psf"])

    if enable_dark:
        path_data_raw += "_dark"

    filenames = read_txt(path_txt)

    if path_psf is None or path_psf == "":
        print("[INFO] Real PSF not exist.")
        # load the learned PSF
        path_psf = os.path.join(
            path_prediction,
            "kernelnet",
            dataset_name_train,
            experiment,
            "kernel_iter_2",
            "kernel_fp.tif",
        )
        assert os.path.exists(path_psf), "[ERROR] Learned forward PSF not exists!"
        print("[INFO] Load learned PSF from:", path_psf)
    else:
        path_psf = win2linux(path_psf)
        assert os.path.exists(path_psf), "[ERROR] PSF not exists!"

    num_samples = len(filenames)
    if id_sample == []:
        id_sample = range(num_samples)
    num_samples_test = len(id_sample)
    print(f"[INFO] Number of test samples: {num_samples_test} | {num_samples}")

    # ------------------------------------------------------------------------------
    params = params_methods[bp_type]
    params["device_id"] = device_id

    print("[INFO] Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # create deconv object
    PSF = io.imread(path_psf).astype(np.float32)
    print("[INFO] PSF shape:", PSF.shape)
    # if data_type == "real-3d":
    #     PSF = np.transpose(PSF, axes=(2, 0, 1))
    DCV = dcv.Deconvolution(PSF=PSF, metrics=None, **params)

    # ------------------------------------------------------------------------------
    path_save_to = os.path.join(path_prediction, bp_type, dataset_name_train)
    os.makedirs(path_save_to, exist_ok=True)
    print("[INFO] Save results to:", path_save_to)
    # save params into a json file
    path_params = os.path.join(path_save_to, "params.json")
    with open(path_params, "w") as f:
        json.dump(params, f, indent=4)

    pbar = tqdm.tqdm(total=num_samples_test, desc=f"Deconvolution ({domain})", ncols=80)
    for i, id in enumerate(id_sample):
        # img_gt = io.imread(os.path.join(path_data_gt, filenames[id])).astype(np.float32)
        img_raw = io.imread(os.path.join(path_data_raw, filenames[id])).astype(
            np.float32
        )

        # PSF_align = dcv.adjust_size(PSF, img_gt.shape)

        path_save_sample = os.path.join(path_save_to, filenames[id].split(".")[0])
        path_save_kernel = os.path.join(path_save_to, "kernel")
        os.makedirs(path_save_sample, exist_ok=True)
        os.makedirs(path_save_kernel, exist_ok=True)

        # ----------------------------------------------------------------------
        num_iter = params["num_iter"]
        outs = DCV.deconv(
            img_raw, num_iter=num_iter, domain=domain, verbose=False, out_all=output_all
        )

        if output_all:
            out, out_all_iters = outs
        else:
            out = outs

        ker_bp = DCV.PSF2

        if data_type == "real-3d-live":
            out = out.astype(np.uint16)
        else:
            out = out.astype(np.float32)

        io.imsave(
            fname=os.path.join(path_save_sample, f"deconv_iter_{num_iter}.tif"),
            arr=out,
            check_contrast=False,
        )

        if output_all:
            io.imsave(
                fname=os.path.join(path_save_sample, f"deconv_iter_{num_iter}_all.tif"),
                arr=out_all_iters,
                check_contrast=False,
            )

        if i == 0:
            io.imsave(
                fname=os.path.join(path_save_kernel, "ker_bp.tif"),
                arr=ker_bp,
                check_contrast=False,
            )
            io.imsave(
                fname=os.path.join(path_save_kernel, "ker_fp.tif"),
                arr=PSF,
                check_contrast=False,
            )
        pbar.update(1)
    pbar.close()
