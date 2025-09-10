"""
Use conventional deconvolution method to restore 3D image.

Requirements:
- Ground truth
- PSF
"""

import numpy as np
import skimage.io as io
import os, pandas, tqdm, json
import methods.deconvolution as dcv
import utils.evaluation as eva
from utils.data import win2linux, read_txt

device_id = "cuda:0"
# device_id = "cpu"
# ------------------------------------------------------------------------------
# dataset_info = ("Microtubule2-3d-1024", "Microtubule2-3d-1024", "fp_n1_r1_bp_n1_r1")
dataset_info = (
    "Nuclear-pore-complex2-1024",
    "Nuclear-pore-complex2-1024",
    "fp_n1_r1_bp_n1_r1",
)

bp_type = "traditional"
# bp_type = "gaussian"
# bp_type = "butterworth"
# bp_type = "wiener-butterworth"

id_sample = [0, 1, 2, 3, 4, 5, 6]
# id_sample = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# ------------------------------------------------------------------------------
dataset_name_test, dataset_name_train, experiment = dataset_info
path_prediction = os.path.join("outputs", "predictions", dataset_name_test)

params_methods = {
    "traditional": {
        "bp_type": "traditional",
        "init": "measured",
        "padding_mode": "reflect",
        "num_iter": 15,  # mt
        # "num_iter": 20,  # npc
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
}

# ------------------------------------------------------------------------------
# load data
# load infomation from excel
info_df = pandas.read_excel("datasets_test.xlsx")
info = info_df[info_df["id"] == dataset_name_test].iloc[0]

path_data_gt = win2linux(info["path_hr"])
path_data_raw = win2linux(info["path_lr"])
path_txt = win2linux(info["path_txt"])
path_psf = win2linux(info["path_psf"])

filenames = read_txt(path_txt)

if path_psf is None or path_psf == "":
    print("[INFO] Real PSF not exist.")
    # load the learned PSF
    path_psf = os.path.join(
        path_prediction,
        "kernelnet",
        dataset_name_train,
        experiment,
        "kernel",
        "kernel_fp.tif",
    )
    assert os.path.exists(path_psf), "[ERROR] Learned forward PSF not exists!"
    print("[INFO] Load learned PSF from:", path_psf)
else:
    path_psf = win2linux(path_psf)
    assert os.path.exists(path_psf), "[ERROR] PSF not exists!"

num_samples_test = len(id_sample)
num_samples = len(filenames)
print(f"[INFO] Number of test samples: {num_samples_test} | {num_samples}")

# ------------------------------------------------------------------------------
# evaluation metrics
cal_ssim = lambda x, y: eva.SSIM(
    img_true=y,
    img_test=x,
    data_range=y.max() - y.min(),
    version_wang=False,
    channle_axis=0,
)
cal_mse = lambda x, y: eva.PSNR(img_true=y, img_test=x, data_range=y.max() - y.min())
cal_ncc = lambda x, y: eva.NCC(img_true=y, img_test=x)
metrics = lambda x: [cal_mse(x, img_gt), cal_ssim(x, img_gt), cal_ncc(x, img_gt)]

# ------------------------------------------------------------------------------
PSF = io.imread(path_psf).astype(np.float32)
PSF = np.transpose(PSF, axes=(2, 0, 1))

DCV = dcv.Deconvolution(
    PSF=PSF, metrics=metrics, device_id=device_id, **params_methods[bp_type]
)

# ------------------------------------------------------------------------------
path_save = os.path.join(path_prediction, bp_type)
os.makedirs(path_save, exist_ok=True)
print("[INFO] Save results to:", path_save)
# save params into a json file
path_params = os.path.join(path_save, "params.json")
params_methods[bp_type]["device_id"] = device_id
with open(path_params, "w") as f:
    json.dump(params_methods[bp_type], f, indent=4)

# ------------------------------------------------------------------------------
pbar = tqdm.tqdm(total=num_samples_test, desc="Deconvolution", ncols=80)
for i, id in enumerate(id_sample):
    img_gt = io.imread(os.path.join(path_data_gt, filenames[id])).astype(np.float32)
    img_raw = io.imread(os.path.join(path_data_raw, filenames[id])).astype(np.float32)

    path_save_sample = os.path.join(path_save, f"sample_{id}")
    path_save_kernel = os.path.join(path_save, "kernel")
    os.makedirs(path_save_sample, exist_ok=True)
    os.makedirs(path_save_kernel, exist_ok=True)

    num_iter = params_methods[bp_type]["num_iter"]
    out = DCV.deconv(img_raw, num_iter=num_iter, domain="direct", verbose=False)
    ker_bp = DCV.PSF2

    io.imsave(
        fname=os.path.join(path_save_sample, f"deconv_{num_iter}.tif"),
        arr=out.astype(np.float32),
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
