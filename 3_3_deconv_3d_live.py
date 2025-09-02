"""
Use conventional deconvolution method to restore 3D image.
Requirements:
- PSF
"""

import numpy as np
import skimage.io as io
import os, tqdm
import methods.deconvolution as dcv
from utils.data import win2linux, center_crop, even2odd, read_txt

# ------------------------------------------------------------------------------
path_root = win2linux(
    "I:\Datasets\ZeroShotDeconvNet\\3D time-lapsing data_LLSM_Mitosis_H2B\\642"
    # "I:\Datasets\ZeroShotDeconvNet\\3D time-lapsing data_LLSM_Mitosis_Mito\\560"
)

device_id = "cuda:0"
# device_id = "cpu"

dataset_name = "ZeroShotDeconvNet"
# bp_type = "traditional"
# bp_type = "gaussian"
# bp_type = "butterworth"
bp_type = "wiener-butterworth"

# id_sample = [0, 346]
# id_sample = [0, 346, 609, 700, 770, 901]
# id_sample = [0, 346, 609, 700, 770, 901]
# id_sample = list(range(1, 500, 10))
# id_sample = range(0, 1000, 4)
id_sample = [0, 1]

path_prediction = os.path.join(
    "outputs",
    "predictions",
    dataset_name,
    "Mitosis",
    os.path.basename(path_root),
    bp_type,
)
os.makedirs(path_prediction, exist_ok=True)

# ------------------------------------------------------------------------------
path_raw = os.path.join(path_root, "raw")
filenames = read_txt(os.path.join(path_root, "all.txt"))

params = {
    "traditional": {
        "bp_type": "traditional",
        "num_iter": 30,
        "init": "measured",
    },
    "gaussian": {
        "bp_type": "gaussian",
        "num_iter": 30,
        "init": "measured",
    },
    "butterworth": {
        "bp_type": "butterworth",
        "num_iter": 30,
        "beta": 0.01,
        "n": 10,
        "res_flag": 1,
        "init": "measured",
    },
    "wiener-butterworth": {
        "bp_type": "wiener-butterworth",
        "num_iter": 2,
        "alpha": 0.005,
        "beta": 0.1,
        "n": 10,
        "res_flag": 1,
        "init": "measured",
    },
}

print(f"[INFO] BP type: {bp_type}")
print(f"[INFO] Parameters: {params[bp_type]}")

# ------------------------------------------------------------------------------
# load PSF
# ------------------------------------------------------------------------------
PSF_raw = io.imread(os.path.join(path_raw, "PSF.tif")).astype(np.float32)
print(f"[INFO] PSF shape (raw): {PSF_raw.shape} (sum = {PSF_raw.sum()})")
PSF_raw = PSF_raw / PSF_raw.sum()

# even shape to odd shape
if PSF_raw.shape[-1] % 2 == 0:
    PSF_odd = even2odd(PSF_raw)
    PSF = center_crop(PSF_odd, size=(101, 101, 101))
    print("[INFO] PSF shape (used): {} (sum = {:>.4f})".format(PSF.shape, PSF.sum()))
    PSF = PSF / PSF.sum()
    path_save_to_psf = os.path.join(path_raw, "PSF_odd.tif")
    io.imsave(fname=path_save_to_psf, arr=PSF, check_contrast=False)
    print("[INFO] Save PSF to", path_save_to_psf)
else:
    PSF = PSF_raw

# ------------------------------------------------------------------------------
# deconvolution
# ------------------------------------------------------------------------------
DCV = dcv.Deconvolution(PSF=PSF, device_id=device_id, **params[bp_type])

num_iter = params[bp_type]["num_iter"]
path_save_to = os.path.join(path_prediction, f"iter_{num_iter}")
os.makedirs(path_save_to, exist_ok=True)

img_tmp = io.imread(os.path.join(path_raw, filenames[id_sample[0]])).astype(np.float32)
print(f"[INFO] Load data from: {path_raw}")
print(f"[INFO] Input shape: {img_tmp.shape}")

pbar = tqdm.tqdm(total=len(id_sample), desc="Deconvolution", ncols=80)
for id in id_sample:
    path_img = os.path.join(path_raw, filenames[id])
    img_raw = io.imread(path_img).astype(np.float32)

    img_deconv = DCV.deconv(img_raw, num_iter=num_iter, domain="fft", verbose=False)
    ker_bp = DCV.PSF2

    io.imsave(
        fname=os.path.join(path_save_to, filenames[id]),
        arr=img_deconv.astype(np.uint16),
        check_contrast=False,
    )
    if id == 0:
        io.imsave(
            fname=os.path.join(path_save_to, "ker_bp.tif"),
            arr=ker_bp,
            check_contrast=False,
        )
    pbar.update(1)
pbar.close()
