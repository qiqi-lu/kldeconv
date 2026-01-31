"""
Generate synthetic data.
- blur and noise.
"""

import os, tqdm
import numpy as np
import skimage.io as io
import methods.deconvolution as dcv
import utils.evaluation as utils_eva
import utils.data as utils_data

path_root = "E:\qiqilu\datasets_2\RLN\\unzip\kldeconv"

# ------------------------------------------------------------------------------
# PARAMETER SETTING
# ------------------------------------------------------------------------------
# dataset_name = 'SimuBeads3D_128'
# dataset_name = 'SimuMix3D_128'
# dataset_name = "SimuMix3D_256"
# dataset_name = 'SimuMix3D_560_382'
# dataset_name = 'SimuMix3D_642_382'
# dataset_name = "SimuMix3D_1024"
dataset_name = "SimuMix3D_512"

std_gauss, poisson, ratio = 0.5, 1, 0.1
# std_gauss, poisson, ratio = 0.5, 1, 0.3
# std_gauss, poisson, ratio = 0.5, 1, 1
# std_gauss, poisson, ratio = 0, 0, 1
scale_factor = 1

# ------------------------------------------------------------------------------
dataset_info = {
    # datsets used for live cell image deconvolution in ZeroShotDeconv
    "SimuMix3D_560_382": {
        "s_crop": 101,
        "size": (101, 101, 101),
    },
    "SimuMix3D_642_382": {
        "s_crop": 101,
        "size": (101, 101, 101),
    },
    # datsets used for simulation data evaluation
    "SimuMix3D_256": {
        "s_crop": 31,
        "size": (31, 31, 31),
    },
    "SimuMix3D_128": {
        "s_crop": 127,
        "size": (127, 127, 127),
    },
    "SimuBeads3D_128": {
        "s_crop": 127,
        "size": (127, 127, 127),
    },
    "SimuMix3D_1024": {
        "s_crop": 31,
        "size": (31, 31, 31),
    },
    "SimuMix3D_512": {
        "s_crop": 31,
        "size": (31, 31, 31),
    },
}

# ------------------------------------------------------------------------------
s_crop = dataset_info[dataset_name]["s_crop"]
size = dataset_info[dataset_name]["size"]
print(
    f"[INFO] std_gauss: {std_gauss}, poisson: {poisson}, ratio: {ratio}, scale_factor: {scale_factor}"
)

# path and file names
path_dataset = utils_data.win2linux(os.path.join(path_root, dataset_name))
path_gt = os.path.join(path_dataset, "gt")
path_filenames = os.path.join(path_dataset, "all.txt")

filenames = utils_data.read_txt(path_filenames)
num_samples = len(filenames)

print(f"[INFO] Load dataset from: {path_dataset}")
print(f"[INFO] Number of samples: {num_samples}")

# ------------------------------------------------------------------------------
# load PSF
path_psf = os.path.join(path_dataset, "PSF.tif")
PSF = io.imread(path_psf).astype(np.float32)
PSF_odd = utils_data.even2odd(PSF)  # make PSF odd
PSF_crop = utils_data.center_crop(PSF_odd, size=size)  # crop PSF

print(f"[INFO] PSF from: {path_psf}")
print(f"[INFO] PSF shape (origin): {PSF.shape}")
print(f"[INFO] PSF after crop: {PSF_crop.shape} sum = {PSF_crop.sum():.4f}")

PSF_crop = PSF_crop / PSF_crop.sum()

# ------------------------------------------------------------------------------
# load single image
img_gt_single = io.imread(os.path.join(path_gt, filenames[0]))
img_gt_single = img_gt_single.astype(np.float32)
print("[INFO] GT shape:", img_gt_single.shape)

# ------------------------------------------------------------------------------
# save to
path_dataset_blur = os.path.join(
    path_dataset,
    f"raw_psf_{s_crop}_gauss_{std_gauss}_poiss_{poisson}_sf_{scale_factor}_ratio_{ratio}",
)
path_psf_crop = os.path.join(path_dataset_blur, "PSF.tif")
os.makedirs(path_dataset_blur, exist_ok=True)

print("[INFO] Save generated images to:", path_dataset_blur)
print("[INFO] Save PSF to:", path_psf_crop)

# ------------------------------------------------------------------------------
# save cropped PSF
io.imsave(path_psf_crop, arr=PSF_crop, check_contrast=False)

# ------------------------------------------------------------------------------
# save synthetic data
pbar = tqdm.tqdm(total=num_samples, desc="DEGRADATION", ncols=80)
for fn in filenames:
    img_gt = io.imread(os.path.join(path_gt, fn)).astype(np.float32)
    # scale to control SNR -------------------------------------------------
    img_gt = img_gt * ratio
    # blur -----------------------------------------------------------------
    img_blur = dcv.convolution(img_gt, PSF_crop, padding_mode="reflect", domain="fft")
    # add noise ------------------------------------------------------------
    img_blur_n = utils_data.add_mix_noise(
        img_blur, poisson=poisson, sigma_gauss=std_gauss, scale_factor=scale_factor
    )
    # SNR
    io.imsave(
        os.path.join(path_dataset_blur, fn),
        arr=img_blur_n,
        check_contrast=False,
    )
    pbar.update(1)
pbar.close()
