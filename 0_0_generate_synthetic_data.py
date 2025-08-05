"""
Generate synthetic data.
blur and noise.
"""

import os, tqdm
import numpy as np
import skimage.io as io
import methods.deconvolution as dcv
import utils.evaluation as utils_eva
import utils.data as utils_data

# ------------------------------------------------------------------------------
# dataset_name = 'SimuBeads3D_128'
# dataset_name = 'SimuMix3D_128'
# dataset_name = 'SimuMix3D_128_2'
# dataset_name = 'SimuMix3D_128_3'
dataset_name = "SimuMix3D_256"
# dataset_name = 'SimuMix3D_256_2'
# dataset_name = 'SimuMix3D_256s'
# dataset_name = 'SimuMix3D_382'

lamb = None
if dataset_name in ["SimuMix3D_382"]:
    # lamb = 642 # wavelength of the emission light
    lamb = 560

# PSF size
if dataset_name in ["SimuBeads3D_128", "SimuMix3D_128"]:
    s_crop = 127
    # s_crop = 63
    size = (s_crop, 127, 127)

if dataset_name in ["SimuMix3D_128_3", "SimuMix3D_256_2", "SimuMix3D_256s"]:
    s_crop = 63
    size = (s_crop, 31, 31)

if dataset_name in ["SimuMix3D_256"]:
    s_crop = 31
    size = (s_crop,) * 3

if dataset_name in ["SimuMix3D_382"]:
    s_crop = 101
    size = (101, s_crop, s_crop)

std_gauss, poisson, ratio = 0.5, 1, 0.1
# std_gauss, poisson, ratio = 0.5, 1, 0.3
# std_gauss, poisson, ratio = 0.5, 1, 1
# std_gauss, poisson, ratio = 0, 0, 1
scale_factor = 1

# ------------------------------------------------------------------------------
# path and file names
path_dataset = os.path.join("F:", os.sep, "Datasets", "RLN", dataset_name)
path_gt = os.path.join(path_dataset, "gt")
print(f"[INFO] dataset from: {path_dataset}")

# load filenames from txt, each line is a filename
path_filenames = os.path.join(path_dataset, "all.txt")
with open(path_filenames, "r") as f:
    filenames = f.read().splitlines()
    if filenames[-1] == "":
        filenames.pop()
num_samples = len(filenames)
print(f"[INFO] Number of samples: {num_samples}")

# ------------------------------------------------------------------------------
if lamb == None:
    path_psf = os.path.join(path_dataset, "PSF.tif")
else:
    path_psf = os.path.join(path_dataset, f"PSF_{lamb}.tif")

PSF = io.imread(path_psf).astype(np.float32)
PSF_odd = utils_data.even2odd(PSF)
PSF_crop = utils_data.center_crop(PSF_odd, size=size)

print(f"[INFO] PSF from: {path_psf}")
print(f"[INFO] PSF shape (origin): {PSF.shape}")
print(f"[INFO] PSF after crop: {PSF_crop.shape} sum = {PSF_crop.sum():.4f}")

PSF_crop = PSF_crop / PSF_crop.sum()

# ------------------------------------------------------------------------------
# single image
img_gt_single = io.imread(os.path.join(path_gt, filenames[0]))
img_gt_single = img_gt_single.astype(np.float32)
print("[INFO] GT shape:", img_gt_single.shape)

# ------------------------------------------------------------------------------
# save to
if lamb == None:
    path_dataset_blur = os.path.join(
        path_dataset,
        f"raw_psf_{s_crop}_gauss_{std_gauss}_poiss_{poisson}_sf_{scale_factor}_ratio_{ratio}",
    )
else:
    path_dataset_blur = os.path.join(
        path_dataset,
        f"raw_psf_{s_crop}_gauss_{std_gauss}_poiss_{poisson}_sf_{scale_factor}_ratio_{ratio}_lambda_{lamb}",
    )
os.makedirs(path_dataset_blur, exist_ok=True)
path_snr = os.path.join(path_dataset_blur, "SNR.txt")

print("[INFO] save generated images to:", path_dataset_blur)

# ------------------------------------------------------------------------------
# save cropped PSF
io.imsave(
    os.path.join(path_dataset_blur, "PSF.tif"), arr=PSF_crop, check_contrast=False
)
# save synthetic data
with open(path_snr, "w") as f:
    pbar = tqdm.tqdm(total=num_samples, desc="GENERATING", ncols=80)
    for fn in filenames:
        img_gt = io.imread(os.path.join(path_gt, fn)).astype(np.float32)
        img_gt = img_gt * ratio  # scale to control SNR
        img_blur = dcv.Convolution(
            img_gt, PSF_crop, padding_mode="reflect", domain="fft"
        )
        img_blur_n = utils_data.add_mix_noise(
            img_blur, poisson=poisson, sigma_gauss=std_gauss, scale_factor=scale_factor
        )  # add noise

        # SNR
        io.imsave(
            os.path.join(path_dataset_blur, fn),
            arr=img_blur_n,
            check_contrast=False,
        )
        pbar.update(1)
        # write snr
        f.write(f"{fn} {utils_eva.SNR(img_blur, img_blur_n)}\n")
    pbar.close()
