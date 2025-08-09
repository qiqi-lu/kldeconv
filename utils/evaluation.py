import skimage.metrics as skim
from utils import data
import numpy as np
from pytorch_msssim import ms_ssim
import torch


def array_input_check(img):
    """
    Check the input array is numpy.ndarray or torch.Tensor.
    And convert it to numpy.ndarray if it is torch.Tensor.
    ### Parameters:
    - img (array): input array.
    ### Returns:
    - img (array): converted array.
    """
    assert img.ndim in [
        2,
        3,
    ], f"[img] must be 2D or 3D array, but got {img.ndim}D array."
    assert isinstance(img, np.ndarray) or isinstance(
        img, torch.Tensor
    ), f"[img] must be numpy.ndarray or torch.Tensor."
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    return img


def SSIM(img_true, img_test, data_range=1.0, version_wang: bool = False):
    """
    Structrual similarity for an single-channel 2D or 3D image.

    ### Parameters:
    - img_true (array): ground truth.
    - img_test (array): predicted image.
    - data_range (float, int): image value range.
    - version_wang (bool): use parameter used in Wang's paper.
    ### Returns:
    - ssim (float): structural similarity.
    """
    img_true = array_input_check(img_true)
    img_test = array_input_check(img_test)

    if version_wang == False:
        ssim = skim.structural_similarity(
            im1=img_true, im2=img_test, data_range=data_range, channel_axis=None
        )

    if version_wang == True:
        ssim = skim.structural_similarity(
            im1=img_true,
            im2=img_test,
            multichannel=False,
            data_range=data_range,
            channel_axis=None,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    return ssim


def MSE(img_true, img_test):
    """Mean Square error for one subject."""
    mse = np.mean((img_test - img_true) ** 2)
    return mse


def RMSE(x, y):
    """
    - y: groud truth
    """
    rmse = np.mean(np.square(y - x)) / np.mean(np.square(y)) * 100
    return rmse


def PSNR(img_true, img_test, data_range=255):
    """
    Peak Signal-to-Noise Ratio (PSNR).
    ### Parameters:
    - img_true (array): ground truth.
    - img_test (array): predicted image.
    - data_range (float, int): image value range.
    ### Returns:
    - psnr (float): peak signal-to-noise ratio.
    """
    img_true = array_input_check(img_true)
    img_test = array_input_check(img_test)

    psnr = skim.peak_signal_noise_ratio(
        image_true=img_true, image_test=img_test, data_range=data_range
    )
    return psnr


def SNR(img_true, img_test, type: int = 0):
    """
    Calculate signal-to-noise ratio (SNR) for an image.
    ### Parameters:
    - `img_true` : ground truth image.
    - `img_test` : test image.
    - `type` : Formula used to calculate the signal-to-noise ratio.
        - `0` for sum of squares-based.
        - `1` for variance-based.
    ### Returns:
    - `snr` : signal-to-noise ratio.
    """
    assert len(img_true.shape) == len(
        img_test.shape
    ), f"The dimensions of the two images are not the same."
    assert type in [0, 1], f"Type must be 0 or 1."

    if type == 0:
        img_true_ss = np.sum(np.square(img_true))
        error_ss = np.sum(np.square(img_true - img_test))
    if type == 1:
        img_true_ss = np.var(img_true)
        error_ss = np.var(img_test - img_true)
    snr = 10 * np.log10(img_true_ss / error_ss) if error_ss != 0 else np.inf
    return snr


def NCC(img_true, img_test):
    """
    Normalized cross-correlation (NCC).
    ### Parameters:
    - img_true (array): ground truth.
    - img_test (array): predicted image.
    ### Returns:
    - ncc (float): normalized cross-correlation.
    """
    img_true = array_input_check(img_true)
    img_test = array_input_check(img_test)

    mean_true = img_true.mean()
    mean_test = img_test.mean()
    sigma_true = img_true.std()
    sigma_test = img_test.std()
    NCC = np.mean(
        (img_true - mean_true) * (img_test - mean_test) / (sigma_true * sigma_test)
    )
    return NCC


def NRMSE(img_true, img_test):
    xmax, xmin = np.max(img_true), np.min(img_true)
    rmse = np.sqrt(np.mean(np.square(img_test - img_true)))
    nrmse = rmse / (xmax - xmin)
    return nrmse


def MSSSIM(img_true, img_test, data_range=255):
    img_true = torch.Tensor(img_true)
    img_test = torch.Tensor(img_test)
    if len(img_true.shape) == 3:
        img_true = img_true[None]
    if len(img_test.shape) == 3:
        img_test = img_test[None]
    img_true = torch.transpose(img_true, dim0=-1, dim1=1)
    img_test = torch.transpose(img_test, dim0=-1, dim1=1)
    msssim = ms_ssim(img_true, img_test, data_range=data_range, size_average=False)
    return msssim


def measure(img_true, img_test, data_range=255):
    """
    Measure metrics of each sample (along the 0 axis) and average.
    Args:
    - img_true (tensor): ground truth.
    - img_test (tensor): test image.
    - data_range (int, optional): The data range of the input images. Default: 255.
    Returns:
    - ave_ssim (float): average ssim.
    - ave_psnr (float): average psnr.
    """
    ssim, psnr = [], []
    if not isinstance(img_true, np.ndarray):
        ToNumpy = data.ToNumpy()
        img_test, img_true = ToNumpy(img_test), ToNumpy(img_true)
        data_range = data_range.cpu().detach().numpy()

    for i in range(img_test.shape[0]):
        if len(img_true.shape) == 4:
            ssim.append(SSIM(img_true[i], img_test[i], data_range=data_range))
        if len(img_true.shape) == 5:
            # ssim.append(SSIM(img_true[i,...,-1], img_test[i,...,-1], data_range=data_range, multichannel=False, channle_axis=None, version_wang=False))
            ssim.append(0)
        psnr.append(PSNR(img_true[i], img_test[i], data_range=data_range))
    ave_ssim, ave_psnr = np.mean(ssim), np.mean(psnr)
    return ave_ssim, ave_psnr


def measure_3d(img_true, img_test, data_range=None):
    ssim, psnr = [], []

    if not isinstance(img_true, np.ndarray):
        ToNumpy = data.ToNumpy()
        img_test, img_true = ToNumpy(img_test), ToNumpy(img_true)
        # if data_range is not None:
        #     data_range = data_range.cpu().detach().numpy()

    for i in range(img_test.shape[0]):
        y, x = img_true[i, ..., 0], img_test[i, ..., 0]
        if data_range == None:
            data_range = y.max() - y.min()
        if y.shape[0] >= 7:
            ssim.append(
                SSIM(
                    img_true=y,
                    img_test=x,
                    data_range=data_range,
                    multichannel=False,
                    channle_axis=None,
                    version_wang=False,
                )
            )
        else:
            ssim.append(
                SSIM(
                    img_true=y,
                    img_test=x,
                    data_range=data_range,
                    multichannel=True,
                    channle_axis=0,
                    version_wang=False,
                )
            )
        psnr.append(PSNR(img_true=y, img_test=x, data_range=data_range))

    ave_ssim, ave_psnr = np.mean(ssim), np.mean(psnr)
    return ave_ssim, ave_psnr


def measure_2d(img_true, img_test, data_range=None):
    ssim, psnr = [], []

    # convert to numpy array
    if not isinstance(img_true, np.ndarray):
        ToNumpy = data.ToNumpy()
        img_test, img_true = ToNumpy(img_test), ToNumpy(img_true)
        # if data_range is not None:
        #     data_range = data_range.cpu().detach().numpy()

    for i in range(img_test.shape[0]):
        y, x = img_true[i, ..., 0], img_test[i, ..., 0]
        if data_range == None:
            data_range = y.max() - y.min()
        ssim.append(
            SSIM(
                img_true=y,
                img_test=x,
                data_range=data_range,
                multichannel=True,
                channle_axis=0,
                version_wang=False,
            )
        )
        psnr.append(PSNR(img_true=y, img_test=x, data_range=data_range))

    ave_ssim, ave_psnr = np.mean(ssim), np.mean(psnr)
    return ave_ssim, ave_psnr


def metrics_batch(img_true, img_test, data_range=255):
    img_true = data.tensor2rgb(img_true)
    img_test = data.tensor2rgb(img_test)
    ssim, psnr = [], []

    for i in range(len(img_true)):
        ssim.append(SSIM(img_true[i], img_test[i], data_range=data_range))
        psnr.append(PSNR(img_true[i], img_test[i], data_range=data_range))
    ave_ssim, ave_psnr = np.mean(ssim), np.mean(psnr)
    return ave_ssim, ave_psnr


def count_parameters(model):
    total_para = sum(p.numel() for p in model.parameters())
    trainbale_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "Total Parameters: {:>10d}, Trainable Parameters: {:>10d}, Non-trainable Parameters: {:>10d}".format(
            total_para, trainbale_para, total_para - trainbale_para
        )
    )
