import skimage.metrics as skim
from utils import data
import numpy as np
from pytorch_msssim import ms_ssim
import torch, itertools, math


def tensor_to_array(img):
    """
    Convert torch Tensor to numpy array.
    ### Parameters:
    - `img`: (torch Tensor/ numpy array) input image.
    ### Returns:
    - `img`: (numpy array) output image.
    """
    if not isinstance(img, np.ndarray):
        img = img.cpu().detach().numpy()
    return img


def generation_combinations(n: int, k: int = 2):
    """
    Gnerate combinations of k elements from n elements.
    ### Parameters:
    - n (int): number of elements.
    - k (int): number of elements in each combination.
    ### Returns::
    - combinations (list): list of combinations.
    """
    assert n >= k, f"[n] must be greater than or equal to [k]."
    combinations = list(itertools.combinations(range(n), k))
    return combinations


def array_input_check(img):
    """
    Check the input array is numpy.ndarray or torch.Tensor.
    And convert it to numpy.ndarray if it is torch.Tensor.
    ### Parameters:
    - img (array): input array.
    ### Returns::
    - img (array): converted array.
    """
    assert img.ndim in [
        2,
        3,
        4,
    ], f"[img] must be 2D or 3D array, but got {img.ndim}D array."
    assert isinstance(img, np.ndarray) or isinstance(
        img, torch.Tensor
    ), f"[img] must be numpy.ndarray or torch.Tensor."
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    return img


def SSIM(
    img_true,
    img_test,
    data_range=None,
    multichannel=False,
    channle_axis=None,
    version_wang=False,
    convert_to_255: bool = False,
):
    """
    Structrual similarity index.

    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.
    - `data_range`: the dynamic range of the images.
    - `multichannel`: whether the image is multi-channel.
    - `channle_axis`: the axis of the channel.
    - `version_wang`: whether to use the Wang et al. version of SSIM.
    - `convert_to_255`: whether to convert the image to [0,255].

    ### Returns:
    - `ssim`: structural similarity index.
    """
    img_true = array_input_check(img_true)
    img_test = array_input_check(img_test)

    if data_range == None:
        data_range = img_true.max() - img_true.min()
    if data_range == 0:
        data_range = 1

    if convert_to_255:
        img_true = (img_true * 255).astype(np.uint8)
        img_test = (img_test * 255).astype(np.uint8)
        data_range = 255

    if version_wang == False:
        ssim = skim.structural_similarity(
            im1=img_true,
            im2=img_test,
            multichannel=multichannel,
            data_range=data_range,
            channel_axis=channle_axis,
        )

    if version_wang == True:
        ssim = skim.structural_similarity(
            im1=img_true,
            im2=img_test,
            multichannel=multichannel,
            data_range=data_range,
            channel_axis=channle_axis,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    return ssim


def SSIM_tb(img_true, img_test, data_range=None, version_wang=False):
    """
    SSIM for a batch of tensor.
    Support 3d and 2d single/multi-channel images.

    ### Parameters:
    - `img_true`: ground truth image. [B, C, [depth], H, W]
    - `img_test`: test image. [B, C, [depth], H, W]
    - `data_range`: the dynamic range of the images. default is None.
    - `version_wang`: whether to use the Wang et al. version of SSIM. default is False.

    ### Returns:
    - `ssim`: structural similarity index (mena of batch).
    """
    # tensor to numpy array
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)

    assert (
        img_true.shape == img_test.shape
    ), "The shape of img_true and img_test must be the same."
    assert len(img_true.shape) in [
        4,
        5,
    ], f"The shape of img_true and img_test must be 2D (4 axis) or 3D (5 axis). But got {len(img_true.shape)}."

    ssims = []

    for i_sample in range(img_true.shape[0]):  # loop through each sample
        x, y = img_test[i_sample], img_true[i_sample]

        if len(y.shape) == 4:  # 3D image
            if y.shape[0] == 1:  # one channel 3D image
                if y.shape[1] >= 7:
                    # SSIM only supports 3D images with more than 7 slices.
                    ssims.append(
                        SSIM(
                            img_true=y[0],
                            img_test=x[0],
                            data_range=data_range,
                            multichannel=False,
                            channle_axis=None,
                            version_wang=version_wang,
                        )
                    )
                else:
                    # if the image is 3D but with less than 7 slices,
                    # calculate SSIM for each slice. And take the mean.
                    tmp = []
                    for i_slice in range(y.shape[1]):  # loop through each slice
                        tmp.append(
                            SSIM(
                                img_true=y[0][i_slice],
                                img_test=x[0][i_slice],
                                data_range=data_range,
                                multichannel=False,
                                channle_axis=None,
                                version_wang=version_wang,
                            )
                        )
                    ssims.append(np.mean(tmp))
            else:  # multi-channel 3D image
                if y.shape[1] > 7:  # multi-channel 3D image with more than 7 slices.
                    ssims.append(
                        SSIM(
                            img_true=y,
                            img_test=x,
                            data_range=data_range,
                            multichannel=True,
                            channle_axis=0,
                            version_wang=version_wang,
                        )
                    )
                else:
                    # if the image is 3D but with less than 7 slices,
                    # calculate SSIM for each sclice. And take the mean.
                    tmp = []
                    for i_slice in range(y.shape[1]):
                        tmp.append(
                            SSIM(
                                img_true=y[:, i_slice, ...],
                                img_test=x[:, i_slice, ...],
                                data_range=data_range,
                                multichannel=True,
                                channle_axis=0,
                                version_wang=version_wang,
                            )
                        )
                    ssims.append(np.mean(tmp))

        if len(y.shape) == 3:  # 2D
            if y.shape[0] == 1:  # single-channel
                ssims.append(SSIM(img_true=y[0], img_test=x[0], data_range=data_range))
            else:  # mutli-channel
                ssims.append(
                    SSIM(
                        img_true=y,
                        img_test=x,
                        data_range=data_range,
                        multichannel=True,
                        channle_axis=0,
                        version_wang=False,
                    )
                )

    return np.mean(ssims)


def MSE(img_true, img_test):
    """
    Mean square error.

    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.

    ### Returns:
    - `err`: mean square error.
    """
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)
    err = np.mean((img_test - img_true) ** 2)
    return err


def RMSE(x, y):
    """
    - y: groud truth
    """
    rmse = np.mean(np.square(y - x)) / np.mean(np.square(y)) * 100
    return rmse


def PSNR(img_true, img_test, data_range=None, convert_to_255=False):
    """
    Peak signal-to-noise ratio.

    ### Parameters:
    - `img_true`: ground truth image.
    - `img_test`: test image.
    - `data_range`: the dynamic range of the images.

    ### Returns:
    - `psnr`: peak signal-to-noise ratio.
    """
    img_true = array_input_check(img_true)
    img_test = array_input_check(img_test)

    if data_range == None:
        data_range = img_true.max() - img_true.min()
    if data_range == 0:
        data_range = 1

    if convert_to_255:
        img_true = (img_true * 255).astype(np.uint8)
        img_test = (img_test * 255).astype(np.uint8)
        data_range = 255

    mse = np.mean((img_true - img_test) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = skim.peak_signal_noise_ratio(
            image_true=img_true, image_test=img_test, data_range=data_range
        )
    return psnr


def PSNR_tb(img_true, img_test, data_range=None):
    """
    PSNR for a batch of np tensor, the input should be [B, C, [depth], H, W].
    Support 3d and 2d single/multi-channel images.

    ### Parameters:
    - `img_true`: ground truth image. [B, C, [depth], H, W]
    - `img_test`: test image. [B, C, [depth], H, W]
    - `data_range`: the dynamic range of the images. default is None.

    ### Returns:
    - `psnr`: peak signal-to-noise ratio (mean of the batch).
    """
    # tensor to numpy array
    img_true = tensor_to_array(img_true)
    img_test = tensor_to_array(img_test)

    assert (
        img_true.shape == img_test.shape
    ), "The shape of img_true and img_test must be the same."
    assert len(img_true.shape) in [
        4,
        5,
    ], f"The shape of img_true and img_test must be 2D (4 axis) or 3D (5 axis). But got {len(img_true.shape)}."

    psnrs = []
    for i in range(img_true.shape[0]):
        psnrs.append(
            PSNR(img_true=img_true[i], img_test=img_test[i], data_range=data_range)
        )

    # only calculate no inf value.
    psnrs_filtered = [v for v in psnrs if not math.isinf(v)]

    if not psnrs_filtered:  # check whether list is empty
        psnrs_filtered = psnrs

    return np.mean(psnrs_filtered)


def SNR(img_true, img_test, type: int = 0):
    """
    Calculate signal-to-noise ratio (SNR) for an image.
    ### Parameters:
    - `img_true` : ground truth image.
    - `img_test` : test image.
    - `type` : Formula used to calculate the signal-to-noise ratio.
        - `0` for sum of squares-based.
        - `1` for variance-based.
    ### Returns::
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
    ### Returns::
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
    ### Parameters:
    - img_true (tensor): ground truth.
    - img_test (tensor): test image.
    - data_range (int, optional): The data range of the input images. Default: 255.
    ### Returns:
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
    """
    Measure metrics of each sample (along the 0 axis) and average.
    ### Parameters:
    - `img_true` (tensor): ground truth.
    - `img_test` (tensor): test image.
    - `data_range` (int, optional): The data range of the input images. Default: 255.
    ### Returns::
    - `ave_ssim` (float): average ssim.
    - `ave_psnr` (float): average psnr.
    """
    ssim, psnr = [], []

    # convert to numpy array
    if not isinstance(img_true, np.ndarray):
        ToNumpy = data.ToNumpy()
        img_test, img_true = ToNumpy(img_test), ToNumpy(img_true)

    # loop through each sample
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
            # when the number of slices is less than 7, use multichannel=True, treat the slices as channels
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
    """
    Measure metrics of each sample (along the 0 axis) and average.
    ### Parameters:
    - `img_true` (tensor): ground truth.
    - `img_test` (tensor): test image.
    - `data_range` (int, optional): The data range of the input images. Default: 255.
    ### Returns::
    - `ave_ssim` (float): average ssim.
    - `ave_psnr` (float): average psnr.
    """
    ssim, psnr = [], []

    # convert to numpy array
    if not isinstance(img_true, np.ndarray):
        ToNumpy = data.ToNumpy()
        img_test, img_true = ToNumpy(img_test), ToNumpy(img_true)

    # loop through each sample
    for i in range(img_test.shape[0]):
        y, x = img_true[i, ..., 0], img_test[i, ..., 0]
        if data_range == None:
            data_range = y.max() - y.min()
        ssim.append(
            SSIM(img_true=y, img_test=x, data_range=data_range, version_wang=False)
        )
        psnr.append(PSNR(img_true=y, img_test=x, data_range=data_range))

    # calculate average ssim and psnr
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
    """
    Count the number of parameters in the model.\n
    print the number of trainable parameters and non-trainable parameters.
    ### Parameters:
    - model (nn.Module): model.
    """
    total_para = sum(p.numel() for p in model.parameters())
    trainbale_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[INFO] Total Parameters: {total_para:>10d}, Trainable Parameters: {trainbale_para:>10d}, Non-trainable Parameters: {total_para - trainbale_para:>10d}"
    )
