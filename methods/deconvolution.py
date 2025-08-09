import numpy as np
import methods.back_projector as back_projector
import tqdm, torch
from fft_conv_pytorch import fft_conv


# ------------------------------------------------------------------------------
# def align_size(img, size, pad_value=0):
#     """
#     Align image size to the given size.
#     ### Parameters:
#     - `img` : image, numpy array.
#     - `size` : target size, tuple.
#     - `pad_value` : padding value, float.
#     ### Returns:
#     - `img2` : aligned image, numpy array.
#     """
#     dim = len(img.shape)
#     assert dim in [2, 3], "Only support 2D and 3D image."
#     assert len(size) == dim, "Size and image dimension mismatch."

#     if dim == 3:
#         Nz_1, Ny_1, Nx_1 = img.shape
#         Nz_2, Ny_2, Nx_2 = size
#         Nz, Ny, Nx = (
#             np.maximum(Nz_1, Nz_2),
#             np.maximum(Ny_1, Ny_2),
#             np.maximum(Nx_1, Nx_2),
#         )

#         imgTemp = np.ones(shape=(Nz, Ny, Nx)) * pad_value
#         imgTemp = imgTemp.astype(img.dtype)

#         Nz_o, Ny_o, Nx_o = (
#             int(np.round((Nz - Nz_1) / 2)),
#             int(np.round((Ny - Ny_1) / 2)),
#             int(np.round((Nx - Nx_1) / 2)),
#         )
#         imgTemp[Nz_o : Nz_o + Nz_1, Ny_o : Ny_o + Ny_1, Nx_o : Nx_o + Nx_1] = img

#         Nz_o, Ny_o, Nx_o = (
#             int(np.round((Nz - Nz_2) / 2)),
#             int(np.round((Ny - Ny_2) / 2)),
#             int(np.round((Nx - Nx_2) / 2)),
#         )
#         img2 = imgTemp[Nz_o : Nz_o + Nz_2, Ny_o : Ny_o + Ny_2, Nx_o : Nx_o + Nx_2]

#     if dim == 2:
#         Ny_1, Nx_1 = img.shape
#         Ny_2, Nx_2 = size
#         Ny, Nx = np.maximum(Ny_1, Ny_2), np.maximum(Nx_1, Nx_2)

#         imgTemp = np.ones(shape=(Ny, Nx)) * pad_value
#         imgTemp = imgTemp.astype(img.dtype)

#         Ny_o, Nx_o = int(np.round((Ny - Ny_1) / 2)), int(np.round((Nx - Nx_1) / 2))
#         imgTemp[Ny_o : Ny_o + Ny_1, Nx_o : Nx_o + Nx_1] = img

#         Ny_o, Nx_o = int(np.round((Ny - Ny_2) / 2)), int(np.round((Nx - Nx_2) / 2))
#         img2 = imgTemp[Ny_o : Ny_o + Ny_2, Nx_o : Nx_o + Nx_2]

#     return img2


def adjust_size(img, size, pad_value=0):
    """
    Adjust image size to the given size, the image will be cropped or padded to the given size.
    ### Parameters:
    - `img` : image, numpy array.
    - `size` : target size, tuple.
    - `pad_value` : padding value, float.
    ### Returns:
    - `img2` : adjusted image, numpy array.
    """
    dim = img.ndim
    assert dim in [2, 3], "Only support 2D and 3D image."
    assert len(size) == dim, "Size and image dimension mismatch."

    original_shape = img.shape
    max_shape = tuple(np.maximum(original_shape, size))

    pad_width = [
        (int(np.round((max_dim - orig_dim) / 2)),)
        for max_dim, orig_dim in zip(max_shape, original_shape)
    ]
    img_padded = np.pad(img, pad_width, mode="constant", constant_values=pad_value)

    # crop to target size
    crop_width = [
        (int(np.round((max_dim - target_dim) / 2)),)
        for max_dim, target_dim in zip(max_shape, size)
    ]
    slices = tuple(
        slice(crop[0], crop[0] + target_dim)
        for crop, target_dim in zip(crop_width, size)
    )

    return img_padded[slices].astype(img.dtype)


# def ConvFFT3_S(inVol, OTF):
#     outVol = np.fft.ifftn(np.fft.fftn(inVol) * OTF)
#     return outVol.real


def convolution(x, PSF, padding_mode="reflect", domain="direct", device="cpu"):
    """
    Convolution between image and PSF.
    ### Parameters:
    - `x` : image, numpy array.
    - `PSF` : point spread function, numpy array.
    - `padding_mode` : padding mode, `reflect` or `constant`.
    - `domain` : convolution domain, `direct` or `fft`.
        - `direct` : direct convolution.
        - `fft` : FFT convolution. `FT(x) * FT(PSF) = FT(x * PSF)`.
    ### Returns:
    - `out` : convolved image, numpy array.
    """
    dim, dim_x = PSF.ndim, x.ndim
    assert dim in [2, 3] and dim_x in [2, 3], "Only support 2D and 3D convolution."
    assert dim == dim_x, "Dimensions of image and PSF mismatch."
    assert domain in ["direct", "fft"], "Only support 'direct' and 'fft' convolution."
    assert (device == "cpu") or (
        "cuda" in device
    ), "Only support 'cpu' and 'cuda' device."

    device = torch.device(device)

    ks = PSF.shape
    PSF, x = torch.tensor(PSF[None, None]), torch.tensor(x[None, None])
    PSF = torch.round(PSF, decimals=16)

    # move to device
    PSF = PSF.to(device)
    x = x.to(device)

    # 3D convolution -----------------------------------------------------------
    if dim == 3:
        # padding
        x_pad = torch.nn.functional.pad(
            input=x,
            pad=(
                ks[2] // 2,
                ks[2] // 2,
                ks[1] // 2,
                ks[1] // 2,
                ks[0] // 2,
                ks[0] // 2,
            ),
            mode=padding_mode,
        )
        if domain == "direct":
            x_conv = torch.nn.functional.conv3d(input=x_pad, weight=PSF)
        if domain == "fft":
            x_conv = fft_conv(signal=x_pad, kernel=PSF)
    # 2D convolution -----------------------------------------------------------
    if dim == 2:
        x_pad = torch.nn.functional.pad(
            input=x,
            pad=(ks[1] // 2, ks[1] // 2, ks[0] // 2, ks[0] // 2),
            mode=padding_mode,
        )
        if domain == "direct":
            x_conv = torch.nn.functional.conv2d(input=x_pad, weight=PSF)
        if domain == "fft":
            x_conv = fft_conv(signal=x_pad, kernel=PSF)
    if device == "cpu":
        out = x_conv.numpy()
    else:
        out = x_conv.cpu().numpy()
    out = out[0, 0]
    del x_pad, x_conv, PSF, x
    return out


class Convolution(torch.nn.Module):
    """
    Convolution class.
    ### Parameters:
    - `PSF`(torch tensor) : point spread function with a shape of (Sx, Sy, Sz) or (Sx, Sy).
    - `padding_mode` (str) : padding mode, `reflect` or `constant`.
    - `domain` (str) : convolution domain, `direct` or `fft`.
        - `direct` : direct convolution.
        - `fft` : FFT convolution. `FT(x) * FT(PSF) = FT(x * PSF)`.
    """

    def __init__(self, PSF, padding_mode="reflect", domain="direct", **kwargs):
        super().__init__()
        self.domain = domain
        self.padding_mode = padding_mode

        assert torch.is_tensor(PSF), "PSF must be a torch tensor."
        self.dim = PSF.ndim
        assert self.dim in [2, 3], "Only support 2D and 3D convolution."
        assert self.domain in [
            "direct",
            "fft",
        ], "Only support 'direct' and 'fft' convolution."
        assert self.padding_mode in [
            "reflect",
            "constant",
        ], "Only support 'reflect' and 'constant' padding mode."

        self.ks = PSF.shape
        self.PSF = torch.round(PSF, decimals=16)[None, None]

    def forward(self, x):
        """
        Run convolution x * PSF.
        ### Parameters:
        - `x` (torch tensor): image with a shape of (Sx, Sy, Sz) or (Sx, Sy).
        ### Returns:
        - `out` (torch tensor): convolved image with a shape of (Sx, Sy, Sz) or (Sx, Sy).
        """
        assert torch.is_tensor(x), "Input must be a torch tensor."
        dim_x = x.ndim
        assert dim_x in [2, 3], "Only support 2D and 3D convolution."
        assert dim_x == self.dim, "Dimensions of image and PSF mismatch."

        x = x[None, None]

        # 3D convolution -------------------------------------------------------
        if self.dim == 3:
            # padding
            x_pad = torch.nn.functional.pad(
                input=x,
                pad=(
                    self.ks[2] // 2,
                    self.ks[2] // 2,
                    self.ks[1] // 2,
                    self.ks[1] // 2,
                    self.ks[0] // 2,
                    self.ks[0] // 2,
                ),
                mode=self.padding_mode,
            )
            if self.domain == "direct":
                x_conv = torch.nn.functional.conv3d(input=x_pad, weight=self.PSF)
            if self.domain == "fft":
                x_conv = fft_conv(signal=x_pad, kernel=self.PSF)
        # 2D convolution -------------------------------------------------------
        if self.dim == 2:
            x_pad = torch.nn.functional.pad(
                input=x,
                pad=(
                    self.ks[1] // 2,
                    self.ks[1] // 2,
                    self.ks[0] // 2,
                    self.ks[0] // 2,
                ),
                mode=self.padding_mode,
            )
            if self.domain == "direct":
                x_conv = torch.nn.functional.conv2d(input=x_pad, weight=self.PSF)
            if self.domain == "fft":
                x_conv = fft_conv(signal=x_pad, kernel=self.PSF)
        out = x_conv[0, 0]
        return out


class Deconvolution(object):
    """
    Deconvolution class.

    ### Parameters:
    - `PSF` (numpy array): point spread function with a shape of (Sx, Sy, Sz) or (Sx, Sy).
    - `bp_type` (str): back projection type, `traditional`, `gaussian`, `butterworth`, `wiener-butterworth`.
    - `alpha` (float): alpha parameter for `wiener-butterworth` filter.
    - `beta` (float): beta parameter for `butterworth` filter.
    - `n` (int): n parameter for `butterworth` filter.
    - `res_flag` : residual flag for `butterworth` filter.
    - `i_res` : i_res parameter for `butterworth` filter.
    - `init` : initialization method, `measured` or `constant`.
    - `metrics` : evaluation metrics, `None` or `lambda x: [mse, ssim, ncc]`.
    - `padding_mode` : padding mode, `reflect` or `constant`.

    ### Returns:
    - `stack_estimate` : deconvolved image, numpy array.
    - `out_gaus_metrics` : evaluation metrics, numpy array.
    """

    def __init__(
        self,
        PSF,
        bp_type: str = "traditional",
        alpha: float = 0.05,
        beta: float = 1,
        n: int = 10,
        res_flag: int = 1,
        i_res=[2.44, 2.44, 10],
        init: str = "measured",
        metrics=None,
        padding_mode: str = "reflect",
        device_id="cpu",
        **kwargs,
    ):
        assert PSF.ndim in [2, 3], "Only support 2D and 3D PSF."
        type_list = [
            "traditional",
            "gaussian",
            "butterworth",
            "wiener-butterworth",
        ]
        assert bp_type in type_list, f"Only support {type_list} back projection type."
        assert init in [
            "measured",
            "constant",
        ], "Only support 'measured' and 'constant' initialization."
        assert padding_mode in [
            "reflect",
            "constant",
        ], "Only support 'reflect' and 'constant' padding mode."
        assert (device_id == "cpu") or (
            "cuda" in device_id
        ), "Only support 'cpu' and 'cuda' device."

        self.device_id = device_id
        self.device = torch.device(device_id)

        self.padding_mode = padding_mode
        self.bp_type = bp_type
        self.smallValue = 0.001
        self.init = init
        self.metrics = metrics
        self.metrics_value = []

        # forward PSF
        self.PSF1 = PSF / np.sum(PSF)  # normalize PSF

        # generate backward PSF
        self.PSF2, _ = back_projector.BackProjector(
            PSF_fp=PSF,
            bp_type=bp_type,
            alpha=alpha,
            beta=beta,
            n=n,
            res_flag=res_flag,
            i_res=i_res,
            verbose_flag=0,
        )
        self.PSF2 = self.PSF2 / np.sum(self.PSF2)

        self.OTF_fp = None
        self.OTF_bp = None

    def measure(self, stack):
        self.metrics_value.append(self.metrics(stack))

    def deconv(self, stack, num_iter: int = 2, domain="fft", verbose: bool = True):
        """
        Deconvolution function.

        ### Parameters:
        - `stack` : stack of images, numpy array.
        - `num_iter` : number of iterations.
        - `domain` : convolution domain, `direct` or `fft`.
            - `direct` : direct convolution.
            - `fft` : FFT convolution. `FT(x) * FT(PSF) = FT(x * PSF)`.

        ### Returns:
        - `stack_estimate` : deconvolved image, numpy array.
        """
        assert domain in ["direct", "fft"], "Only support direct and fft convolution."
        dim_img = stack.ndim
        assert dim_img in [2, 3], "Only support 2D and 3D image."
        assert dim_img == self.PSF1.ndim, "PSF and image dimension mismatch."

        self.metrics_value = []

        log = print if verbose else lambda *args, **kwargs: None

        log("-" * 50)
        log("[INFO] Convolution in", domain, "domain.")
        log("[INFO] BP Type:", self.bp_type)
        log("[INFO] PSF shape (FP/BP):", self.PSF1.shape, self.PSF2.shape)

        size = stack.shape
        PSF_fp = adjust_size(self.PSF1, size)
        PSF_bp = adjust_size(self.PSF2, size)

        self.OTF_fp = np.fft.fftn(np.fft.ifftshift(PSF_fp))
        self.OTF_bp = np.fft.fftn(np.fft.ifftshift(PSF_bp))

        stack = np.maximum(stack, self.smallValue)

        # initialization, the inital guess of the deconvolved image.
        log("[INFO] Initialization : ", self.init)
        if self.init == "constant":
            stack_estimate = np.ones(shape=stack.shape) * np.mean(stack)
        else:
            stack_estimate = stack

        if self.metrics is not None:
            self.measure(stack_estimate)

        # ----------------------------------------------------------------------
        dict_conv = dict(
            padding_mode=self.padding_mode, domain=domain, device=self.device_id
        )

        # move to device
        stack = torch.tensor(stack).to(self.device)
        stack_estimate = torch.tensor(stack_estimate).to(self.device)
        smallValue = torch.tensor(self.smallValue).to(self.device)
        PSF_fp = torch.tensor(PSF_fp).to(self.device)
        PSF_bp = torch.tensor(PSF_bp).to(self.device)
        conv_fp = Convolution(PSF_fp, **dict_conv).to(self.device)
        conv_bp = Convolution(PSF_bp, **dict_conv).to(self.device)

        # iterations
        pbar = tqdm.tqdm(desc="DECONV", total=num_iter, ncols=50, disable=not verbose)
        for i in range(num_iter):
            # ------------------------------------------------------------------
            # fp = ConvFFT3_S(stack_estimate, self.OTF_fp)
            # # dv = stack / (fp + self.smallValue)
            # dv = stack / fp
            # bp = ConvFFT3_S(dv, self.OTF_bp)
            # ------------------------------------------------------------------
            fp = conv_fp(stack_estimate)
            dv = stack / (fp + smallValue)
            bp = conv_bp(dv)
            stack_estimate = stack_estimate * bp
            stack_estimate = torch.maximum(stack_estimate, smallValue)

            if self.metrics is not None:
                self.measure(stack_estimate)

            pbar.update(1)
        pbar.close()

        if self.device_id == "cpu":
            stack_estimate = stack_estimate.numpy()
        elif "cuda" in self.device_id:
            stack_estimate = stack_estimate.cpu().numpy()
        return stack_estimate

    def get_metrics(self):
        return np.stack(self.metrics_value)
