import numpy as np
import methods.back_projector as back_projector
import tqdm, torch
from fft_conv_pytorch import fft_conv


# ------------------------------------------------------------------------------
def align_size(img, size, pad_value=0):
    """
    Align image size to the given size.
    ### Parameters:
    - `img` : image, numpy array.
    - `size` : target size, tuple.
    - `pad_value` : padding value, float.
    ### Returns:
    - `img2` : aligned image, numpy array.
    """
    dim = len(img.shape)
    assert dim in [2, 3], "Only support 2D and 3D image."
    assert len(size) == dim, "Size and image dimension mismatch."

    if dim == 3:
        Nz_1, Ny_1, Nx_1 = img.shape
        Nz_2, Ny_2, Nx_2 = size
        Nz, Ny, Nx = (
            np.maximum(Nz_1, Nz_2),
            np.maximum(Ny_1, Ny_2),
            np.maximum(Nx_1, Nx_2),
        )

        imgTemp = np.ones(shape=(Nz, Ny, Nx)) * pad_value
        imgTemp = imgTemp.astype(img.dtype)

        Nz_o, Ny_o, Nx_o = (
            int(np.round((Nz - Nz_1) / 2)),
            int(np.round((Ny - Ny_1) / 2)),
            int(np.round((Nx - Nx_1) / 2)),
        )
        imgTemp[Nz_o : Nz_o + Nz_1, Ny_o : Ny_o + Ny_1, Nx_o : Nx_o + Nx_1] = img

        Nz_o, Ny_o, Nx_o = (
            int(np.round((Nz - Nz_2) / 2)),
            int(np.round((Ny - Ny_2) / 2)),
            int(np.round((Nx - Nx_2) / 2)),
        )
        img2 = imgTemp[Nz_o : Nz_o + Nz_2, Ny_o : Ny_o + Ny_2, Nx_o : Nx_o + Nx_2]

    if dim == 2:
        Ny_1, Nx_1 = img.shape
        Ny_2, Nx_2 = size
        Ny, Nx = np.maximum(Ny_1, Ny_2), np.maximum(Nx_1, Nx_2)

        imgTemp = np.ones(shape=(Ny, Nx)) * pad_value
        imgTemp = imgTemp.astype(img.dtype)

        Ny_o, Nx_o = int(np.round((Ny - Ny_1) / 2)), int(np.round((Nx - Nx_1) / 2))
        imgTemp[Ny_o : Ny_o + Ny_1, Nx_o : Nx_o + Nx_1] = img

        Ny_o, Nx_o = int(np.round((Ny - Ny_2) / 2)), int(np.round((Nx - Nx_2) / 2))
        img2 = imgTemp[Ny_o : Ny_o + Ny_2, Nx_o : Nx_o + Nx_2]

    return img2


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


def ConvFFT3_S(inVol, OTF):
    outVol = np.fft.ifftn(np.fft.fftn(inVol) * OTF)
    return outVol.real


def Convolution(x, PSF, padding_mode="reflect", domain="direct"):
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
    ks, xs = PSF.shape, x.shape
    dim, dim_x = len(ks), len(xs)

    # check dimensions
    assert dim in [2, 3] and dim_x in [2, 3], "Only support 2D and 3D convolution."
    assert domain in ["direct", "fft"], "Only support direct and fft convolution."

    PSF, x = torch.tensor(PSF[None, None]), torch.tensor(x[None, None])
    PSF = torch.round(PSF, decimals=16)
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
    out = x_conv.numpy()[0, 0]
    return out


class Deconvolution(object):
    """
    Deconvolution class.
    ### Parameters:
    - `PSF` : point spread function, numpy array.
    - `bp_type` : back projection type, `traditional`, `gaussian`, `butterworth`, `wiener-butterworth`.
    - `alpha` : alpha parameter for wiener-butterworth filter.
    - `beta` : beta parameter for butterworth filter.
    - `n` : n parameter for butterworth filter.
    - `res_flag` : residual flag for butterworth filter.
    - `i_res` : i_res parameter for butterworth filter.
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
    ):

        self.padding_mode = padding_mode
        self.bp_type = bp_type
        # forward PSF
        self.PSF1 = PSF / np.sum(PSF)  # normalize PSF
        # backward PSF
        self.PSF2, _ = back_projector.BackProjector(
            PSF_fp=PSF,
            bp_type=bp_type,
            alpha=alpha,
            beta=beta,
            n=n,
            res_flag=res_flag,
            i_res=i_res,
        )
        self.PSF2 = self.PSF2 / np.sum(self.PSF2)

        self.smallValue = 0.001
        self.init = init

        self.metrics = metrics
        self.metrics_value = []

        self.OTF_fp = None
        self.OTF_bp = None

    def measure(self, stack):
        self.metrics_value.append(self.metrics(stack))

    def deconv(self, stack, num_iter, domain="fft"):
        self.metrics_value = []
        print("=" * 80)
        print(">> Convolution in", domain, "domain.")
        print(">> BP Type:", self.bp_type)
        print(">> PSF shape (FP/BP):", self.PSF1.shape, self.PSF2.shape)

        size = stack.shape
        PSF_fp = align_size(self.PSF1, size)
        PSF_bp = align_size(self.PSF2, size)

        self.OTF_fp = np.fft.fftn(np.fft.ifftshift(PSF_fp))
        self.OTF_bp = np.fft.fftn(np.fft.ifftshift(PSF_bp))

        stack = np.maximum(stack, self.smallValue)

        # initialization
        print(">> Initialization : ", self.init)
        if self.init == "constant":
            stack_estimate = np.ones(shape=stack.shape) * np.mean(stack)
        else:
            stack_estimate = stack

        # iterations
        pbar = tqdm.tqdm(desc="Deconvolution", total=num_iter, ncols=80)

        if self.metrics is not None:
            self.measure(stack_estimate)

        for i in range(num_iter):
            # if domain == 'frequency':
            #     fp = ConvFFT3_S(stack_estimate, self.OTF_fp)
            #     # dv = stack / (fp + self.smallValue)
            #     dv = stack / fp
            #     bp = ConvFFT3_S(dv, self.OTF_bp)

            # if domain == 'spatial':
            fp = Convolution(
                stack_estimate,
                PSF=self.PSF1,
                padding_mode=self.padding_mode,
                domain=domain,
            )
            dv = stack / (fp + self.smallValue)
            bp = Convolution(
                dv, PSF=self.PSF2, padding_mode=self.padding_mode, domain=domain
            )
            stack_estimate = stack_estimate * bp
            stack_estimate = np.maximum(stack_estimate, self.smallValue)

            if self.metrics is not None:
                self.measure(stack_estimate)

            pbar.update(1)
        pbar.close()
        return stack_estimate

    def get_metrics(self):
        return np.stack(self.metrics_value)
