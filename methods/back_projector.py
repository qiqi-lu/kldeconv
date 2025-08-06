"""
Back projector construction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths
from skimage import io
import os


def align_size(img1, Sx2, Sy2, Sz2, padValue=0):
    Sx1, Sy1, Sz1 = img1.shape
    Sx, Sy, Sz = np.maximum(Sx1, Sx2), np.maximum(Sy1, Sy2), np.maximum(Sz1, Sz2)
    imgTemp = np.ones(shape=(Sx, Sy, Sz)) * padValue

    Sox, Soy, Soz = (
        int(np.round((Sx - Sx1) / 2)),
        int(np.round((Sy - Sy1) / 2)),
        int(np.round((Sz - Sz1) / 2)),
    )
    imgTemp[Sox : Sox + Sx1, Soy : Soy + Sy1, Soz : Soz + Sz1] = img1

    Sox, Soy, Soz = (
        int(np.round((Sx - Sx2) / 2)),
        int(np.round((Sy - Sy2) / 2)),
        int(np.round((Sz - Sz2) / 2)),
    )
    img2 = imgTemp[Sox : Sox + Sx2, Soy : Soy + Sy2, Soz : Soz + Sz2]
    return img2


def lin_interp(x, y, i, half):
    """
    Linear interpolation.
    ```
    |---- y(i) ---- half ---- y(i+1) ----|
    |---- x(i) -- x_interp -- x(i+1) ----|
    ```
    ### Parameters:
    - `x` : x-axis, numpy array.
    - `y` : y-axis, numpy array.
    - `i` : index of x-axis.
    - `half` : half value of y-axis.
    ### Returns:
    - `x_interp` : interpolated value of x-axis.
    """
    assert len(x) == len(y), "x and y should have the same length."
    assert i >= 0 and i < len(x) - 1, "i should be in the range of [0, len(x)-1]."
    assert (half >= y[i] and half <= y[i + 1]) or (
        half >= y[i + 1] and half <= y[i]
    ), "half should be in the range of [y[i], y[i+1]]."

    x_interp = x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))
    return x_interp


def FWHM_1d(y):
    """
    Calculate FWHM of 1D signal.
    ### Parameters:
    - `y` : 1D signal, numpy array.
    ### Returns:
    - `dist` : FWHM of 1D signal.
    """
    y = np.squeeze(y)
    assert len(y.shape) == 1, "Input signal should be 1D."

    # find the two zero crossings
    x = np.linspace(start=0, stop=y.shape[0] - 1, num=y.shape[0])
    half = max(y) / 2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = signs[0:-2] != signs[1:-1]
    zero_crossings_i = np.where(zero_crossings)[0]

    if len(zero_crossings_i) != 2:
        raise ValueError(
            f"Signal should contain exactly 2 zero crossings, but got {len(zero_crossings_i)}"
        )

    # calculate the distance between the two zero crossings
    dist = lin_interp(x, y, zero_crossings_i[1], half) - lin_interp(
        x, y, zero_crossings_i[0], half
    )
    return dist


def FWHM_PSF(PSF, pixel_size: float | int = 1.0):
    """
    Calculate FWHM of PSF.
    ### Parameters:
    - `PSF` : point spread function, numpy array.
    - `pixel_size` : pixel size in spatial domain.
    ### Returns:
    - `FWHMs` : FWHM of PSF in spatial domain.
        - if `PSF` is 2D, `FWHMs` is a list of two elements, i.e., `[FWHMx, FWHMy]`.
        - if `PSF` is 3D, `FWHMs` is a list of three elements, i.e., `[FWHMx, FWHMy, FWHMz]`.
    """
    dim = len(PSF.shape)
    assert dim in [2, 3], "Only support PSF with shape (Sx, Sy) or (Sx, Sy, Sz)."
    if dim == 2:
        Sx, Sy = PSF.shape
        indx, indy = Sx // 2, Sy // 2
        FWHMx = FWHM_1d(PSF[:, indy]) * pixel_size
        FWHMy = FWHM_1d(PSF[indx]) * pixel_size
        FWHMs = [FWHMx, FWHMy]

    if dim == 3:
        Sx, Sy, Sz = PSF.shape
        indx, indy, indz = Sx // 2, Sy // 2, Sz // 2
        FWHMx = FWHM_1d(PSF[:, indy, indz]) * pixel_size
        FWHMy = FWHM_1d(PSF[indx, :, indz]) * pixel_size
        FWHMz = FWHM_1d(PSF[indx, indy]) * pixel_size
        FWHMs = [FWHMx, FWHMy, FWHMz]
    return FWHMs


def sigma2FWHM(sigmas):
    """
    Convert sigma to FWHM.
    The conversion formula used is: FWHM = sigma * 2\sqrt(2ln2) ≈ sigma * 2.3548
    ### Parameters:
    - `sigmas` (list) : sigma of PSF in spatial domain.
    ### Returns:
    - `FWHMs` (list) : FWHM of PSF in spatial domain.
    """
    sigmas = np.array(sigmas) * 2.3548
    return sigmas.tolist()


def FWHM2sigma(FWHM):
    """
    Convert FWHM to sigma.
    The conversion formula used is: sigma = FWHM / (2\sqrt(2ln2)) ≈ FWHM / 2.3548
    ### Parameters:
    - `FWHM` (list) : FWHM of PSF in spatial domain.
    ### Returns:
    - `sigmas` (list) : sigma of PSF in spatial domain.
    """
    FWHM = np.array(FWHM) / 2.3548
    return FWHM.tolist()


def PSF_gaussian(size, sigmas):
    dim = len(size)
    assert (
        len(sigmas) == dim
    ), ">> The length of FWHMs should be same as the length of size."
    if dim == 2:
        Sx, Sy = size
        sigma_x, sigma_y = sigmas
        coef = 1 / (2 * np.pi * sigma_x * sigma_y)

        i, j = np.mgrid[-Sx // 2 + 1 : Sx // 2 + 1, -Sy // 2 + 1 : Sy // 2 + 1]
        if (Sx % 2) == 0:
            i = (i - 0.5).astype(np.float32)
        if (Sy % 2) == 0:
            j = (j - 0.5).astype(np.float32)
        PSF = np.exp(-((i**2 / (2.0 * sigma_x**2) + j**2 / (2.0 * sigma_y**2))))

    if dim == 3:
        Sx, Sy, Sz = size
        sigma_x, sigma_y, sigma_z = sigmas
        coef = 1 / ((2 * np.pi) ** (3 / 2) * sigma_x * sigma_y * sigma_z)

        i, j, k = np.mgrid[
            -Sx // 2 + 1 : Sx // 2 + 1,
            -Sy // 2 + 1 : Sy // 2 + 1,
            -Sz // 2 + 1 : Sz // 2 + 1,
        ]
        if (Sx % 2) == 0:
            i = (i - 0.5).astype(np.float32)
        if (Sy % 2) == 0:
            j = (j - 0.5).astype(np.float32)
        if (Sz % 2) == 0:
            k = (k - 0.5).astype(np.float32)
        PSF = np.exp(
            -(
                (
                    i**2 / (2.0 * sigma_x**2)
                    + j**2 / (2.0 * sigma_y**2)
                    + k**2 / (2.0 * sigma_z**2)
                )
            )
        )

    PSF = coef * PSF
    return PSF


def BackProjector(
    PSF_fp,
    bp_type="traditional",
    alpha=0.001,
    beta=1,
    n=10,
    res_flag=1,
    i_res=[0, 0, 0],
    verbose_flag=0,
):
    """
    Gnerator backrpojector according to the given PSF.
    ### Parameters:
    - `PSF_fp`: Forward projector (i.e., PSF).
    - `bp_type`: `traditional`, `gaussian`, `butterworth`, `wiener`, `wiener-butterworth`.
    - `alpha`: [0.0001, 0.001] or 1 (use OTF (optical transfer function) value of the PSF_bp at resolution limit).
    - `beta`: [0.001, 0.01] or 1 (use OTF value of PSF_bp at resolution limit).
    - `n`: [4, 15], order of Butterworth filter.
    - `res_flag`:
        - `0` (use PSF_fp FWHM/root(2) as resolution limit (for iSIM)),
        - `1` (use PSF_fp FWHM as resoltuion limit),
        - `2` (use input values (iRes) as resoltuion limit).
    - `i_res`:    input resolution limit in 3 dimensions in terms of pixels.
    - `verbose_flag`:
        - `0` : hide log and intermediate results,
        - `1` : show
    ### Output:
    - `PSF_bp`: BP kernel.
    - `OTF_bp`: OTF of BP kernel.
    """

    dim = len(PSF_fp.shape)
    assert dim in [
        2,
        3,
    ], "[ERROR] Only support PSF with shape (Sx, Sy) or (Sx, Sy, Sz)."

    if verbose_flag:
        print("-" * 80)
        print(f"[INFO] Back projector type: {bp_type} ({dim}D)")

    type_list = [
        "traditional",
        "gaussian",
        "butterworth",
        "wiener",
        "wiener-butterworth",
    ]

    assert (
        bp_type in type_list
    ), f"[ERROR] Unsupported back projector type. Only support {type_list}"

    assert res_flag in [
        0,
        1,
        2,
    ], "[ERROR] Unsupported res_flag value. Only support 0, 1, 2."

    # --------------------------------------------------------------------------
    # Traditional backward projection, i.e., PSF_bp = PSF_fp^T
    if bp_type == "traditional":
        PSF_bp = np.flip(PSF_fp)
        OTF_bp = np.fft.fftn(np.fft.ifftshift(PSF_bp))
        if verbose_flag:
            print("-" * 80)
        return PSF_bp.astype(np.float32), OTF_bp.astype(np.complex64)
    # --------------------------------------------------------------------------
    if dim == 2:
        Sx, Sy = PSF_fp.shape
        Scx, Scy = (Sx - 1) / 2, (Sy - 1) / 2
        Sox, Soy = int(np.round((Sx - 1) / 2)), int(np.round((Sy - 1) / 2))

        # Calculate PSF and OTF size
        # calculate the FWHM of PSF
        FWHMx, FWHMy = FWHM_PSF(PSF_fp)
        sigma_x, sigma_y = FWHM2sigma(FWHMx), FWHM2sigma(FWHMy)

        if verbose_flag:
            print(f"[INFO] FWHM  = {FWHMx:>.4f} x {FWHMy:>.4f}")
            print(f"[INFO] Sigma = {sigma_x:>.4f} x {sigma_y:>.4f}")

        # set resolution cutoff
        if res_flag == 0:
            # set resolution as 1/root(2) of PSF_fp FWHM: iSIM case
            resx, resy = FWHMx / (2**0.5), FWHMy / (2**0.5)
        if res_flag == 1:
            # set resolution as PSF_fp FWHM
            resx, resy = FWHMx, FWHMy
        if res_flag == 2:
            # set resolution based on input values
            resx, resy = i_res

        # pixel size in Fourier domain
        px, py = 1 / Sx, 1 / Sy

        # frequency cutoff in terms of pixels
        tx, ty = (1 / resx) / px, (1 / resy) / py

        if verbose_flag:
            print(
                f"[INFO] Resolution cut-off in spatial domain : {resx:.4f} x {resy:.4f}"
            )
            print(f"[INFO] Resolution cut-off in Fourier domain : {tx:.4f} x {ty:.4f}")

    if dim == 3:
        Sx, Sy, Sz = PSF_fp.shape
        Scx, Scy, Scz = (Sx - 1) / 2, (Sy - 1) / 2, (Sz - 1) / 2
        Sox, Soy, Soz = (
            int(np.round((Sx - 1) / 2)),
            int(np.round((Sy - 1) / 2)),
            int(np.round((Sz - 1) / 2)),
        )

        # Calculate PSF and OTF size
        FWHMx, FWHMy, FWHMz = FWHM_PSF(PSF_fp)
        sigma_x, sigma_y, sigma_z = (
            FWHM2sigma(FWHMx),
            FWHM2sigma(FWHMy),
            FWHM2sigma(FWHMz),
        )

        if verbose_flag:
            print(f"[INFO] FWHM  = {FWHMx:>.4f} x {FWHMy:>.4f} x {FWHMz:>.4f} (pixels)")
            print(
                f"[INFO] Sigma = {sigma_x:>.4f} x {sigma_y:>.4f} x {sigma_z:>.4f} (pixels)"
            )

        # set resolution cutoff
        if res_flag == 0:
            resx, resy, resz = FWHMx / (2**0.5), FWHMy / (2**0.5), FWHMz / (2**0.5)
        if res_flag == 1:
            resx, resy, resz = FWHMx, FWHMy, FWHMz
        if res_flag == 2:
            resx, resy, resz = i_res

        # pixel size in Fourier domain
        px, py, pz = 1 / Sx, 1 / Sy, 1 / Sz

        # frequency cutoff in terms of pixels
        tx, ty, tz = (1 / resx) / px, (1 / resy) / py, (1 / resz) / pz

        if verbose_flag:
            print(
                f"[INFO] Resolution cutoff in spatial domain : {resx:.4f} x {resy:.4f} x {resz:.4f}"
            )
            print(
                f"[INFO] Resolution cutoff in Fourier domain : {tx:.4f} x {ty:.4f} x {tz:.4f}"
            )

    # normalize flipped PSF: traditional back projector
    PSF_flipped = np.flip(PSF_fp)
    OTF_flip = np.fft.fftn(np.fft.ifftshift(PSF_flipped))
    OTF_abs = np.fft.fftshift(np.abs(OTF_flip))
    OTF_max = np.max(OTF_abs)
    M = OTF_max
    OTF_abs_norm = OTF_abs / M

    # check cutoff gains of traditional back projector
    if dim == 2:
        tline = np.max(OTF_abs_norm, axis=1)
        to1 = int(np.maximum(np.round(Scx - tx), 0))
        to2 = int(np.minimum(np.round(Scx + tx), Sx - 1))
        beta_fpx = (tline[to1] + tline[to2]) / 2  # OTF frequency intensity as cutoff: y

        tline = np.max(OTF_abs_norm, axis=0)
        to1 = int(np.maximum(np.round(Scy - ty), 0))
        to2 = int(np.minimum(np.round(Scy + ty), Sy - 1))
        beta_fpy = (tline[to1] + tline[to2]) / 2  # OTF frequency intensity as cutoff: x

        beta_fp = (beta_fpx + beta_fpy) / 2
        if verbose_flag:
            print(
                f"[INFO] Cutoff gain of forward projector : {beta_fpx:>.4f} x {beta_fpy:>.4f}, average = {beta_fp:>.4f}"
            )

    if dim == 3:
        tplane = np.max(OTF_abs_norm, axis=2)
        tline = np.max(tplane, axis=1)
        to1 = int(np.maximum(np.round(Scx - tx), 0))
        to2 = int(np.minimum(np.round(Scx + tx), Sx - 1))
        beta_fpx = (tline[to1] + tline[to2]) / 2  # OTF frequency intensity as cutoff: x

        tplane = np.max(OTF_abs_norm, axis=2)
        tline = np.max(tplane, axis=0)
        to1 = int(np.maximum(np.round(Scy - ty), 0))
        to2 = int(np.minimum(np.round(Scy + ty), Sy - 1))
        beta_fpy = (tline[to1] + tline[to2]) / 2  # OTF frequency intensity as cutoff: y

        tplane = np.max(OTF_abs_norm, axis=0)
        tline = np.max(tplane, axis=0)
        to1 = int(np.maximum(np.round(Scz - tz), 0))
        to2 = int(np.minimum(np.round(Scz + tz), Sz - 1))
        beta_fpz = (tline[to1] + tline[to2]) / 2  # OTF frequency intensity as cutoff: z

        beta_fp = (beta_fpx + beta_fpy + beta_fpz) / 3
        if verbose_flag:
            print(
                f"[INFO] Cutoff gain of forward projector : {beta_fpx:>.4f} x {beta_fpy:>.4f} x {beta_fpz:>.4f}, average = {beta_fp:>.4f}"
            )

    if alpha == 1:
        alpha = beta_fp
        if verbose_flag:
            print(
                f"[INFO] wiener parameter adjusted as traditional BP cutoff gain: alpha = {alpha:>.4f}"
            )
    else:
        if verbose_flag:
            print(f"[INFO] Wiener parameter set as input: alpha = {alpha:>.4f}")

    if beta == 1:
        beta = beta_fp
        if verbose_flag:
            print(
                f"[INFO] Cutoff gain adjusted as traditional BP cutoff gain: beta = {beta:>.4f}"
            )
    else:
        if verbose_flag:
            print(f"[INFO] Cutoff gain set as input: beta = {beta:>.4f}")

    # order of Butterworth filter
    if verbose_flag:
        print(f"[INFO] Butterworth order (slope parameter) set as: n = {n}")

    # --------------------------------------------------------------------------
    if bp_type == "gaussian":
        if dim == 2:
            resx, resy = FWHMx, FWHMy
            PSF_bp = PSF_gaussian(size=[Sx, Sy], sigmas=FWHM2sigma([resx, resy]))

        if dim == 3:
            resx, resy, resz = FWHMx, FWHMy, FWHMz
            PSF_bp = PSF_gaussian(
                size=[Sx, Sy, Sz], sigmas=FWHM2sigma([resx, resy, resz])
            )

        OTF_bp = np.fft.fftn(np.fft.ifftshift(PSF_bp))

    # --------------------------------------------------------------------------
    if bp_type == "butterworth":
        ee = 1 / beta**2 - 1
        # create Butterworth filter (2D)
        if dim == 2:
            kcx, kcy = tx, ty  # width of Butterworth Filter
            i, j = np.mgrid[0 : Sx - 1, 0 : Sy - 1]
            w = ((i - Scx) / kcx) ** 2 + ((j - Scy) / kcy) ** 2
            mask = 1 / np.sqrt(1 + ee * (w**n))

        # create Butterworth filter (3D)
        if dim == 3:
            kcx, kcy, kcz = tx, ty, tz  # width of Butterworth Filter
            i, j, k = np.mgrid[0:Sx, 0:Sy, 0:Sz]
            w = ((i - Scx) / kcx) ** 2 + ((j - Scy) / kcy) ** 2 + ((k - Scz) / kcz) ** 2
            mask = 1 / np.sqrt(1 + ee * w**n)  # w^n = (kx/kcx)^pn

        OTF_bp = np.fft.ifftshift(mask)
        PSF_bp = np.fft.fftshift(np.real(np.fft.ifftn(OTF_bp)))

    # --------------------------------------------------------------------------
    if bp_type == "wiener":
        OTF_flip_norm = OTF_flip / M  # Normalized OTF_flip

        OTF_bp = OTF_flip_norm / (abs(OTF_flip_norm) ** 2 + alpha)  # Wiener filter
        PSF_bp = np.fft.fftshift(np.real(np.fft.ifftn(OTF_bp)))

    # --------------------------------------------------------------------------
    if bp_type == "wiener-butterworth":
        # create Wiener filter
        OTF_flip_norm = OTF_flip / M
        OTF_Wiener = OTF_flip_norm / (np.abs(OTF_flip_norm) ** 2 + alpha)

        # cut_off gain for wiener filter
        OTF_Wiener_abs = np.fft.fftshift(np.abs(OTF_Wiener))
        if dim == 2:
            tplane = np.abs(OTF_Wiener_abs)
        if dim == 3:
            tplane = np.abs(OTF_Wiener_abs[:, :, Soz])  # central slice

        tline = np.max(tplane, axis=1)
        to1 = int(np.maximum(np.round(Scx - tx), 0))
        to2 = int(np.minimum(np.round(Scx + tx), Sx - 1))
        beta_wienerx = (
            tline[to1] + tline[to2]
        ) / 2  # OTF frequency intensity at cutoff

        if verbose_flag:
            print(f"[INFO] Wiener cutoff gain: beta_wienerx  = {beta_wienerx}")

        ee = beta_wienerx / beta**2 - 1
        # create Butterworth filter (2D)
        if dim == 2:
            kcx, kcy = tx, ty  # width of Butterworth filter
            i, j = np.mgrid[0 : Sx - 1, 0 : Sy - 1]
            w = ((i - Scx) / kcx) ** 2 + ((j - Scy) / kcy) ** 2
            mask = 1 / np.sqrt(1 + ee * w**n)  # w^n = (kx/kcx)^pn

        # create Butterworth filter (3D)
        if dim == 3:
            kcx, kcy, kcz = tx, ty, tz  # width of Butterworth filter
            i, j, k = np.mgrid[0:Sx, 0:Sy, 0:Sz]
            w = ((i - Scx) / kcx) ** 2 + ((j - Scy) / kcy) ** 2 + ((k - Scz) / kcz) ** 2
            mask = 1 / np.sqrt(1 + ee * w**n)  # w^n = (kx/kcx)^pn

        mask = np.fft.ifftshift(mask)

        # create Wiener-Butterworth filter
        OTF_bp = mask * OTF_Wiener
        PSF_bp = np.fft.fftshift(np.real(np.fft.ifftn(OTF_bp)))

    if verbose_flag:
        print("-" * 50)
    return PSF_bp.astype(np.float32), OTF_bp.astype(np.complex64)


if __name__ == "__main__":
    std = None

    # std = [1.18, 1.18, 4.41]
    # size_image = [127, 127, 217]
    # size = np.array([127, 127, 127])
    # PSF = PSF_gaussian(size, std)

    # io.imsave('PSF_gauss.tif', PSF)

    # size_image = np.array([128, 128])
    # size = np.array([128, 128])
    # PSF = PSF_gaussian(size,[1.18, 1.18])

    PSF = io.imread("methods/PSF_iSIM.tif")
    PSF = np.transpose(PSF, axes=(2, 1, 0))

    # ------------------------------------------------------------------------------------===
    # padding PSF in the spatial domain == interpolation in the Fourier domain
    size = (np.array(PSF.shape).max(),) * 3
    PSF = align_size(PSF, size[0], size[1], size[2])

    size = np.array(PSF.shape)
    PSF = PSF / np.sum(PSF)
    dim = len(PSF.shape)
    if std == None:
        std = FWHM2sigma(FWHM_PSF(PSF=PSF))

    PSF_FT = np.fft.fftn(np.fft.ifftshift(PSF))
    PSF_FT_shift_abs = np.abs(np.fft.fftshift(PSF_FT))

    BP_trad, BP_trad_OTF = BackProjector(
        PSF,
        bp_type="traditional",
        alpha=0.001,
        beta=1,
        n=10,
        res_flag=1,
        i_res=[0, 0, 0],
        verbose_flag=1,
    )
    BP_gauss, BP_gauss_OTF = BackProjector(
        PSF,
        bp_type="gaussian",
        alpha=0.001,
        beta=1,
        n=10,
        res_flag=0,
        i_res=[0, 0, 0],
        verbose_flag=0,
    )
    BP_bw, BP_bw_OTF = BackProjector(
        PSF,
        bp_type="butterworth",
        alpha=0.001,
        beta=0.001,
        n=15,
        res_flag=0,
        i_res=[0, 0, 0],
        verbose_flag=0,
    )
    BP_wiener, BP_wiener_OTF = BackProjector(
        PSF,
        bp_type="wiener",
        alpha=0.001,
        beta=1,
        n=10,
        res_flag=1,
        i_res=[0, 0, 0],
        verbose_flag=0,
    )
    BP_wb, BP_wb_OTF = BackProjector(
        PSF,
        bp_type="wiener-butterworth",
        alpha=0.001,
        beta=0.001,
        n=12,
        res_flag=0,
        i_res=[0, 0, 0],
        verbose_flag=0,
    )

    # ------------------------------------------------------------------------------------===
    print(PSF.shape)
    print(np.sum(PSF))

    delta_x = 55
    delta_z = 55
    detla_u = 1.0 / (delta_x * size[0])

    # ------------------------------------------------------------------------------------===
    if dim == 3:

        def plot_curve(x_ft, axes, name, color):
            x_ft_shift_abs = np.abs(np.fft.fftshift(x_ft))
            x_ft_x_psf_ft_shift_abs = np.abs(np.fft.fftshift(x_ft * PSF_FT))
            axes[0].plot(
                x_ft_shift_abs[size[0] // 2, size[0] // 2 :, size[2] // 2],
                label=name,
                color=color,
            )
            axes[1].plot(
                x_ft_x_psf_ft_shift_abs[size[0] // 2, size[0] // 2 :, size[2] // 2],
                label=name,
                color=color,
            )

        nc, nr = 3, 1
        fig, axes = plt.subplots(
            nrows=nr,
            ncols=nc,
            dpi=300,
            figsize=(4.0 * nc, 4.0 * nr),
            constrained_layout=True,
        )
        axes[0].set_title("|FT(PSF)|")
        axes[0].plot(
            PSF_FT_shift_abs[size[0] // 2, size[1] // 2 :, size[2] // 2], color="blue"
        )
        axes[1].set_title("|FT(BP)|")
        axes[2].set_title("|FT(BP) x FT(PSF)|")
        plot_curve(BP_trad_OTF, [axes[1], axes[2]], name="traditional", color="blue")
        plot_curve(BP_gauss_OTF, [axes[1], axes[2]], name="gaussian", color="cyan")
        plot_curve(BP_bw_OTF, [axes[1], axes[2]], name="butterworth", color="orange")
        # plot_curve(BP_wiener_OTF, [axes[1], axes[2]], name='wiener', color='green')
        plot_curve(
            BP_wb_OTF, [axes[1], axes[2]], name="wiener-butterworth", color="orangered"
        )
        axes[1].legend()
        axes[2].legend()
        for ax in [axes[0], axes[1], axes[2]]:
            ax.set_xlim([0, None])
            ax.set_ylim([0, None])
            ax.set_xlabel("Frequency (kx, nm-1)")
            ax.set_ylabel("Normalized value")
            x_freq = np.array([0.0, 1.0 / 480.0, 1.0 / 240.0, 1.0 / 160.0, 1.0 / 120.0])
            x_freq_txt = ["0", "1/480", "1/240", "1/160", "1/120"]
            x_ticks = x_freq / detla_u
            ax.set_xticks(x_ticks, labels=x_freq_txt)

    plt.savefig(os.path.join("methods", "output", "BP_curves.png"))

    # os._exit(0)
    # ------------------------------------------------------------------------------------===
    if dim == 2:

        def show_kernel(x, x_ft, name, axes):
            x_ft_shift_abs = np.abs(np.fft.fftshift(x_ft))
            x_ft_x_psf_ft_shift_abs = np.abs(np.fft.fftshift(x_ft * PSF_FT))
            axes[0].imshow(x, cmap="gray", vmin=np.min(x), vmax=np.max(x)), axes[
                0
            ].set_title("BP ({})".format(name))
            axes[1].imshow(
                x_ft_shift_abs, cmap="gray", vmin=0, vmax=np.max(x_ft_shift_abs)
            ), axes[1].set_title("|FT(BP)|")
            axes[2].imshow(
                x_ft_x_psf_ft_shift_abs,
                cmap="gray",
                vmin=0,
                vmax=np.max(x_ft_x_psf_ft_shift_abs),
            ), axes[2].set_title("|FT(BP) x FT(PSF)|")
            axes[3].plot(
                x_ft_shift_abs[size[0] // 2, size[0] // 2 :], color="blue"
            ), axes[3].set_title("|FT(BP)|")
            axes[4].plot(
                x_ft_x_psf_ft_shift_abs[size[0] // 2, size[0] // 2 :], color="blue"
            ), axes[4].set_title("|FT(BP) x FT(PSF)|")

        nr, nc = 5, 6
        fig, axes = plt.subplots(
            nrows=nr,
            ncols=nc,
            dpi=300,
            figsize=(2.4 * nc, 2.4 * nr),
            constrained_layout=True,
        )

        axes[0, 0].imshow(PSF, cmap="gray", vmin=0, vmax=np.max(PSF)), axes[
            0, 0
        ].set_title("PSF")
        axes[1, 0].imshow(
            PSF_FT_shift_abs, cmap="gray", vmin=0, vmax=np.max(PSF_FT_shift_abs)
        ), axes[1, 0].set_title("|FT(PSF)|")
        axes[2, 0].set_axis_off()
        axes[3, 0].plot(PSF_FT_shift_abs[size[0] // 2, size[0] // 2 :]), axes[
            3, 0
        ].set_title("|FT(PSF)|", color="blue")
        axes[4, 0].set_axis_off()

        show_kernel(BP_trad, BP_trad_OTF, "traditional", axes=axes[:, 1])
        show_kernel(BP_gauss, BP_gauss_OTF, "gaussian", axes=axes[:, 2])
        show_kernel(BP_bw, BP_bw_OTF, "butterworth", axes=axes[:, 3])
        show_kernel(BP_wiener, BP_wiener_OTF, "wiener", axes=axes[:, 4])
        show_kernel(BP_wb, BP_wb_OTF, "wiener-butterworth", axes=axes[:, 5])

    if dim == 3:
        nr, nc = 4, 12

        def show_kernel_3d(x, x_ft, name, axes):
            x_ft_shift_abs = np.abs(np.fft.fftshift(x_ft))
            x_ft_x_psf_ft_shift_abs = np.abs(np.fft.fftshift(x_ft * PSF_FT))

            axes[0, 0].imshow(
                x[..., size[-1] // 2].transpose(),
                cmap="gray",
                vmin=np.min(x),
                vmax=np.max(x),
            ), axes[0, 0].set_title("BP ({}) (xy)".format(name))
            axes[0, 1].imshow(
                x[size[1] // 2, :, :].transpose(),
                cmap="gray",
                vmin=np.min(x),
                vmax=np.max(x),
            ), axes[0, 1].set_title("BP ({}) (xz)".format(name))
            axes[1, 0].imshow(
                x_ft_shift_abs[..., size[-1] // 2].transpose(),
                cmap="gray",
                vmin=0,
                vmax=np.max(x_ft_shift_abs[..., size[-1] // 2]),
            ), axes[1, 0].set_title("|FT(BP)| (kxky)")
            axes[1, 1].imshow(
                x_ft_shift_abs[size[1] // 2, :, :].transpose(),
                cmap="gray",
                vmin=0,
                vmax=np.max(x_ft_shift_abs[size[1] // 2, :, :]),
            ), axes[1, 1].set_title("|FT(BP)| (kxkz)")
            axes[2, 0].imshow(
                x_ft_x_psf_ft_shift_abs[..., size[-1] // 2].transpose(),
                cmap="gray",
                vmin=0,
                vmax=np.max(x_ft_x_psf_ft_shift_abs),
            ), axes[2, 0].set_title("|FT(BP) x FT(PSF)| (kxky)")
            axes[2, 1].imshow(
                x_ft_x_psf_ft_shift_abs[size[1] // 2, :, :].transpose(),
                cmap="gray",
                vmin=0,
                vmax=np.max(x_ft_x_psf_ft_shift_abs),
            ), axes[2, 1].set_title("|FT(BP) x FT(PSF)| (kxkz)")
            axes[3, 0].plot(
                x_ft_shift_abs[size[0] // 2, size[0] // 2 :, size[2] // 2], color="blue"
            ), axes[3, 0].set_title("|FT(BP)|")
            axes[3, 1].plot(
                x_ft_x_psf_ft_shift_abs[size[0] // 2, size[0] // 2 :, size[2] // 2],
                color="blue",
            ), axes[3, 1].set_title("|FT(BP) x FT(PSF)|")
            for ax in [axes[3, 0], axes[3, 1]]:
                ax.set_xlim([0, None])
                ax.set_ylim([0, None])

        fig, axes = plt.subplots(
            nrows=nr,
            ncols=nc,
            dpi=300,
            figsize=(2.4 * nc, 2.4 * nr),
            constrained_layout=True,
        )
        norm_max = lambda x: x / np.max(x)

        axes[0, 0].imshow(
            PSF[..., size[-1] // 2].transpose(),
            cmap="gray",
            vmin=np.min(PSF),
            vmax=np.max(PSF),
        ), axes[0, 0].set_title("PSF (xy) ({:>.2f}x{:>.2f})".format(std[1], std[0]))
        axes[0, 1].imshow(
            PSF[size[1] // 2, :, :].transpose(),
            cmap="gray",
            vmin=np.min(PSF),
            vmax=np.max(PSF),
        ), axes[0, 1].set_title("PSF (xz) ({:>.2f}x{:>.2f})".format(std[1], std[2]))
        axes[1, 0].imshow(
            PSF_FT_shift_abs[..., size[-1] // 2].transpose(),
            cmap="gray",
            vmin=0,
            vmax=np.max(PSF_FT_shift_abs),
        ), axes[1, 0].set_title("|FT(PSF)| (kxky)")
        axes[1, 1].imshow(
            PSF_FT_shift_abs[size[1] // 2, :, :].transpose(),
            cmap="gray",
            vmin=0,
            vmax=np.max(PSF_FT_shift_abs),
        ), axes[1, 1].set_title("|FT(PSF)| (kxkz)")
        axes[2, 0].plot(
            norm_max(PSF[size[0] // 2, :, size[2] // 2]), color="blue"
        ), axes[2, 0].set_title("PSF (x)")
        axes[2, 1].plot(
            norm_max(PSF[size[0] // 2, size[1] // 2, :]), color="blue"
        ), axes[2, 1].set_title("PSF (z)")
        axes[3, 0].plot(
            PSF_FT_shift_abs[size[0] // 2, size[1] // 2 :, size[2] // 2], color="blue"
        ), axes[3, 0].set_title("|FT(PSF)|")
        axes[3, 0].set_xlim([0, None])
        axes[3, 0].set_ylim([0, None])
        axes[3, 1].set_axis_off()

        x_dist = np.array([-600, -300, 0, 300, 600])
        x_dist_txt = ["-600", "-300", "0", "300", "600"]

        x_ticks = x_dist / delta_x + size[0] / 2 - 0.5
        axes[2, 0].set_xticks(x_ticks, labels=x_dist_txt)
        axes[2, 0].set_xlim(
            [-750 / delta_x + size[0] / 2 - 0.5, 750 / delta_x + size[0] / 2 - 0.5]
        )
        axes[2, 0].set_ylim([0, None])

        x_ticks = x_dist / delta_z + size[2] / 2 - 0.5
        axes[2, 1].set_xticks(x_ticks, labels=x_dist_txt)
        axes[2, 1].set_xlim(
            [-750 / delta_z + size[2] / 2 - 0.5, 750 / delta_z + size[2] / 2 - 0.5]
        )
        axes[2, 1].set_ylim([0, None])

        show_kernel_3d(BP_trad, BP_trad_OTF, "traditional", axes=axes[:, 2:4])
        show_kernel_3d(BP_gauss, BP_gauss_OTF, "gaussian", axes=axes[:, 4:6])
        show_kernel_3d(BP_bw, BP_bw_OTF, "butterworth", axes=axes[:, 6:8])
        show_kernel_3d(BP_wiener, BP_wiener_OTF, "wiener", axes=axes[:, 8:10])
        show_kernel_3d(BP_wb, BP_wb_OTF, "wiener-butterworth", axes=axes[:, 10:12])

    plt.savefig(os.path.join("methods", "output", "PSF_BPs.png"))
