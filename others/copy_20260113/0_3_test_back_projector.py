"""
Test back projector funcitons.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from methods.back_projector import BackProjector, FWHM2sigma, FWHM_PSF, PSF_gaussian
from methods.deconvolution import adjust_size

path_fig = os.path.join("outputs", "figures", "test")
os.makedirs(path_fig, exist_ok=True)

# ------------------------------------------------------------------------------
# get PSF
# ------------------------------------------------------------------------------
# sythetic 3D gaussian PSF
# size, std = (127, 127, 127), (1.18, 1.18, 4.41)
# sythetic 2D gaussian PSF
# std, size = (1.18, 1.18), (128, 128)

# PSF = PSF_gaussian(size, std)

# real PSF from iSIM (maybe not real, from RLD algorithm)
std = None
PSF = io.imread(os.path.join("methods", "PSF_iSIM.tif"))
PSF = np.transpose(PSF, axes=(1, 2, 0))  # (Sx, Sy, Sz)

# ------------------------------------------------------------------------------
# padding PSF in the spatial domain is equivalent to
# interpolation in the Fourier domain
size = (np.array(PSF.shape).max(),) * 3
PSF = adjust_size(PSF, size)
PSF = PSF / np.sum(PSF)
size = PSF.shape
dim = PSF.ndim
std = FWHM2sigma(FWHM_PSF(PSF=PSF))
print(size)
print(np.sum(PSF))

delta_x = 55  # pixel size in spatial domain (xy plane)
delta_z = 55  # pixel size in spatial domain (z-axis)
detla_u = 1.0 / (delta_x * size[0])  # pixel size in Fourier domain (xy plane)

# ------------------------------------------------------------------------------
# generate backward projection kernel
# ------------------------------------------------------------------------------
PSF_FT = np.fft.fftn(np.fft.ifftshift(PSF))
PSF_FT_shift_abs = np.abs(np.fft.fftshift(PSF_FT))

dict_params_common = dict(PSF_fp=PSF, i_res=[0, 0, 0], verbose_flag=1)

BP_trad, BP_trad_OTF = BackProjector(bp_type="traditional", **dict_params_common)
BP_gauss, BP_gauss_OTF = BackProjector(bp_type="gaussian", **dict_params_common)
BP_bw, BP_bw_OTF = BackProjector(
    bp_type="butterworth",
    alpha=0.001,
    beta=0.001,
    n=15,
    res_flag=0,
    **dict_params_common,
)
BP_wiener, BP_wiener_OTF = BackProjector(
    bp_type="wiener",
    alpha=0.001,
    beta=1,
    n=10,
    res_flag=1,
    **dict_params_common,
)
BP_wb, BP_wb_OTF = BackProjector(
    bp_type="wiener-butterworth",
    alpha=0.001,
    beta=0.001,
    n=12,
    res_flag=0,
    **dict_params_common,
)

# ------------------------------------------------------------------------------
# plot PSF and BP in spatial domain and Fourier domain
# ------------------------------------------------------------------------------
dict_fig = dict(dpi=300, constrained_layout=True)

if dim == 3:

    def plot_profile(x_ft, axes, name, color):
        x_ft_shift_abs = np.abs(np.fft.fftshift(x_ft))
        x_ft_x_psf_ft_shift_abs = np.abs(np.fft.fftshift(x_ft * PSF_FT))
        dict_line = dict(label=name, color=color)
        axes[0].plot(
            x_ft_shift_abs[size[0] // 2, size[0] // 2 :, size[2] // 2], **dict_line
        )
        axes[1].plot(
            x_ft_x_psf_ft_shift_abs[size[0] // 2, size[0] // 2 :, size[2] // 2],
            **dict_line,
        )

    # --------------------------------------------------------------------------
    nc, nr = 3, 1
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    axes[0].set_title("|FT(PSF)|", loc="left")
    axes[1].set_title("|FT(BP)|", loc="left")
    axes[2].set_title("|FT(BP) x FT(PSF)|", loc="left")

    axes[0].plot(
        PSF_FT_shift_abs[size[0] // 2, size[1] // 2 :, size[2] // 2], color="blue"
    )
    plot_profile(BP_trad_OTF, [axes[1], axes[2]], name="Traditional", color="blue")
    plot_profile(BP_gauss_OTF, [axes[1], axes[2]], name="Gaussian", color="cyan")
    plot_profile(BP_bw_OTF, [axes[1], axes[2]], name="Butterworth", color="orange")
    # plot_profile(BP_wiener_OTF, [axes[1], axes[2]], name='wiener', color='green')
    plot_profile(BP_wb_OTF, [axes[1], axes[2]], name="WB", color="orangered")
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
        # turn off upper and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

plt.savefig(os.path.join(path_fig, "BP_profiles.png"))

# ------------------------------------------------------------------------------
# show kernel in spatial domain and Fourier domain
if dim == 2:

    def show_kernel(x, x_ft, name, axes):
        x_ft_shift_abs = np.abs(np.fft.fftshift(x_ft))
        x_ft_x_psf_ft_shift_abs = np.abs(np.fft.fftshift(x_ft * PSF_FT))

        dict_img = dict(cmap="gray", vmin=0)

        axes[0].imshow(x, cmap="gray", vmin=np.min(x), vmax=np.max(x))
        axes[1].imshow(x_ft_shift_abs, vmax=np.max(x_ft_shift_abs), **dict_img)
        axes[2].imshow(
            x_ft_x_psf_ft_shift_abs, vmax=np.max(x_ft_x_psf_ft_shift_abs), **dict_img
        )
        axes[3].plot(x_ft_shift_abs[size[0] // 2, size[0] // 2 :], color="blue")
        axes[4].plot(
            x_ft_x_psf_ft_shift_abs[size[0] // 2, size[0] // 2 :], color="blue"
        )

        axes[0].set_title("BP ({})".format(name))
        axes[1].set_title("|FT(BP)|")
        axes[2].set_title("|FT(BP) x FT(PSF)|")
        axes[3].set_title("|FT(BP)|")
        axes[4].set_title("|FT(BP) x FT(PSF)|")

    # --------------------------------------------------------------------------
    nr, nc = 5, 6
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    dict_img = dict(cmap="gray", vmin=0)

    axes[0, 0].imshow(PSF, vmax=np.max(PSF), **dict_img)
    axes[1, 0].imshow(PSF_FT_shift_abs, vmax=np.max(PSF_FT_shift_abs), **dict_img)
    axes[3, 0].plot(PSF_FT_shift_abs[size[0] // 2, size[0] // 2 :])

    axes[0, 0].set_title("PSF")
    axes[1, 0].set_title("|FT(PSF)|")
    axes[3, 0].set_title("|FT(PSF)|", color="blue")

    axes[2, 0].set_axis_off()
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

        dict_img_min = dict(cmap="gray", vmin=np.min(x), vmax=np.max(x))
        dict_img = dict(cmap="gray", vmin=0)

        axes[0, 0].set_title(f"BP ({name}) (xy)")
        axes[0, 1].set_title(f"BP ({name}) (xz)")
        axes[1, 0].set_title("|FT(BP)| (kxky)")
        axes[1, 1].set_title("|FT(BP)| (kxkz)")
        axes[2, 0].set_title("|FT(BP) x FT(PSF)| (kxky)")
        axes[2, 1].set_title("|FT(BP) x FT(PSF)| (kxkz)")
        axes[3, 0].set_title("|FT(BP)|")
        axes[3, 1].set_title("|FT(BP) x FT(PSF)|")

        axes[0, 0].imshow(x[..., size[-1] // 2].transpose(), **dict_img_min)
        axes[0, 1].imshow(x[size[1] // 2, :, :].transpose(), **dict_img_min)

        for ax, ker in [
            (axes[1, 0], x_ft_shift_abs[..., size[-1] // 2]),
            (axes[1, 1], x_ft_shift_abs[size[1] // 2, :, :]),
            (axes[2, 0], x_ft_x_psf_ft_shift_abs[..., size[-1] // 2]),
            (axes[2, 1], x_ft_x_psf_ft_shift_abs[size[1] // 2, :, :]),
        ]:
            ax.imshow(ker.transpose(), vmax=np.max(ker), **dict_img)

        for ax, profile in [
            (
                axes[3, 0],
                x_ft_shift_abs[size[0] // 2, size[0] // 2 :, size[2] // 2],
            ),
            (
                axes[3, 1],
                x_ft_x_psf_ft_shift_abs[size[0] // 2, size[0] // 2 :, size[2] // 2],
            ),
        ]:
            ax.plot(profile, color="blue")
            ax.set_xlim([0, None])
            ax.set_ylim([0, None])

    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
    norm_max = lambda x: x / np.max(x)

    dict_img = dict(cmap="gray", vmin=0)
    # --------------------------------------------------------------------------
    # PSF
    axes[0, 0].set_title(f"PSF (xy) ({std[1]:>.2f}x{std[0]:>.2f})")
    axes[0, 1].set_title(f"PSF (xz) ({std[1]:>.2f}x{ std[2]:>.2f})")
    axes[1, 0].set_title("|FT(PSF)| (kxky)")
    axes[1, 1].set_title("|FT(PSF)| (kxkz)")

    axes[0, 0].imshow(
        PSF[..., size[-1] // 2].transpose(),
        cmap="gray",
        vmin=np.min(PSF),
        vmax=np.max(PSF),
    )
    axes[0, 1].imshow(
        PSF[size[1] // 2, :, :].transpose(),
        cmap="gray",
        vmin=np.min(PSF),
        vmax=np.max(PSF),
    )
    axes[1, 0].imshow(
        PSF_FT_shift_abs[..., size[-1] // 2].transpose(),
        vmax=np.max(PSF_FT_shift_abs),
        **dict_img,
    )
    axes[1, 1].imshow(
        PSF_FT_shift_abs[size[1] // 2, :, :].transpose(),
        vmax=np.max(PSF_FT_shift_abs),
        **dict_img,
    )
    axes[2, 0].set_title("PSF (x)")
    axes[2, 1].set_title("PSF (z)")
    axes[3, 0].set_title("|FT(PSF)|")

    axes[2, 0].plot(norm_max(PSF[size[0] // 2, :, size[2] // 2]), color="blue")
    axes[2, 1].plot(norm_max(PSF[size[0] // 2, size[1] // 2, :]), color="blue")
    axes[3, 0].plot(
        PSF_FT_shift_abs[size[0] // 2, size[1] // 2 :, size[2] // 2], color="blue"
    )
    axes[3, 0].set_xlim([0, None])
    axes[3, 0].set_ylim([0, None])
    axes[3, 1].set_axis_off()

    # set xticks for profile plot
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
    # --------------------------------------------------------------------------
    show_kernel_3d(BP_trad, BP_trad_OTF, "traditional", axes=axes[:, 2:4])
    show_kernel_3d(BP_gauss, BP_gauss_OTF, "gaussian", axes=axes[:, 4:6])
    show_kernel_3d(BP_bw, BP_bw_OTF, "butterworth", axes=axes[:, 6:8])
    show_kernel_3d(BP_wiener, BP_wiener_OTF, "wiener", axes=axes[:, 8:10])
    show_kernel_3d(BP_wb, BP_wb_OTF, "wiener-butterworth", axes=axes[:, 10:12])

plt.savefig(os.path.join(path_fig, "PSF_BP_images.png"))
