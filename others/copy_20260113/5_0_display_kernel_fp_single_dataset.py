"""
Display the learned forward and backward kernels for a single dataset.
"""

import matplotlib.pyplot as plt
import skimage.io as io
import os
from utils.data import win2linux

plt.rcParams["svg.fonttype"] = "none"
# ------------------------------------------------------------------------------

path_kernels = win2linux(
    "E:\qiqilu\Project\\2023 cytoSR\code\outputs\predictions\SirDNA-1024\kernelnet\SirDNA-1024\\fp_n1_r1_bp_n1_r1\kernel"
)  # the figure will be saved in this folder

# ------------------------------------------------------------------------------
# load kernels
# ------------------------------------------------------------------------------
kf_true = io.imread(os.path.join(path_kernels, "kernel_true.tif"))
kf_est = io.imread(os.path.join(path_kernels, "kernel_fp.tif"))
kb_est = io.imread(os.path.join(path_kernels, "kernel_bp.tif"))
k_init = io.imread(os.path.join(path_kernels, "kernel_init.tif"))

Nz, Ny, Nx = kf_true.shape

print(f"[INFO] Kernel shape : {kf_true.shape}")
print(f"[INFO] estimated forward kernel sum : {kf_est.sum():.4f}")
print(f"[INFO] estimated backward kernel sum : {kb_est.sum():.4f}")
print(f"[INFO] initial kernel sum : {k_init.sum():.4f}")

# ------------------------------------------------------------------------------
# display kernels
# ------------------------------------------------------------------------------
nr, nc = 2, 2
dict_fig = {"dpi": 300, "constrained_layout": True}
fig, axes = plt.subplots(nr, nc, figsize=(nc * 3, nr * 3), **dict_fig)


# show the center profile
axes[0, 0].plot(kf_true[Nz // 2, :, Nx // 2], label="True", color="black")
axes[0, 0].plot(kf_est[Nz // 2, :, Nx // 2], label="Estimated", color="red")
axes[0, 0].plot(k_init[Nz // 2, :, Nx // 2], label="Initial", color="blue")
axes[0, 0].set_title("Forward Kernel (x)")

axes[1, 0].plot(kf_true[:, Ny // 2, Nx // 2], label="True", color="black")
axes[1, 0].plot(kf_est[:, Ny // 2, Nx // 2], label="Estimated", color="red")
axes[1, 0].plot(k_init[:, Ny // 2, Nx // 2], label="Initial", color="blue")
axes[1, 0].set_title("Forward Kernel (z)")

axes[0, 1].plot(kb_est[Nz // 2, :, Nx // 2], label="Estimated", color="red")
axes[0, 1].plot(k_init[Nz // 2, :, Nx // 2], label="Initial", color="blue")
axes[0, 1].set_title("Backward Kernel (x)")

axes[1, 1].plot(kb_est[:, Ny // 2, Nx // 2], label="Estimated", color="red")
axes[1, 1].plot(k_init[:, Ny // 2, Nx // 2], label="Initial", color="blue")
axes[1, 1].set_title("Backward Kernel (z)")

for ax in axes.flatten():
    ax.legend()

plt.savefig(os.path.join(path_kernels, "kernel_profile.png"))
