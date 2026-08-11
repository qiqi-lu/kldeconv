"""
Test the metrics.
"""

import os
import numpy as np
import skimage.io as io
from utils.data import win2linux, NormalizePercentile
import matplotlib.pyplot as plt

path_img_rln = "outputs\predictions\\biotisr-3d-mito-2\\rln\\biotisr-3d-mito-2\\n1_r1\Cell_001_0\y_pred.tif"
path_img_kld = "outputs\predictions\\biotisr-3d-mito-2\kernelnet\\biotisr-3d-mito-2\\fp_n1_r1_bp_n1_r1\\train_iter_2\Cell_001_0\y_pred_all.tif"
path_gt = "outputs\predictions\\biotisr-3d-mito-2\kernelnet\\biotisr-3d-mito-2\\fp_n1_r1_bp_n1_r1\\train_iter_2\Cell_001_0\y.tif"
path_figures = "outputs/figures/test"

# read images
img_rln = io.imread(win2linux(path_img_rln))
img_kld = io.imread(win2linux(path_img_kld))[-1]
img_gt = io.imread(win2linux(path_gt))

print(f"Image (rln) : {img_rln.shape}")
print(f"Image (kld) : {img_kld.shape}")
print(f"Image (gt)  : {img_gt.shape}")

ndim = 3
normalizer = NormalizePercentile(p_low=0.03, p_high=0.995, ndim=ndim)
dict_clip = {"a_min": 0, "a_max": 2.5}
data_range = dict_clip["a_max"] - dict_clip["a_min"]


img_rln = np.clip(normalizer(img_rln), **dict_clip)
img_kld = np.clip(normalizer(img_kld), **dict_clip)
img_gt = np.clip(normalizer(img_gt), **dict_clip)

res_rln = img_rln - img_gt
res_kld = img_kld - img_gt


mse_rln = np.mean((res_rln) ** 2)
mse_kld = np.mean((res_kld) ** 2)

print(f"MSE (rln) : {mse_rln:.4f}")
print(f"MSE (kld) : {mse_kld:.4f}")

# ------------------------------------------------------------------------------
id_slice = 7
nr, nc = 3, 3
dict_fig = dict(dpi=300, constrained_layout=True)
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)

dict_img = dict(cmap="hot", vmin=0, vmax=1.5)
dict_res = dict(cmap="seismic", vmin=-1.5, vmax=1.5)

axes[0, 0].imshow(img_rln[id_slice], **dict_img)
axes[0, 0].set_title(f"RLN (MSE={mse_rln:.4f})")
axes[0, 1].imshow(img_kld[id_slice], **dict_img)
axes[0, 1].set_title(f"KLD (MSE={mse_kld:.4f})")
axes[0, 2].imshow(img_gt[id_slice], **dict_img)
axes[0, 2].set_title(f"GT")

# show res in second row
axes[1, 0].imshow(res_rln[id_slice], **dict_res)
axes[1, 0].set_title(f"RLN - GT")
axes[1, 1].imshow(res_kld[id_slice], **dict_res)
axes[1, 1].set_title(f"KLD - GT")

# show profile
pos_y = 350
pos_x = (300, 400)


axes[2, 0].plot(img_rln[id_slice, pos_y, pos_x[0] : pos_x[1]], label="RLN")
axes[2, 0].plot(img_gt[id_slice, pos_y, pos_x[0] : pos_x[1]], label="GT")
axes[2, 0].legend()
axes[2, 1].plot(img_kld[id_slice, pos_y, pos_x[0] : pos_x[1]], label="KLD")
axes[2, 1].plot(img_gt[id_slice, pos_y, pos_x[0] : pos_x[1]], label="GT")
axes[2, 1].legend()
for ax in [axes[2, 0], axes[2, 1]]:
    ax.set_ylim(0, 1)

fig.savefig(os.path.join(path_figures, "test_metrics.png"))
