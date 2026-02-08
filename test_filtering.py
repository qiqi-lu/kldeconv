"""
Explore the influence of filtering on blurred images.
"""

import numpy as np
from skimage.filters import gaussian
from scipy.ndimage import median_filter
from skimage.restoration import denoise_bilateral
import matplotlib.pyplot as plt
import os

# create a test image with black background and a white circle and a white line
img_size = 512
img = np.zeros((img_size, img_size))
# add a circular ring in image
radius_out = 100
radius_in = 90
center = (img_size // 2, img_size // 2)
y, x = np.ogrid[:img_size, :img_size]
mask_out = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius_out**2
mask_in = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius_in**2
mask = mask_out & ~mask_in
img[mask] = 1

# add a white line in image
line_with = 10
# horizontal line
pos_y = img_size // 2
img[pos_y - line_with // 2 : pos_y + line_with // 2, :] = 1
# vertical line
pos_x = img_size // 2
img[:, pos_x - line_with // 2 : pos_x + line_with // 2] = 1

# add a white square in image
square_size = 50
img[
    pos_y - square_size // 2 : pos_y + square_size // 2,
    pos_x - square_size // 2 : pos_x + square_size // 2,
] = 1

# set intensity
intensity = 100
img = img * intensity

# blur the image
sigma = 10
scale = 0.5
img_blur = gaussian(img, sigma=sigma) * scale

# add poisson noise and gaussian noise
noise_gauss = np.random.normal(0, 1, img.shape)
noise_level_gauss = 10

img_blur_noise = np.random.poisson(img_blur) + noise_gauss * noise_level_gauss
img_blur_noise = np.clip(img_blur_noise, 0, None)

img_blur_noise_median = median_filter(img_blur_noise, size=7)
diff_median = img_blur_noise_median - img_blur
mae_median = np.mean(np.abs(diff_median))
print(f"MAE (Median): {mae_median:.4f}")

# bilateral filter
img_blur_noise_bilateral = denoise_bilateral(img_blur_noise)
# img_blur_noise_bilateral = denoise_bilateral(
#     img_blur_noise, sigma_color=0.1, sigma_spatial=15
# )
diff_bilateral = img_blur_noise_bilateral - img_blur
mae_bilateral = np.mean(np.abs(diff_bilateral))
print(f"MAE (Bilateral): {mae_bilateral:.4f}")

diff_noise = img_blur_noise - img_blur
mae_noise = np.mean(np.abs(diff_noise))
print(f"MAE (Noise): {mae_noise:.4f}")


# ------------------------------------------------------------------------------
dict_fig = dict(dpi=300, constrained_layout=True)
nr, nc = 3, 3
fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(3 * nc, 3 * nr), **dict_fig)
axes = axes.flatten()
for ax in axes:
    ax.set_axis_off()

dict_img = dict(cmap="gray", vmin=0, vmax=intensity)
dict_img_blur = dict(cmap="gray", vmin=0, vmax=intensity * scale)
dict_img_diff = dict(cmap="seismic", vmin=intensity * (-scale), vmax=intensity * scale)

axes[0].imshow(img, **dict_img)
axes[0].set_title("Original")

axes[1].imshow(img_blur, **dict_img_blur)
axes[1].set_title("Blurred")

axes[2].imshow(img_blur_noise, **dict_img_blur)
axes[2].set_title("Blurred + Noise")

axes[3].imshow(img_blur_noise_median, **dict_img_blur)
axes[3].set_title("Blurred + Noise + Median")

axes[4].imshow(diff_median, **dict_img_diff)
axes[4].set_title(f"Difference (Median) (mae: {mae_median:.4f})")

axes[5].imshow(img_blur_noise_bilateral, **dict_img_blur)
axes[5].set_title("Blurred + Noise + Bilateral")
axes[6].imshow(diff_bilateral, **dict_img_diff)
axes[6].set_title(f"Difference (Bilateral) (mae: {mae_bilateral:.4f})")

pos_profile = 200
# axes[7].plot(img[pos_profile, :], label="Original")
axes[7].plot(img_blur[pos_profile, :], label="Blurred")
# axes[7].plot(img_blur_noise[pos_profile, :], label="Blurred + Noise")
axes[7].plot(img_blur_noise_median[pos_profile, :], label="Blurred + Noise + Median")
# axes[7].plot(
#     img_blur_noise_bilateral[pos_profile, :], label="Blurred + Noise + Bilateral"
# )
axes[7].set_title(f"Profile (y: {pos_profile})")
axes[7].legend()

axes[8].imshow(diff_noise, **dict_img_diff)
axes[8].set_title(f"Difference (Noise) (mae: {mae_noise:.4f})")


path_save_figure = os.path.join("outputs", "figures", "test")
if not os.path.exists(path_save_figure):
    os.makedirs(path_save_figure)
plt.savefig(os.path.join(path_save_figure, "test_filtering.png"))
