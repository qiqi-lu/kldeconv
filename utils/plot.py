import skimage.exposure as exposure
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from skimage.measure import profile_line
import torch


def image_combine_2d(image1, image2, flip=True):
    """
    Combine two 2D images into one 2D image.
    The left top half of the image is the first image,
    and the right bottom half of the image is the second image.
    ### Parameters:
    - `image1`: numpy array, shape (H, W), the first image.
    - `image2`: numpy array, shape (H, W), the second image.
    ### Returns:
    - `image_combined`: numpy array, shape (H, W).
    """
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    assert image1.ndim == 2 and image2.ndim == 2, "Only support 2D images."
    assert image1.shape == image2.shape, "The two images should have the same shape."

    H, W = image1.shape
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    mask = (jj * H) >= (ii * W)  # Diagonal from top-left to bottom-right
    # flip mask left tight
    if flip:
        mask = np.flip(mask, axis=1)
    image_combined = np.where(mask, image1, image2)
    return image_combined


def render_color(img, vmax=None):
    # norlaization
    if vmax == None:
        vmax = np.array(
            [np.percentile(img[..., 0], 99), np.percentile(img[..., 1], 99)]
        )
        # vmax = np.percentile(img, 99.99)
        # vmax = np.percentile(img, 99)
        # vmax = np.max(img)
    img = img / vmax
    img = np.clip(img, 0.0, 1.0)

    # gamma correction
    # img = exposure.adjust_gamma(img, gamma=0.5)

    # set colors
    color_magenta = np.asarray([255, 0, 255]).reshape((1, 1, -1))  # 642
    color_green = np.asarray([0, 255, 0]).reshape((1, 1, -1))  # 560

    img_color = (
        img[..., 0][..., None] * color_magenta + img[..., 1][..., None] * color_green
    )
    img_color = np.clip(img_color, 0, 255)

    return img_color.astype(np.uint8)


def add_scale_bar(
    ax, image, pixel_size, bar_length, bar_height=0.01, bar_color="white", pos=(20, 20)
):
    """
    Add a scale bar to the given axes.

    ### Parameters:
    - `ax` (matplotlib.axes.Axes): The axes to which the scale bar will be added.
    - `image` (np.ndarray): The 2D image array. The shape of the image array should be (H, W).
    - `pixel_size` (float): The size of each pixel in the image (um).
    - `bar_length` (float): The desired length of the scale bar (um).
    - `bar_height` (float, optional): The height of the scale bar as a fraction
            of the image height. Default is 0.02.
    - `bar_color` (str, optional): The color of the scale bar. Default is 'white'.
    - `pos` (tuple, optional): The position of the scale bar in the image (pixels). Default is (20, 20).
    """
    # Calculate the number of pixels corresponding to the bar length
    bar_pixels = bar_length / pixel_size

    # Get the image dimensions
    image_height, image_width = image.shape[-2:]

    # Calculate the physical height of the bar
    bar_physical_height = bar_height * image_height

    # Add the scale bar rectangle
    rect = plt.Rectangle(
        pos, bar_pixels, bar_physical_height, color=bar_color, zorder=10
    )
    ax.add_patch(rect)


def add_patch(
    ax: plt.Axes,
    image: np.ndarray,
    pos: tuple,
    size: int,
    percent: float = 0.45,
    show_box=False,
    axes_lw=1,
    box_lw=0.5,
    box_color="white",
):
    """
    Add a patch to the given axes.
    ### Parameters:
    - `ax` (matplotlib.axes.Axes): The axes to which the patch will be added.
    - `image` (np.ndarray): The 2D image array. The shape of the image array should be (H, W) or (H, W, 3).
    - `pos` (tuple): The position of the patch in the image (pixels).
    - `size` (tuple): The size of the patch in the image (pixels).
    - `percent` (float): The percentage of the patch to be displayed. Default is 0.45.
    - `show_box` (bool): Whether to show the box in image. Default is False.
    - `axes_lw` (float): The linewidth of the axes. Default is 1.
    """

    assert image.ndim in [2, 3], "The shape of image should be (H, W) or (H, W, 3)"
    if image.ndim == 3:
        assert image.shape[-1] == 3, "The shape of image should be (H, W, 3)"

    # add box in the image -----------------------------------------------------
    dict_box = dict(fill=False, edgecolor=box_color, linewidth=box_lw)

    if show_box:
        box = plt.Rectangle(xy=(pos[0], pos[1]), width=size, height=size, **dict_box)
        ax.add_patch(box)

    # crop patch ---------------------------------------------------------------
    patch = image[pos[1] : pos[1] + size, pos[0] : pos[0] + size]
    # add patch to the figure
    patch = np.flipud(patch)
    # set the size of patch
    img_shape = image.shape
    w_box, h_box = (max(img_shape[1] * percent, img_shape[0] * percent),) * 2
    ax_patch = ax.inset_axes(
        [img_shape[1] - w_box, img_shape[0] - h_box, w_box, h_box],
        transform=ax.transData,
    )
    ax_patch.imshow(patch)

    # adjust apperence ---------------------------------------------------------
    ax_patch.set_xlim(0, size - 1)
    ax_patch.set_ylim(0, size - 1)
    # del the ticks and their labels
    ax_patch.set_xticks([])
    ax_patch.set_yticks([])
    ax_patch.set_xticklabels([])
    ax_patch.set_yticklabels([])
    # set the color of axes to white and width to 1
    ax_patch.spines["top"].set_color("white")
    ax_patch.spines["left"].set_color("white")
    ax_patch.spines["top"].set_linewidth(axes_lw)
    ax_patch.spines["left"].set_linewidth(axes_lw)
    # del the left and bottom spines
    ax_patch.spines["right"].set_visible(False)
    ax_patch.spines["bottom"].set_visible(False)


def add_significant_bars(ax, x1, x2, y, p_value, dict_line={}, dict_asterisks={}):
    """
    Add significant bars to the given axes.

    ### Parameters:
    - `ax`: matplotlib.axes.Axes, the axes to which the significant bars will be added.
    - `x1`: float, the x-coordinate of the left edge of the bar.
    - `x2`: float, the x-coordinate of the right edge of the bar.
    - `y`: float, the y-coordinate of the bar.
    - `p_value`: float, the p-value of the comparison.
    - `significant_level`: float, the significance level. Default is 0.05.
    """
    if p_value <= 0.0001:
        asterisks = "****"
    elif p_value <= 0.001:
        asterisks = "***"
    elif p_value <= 0.01:
        asterisks = "**"
    elif p_value <= 0.05:
        asterisks = "*"
    else:
        asterisks = "ns"

    offset = 0.05
    dict_l = {"color": "black", "linewidth": 1}
    dict_a = {"ha": "left", "va": "bottom", "fontsize": 10, "color": "black"}

    if dict_line:
        dict_l.update(dict_line)
    if dict_asterisks:
        dict_a.update(dict_asterisks)

    # ax.plot([x1, x1, x2, x2], [y, y + offset, y + offset, y], **dict_l)
    ax.plot([x1, x2], [y, y], **dict_l)
    # ax.text((x1 + x2) / 2, y * 1.001, asterisks, **dict_a)
    ax.text(x1, y * 0.98, asterisks, **dict_a)


def add_significant_star(ax, x, y, p_value, dict_asterisks={}):
    """
    Add significant stars to the given axes at a specific position.

    ### Parameters:
    - `ax`: matplotlib.axes.Axes, the axes to which the significant stars will be added.
    - `x`: float, the x-coordinate of the star.
    - `y`: float, the y-coordinate of the star.
    - `p_value`: float, the p-value of the comparison.
    """
    if p_value <= 0.0001:
        asterisks = "**"
    elif p_value <= 0.001:
        asterisks = "**"
    elif p_value <= 0.01:
        asterisks = "**"
    elif p_value <= 0.05:
        asterisks = "*"
    else:
        asterisks = "ns"
    dict_a = {"ha": "center", "va": "bottom", "fontsize": 8, "color": "black"}
    if dict_asterisks:
        dict_a.update(dict_asterisks)
    ax.text(x, y, asterisks, **dict_a)


def add_line_profile(
    ax: plt.Axes,
    image: np.ndarray,
    line_pos: tuple,
    profile_pos: tuple,
    profiel_ylim=2.0,
    line_color="white",
    line_width=1,
    show_line=False,
):
    """
    Add a line profile to the given axes.
    ### Parameters:
    - `ax` (matplotlib.axes.Axes): The axes to which the line profile will be added.
    - `image` (np.ndarray): The 2D image array. The shape of the image array should be (H, W).
    - `line_pos` (tuple): The position of the line profile in the image (pixels).
        (x1,y1, x2, y2).
    - `profile_pos` (tuple): The position of the line profile in the figure (pixels).
        (w_box, h_box).
    - `line_color` (str): The color of the line profile. Default is 'white'.
    - `line_width` (float): The width of the line profile. Default is 1.
    """
    assert image.ndim == 2, "The shape of image should be (H, W)"
    assert len(line_pos) == 4, "The length of line_pos should be 4"

    dict_line = dict(color=line_color, linewidth=line_width)
    x1, y1, x2, y2 = line_pos
    w_box, h_box = profile_pos
    img_shape = image.shape

    # plot the line in the image
    if show_line:
        ax.plot([x1, x2], [y1, y2], linestyle="--", **dict_line)

    profile = profile_line(
        # np.mean(img_color, axis=-1), (y1, x1), (y2, x2), linewidth=1
        image,
        (y1, x1),
        (y2, x2),
        linewidth=1,
    )
    profile_ax = ax.inset_axes(
        [img_shape[1] - w_box, img_shape[0] - h_box * 1.5, w_box, h_box * 0.5],
        transform=ax.transData,
    )
    profile_ax.plot(profile, **dict_line)
    profile_ax.set_ylim((0, profiel_ylim))
    profile_ax.set_axis_off()


def colorize(image, vmin=0, vmax=1, color=(0, 255, 0)):
    """
    Create an RGB image from a single-channel image using a
    specific color.

    ### Parameters:
    - `image`: numpy array, shape (H, W), single channel image.
    - `vmin`: float, the minimum value of the image.
    - `vmax`: float, the maximum value of the image.
    - `color`: tuple, the color to use for the image.

    ### Returns:
    - `image`: numpy array, shape (H, W, 3), RGB image.
    """
    # Rescale the image
    image_clip = np.clip(image, vmin, vmax)
    image_clip = (image_clip - vmin) / (vmax - vmin)
    image_clip_3 = np.repeat(image_clip[..., None], 3, axis=-1)
    image_clip_3 = image_clip_3 * color
    return image_clip_3.astype(np.uint8)


def interp_iso_z(x, ps_xy=25, ps_z=160):
    """
    Interpolate the image to the isotropic z-axis.
    ### Parameters:
    - `x`: numpy array, shape (D, H, W), the image to be interpolated.
    - `ps_xy`: float, the pixel size in the xy-axis.
    - `ps_z`: float, the pixel size in the z-axis.
    ### Returns:
    - `x`: numpy array, shape (D, H, W), the interpolated image.
    """
    assert x.ndim == 3, "The shape of x should be (D, H, W)"
    z_scale = ps_z / ps_xy
    x = torch.tensor(x)[None, None]
    x = torch.nn.functional.interpolate(x, scale_factor=(z_scale, 1, 1), mode="nearest")
    x = x.numpy()[0, 0]
    return x


def normalization_01(img_gray, vmin: tuple | list, vmax: tuple | list):
    """
    Normalize image to 0-1, the normalized image is clip to (0,1).

    ### Parameters:
    - `img_gray`: numpy array, shape (H, W, C), the image to be normalized.
    - `vmin`: tuple, the minimum value of the image.
    - `vmax`: tuple, the maximum value of the image.
    ### Returns:
    - `img_gray`: numpy array, shape (H, W, C), the normalized image.
    """
    assert img_gray.ndim == 3, "The shape of img_gray should be (H, W, C)"
    assert len(vmin) == len(vmax), "The length of vmin and vmax should be the same"
    assert img_gray.shape[-1] == len(
        vmax
    ), "The number of channels of img_gray should have the same length as vmax"

    vmin, vmax = np.array(vmin), np.array(vmax)
    img_norm = (img_gray - vmin) / (vmax - vmin)
    img_norm = np.clip(img_norm, 0, 1)
    return img_norm


def look_up(img_gray, lut, rgb_type="rgb"):
    """
    Apply LUT to gray scale image.

    ### Parameters:
    - `img_gray`: numpy array, shape (H, W), the gray scale image.
    - `lut`: look up table.
    - `rgb_type`: str, the type of the image. Default is 'rgb'.
    ### Returns:
    - `rgb`: numpy array, shape (H, W, 3), the RGB image.
    """
    img_gray_flat = img_gray.reshape(-1)
    img_color = lut(img_gray_flat)
    img_color = img_color.reshape(img_gray.shape + (4,))
    # remove the alpha channel, which control the transparency of the color
    img_color = img_color[..., :-1]
    if rgb_type == "bgr":
        img_color = np.flip(img_color, axis=-1)
    return img_color


def merge(imgs):
    """
    Merge colored image to a single image. The shape of the image should be (H, W, 3).
    ### Parameters:
    - `imgs`: list of numpy array, shape (H, W, 3), the colored image.
    ### Returns:
    - `img_merge`: numpy array, shape (H, W, 3), the merged image.
    """
    assert isinstance(imgs, list), "imgs should be a list of numpy array"
    assert len(imgs) > 0, "imgs should not be empty"

    if len(imgs) > 1:
        merged = 0
        for img in imgs:
            merged += img.astype(np.float64)
    else:
        merged = imgs[0].astype(np.float64)
    merged = merged * 255.0
    merged[merged > 255] = 255
    return merged.astype(np.uint8)


def create_cmap(color: tuple | list = (255, 255, 255)):
    """
    Create a colormap from a single color.
    Start from black to the given color.
    ### Parameters:
    - `color`: tuple, the color to use for the colormap. Default is (255,255,255).
    ### Returns:
    - `cmap`: matplotlib.colors.ListedColormap, the colormap.
    """
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(0, color[0] / N, N)
    vals[:, 1] = np.linspace(0, color[1] / N, N)
    vals[:, 2] = np.linspace(0, color[2] / N, N)
    newcmp = ListedColormap(vals)
    return newcmp


def render(
    img, cmaps, plow: tuple | list = None, phigh: tuple | list = None, rgb_type="rgb"
):
    """
    Render image with color map.

    ### Parameters:
    - `img`: numpy array, shape (H, W, C), the image to be rendered.
    - `cmaps`: list, the color map to be used. Default is ['gray'].
    - `vmin`: tuple, the minimum value of the image. Default is None.
    - `vmax`: tuple, the maximum value of the image. Default is None.
    - `rgb_type`: str, the type of the RGB image. Default is 'rgb'.
    ### Returns:
    - `img`: numpy array, shape (H, W, C), the rendered image.
    """
    assert img.ndim == 3, "The shape of img should be (H, W, C)"
    assert len(cmaps) == img.shape[-1], "The length of cmaps should be the same as C"

    num_channel = img.shape[-1]

    # if vmin and vmax is not given,
    # use the minimum and maximum value of the image to normalize the image.
    if plow is None:
        plow = (0,) * num_channel
    if phigh is None:
        phigh = (100,) * num_channel

    vmin, vmax = [], []
    for i in range(num_channel):
        vmin.append(np.percentile(img[..., i], plow[i]))
        vmax.append(np.percentile(img[..., i], phigh[i]))

    # Normalise image
    img_norm = normalization_01(img, vmin, vmax)

    # Grayscale images converted to rgb with a LUT
    imgs = []
    for i in range(num_channel):
        img_sc = img_norm[..., i]
        imgs.append(look_up(img_sc, cmaps[i], rgb_type=rgb_type))

    img_merged = merge(imgs)
    return img_merged
