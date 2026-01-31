import torch, os, pydicom
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk

# from nanopyx.core.transform.error_map import ErrorMap


# def registration(img, img_ref):
#     """
#     Registration of super-resolution reconstructions against the reference image.
#     Use the method in SQUIRREL algorithm.
#     """
#     img = np.squeeze(img)
#     img_ref = np.squeeze(img_ref)
#     ndim_img = img.ndim
#     ndim_img_ref = img_ref.ndim
#     assert ndim_img == 2 and ndim_img_ref == 2, "[ERROR] Only support 2D image."

#     emc = ErrorMap()
#     emc.optimize(img_ref, img)


def bkg_estimation_const(img, min_obj_size=32, open_radius=2):
    """
    Estimate the background noise level of an image.
    ### Parameters:
    - `img` (numpy.ndarray): Input image with a shape of `(Ny, Nx)` or `(Nz, Ny, Nx)`.
    - `min_obj_size` (int): Minimum object size.
    - `open_radius` (int): Radius of the disk structuring element for opening.
    """
    img = np.array(img, dtype=np.float32)
    img = np.squeeze(img)
    ndim = img.ndim
    assert ndim in [2, 3], "[ERROR] Only support 2D or 3D image."

    # detect the object
    thr = threshold_otsu(img)
    obj = img > thr
    obj = remove_small_objects(obj, min_size=min_obj_size)
    obj = binary_opening(obj, disk(open_radius))
    bkg = ~obj

    if bkg.sum() < 100:
        bkg_constant = np.percentile(img, 2)
    else:
        bkg_constant = np.median(img[bkg])

    return bkg_constant


def normalization(image, p_low, p_high):
    vmin = np.percentile(a=image, q=p_low * 100)
    vmax = np.percentile(a=image, q=p_high * 100)
    if vmax == 0:
        image *= 0.0
    else:
        amp = vmax - vmin
        if amp == 0:
            amp = 1
        image = (image - vmin) / amp

    return image, vmin, vmax


class NormalizePercentile(object):
    """
    Percentile-based normalization.
    Support 2D and 3D images.

    ### Parameters:
    - `p_low` : float, lower percentile.
    - `p_high` : float, upper percentile.
    """

    def __init__(self, p_low=0.0, p_high=1.0, ndim=2):
        self.p_low = p_low
        self.p_high = p_high
        self.ndim = ndim

    def __call__(self, image):
        """
        ### Parameters:
        - `image` : numpy array, image to be normalized.
        ### Returns:
        - `image` : numpy array, normalized image.
        """
        if isinstance(image, np.ndarray):
            if self.ndim == 2:
                dict_perc = {"axis": (-2, -1), "keepdims": True}
            else:
                dict_perc = {"axis": (-3, -2, -1), "keepdims": True}
            vmin = np.percentile(a=image, q=self.p_low * 100, **dict_perc)
            vmax = np.percentile(a=image, q=self.p_high * 100, **dict_perc)

        if isinstance(image, torch.Tensor):
            if self.ndim == 2:
                image_flat = image.flatten(start_dim=-2)
            elif self.ndim == 3:
                image_flat = image.flatten(start_dim=-3)
            else:
                raise ValueError(f"[ERROR] ndim should be 2 or 3, but got {self.ndim}")
            # if the number of element is larger than 16M, use np.qualtile
            if image_flat.shape[-1] > 16 * 1024 * 1024:
                dict_perc = {"axis": -1, "keepdims": True}
                vmin = np.quantile(
                    a=image_flat.cpu().numpy(), q=self.p_low, **dict_perc
                )
                vmax = np.quantile(
                    a=image_flat.cpu().numpy(), q=self.p_high, **dict_perc
                )
                vmin = torch.tensor(
                    vmin, dtype=image_flat.dtype, device=image_flat.device
                )
                vmax = torch.tensor(
                    vmax, dtype=image_flat.dtype, device=image_flat.device
                )
            else:
                dict_perc = {"dim": -1, "keepdim": True}
                vmin = torch.quantile(input=image_flat, q=self.p_low, **dict_perc)
                vmax = torch.quantile(input=image_flat, q=self.p_high, **dict_perc)

            # restore the dimension
            vmin = vmin.view(*vmin.shape[:-1], *([1] * self.ndim))
            vmax = vmax.view(*vmax.shape[:-1], *([1] * self.ndim))

        # avoid divide by zero
        amp = vmax - vmin
        amp[amp == 0] = 1
        # normalize
        image = (image - vmin) / amp
        return image


def pad_img_xy(img, n_pad):
    """
    Pad the image with edge values. only pad the last two dimensions.
    ### Parameters:
    - `img` (numpy.ndarray): Input image with a shape of `(Nz, Ny, Nx)`.
    - `n_pad` (int): Number of pixels to pad.
    ### Returns:
    - `img` (numpy.ndarray): Padded image.
    """
    if img.ndim == 3:
        Nz, Ny, Nx = img.shape
        img = np.pad(
            img, pad_width=((0, 0), (0, n_pad - Ny), (0, n_pad - Nx)), mode="edge"
        )
    elif img.ndim == 2:
        Ny, Nx = img.shape
        img = np.pad(img, pad_width=((0, n_pad - Ny), (0, n_pad - Nx)), mode="edge")
    else:
        raise ValueError(f"[ERROR] Input image should be 2D or 3D, but got {img.ndim}D")
    return img


def text2tuple(text: str):
    """
    Convert a string of numbers separated by commas into a tuple of integers.
    ### Parameters:
    - `text` (str): A string of numbers separated by commas, such as '(25,25)'.
    ### Returns:
    - `tuple`: A tuple of integers. (25,25)
    """
    text = text.replace("(", "").replace(")", "")
    text = text.split(",")
    text_tuple = tuple([int(i) if "." not in i else float(i) for i in text])
    return text_tuple


def win2linux(win_path):
    """
    Convert a Windows path to a Linux path if the current operating system is Linux,
    otherwise return the original path.
    ### Parameters:
        - `win_path` (str): The Windows path to be converted.
    ### Returns:
        - (str): The converted Linux path if the current operating system is Linux,
               otherwise the original path.
    """
    if not isinstance(win_path, str):
        return ""
    elif os.name == "posix":
        linux_path = win_path.replace("\\", "/")
        if len(linux_path) > 1 and linux_path[1] == ":":
            drive_letter = linux_path[0].lower()
            linux_path = "/mnt/" + drive_letter + linux_path[2:]
        return linux_path
    else:
        return win_path


def read_txt(path_txt):
    """
    Read txt file consisting of info in each line.
    ### Parameters:
    - `path_txt` : str, path of the txt file.
    ### Returns:
    - `lines` : list, info in each line.
    """
    if os.name == "posix":
        path_txt = win2linux(path_txt)

    with open(path_txt) as f:
        lines = f.read().splitlines()

    if lines[-1] == "":
        lines.pop()

    return lines


def interp(x, ps_xy=1, ps_z=1):
    x = np.array(x, dtype=np.float32)
    num_dim = len(x.shape)

    if num_dim == 3:
        z_scale = ps_z / ps_xy
        x = torch.tensor(x)[None, None]
        x = torch.nn.functional.interpolate(
            x, scale_factor=(z_scale, 1, 1), mode="nearest"
        )
        x = x.numpy()[0, 0]

    if num_dim == 2:
        z_scale = ps_z / ps_xy
        x = torch.tensor(x)[None, None]
        x = torch.nn.functional.interpolate(
            x, scale_factor=(z_scale, 1), mode="nearest"
        )
        x = x.numpy()[0, 0]
    return x


def gauss_kernel_1d(shape=3, std=1.0):
    x = torch.linspace(start=0, end=shape - 1, steps=shape)
    x_center = (shape - 1) / 2

    g = torch.exp(-((x - x_center) ** 2 / (2.0 * std**2)))
    g = g / torch.sum(g)  # shape = 3
    return g


def gauss_kernel_2d(shape=(3, 3), std=(1.0, 1.0), pixel_size=(1.0, 1.0)):
    """
    Generate a 2D Gaussian kernel.
    ### Parameters:
    - `shape` (tuple): The shape of the kernel. Default: (3, 3).
    - `std` (tuple): The standard deviation of the kernel. Default: (1.0, 1.0).
    - `pixel_size` (tuple): The pixel size of the image. Default: (1.0, 1.0).
    ### Returns:
    - `g` (torch.Tensor): The 2D Gaussian kernel.
    """
    y = torch.linspace(start=0, end=shape[0] - 1, steps=shape[0])
    x = torch.linspace(start=0, end=shape[1] - 1, steps=shape[1])
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    y_center, x_center = (shape[0] - 1) / 2, (shape[1] - 1) / 2

    g = torch.exp(
        -(
            ((y_grid - y_center) * pixel_size[0]) ** 2 / (2.0 * std[0] ** 2)
            + ((x_grid - x_center) * pixel_size[1]) ** 2 / (2.0 * std[1] ** 2)
        )
    )
    g = g / torch.sum(g)
    return g


def gauss_kernel_3d(shape=(3, 3, 3), std=(1.0, 1.0, 1.0), pixel_size=(1.0, 1.0, 1.0)):
    """
    Generate a 3D Gaussian kernel.
    ### Parameters:
    - `shape` (tuple): The shape of the kernel. Default: (3, 3, 3).
    - `std` (tuple): The standard deviation of the kernel. Default: (1.0, 1.0, 1.0).
    - `pixel_size` (tuple): The pixel size of the image. Default: (1.0, 1.0, 1.0).
    ### Returns:
    - `g` (torch.Tensor): The 3D Gaussian kernel.
    """
    grids = [torch.linspace(start=0, end=int(dim) - 1, steps=int(dim)) for dim in shape]
    z_grid, y_grid, x_grid = torch.meshgrid(grids, indexing="ij")
    z_center, y_center, x_center = ((dim - 1) / 2 for dim in shape)

    g = torch.exp(
        -(
            ((z_grid - z_center) * pixel_size[0]) ** 2 / (2.0 * std[0] ** 2)
            + ((y_grid - y_center) * pixel_size[1]) ** 2 / (2.0 * std[1] ** 2)
            + ((x_grid - x_center) * pixel_size[2]) ** 2 / (2.0 * std[2] ** 2)
        )
    )
    g = g / torch.sum(g)  # shape = [3, 3, 3]
    return g


def padding_kernel(x, y):
    """
    Padding the kernel to the same size as the ground-truth kernel.
    ### Parameters:
    - `x` (numpy.ndarray): Input kernel.
    - `y` (numpy.ndarray): Ground-truth kernel.
    ### Returns:
    - `x` (numpy.ndarray): Padded kernel.
    """
    dim = y.ndim
    if dim == 3:
        i_x, j_x, k_x = x.shape
        i_y, j_y, k_y = y.shape
        if (j_x <= j_y) & (i_x <= i_y):
            x = np.pad(
                x,
                pad_width=(
                    ((i_y - i_x) // 2,) * 2,
                    ((j_y - j_x) // 2,) * 2,
                    ((k_y - k_x) // 2,) * 2,
                ),
            )
    if dim == 2:
        j_x, k_x = x.shape
        j_y, k_y = y.shape
        if j_x <= j_y:
            x = np.pad(x, pad_width=(((j_y - j_x) // 2,) * 2, ((k_y - k_x) // 2,) * 2))
    return x


def ave_pooling(x, scale_factor: int = 1):
    """
    Average pooling for 2D/3D image.
    ### Parameters:
    - `x` (numpy.ndarray): Input image with a shape of `(Ny, Nx)` or `(Nz, Ny, Nx)`.
    - `scale_factor` (int): Downsampling factor.
    ### Returns:
    - `x` (numpy.ndarray): Downsampled image.
    """
    dim = len(x.shape)
    assert dim in [2, 3], "[ERROR] Only support 2D or 3D image."
    x = torch.tensor(x, dtype=torch.float32)[None, None]
    if dim == 2:
        x = torch.nn.functional.avg_pool2d(x, kernel_size=scale_factor)
    if dim == 3:
        x = torch.nn.functional.avg_pool3d(x, kernel_size=scale_factor)
    x = x.numpy()[0, 0]
    return x


def add_mix_noise(x, poisson=0, sigma_gauss=0, scale_factor: int = 1):
    """
    Add Poisson and Gaussian noise.
    ### Parameters:
    - `x` (numpy.ndarray): Input image with any shape.
    - `poisson` (int): Add Poisson noise or not.
    - `sigma_gauss` (float): Standard deviation of Gaussian noise.
    - `scale_factor` (int): Downsampling factor.
    ### Returns:
    - `x_n` (numpy.ndarray): Noisy image.
    """
    # clip to non-negative
    x = np.maximum(x, 0.0)

    # add poisson noise
    x_poi = np.random.poisson(lam=x) if poisson == 1 else x

    # downsampling
    if scale_factor > 1:
        x_poi = ave_pooling(x_poi, scale_factor=scale_factor)

    # add gaussian noise
    # based on the code from RLN
    # https://github.com/MeatyPlus/Richardson-Lucy-Net/blob/77256d1019dae7db7c4763b2659aa19a8a0e666f/Phantom_generate/Generation_of_anisotropic_input.m#L42
    if sigma_gauss > 0:
        max_signal = np.max(x_poi)
        x_poi_norm = x_poi / max_signal
        x_poi_gaus = x_poi_norm + np.random.normal(
            loc=0, scale=sigma_gauss / max_signal, size=x_poi_norm.shape
        )
        x_n = x_poi_gaus * max_signal
    else:
        x_n = x_poi
    x_n = x_n.astype(np.float32)
    return x_n


def fft_n(kernel, s=None):
    """
    Compute the Fourier transform of the kernel.
    ### Parameters:
    - `kernel` (numpy.ndarray): Input kernel with any shape.
    - `s` (tuple): The size of the output Fourier transform.
    ### Returns:
    - `kernel_fft` (numpy.ndarray): Fourier transform of the kernel.
    """
    kernel_fft = np.abs(np.fft.fftshift(np.fft.fftn(kernel, s=s)))
    return kernel_fft


def center_crop(x, size):
    """
    Crop the center region of image.
    ### Parameters:
    - `x` (numpy.ndarray): Input image with a shape of `(Ny, Nx)` or `(Nz, Ny, Nx)`.
    - `size` (tuple): The size of the cropped region, length of `size` should be 2 or 3.
        - For 2D image, `size` is a tuple of length 2.
        - For 3D image, `size` is a tuple of length 3.
    ### Returns:
    - `out` (numpy.ndarray): Cropped image.
    """
    dim = len(x.shape)
    assert dim in [2, 3], "[ERROR] Only support 2D or 3D image."
    if dim == 3:
        Nz, Ny, Nx = x.shape
        assert (
            len(size) == 3
        ), f"[ERROR] The size of cropped region should be 3 for 3D image, but got {len(size)}"
        assert (
            size[0] <= Nz and size[1] <= Ny and size[2] <= Nx
        ), f"[ERROR] The size of cropped region should be smaller than the size of image. but got {size} when image size is {x.shape}"
        z_start = (Nz - size[0]) // 2
        y_start = (Ny - size[1]) // 2
        x_start = (Nx - size[2]) // 2
        out = x[
            z_start : z_start + size[0],
            y_start : y_start + size[1],
            x_start : x_start + size[2],
        ]
    if dim == 2:
        Ny, Nx = x.shape
        assert (
            len(size) == 2
        ), f"[ERROR] The size of cropped region should be 2 for 2D image, but got {len(size)}"
        assert (
            size[0] <= Ny and size[1] <= Nx
        ), f"[ERROR] The size of cropped region should be smaller than the size of image, but got {size} when image size is {x.shape}"
        y_start = (Ny - size[0]) // 2
        x_start = (Nx - size[1]) // 2
        out = x[y_start : y_start + size[0], x_start : x_start + size[1]]
    return out


def even2odd(x):
    """
    Convert the input PSF to an odd-shape PSF by interpolation.
    ### Parameters:
    - `x` (numpy.ndarray): Input PSF with a shape of `(Ny, Nx)` or `(Nz, Ny, Nx)`.
    ### Returns:
    - `x_inter` (numpy.ndarray): Output PSF.
    """
    dim = len(x.shape)
    assert dim in [2, 3], "[ERROR] Only support 2D or 3D PSF."
    # 3D PSF
    if dim == 3:
        i, j, k = x.shape
        if i % 2 == 0:
            i = i - 1
        if j % 2 == 0:
            j = j - 1
        if k % 2 == 0:
            k = k - 1
        dict_inter = dict(size=(i, j, k), mode="trilinear")
    # 2D PSF
    if dim == 2:
        i, j = x.shape
        if i % 2 == 0:
            i = i - 1
        if j % 2 == 0:
            j = j - 1
        dict_inter = dict(size=(i, j), mode="bilinear")

    x = torch.tensor(x)[None, None]
    x_inter = torch.nn.functional.interpolate(x, **dict_inter)
    x_inter = x_inter / x_inter.sum()  # normalize PSF to have a sum of 1.0.
    x_inter = x_inter.numpy()[0, 0]
    print(f"[INFO] convert PSF shape from {x.numpy()[0, 0].shape} to {x_inter.shape}")
    return x_inter


def percentile_norm(x, p_low=0, p_high=100):
    """percentile-based normalization."""
    xmax, xmin = np.percentile(x, p_high), np.percentile(x, p_low)
    x = (x - xmin) / (xmax - xmin)
    x = np.clip(x, a_min=0.0, a_max=1.0)
    return x


def linear_transform(x, y):
    """
    Linear transformation between two images.
    ### Parameters:
    - `x` (numpy.ndarray): Input image with a shape of `(Ny, Nx)` or `(Nz, Ny, Nx)`.
    - `y` (numpy.ndarray): Target image with a shape of `(Ny, Nx)` or `(Nz, Ny, Nx)`.
    ### Returns:
    - `x` (numpy.ndarray): Transformed image.
    """
    assert (
        x.shape == y.shape
    ), f"[ERROR] The shape of x and y should be the same, but got {x.shape} and {y.shape}"
    # check x and y whether have same value
    if np.all(x == y):
        return x
    n = np.size(x)
    m_x = np.mean(x)
    m_y = np.mean(y)
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    x_linear = b_1 * x + b_0
    return x_linear


def read_image(img_path: str, normalization: bool = False, data_range: tuple = None):
    """
    Read image.
    ### Parameters:
    - `img_path` (str): Image path.
    - `normalization` (bool): Normalize data into (0,1).
    - `data_range` (tuple): (min, max) value of data.
    ### Returns:
    - `img` (numpy.ndarray): Image data.
    """
    # check file type, get extension of file
    _, ext = os.path.splitext(img_path)

    # DICOM data
    if ext == ".dcm":
        img_dcm = pydicom.dcmread(img_path)
        img = img_dcm.pixel_array
        img = img.astype(np.float32)

    # TIFF data
    if ext == ".tif":
        img = io.imread(img_path)

    if len(img.shape) in (2, 3):
        img = np.expand_dims(img, axis=0)

    # Image normalization
    if normalization == True:
        if data_range == None:
            img_max, img_min = img.max(), img.min()
            img = (img - img_min) / (img_max - img_min)
        if type(data_range) == tuple:
            assert (
                len(data_range) == 2
            ), f"[ERROR] data_range should be a tuple of length 2, but got {len(data_range)}"
            assert (
                data_range[0] < data_range[1]
            ), f"[ERROR] data_range should be (min, max), but got {data_range}"

            img = (img - data_range[0]) / (data_range[1] - data_range[0])

    return img.astype(np.float32)


class SRDataset(Dataset):
    """
    Super-resolution dataset used to get low-resolution and hig-resolution data.

    ### Parameters:
    - `hr_root_path` (str): root path for high-resolution data.
    - `lr_root_path` (str): root path for low-resolution data.
    - `hr_txt_file_path` (str): path of file saving path of high-resolution data.
    - `lr_txt_file_path` (str): path of file saving path of low-resolution data.
    - `id_range` (tuple): extract part of the data.
                        Default: None, all the data in dataset.
    - `transform` (bool): data transformation. Default: None.
    - `normalization` (tuple[bool]): whether to normalize the data
                    when read image (lr, hr). Default: (False, False).
    ### Returns:
    - sample (dict): {
        'lr': low-resolution image,
        'hr': high-resolution image
    }
    """

    def __init__(
        self,
        hr_root_path: str,
        lr_root_path: str,
        hr_txt_file_path: str,
        lr_txt_file_path: str,
        id_range=None,
        transform=None,
        normalization=(False, False),
    ):
        super().__init__()
        self.lr_root_path = lr_root_path
        self.hr_root_path = hr_root_path
        self.transform = transform
        self.normalization = normalization

        with open(lr_txt_file_path) as f:
            self.file_names_lr = f.read().splitlines()
        with open(hr_txt_file_path) as f:
            self.file_names_hr = f.read().splitlines()

        data_size_all = len(self.file_names_lr)
        if id_range != None:
            self.file_names_lr = self.file_names_lr[id_range[0] : id_range[1]]
            self.file_names_hr = self.file_names_hr[id_range[0] : id_range[1]]

        # ----------------------------------------------------------------------
        print("-" * 80)
        print(f"[INFO] Use datasets. ({len(self.file_names_lr)}|{data_size_all})")
        if self.transform is not None:
            print(f"[INFO] Enable data transformation.")
        else:
            print(f"[INFO] Disable data transformation.")
        if self.normalization[0] == True:
            print(f"[INFO] Normalize LR data.")
        else:
            print(f"[INFO] Not normalize LR data.")
        if self.normalization[1] == True:
            print(f"[INFO] Normalize HR data.")
        else:
            print(f"[INFO] Not normalize HR data.")
        print("-" * 80)

    def __len__(self):
        return len(self.file_names_lr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path_lr = os.path.join(self.lr_root_path, self.file_names_lr[idx])
        img_path_hr = os.path.join(self.hr_root_path, self.file_names_hr[idx])

        image_lr = read_image(img_path_lr, normalization=self.normalization[0])
        image_hr = read_image(img_path_hr, normalization=self.normalization[1])

        if self.transform is not None:
            image_lr = self.transform(image_lr)
            image_hr = self.transform(image_hr)

        # scale = np.percentile(image_hr, 95)
        # return {'lr': torch.tensor(image_lr/scale), 'hr': torch.tensor(image_hr/scale)}
        return {"lr": torch.tensor(image_lr), "hr": torch.tensor(image_hr)}


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image_lr, image_hr = sample["lr"], sample["hr"]

        h, w = image_lr.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image_lr_new = transform.resize(image_lr, (new_h, new_w))

        return {"lr": image_lr_new, "hr": image_hr}


class ToNumpy(object):
    """
    Convert pytorch tensor into numpy array,
    and shift the channel axis to the last axis.
    Args:
    - tensor (torch tensor): input tensor.
    """

    def __call__(self, tensor):
        img = tensor.cpu().detach().numpy()
        # move the chennel axis to the last dimension.
        if len(img.shape) == 4:
            img = np.transpose(img, axes=(0, 2, 3, 1))
        if len(img.shape) == 5:
            img = np.transpose(img, axes=(0, 2, 3, 4, 1))
        return img


def tensor2rgb(x):
    x = torch.clamp(x, min=0.0, max=1.0)
    x = (x * 255.0).to(torch.uint8)
    x = x.cpu().detach().numpy()
    if len(x.shape) == 4:
        x = np.transpose(x, axes=(0, 2, 3, 1))
    if len(x.shape) == 5:
        x = np.transpose(x, axes=(0, 1, 3, 4, 2))
    return x


def tensor2gray(x):
    x = x.cpu().detach().numpy()
    if len(x.shape) == 4:
        x = np.transpose(x, axes=(0, 2, 3, 1))
    if len(x.shape) == 5:
        x = np.transpose(x, axes=(0, 1, 3, 4, 2))
    return x


if __name__ == "__main__":
    # data_set_name = 'tinymicro_synth'
    # data_set_name = 'tinymicro_real'
    # data_set_name = 'biosr_real'
    data_set_name = "lung3_synth"
    # data_set_name = 'msi_synth'

    # -------------------------------------------------------------------------------------
    if data_set_name == "tinymicro_synth":
        # TinyMicro (synth)
        hr_root_path = os.path.join("data", "raw", "cyto_potable_microscope", "data1")
        lr_root_path = os.path.join(
            "data",
            "raw",
            "cyto_potable_microscope",
            "data_synth",
            "train",
            "sf_4_k_2.0_gaussian_mix_ave",
        )  # TinyMicro (synth)

        hr_txt_file_path = os.path.join(
            "data", "raw", "cyto_potable_microscope", "train_txt", "hr.txt"
        )
        lr_txt_file_path = os.path.join(
            "data", "raw", "cyto_potable_microscope", "train_txt", "lr.txt"
        )
        normalization = (False, False)

    if data_set_name == "tinymicro_real":
        # TinyMicro (real)
        hr_root_path = os.path.join("data", "raw", "cyto_potable_microscope", "data1")
        lr_root_path = os.path.join("data", "raw", "cyto_potable_microscope", "data1")

        hr_txt_file_path = os.path.join(
            "data", "raw", "cyto_potable_microscope", "train_txt", "hr.txt"
        )
        lr_txt_file_path = os.path.join(
            "data", "raw", "cyto_potable_microscope", "train_txt", "lr.txt"
        )
        normalization = (False, False)

    if data_set_name == "biosr_real":
        pass
    if data_set_name == "lung3_synth":
        # Lung3 (synth)
        # F:\Datasets\Lung3\manifest-41uMmeOh151290643884877939
        # F:\Datasets\Lung3\manifest-41uMmeOh151290643884877939\data_synth\train\sf_4_k_2.0_gaussian_mix_ave
        hr_root_path = os.path.join(
            "F:", os.sep, "Datasets", "Lung3", "manifest-41uMmeOh151290643884877939"
        )
        lr_root_path = os.path.join(
            "F:",
            os.sep,
            "Datasets",
            "Lung3",
            "manifest-41uMmeOh151290643884877939",
            "data_synth",
            "train",
            "sf_4_k_2.0_gaussian_mix_ave",
        )

        hr_txt_file_path = os.path.join(
            "F:",
            os.sep,
            "Datasets",
            "Lung3",
            "manifest-41uMmeOh151290643884877939",
            "train_txt",
            "hr.txt",
        )
        lr_txt_file_path = os.path.join(
            "F:",
            os.sep,
            "Datasets",
            "Lung3",
            "manifest-41uMmeOh151290643884877939",
            "train_txt",
            "lr.txt",
        )
        normalization = (False, True)
    if data_set_name == "msi_synth":
        pass

    fig_dir = os.path.join("outputs", "figures")

    # -------------------------------------------------------------------------------------
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # -------------------------------------------------------------------------------------
    paired_dataset = SRDataset(
        hr_root_path=hr_root_path,
        lr_root_path=lr_root_path,
        hr_txt_file_path=hr_txt_file_path,
        lr_txt_file_path=lr_txt_file_path,
        transform=trans,
        id_range=[0, 1000],
        normalization=normalization,
    )

    print("Datasize: ", paired_dataset.__len__())

    dataloader = DataLoader(
        dataset=paired_dataset, batch_size=5, shuffle=False, num_workers=0
    )

    # -------------------------------------------------------------------------------------
    i_batch_show = 0
    for i_batch, sample in enumerate(dataloader):
        print(
            i_batch,
            sample["lr"].size(),
            sample["hr"].size(),
            "max: ",
            torch.max(sample["hr"]).item(),
            "min: ",
            torch.min(sample["hr"]).item(),
        )
        if i_batch == i_batch_show:
            fig, axes = plt.subplots(
                nrows=2, ncols=5, figsize=(12, 5), dpi=600, constrained_layout=True
            )
            [ax.set_axis_off() for ax in axes.ravel()]

            images_lr_batch, images_hr_batch = sample["lr"], sample["hr"]
            if images_hr_batch.shape[1] == 1:
                cm = "gray"
            else:
                cm = None
            for i in range(5):
                axes[0, i].imshow(
                    images_lr_batch[i].transpose(0, -1).transpose(0, 1),
                    cmap=cm,
                    vmin=0.0,
                    vmax=1.0,
                )
                axes[1, i].imshow(
                    images_hr_batch[i].transpose(0, -1).transpose(0, 1),
                    cmap=cm,
                    vmin=0.0,
                    vmax=1.0,
                )

            save_to = os.path.join(fig_dir, data_set_name)
            if os.path.exists(save_to) == False:
                os.makedirs(save_to, exist_ok=True)
            plt.savefig(
                os.path.join(
                    fig_dir, data_set_name, "sample_batch_{}".format(i_batch_show)
                )
            )
            break
    print("end")
