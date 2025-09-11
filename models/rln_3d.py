import torch, torchinfo
import torch.nn as nn
import numpy as np


def gauss_kernel_3d(shape=[3, 3, 3], sigma=1.0):
    """
    Create  Gaussian kernel.

    ### Parameter:
    - `shape` (tuple[int]): kernel shape. Default: (3, 3, 3).
    - `sigma` (float): kernel std. Default: 1.0.

    ### Return:
    - `g` (torch.Tensor): kernel.
    """
    x_data, y_data, z_data = np.mgrid[
        -shape[0] // 2 + 1 : shape[0] // 2 + 1,
        -shape[1] // 2 + 1 : shape[1] // 2 + 1,
        -shape[2] // 2 + 1 : shape[2] // 2 + 1,
    ]

    x_data = np.expand_dims(np.expand_dims(x_data, axis=0), axis=0)
    y_data = np.expand_dims(np.expand_dims(y_data, axis=0), axis=0)
    z_data = np.expand_dims(np.expand_dims(z_data, axis=0), axis=0)

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    z = torch.tensor(z_data, dtype=torch.float32)

    g = torch.exp(-((x**2 + y**2 + z**2) / (2.0 * sigma**2)))
    # normalization
    g = g / torch.max(g)  # [1,1,3,3,3]
    return g


def gauss_kernel_3d_multichannel(shape=[4, 3, 3, 3, 3], stddev=[2.0, 0.5, 1.0, 1.5]):
    """
    Create multi-channel  Gaussian kernel for initialization of convolutional layer.
    ### Parameter:
    - shape (tuple[int]): kernel shape (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]).
    - stddev (tuple[int]): kernel std in each channel.
    """
    kernels = []
    for i in range(shape[0]):
        kernels.append(
            gauss_kernel_3d(shape=[shape[2], shape[3], shape[4]], sigma=stddev[i])
        )  # [1,1,ker0,ker1,ker2]
    init = torch.cat(kernels, dim=0)  # [out_channels, 1, ker0, ker1, ker2]
    init = init.repeat(
        1, shape[1], 1, 1, 1
    )  # [out_channels, in_channels, ker0, ker1, ker2]

    # random weights to increase randomness.
    minval, maxval = 0.0, 1.0
    rad = (maxval - minval) * torch.rand(
        size=[1, shape[1], 1, 1, 1], dtype=torch.float32
    ) + minval
    init = init * rad
    return init


class FP1(nn.Module):
    """
    ### Parameter:
    - in_channels (int): channel number of input image. Default: 3.
    - n_features (int): number of features. Default: 4.
    - kernel_size (int): kernel size. Default: 3.
    - bias (bool): bias in convolutional layer. Default: False.
    """

    def __init__(self, in_channels=3, n_features=4, kernel_size=3, bias=False):
        super().__init__()
        self.n_features = n_features

        dict_conv = dict(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2),
            bias=bias,
        )

        # ----------------------------------------------------------------------
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=n_features, **dict_conv
        )
        self.bn_act1 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.conv2 = nn.Conv3d(
            in_channels=n_features, out_channels=n_features, **dict_conv
        )
        self.bn_act2 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.conv3 = nn.Conv3d(
            in_channels=n_features * 2, out_channels=in_channels * 2, **dict_conv
        )
        self.bn_act3 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.act0 = nn.Softplus()

        # initialization of convolution filters --------------------------------
        # (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        with torch.no_grad():
            self.conv1.weight.data = gauss_kernel_3d_multichannel(
                shape=[n_features, in_channels] + [kernel_size] * 3,
                stddev=np.linspace(0.5, 2.0, n_features),
            )
            self.conv2.weight.data = gauss_kernel_3d_multichannel(
                shape=[n_features, n_features] + [kernel_size] * 3,
                stddev=np.linspace(0.5, 2.0, n_features),
            )
            self.conv3.weight.data = gauss_kernel_3d_multichannel(
                shape=[n_features, n_features * 2] + [kernel_size] * 3,
                stddev=np.linspace(0.5, 2.0, n_features),
            )

    def forward(self, x: torch.Tensor):
        out_1 = self.bn_act1(self.conv1(x))
        out_2 = self.bn_act2(self.conv2(out_1))
        cat_12 = torch.cat([out_1, out_2], dim=1)  # concat along channel dimension
        out_3 = self.bn_act3(self.conv3(cat_12))
        out = out_3 + self.act0(x.repeat(1, self.n_features, 1, 1, 1))
        return out


class FP2(nn.Module):
    """
    ### Parameter:
    - `in_channels` (int): channel number of input image. Default: 1.
    - `n_features` (int): number of features. Default: 4.
    - `kernel_size` (int): kernel size. Default: 3.
    - `bias` (bool): bias in convolutional layer. Default: False.
    """

    def __init__(self, in_channels=1, n_features=4, kernel_size=3, bias=False):
        super().__init__()
        self.n_features = n_features

        dict_conv = dict(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2),
            bias=bias,
        )

        # ----------------------------------------------------------------------
        self.conv1 = nn.Conv3d(in_channels, n_features, **dict_conv)
        self.bn_act1 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.conv2 = nn.Conv3d(n_features, n_features, **dict_conv)
        self.bn_act2 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.act0 = nn.Softplus()

        # ----------------------------------------------------------------------
        with torch.no_grad():
            self.conv1.weight.data = gauss_kernel_3d_multichannel(
                shape=[n_features, in_channels] + [kernel_size] * 3,
                stddev=np.linspace(0.5, 2.0, n_features),
            )
            self.conv2.weight.data = gauss_kernel_3d_multichannel(
                shape=[n_features, n_features] + [kernel_size] * 3,
                stddev=np.linspace(0.5, 2.0, n_features),
            )

    def forward(self, x: torch.Tensor):
        out_1 = self.bn_act1(self.conv1(x))
        out_2 = self.bn_act2(self.conv2(out_1))
        out = out_2 + self.act0(x.repeat(1, self.n_features, 1, 1, 1))
        return out


class BP1(nn.Module):
    """
    ### Parameter:
    - `in_channels` (int): channel number of input image. Default: 3.
    - `n_features` (int): number of features. Default: 8.
    - `kernel_size` (int): kernel size. Default: 3.
    - `init_w_std` (float): std of the weight used for convolutional layer initialization. Default: 1.0.
    - `bias` (bool): bias in convolutional layer. Default: False.
    """

    def __init__(
        self, in_channels=1, n_features=8, kernel_size=3, init_w_std=1.0, bias=False
    ):
        super().__init__()

        dict_conv = dict(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2),
            bias=bias,
        )

        # ----------------------------------------------------------------------
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=n_features, **dict_conv
        )
        self.bn_act1 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.conv2 = nn.Conv3d(
            in_channels=n_features, out_channels=n_features, **dict_conv
        )
        self.bn_act2 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.conv3 = nn.Conv3d(
            in_channels=n_features * 2, out_channels=n_features, **dict_conv
        )
        self.bn_act3 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        # initialization of the weight of filters ------------------------------
        nn.init.trunc_normal_(tensor=self.conv1.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv2.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv3.weight, mean=0.0, std=init_w_std)

    def forward(self, x: torch.Tensor):
        out_1 = self.bn_act1(self.conv1(x))
        out_2 = self.bn_act2(self.conv2(out_1))
        cat_12 = torch.cat(tensors=[out_2, out_1], dim=1)
        out_3 = self.bn_act3(self.conv3(cat_12))
        return out_3


class BP2(nn.Module):
    """
    ### Parameter:
    - `in_channels` (int): channel number of input image. Default: 3.
    - `n_features` (int): number of features. Default: 8.
    - `kernel_size` (int): kernel size. Default: 3.
    - `init_w_std` (float): std of the weight used for convolutional layer initialization. Default: 1.0.
    - `bias` (bool): bias in convolutional layer. Default: False.
    """

    def __init__(
        self, in_channels=3, n_features=8, kernel_size=3, init_w_std=1.0, bias=False
    ):
        super().__init__()

        dict_conv = dict(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2),
            bias=bias,
        )

        # ----------------------------------------------------------------------
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=n_features, **dict_conv
        )
        self.bn_act1 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.conv2 = nn.Conv3d(
            in_channels=n_features, out_channels=n_features, **dict_conv
        )
        self.bn_act2 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        # initialization of the weight of filters ------------------------------
        nn.init.trunc_normal_(tensor=self.conv1.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv2.weight, mean=0.0, std=init_w_std)

    def forward(self, x):
        out_1 = self.bn_act1(self.conv1(x))
        out_2 = self.bn_act2(self.conv2(out_1))
        return out_2


class BP1up(nn.Module):
    """
    ### Parameter:
    - `in_channels` (int): channel number of input image. Default: 8.
    - `n_features` (int): number of features. Default: 4.
    - `kernel_size` (int): kernel size. Default: 3.
    - `init_w_std` (float): std of the weight used for convolutional layer initialization. Default: 1.0.
    - `bias` (bool): bias in convolutional layer. Default: False.
    """

    def __init__(
        self, in_channels=8, n_features=4, kernel_size=3, init_w_std=1.0, bias=False
    ):
        super().__init__()
        self.conv_trans = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=n_features,
            kernel_size=2,
            stride=(2, 2, 2),
            bias=bias,
        )
        self.bn_act_trans = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.conv = nn.Conv3d(
            in_channels=n_features,
            out_channels=n_features,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2),
            bias=bias,
        )
        self.bn_act = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        # initialization of the weight of filters
        nn.init.trunc_normal_(tensor=self.conv_trans.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv.weight, mean=0.0, std=init_w_std)

    def forward(self, x: torch.Tensor):
        out_trans = self.bn_act_trans(self.conv_trans(x))
        out = self.bn_act(self.conv(out_trans))
        return out


class DV(nn.Module):
    """
    output = a / (b + eps).

    ### Parameter:
    - in_channels (int): channel number of input image. Default: 3.
    - eps (float): epsilon. Default: 0.0001.
    """

    def __init__(self, in_channels=3, eps=0.0001):
        super().__init__()
        self.eps = eps
        self.bn = nn.BatchNorm3d(num_features=in_channels)

    def forward(self, a, b):
        dv = self.bn(torch.div(a, b + self.eps))
        return dv


class MUL(nn.Module):
    """
    output = a * b.
    ### Parameter:
    - `in_channels` (int): channel number of input image.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.bn_act = nn.Sequential(
            nn.BatchNorm3d(num_features=in_channels), nn.Softplus()
        )

    def forward(self, a, b):
        mul = self.bn_act(torch.mul(a, b))
        return mul


class Merge(nn.Module):
    """
    ### Parameter:
    - `in_channels` (int): channel number of input image. Default: 1.
    - `n_features` (int): Number of features. Default: 8.
    - `kernel_size` (int): kernel size. Default: 3.
    - `init_w_std` (float): std used for weight initialization. Default: 1.0.
    - `bias` (bool): bias in convolutional layer. Default: False.
    """

    def __init__(
        self, in_channels=1, n_features=8, kernel_size=3, init_w_std=1.0, bias=False
    ):
        super().__init__()

        dict_conv = dict(
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2),
            bias=bias,
        )

        # ----------------------------------------------------------------------
        self.conv1 = nn.Conv3d(
            in_channels=in_channels, out_channels=n_features, **dict_conv
        )
        self.conv2 = nn.Conv3d(
            in_channels=n_features + in_channels + in_channels,
            out_channels=n_features,
            **dict_conv,
        )
        self.conv3 = nn.Conv3d(
            in_channels=(n_features + n_features),
            out_channels=n_features,
            **dict_conv,
        )

        self.bn_act3 = nn.Sequential(
            nn.BatchNorm3d(num_features=n_features), nn.Softplus()
        )

        self.act1 = nn.Softplus()
        self.act2 = nn.Softplus()

        # initialization of the weight of filters ------------------------------
        nn.init.trunc_normal_(tensor=self.conv1.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv2.weight, mean=0.0, std=init_w_std)
        nn.init.trunc_normal_(tensor=self.conv3.weight, mean=0.0, std=init_w_std)

    def forward(self, e1, e2):
        e2_conv = self.act1(self.conv1(e2))

        e_cat = torch.cat(tensors=[e2_conv, e2, e1], dim=1)
        e_cat = self.act2(self.conv2(e_cat))

        merge = torch.cat(tensors=[e2_conv, e_cat], dim=1)
        merge = self.bn_act3(self.conv3(merge))
        return merge


class RLN3D(nn.Module):
    """
    Parameters:
    - scale (int): upsampling scale factor. Default: 1.
        The original RLN can not unsampling image.
    - in_channels (int): channel number of input image. Default: 3.
    - n_features (int): number of features. Default: 4.
    - kernel_size (int): kernel size. Default: 3.
    """

    def __init__(
        self,
        scale: int = 1,
        in_channels: int = 1,
        n_features: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels

        dict_fp = dict(kernel_size=kernel_size, bias=False)
        dict_bp = dict(kernel_size=kernel_size, init_w_std=1.0, bias=False)
        # ----------------------------------------------------------------------
        # H1
        self.ave_pool = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)
        self.FP1 = FP1(in_channels=in_channels, n_features=n_features, **dict_fp)
        self.DV1 = DV(in_channels=in_channels)
        self.BP1 = BP1(in_channels=in_channels, n_features=8, **dict_bp)
        self.BP1up = BP1up(in_channels=8, n_features=n_features, **dict_bp)
        self.MUL1 = MUL(in_channels=in_channels)
        # H2
        self.FP2 = FP2(in_channels=in_channels, n_features=n_features, **dict_fp)
        self.DV2 = DV(in_channels=in_channels)
        self.BP2 = BP2(in_channels=in_channels, n_features=8, **dict_bp)
        self.MUL2 = MUL(in_channels=in_channels)
        # H3
        self.Merge = Merge(in_channels=in_channels, n_features=8, **dict_bp)

        # upsampling
        # if self.scale > 1:
        #     self.upsampler = common.Upsampler(scale=scale,n_features=8,kernel_size=kernel_size,\
        #                         bn=False,act=False,bias=True)

        # ----------------------------------------------------------------------
        if in_channels > 1:
            self.conv_last = nn.Conv3d(
                8,
                in_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size // 2),
                bias=True,
            )

    def forward(self, x):
        # trilinear interpolation ( equivalent of bicubic)
        x = nn.functional.interpolate(
            input=x, scale_factor=self.scale, mode="trilinear", align_corners=False
        )

        # H1 -------------------------------------------------------------------
        Iap = self.ave_pool(x)

        fp1 = self.FP1(Iap)
        fp1 = torch.mean(input=fp1, dim=1, keepdim=True)
        fp1 = torch.cat([fp1] * self.in_channels, dim=1)

        # divide
        dv1 = self.DV1(Iap, fp1)

        bp1 = self.BP1(dv1)
        bp1up = self.BP1up(bp1)
        bp1up = torch.mean(input=bp1up, dim=1, keepdim=True)
        E1 = self.MUL1(x, bp1up)

        # H2 -------------------------------------------------------------------
        fp2 = self.FP2(x)
        fp2 = torch.mean(input=fp2, dim=1, keepdim=True)
        fp2 = torch.cat([fp2] * self.in_channels, dim=1)

        dv2 = self.DV2(x, fp2)

        bp2 = self.BP2(dv2)
        bp2 = bp2 + torch.ones_like(input=bp2)
        bp2 = torch.mean(input=bp2, dim=1, keepdim=True)

        E2 = self.MUL2(E1, bp2)

        # H3 -------------------------------------------------------------------
        merge = self.Merge(E1, E2)

        if self.in_channels > 1:
            out = self.conv_last(merge)
        else:
            out = torch.mean(input=merge, dim=1, keepdim=True)
        return out


if __name__ == "__main__":
    kernel = gauss_kernel_3d(shape=[3, 3, 3], sigma=1.0)
    print(" Gaussian kernel shape:", kernel.shape)
    print(" Gaussian kernel:\n", kernel[0, 0])

    kernels = gauss_kernel_3d_multichannel(
        shape=[4, 3, 3, 3, 3], stddev=np.linspace(0.5, 2.0, 4)
    )
    print("Multi-channel  Gaussian kernels shape:", kernels.shape)
    print("First kernel sample:\n", kernels[0, 0, :, :, 1])  # middle slice

    # --------------------------------------------------------------------------
    inchannle = 1
    input_shape = (4, inchannle, 32, 64, 64)  # (B, C, D, H, W)
    x = torch.zeros(size=input_shape)
    model = RLN3D(scale=1, in_channels=inchannle, n_features=4, kernel_size=3)
    o = model(x)
    torchinfo.summary(model=model, input_size=input_shape)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {o.shape}")
