import torch, time
import torch.nn as nn
from fft_conv_pytorch import fft_conv
import sys

sys.path.append("E:\\Project\\2023 cytoSR")
from utils import data as utils_data


class TV_grad(nn.Module):
    """
    Gradient of TV.
    @ Wang, C. et al. Sparse deconvolution for background noise suppression with
    total variation regularization in light field microscopy. Opt Lett 48, 1894
    (2023).
    """

    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        batch = []
        for i in range(x.shape[0]):
            sample = []
            for j in range(x.shape[1]):
                x_sc = x[i, j]  # single chennel image
                u = torch.gradient(x_sc, edge_order=1)  # gradient
                u0 = u[0] / (torch.sqrt(self.epsilon + u[0] ** 2 + u[1] ** 2))
                u1 = u[1] / (torch.sqrt(self.epsilon + u[0] ** 2 + u[1] ** 2))
                d = torch.gradient(u0)[0] + torch.gradient(u1)[1]  # divergence
                sample.append(d)
            sample = torch.stack(sample, dim=0)
            batch.append(sample)
        batch = -1.0 * torch.stack(batch, dim=0)
        return batch


class RadialSymmetricConv2D(nn.Module):
    """
    2D convolution with radial symetric kernel.
    ### Parameters:
    - `in_channels`: int, default=1, input channels.
    - `kernel_size`: list, default=[25, 25], kernel size.
    - `init`: str, default="gauss", initialization method.
    - `std_init`: list, default=[1.0, 1.0], standard deviation of the kernel.
    - `padding_mode`: str, default="reflect", padding mode.
    - `positive`: bool, default=False, whether to use positive constraints.
    - `interpolation`: bool, default=True, whether to use linear interpolation.
    - `over_sampling`: int, default=2, over sampling factor.
    - `kernel_norm`: bool, default=True, whether to normalize the kernel.
    - `conv_mode`: str, default="direct", convolution mode. 'direct' or 'fft'.
    """

    def __init__(
        self,
        in_channels: int = 1,
        kernel_size=(25, 25),
        init="gauss",
        std_init=(1.0, 1.0),
        padding_mode="reflect",
        positive: bool = False,
        interpolation: bool = True,
        over_sampling=2,
        kernel_norm: bool = True,
        conv_mode="direct",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = torch.tensor(kernel_size)
        self.kernel_norm = kernel_norm
        self.interpolation = interpolation
        self.positive = positive
        self.padding_mode = padding_mode
        self.PSF = None
        self.conv_mode = conv_mode

        # ----------------------------------------------------------------------
        # xy plane
        Ny, Nx = self.kernel_size
        yp, xp = (Ny - 1) / 2, (Nx - 1) / 2  # position of the center of PSF
        maxRadius = (
            torch.round(torch.sqrt(((Nx - 1) - xp) ** 2 + ((Ny - 1) - yp) ** 2)) + 1
        )
        R = (
            torch.linspace(
                start=0,
                end=maxRadius * over_sampling - 1,
                steps=int(maxRadius * over_sampling),
            )
            / over_sampling
        )  # sampling points

        gridx = torch.linspace(start=0, end=Nx - 1, steps=Nx)
        gridy = torch.linspace(start=0, end=Ny - 1, steps=Ny)
        X, Y = torch.meshgrid(gridx, gridy)

        rPixel = torch.sqrt((X - xp) ** 2 + (Y - yp) ** 2)
        index = torch.floor(rPixel * over_sampling).type(torch.int)

        self.register_buffer("index1", index)
        if self.interpolation == True:
            disR = (rPixel - R[index]) * over_sampling
            self.register_buffer("disR_b", disR)
            self.register_buffer("disR1", 1.0 - disR)
            self.register_buffer("index2", index + 1)

        # ----------------------------------------------------------------------
        # initialization
        if init == "gauss":
            psfapp_init = utils_data.gauss_kernel_2d(
                shape=(R.shape[0] * 2 - 1, R.shape[1] * 2 - 1),
                std=std_init,
                pixel_size=(1.0 / over_sampling,) * 2,
            )
            psfapp_init = psfapp_init * (over_sampling**2)

        if init == "ones":
            psfapp_init = torch.ones(size=(R.shape[0] * 2 - 1, R.shape[1] * 2 - 1))
            psfapp_init = psfapp_init / psfapp_init.sum()

        # ----------------------------------------------------------------------
        # extract half PSF
        self.PSF_half = nn.Parameter(data=psfapp_init[R.shape[0] - 1, R.shape[1] - 1 :])

    def get_kernel(self):
        # positive constraints
        PSF_half = torch.abs(self.PSF_half) if self.positive else self.PSF_half

        # linear interpolation / left-nearest interpolation
        if self.interpolation == True:
            self.PSF = (
                PSF_half[self.index2] * self.disR_b + PSF_half[self.index1] * self.disR1
            )
        else:
            self.PSF = PSF_half[self.index1]

        # PSF normalizarion
        if self.kernel_norm == True:
            self.PSF = torch.div(self.PSF, torch.sum(self.PSF))

        weight = self.PSF.repeat(repeats=[self.in_channels, 1, 1, 1])
        return weight

    def forward(self, x):
        weight = self.get_kernel()
        pad_size = (
            self.kernel_size[1] // 2,
            self.kernel_size[1] // 2,
            self.kernel_size[0] // 2,
            self.kernel_size[0] // 2,
        )
        x_pad = nn.functional.pad(input=x, pad=pad_size, mode=self.padding_mode)

        # convolution
        if self.conv_mode == "direct":
            conv = nn.functional.conv2d(
                input=x_pad, weight=weight, groups=self.in_channels
            )
        if self.conv_mode == "fft":
            conv = fft_conv(signal=x_pad, kernel=weight, groups=self.in_channels)
        return conv


class RadialSymmetricConv3D(nn.Module):
    """
    3D convolution with radial symetric kernel.
    ### Parameters:
    - `in_channels`: int, default=1, input channels.
    - `kernel_size`: list, default=[25, 25, 25], kernel size.
    - `init`: str, default="gauss", initialization method.
    - `std_init`: list, default=[1.0, 1.0, 1.0], standard deviation of the kernel.
    - `padding_mode`: str, default="reflect", padding mode.
    - `positive`: bool, default=False, whether to use positive constraints.
    - `interpolation`: bool, default=True, whether to use linear interpolation.
    - `over_sampling`: int, default=2, over sampling factor.
    - `kernel_norm`: bool, default=True, whether to normalize the kernel.
    - `conv_mode`: str, default="direct", convolution mode.
    """

    def __init__(
        self,
        in_channels: int = 1,
        kernel_size=(25, 25, 25),
        init="gauss",
        std_init=(1.0, 1.0, 1.0),
        padding_mode="reflect",
        positive: bool = False,
        interpolation: bool = True,
        over_sampling=2,
        kernel_norm: bool = True,
        conv_mode="direct",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = torch.tensor(kernel_size)
        self.kernel_norm = kernel_norm
        self.interpolation = interpolation
        self.positive = positive
        self.padding_mode = padding_mode  # constant or reflect
        self.PSF = None
        self.conv_mode = conv_mode

        # ----------------------------------------------------------------------
        Nz, Nx, Ny = self.kernel_size

        # xy plane
        xp, yp = (Nx - 1) / 2, (Ny - 1) / 2

        maxRadius = (
            torch.round(torch.sqrt(((Nx - 1) - xp) ** 2 + ((Ny - 1) - yp) ** 2)) + 1
        )
        if over_sampling == 1:
            R = (
                torch.linspace(
                    start=0,
                    end=maxRadius * over_sampling,
                    steps=int(maxRadius * over_sampling + 1),
                )
                / over_sampling
            )
        else:
            R = (
                torch.linspace(
                    start=0,
                    end=maxRadius * over_sampling - 1,
                    steps=int(maxRadius * over_sampling),
                )
                / over_sampling
            )

        gridx = torch.linspace(start=0, end=Nx - 1, steps=Nx)
        gridy = torch.linspace(start=0, end=Ny - 1, steps=Ny)
        Y, X = torch.meshgrid(gridx, gridy)

        rPixel = torch.sqrt((X - xp) ** 2 + (Y - yp) ** 2)

        if self.interpolation == False:
            index = torch.round(rPixel * over_sampling).type(torch.int)
        else:
            index = torch.floor(rPixel * over_sampling).type(torch.int)

        # z direction
        index = index[None].repeat(Nz, 1, 1)
        index_slice = torch.linspace(start=0, end=Nz - 1, steps=Nz)[..., None, None]
        index_slice = index_slice.repeat(1, Nx, Ny).type(torch.int)
        self.register_buffer("index_slice", index_slice)

        self.register_buffer("index1", index)
        if self.interpolation == True:
            disR = (rPixel - R[index]) * over_sampling
            self.register_buffer("disR_b", disR)
            self.register_buffer("disR1", 1.0 - disR)
            self.register_buffer("index2", index + 1)

        # ----------------------------------------------------------------------
        # initialization
        if init == "gauss":
            psfapp_init = utils_data.gauss_kernel_3d(
                shape=[Nz.numpy(), R.shape[0] * 2 - 1, R.shape[0] * 2 - 1],
                std=[std_init[0], std_init[1], std_init[2]],
                pixel_size=[1.0, 1.0 / over_sampling, 1.0 / over_sampling],
            )
            # to make the constructed PSF have a 1 sum
            psfapp_init = psfapp_init * (over_sampling**2)

        if init == "ones":
            psfapp_init = torch.ones(
                size=[Nz.numpy(), R.shape[0] * 2 - 1, R.shape[0] * 2 - 1]
            )
            psfapp_init = psfapp_init / psfapp_init.sum()

        if self.positive == True:
            # psfapp_init = torch.abs(psfapp_init)
            psfapp_init = torch.pow(100.0 * psfapp_init, 1 / 2)

        # ----------------------------------------------------------------------
        # extract half PSF
        self.PSF_half = nn.Parameter(
            data=psfapp_init[:, R.shape[0] - 1, R.shape[0] - 1 :]
        )

    def get_kernel(self):
        """
        Get the kernel.
        ### Returns:
        - `weight`: torch.Tensor, kernel.
        """
        # positive constraints
        # if self.positive == True:
        #     PSF_half = self.PSF_half
        #     # PSF_half = torch.abs(self.PSF_half)
        #     # tmp = torch.where(self.PSF_half == 0, 0.01, self.PSF_half)
        #     # PSF_half = 0.01*torch.square(tmp)
        # else:
        PSF_half = self.PSF_half

        # linear interpolation / left-nearest interpolation
        if self.interpolation == True:
            self.PSF = (
                PSF_half[self.index_slice, self.index2] * self.disR_b
                + PSF_half[self.index_slice, self.index1] * self.disR1
            )
        else:
            self.PSF = PSF_half[self.index_slice, self.index1]

        if self.positive == True:
            tmp = self.PSF
            self.PSF = 0.01 * torch.square(tmp)
            # self.PSF = torch.abs(tmp)

        # PSF normalization
        if self.kernel_norm == True:
            self.PSF = torch.div(self.PSF, torch.sum(self.PSF))

        weight = self.PSF.repeat(repeats=[self.in_channels, 1, 1, 1, 1])
        return weight

    def forward(self, x):
        """
        ### Parameters:
        - `x`: torch.Tensor, input tensor.
        ### Returns:
        - `x`: torch.Tensor, output tensor.
        """
        weight = self.get_kernel()
        pad_size = (
            self.kernel_size[2] // 2,
            self.kernel_size[2] // 2,
            self.kernel_size[1] // 2,
            self.kernel_size[1] // 2,
            self.kernel_size[0] // 2,
            self.kernel_size[0] // 2,
        )
        x_pad = nn.functional.pad(input=x, pad=pad_size, mode=self.padding_mode)

        # convolution
        if self.conv_mode == "direct":
            conv = nn.functional.conv3d(
                input=x_pad, weight=weight, groups=self.in_channels
            )
        if self.conv_mode == "fft":
            conv = fft_conv(signal=x_pad, kernel=weight, groups=self.in_channels)
        return conv


# ---------------------------------------------------------------------------
class ForwardProject(nn.Module):
    """
    Forward kernel construction.
    ### Parameters:
    - `in_channels`: int, default=1, input channels.
    - `scale_factor`: int, default=1, scale factor.
    - `dim`: int, default=2, dimension of the kernel.
    - `kernel_size`: list, default=None, kernel size.
    - `std_init`: list, default=None, standard deviation of the kernel.
    - `init`: str, default="gauss", initialization method.
    - `trainable`: bool, default=True, whether the kernel is trainable.
    - `padding_mode`: str, default="reflect", padding mode.
    - `interpolation`: bool, default=True, whether to use linear interpolation.
    - `over_sampling`: int, default=2, over sampling factor.
    - `kernel_norm`: bool, default=True, whether to normalize the kernel.
    - `conv_mode`: str, default="direct", convolution mode.
    """

    def __init__(
        self,
        in_channels: int = 1,
        scale_factor=1,
        dim: int = 3,
        kernel_size=None,
        std_init=None,
        init="gauss",
        trainable: bool = True,
        padding_mode="reflect",
        interpolation: bool = True,
        over_sampling=2,
        kernel_norm: bool = True,
        conv_mode="direct",
    ):
        super().__init__()

        # ----------------------------------------------------------------------
        if dim == 2:
            if std_init == None:
                std_init = (1.0, 1.0)
            if kernel_size == None:
                kernel_size == (25, 25)

            self.conv = RadialSymmetricConv2D(
                in_channels=in_channels,
                kernel_size=kernel_size,
                init=init,
                std_init=std_init,
                positive=True,
                padding_mode=padding_mode,
                interpolation=interpolation,
                over_sampling=over_sampling,
                kernel_norm=kernel_norm,
                conv_mode=conv_mode,
            )
            self.pooling = nn.AvgPool2d(kernel_size=scale_factor)

        # ----------------------------------------------------------------------
        if dim == 3:
            if std_init == None:
                std_init = (1.0, 1.0, 1.0)
            if kernel_size == None:
                kernel_size = (25, 25, 25)

            self.conv = RadialSymmetricConv3D(
                in_channels=in_channels,
                kernel_size=kernel_size,
                init=init,
                std_init=std_init,
                positive=True,
                padding_mode=padding_mode,
                interpolation=interpolation,
                over_sampling=over_sampling,
                kernel_norm=kernel_norm,
                conv_mode=conv_mode,
            )
            self.pooling = nn.AvgPool3d(kernel_size=scale_factor)
        # ----------------------------------------------------------------------

        for param in self.conv.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        """
        ### Parameters:
        - `x`: torch.Tensor, input tensor.
        ### Returns:
        - `x`: torch.Tensor, output tensor.
        """
        x = self.conv(x)
        x = self.pooling(x)
        x = torch.maximum(x, torch.tensor([0.0]).to(x.device))
        return x


# ------------------------------------------------------------------------------
class BackwardProject(nn.Module):
    """
    Backward kernel construction.
    ### Parameters:
    - `in_channels`: int, default=1, input channels.
    - `scale_factor`: int, default=1, scale factor.
    - `dim`: int, default=2, dimension of the kernel.
    - `kernel_size`: list, default=None, kernel size.
    - `std_init`: list, default=None, standard deviation of the kernel.
    - `init`: str, default="gauss", initialization method.
    - `trainable`: bool, default=True, whether the kernel is trainable.
    - `padding_mode`: str, default="reflect", padding mode.
    - `interpolation`: bool, default=True, whether to use linear interpolation.
    - `over_sampling`: int, default=2, over sampling factor.
    - `kernel_norm`: bool, default=True, whether to normalize the kernel.
    - `conv_mode`: str, default="direct", convolution mode.
    """

    def __init__(
        self,
        in_channels: int = 1,
        scale_factor=1,
        dim: int = 2,
        kernel_size=None,
        std_init=None,
        init="gauss",
        trainable: bool = True,
        padding_mode="reflect",
        interpolation: bool = True,
        over_sampling=2,
        kernel_norm: bool = True,
        conv_mode="direct",
    ):
        super().__init__()

        self.scale_factor = scale_factor

        # ----------------------------------------------------------------------
        if dim == 2:
            if std_init == None:
                std_init = (1.0, 1.0)
            if kernel_size == None:
                kernel_size = (25, 25)

            self.conv = RadialSymmetricConv2D(
                in_channels=in_channels,
                kernel_size=kernel_size,
                init=init,
                std_init=std_init,
                positive=False,
                padding_mode=padding_mode,
                interpolation=interpolation,
                over_sampling=over_sampling,
                kernel_norm=kernel_norm,
                conv_mode=conv_mode,
            )

        if dim == 3:
            if std_init == None:
                std_init = (1.0, 1.0, 1.0)
            if kernel_size == None:
                kernel_size = (25, 25, 25)

            self.conv = RadialSymmetricConv3D(
                in_channels=in_channels,
                kernel_size=kernel_size,
                init=init,
                std_init=std_init,
                positive=False,
                padding_mode=padding_mode,
                interpolation=interpolation,
                over_sampling=over_sampling,
                kernel_norm=kernel_norm,
                conv_mode=conv_mode,
            )
        # ----------------------------------------------------------------------
        for param in self.conv.parameters():
            param.requires_grad = trainable

        print("[INFO] Initial SD of backward kernel: ", std_init)
        print("[INFO] Kernel size of backward kernel: ", kernel_size)
        print("[INFO] Trainable backward kernel: ", trainable)

    def forward(self, x):
        x = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode="nearest-exact"
        )
        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
class KernelNet(nn.Module):
    """
    KernelNet.
    ### Parameters:
    - `in_channels`: int, default=1, input channels.
    - `scale_factor`: int, default=1, scale factor.
    - `dim`: int, default=2, dimension of the kernel.
    - `num_iter`: int, default=1, number of iterations.
    - `kernel_size_fp`: list, default=None, kernel size of the forward projection.
    - `kernel_size_bp`: list, default=None, kernel size of the backward projection.
    - `init`: str, default="gauss", initialization method.
    - `std_init`: list, default=None, standard deviation of the kernel.
    - `padding_mode`: str, default="reflect", padding mode.
    - `FP`: nn.Module, default=None, forward projection module.
    - `BP`: nn.Module, default=None, backward projection module.
    - `shared_bp`: bool, default=True, whether to share the backward projection kernel.
    - `lam`: float, default=0.0, regularization parameter for the prior.
    - `interpolation`: bool, default=True, whether to use linear interpolation.
    - `over_sampling`: int, default=2, over sampling factor.
    - `kernel_norm`: bool, default=True, whether to normalize the kernel.
    - `conv_mode`: str, default="direct", convolution mode.
    - `return_inter`: bool, default=False, whether to return the deconvolution
            results of intermediate iterations. The first one will be the input.
    - `multi_out`: bool, default=False, whether to return multiple outputs which
            are the deconvolution results of intermediate iterations. (without the input)
    - `self_supervised`: bool, default=False, whether to use self-supervised learning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        scale_factor=1,
        dim: int = 2,
        num_iter: int = 1,
        kernel_size_fp=None,
        kernel_size_bp=None,
        init="gauss",
        std_init=None,
        padding_mode="reflect",
        FP=None,
        BP=None,
        shared_bp=True,
        lam: float = 0.0,
        interpolation: bool = True,
        over_sampling=2,
        kernel_norm: bool = True,
        conv_mode="direct",
        return_inter: bool = False,
        multi_out: bool = False,
        self_supervised: bool = False,
    ):
        super().__init__()

        if dim == 2:
            if kernel_size_fp is None:
                kernel_size_fp = [25, 25]
            if kernel_size_bp is None:
                kernel_size_bp = [25, 25]
        elif dim == 3:
            if kernel_size_fp is None:
                kernel_size_fp = [25, 25, 25]
            if kernel_size_bp is None:
                kernel_size_bp = [25, 25, 25]

        self.scale_factor = scale_factor
        self.num_iter = num_iter
        self.lam = lam
        self.eps = 0.000001

        self.shared_bp = shared_bp
        self.self_supervised = self_supervised
        self.return_inter = return_inter
        self.multi_out = multi_out

        # ----------------------------------------------------------------------
        # Forward Projector
        if FP == None:
            print("[INFO] Create Forward Projector")
            self.FP = ForwardProject(
                dim=dim,
                in_channels=in_channels,
                scale_factor=scale_factor,
                kernel_size=kernel_size_fp,
                std_init=std_init,
                init=init,
                padding_mode=padding_mode,
                trainable=True,
                interpolation=interpolation,
                over_sampling=over_sampling,
                kernel_norm=kernel_norm,
                conv_mode=conv_mode,
            )
        else:
            print("[INFO] Use the given Forward Projector")
            self.FP = FP

        # ----------------------------------------------------------------------
        # Backward Projector
        if BP == None:
            print("[INFO] Create Backward Projector")
            if self.shared_bp == True:
                print("[INFO] Use the same kernel for every iteration.")
                self.BP = BackwardProject(
                    dim=dim,
                    in_channels=in_channels,
                    scale_factor=scale_factor,
                    kernel_size=kernel_size_bp,
                    std_init=std_init,
                    init=init,
                    padding_mode=padding_mode,
                    trainable=True,
                    interpolation=interpolation,
                    over_sampling=over_sampling,
                    kernel_norm=kernel_norm,
                    conv_mode=conv_mode,
                )
            else:
                print("[INFO] Use different kernel for different iteration")
                self.BP = nn.ModuleList()
                for _ in range(self.num_iter):
                    bp = BackwardProject(
                        dim=dim,
                        in_channels=in_channels,
                        scale_factor=scale_factor,
                        kernel_size=kernel_size_bp,
                        std_init=std_init,
                        init=init,
                        padding_mode=padding_mode,
                        trainable=True,
                        interpolation=interpolation,
                        over_sampling=over_sampling,
                        kernel_norm=kernel_norm,
                        conv_mode=conv_mode,
                    )
                    self.BP.append(bp)
        else:
            print("[INFO] Use the given Backward Projector")
            self.BP = BP
        # ---------------------------------------------------------------------------
        # Prior
        if self.lam > 0:
            print("[INFO] Use a Prior")
            self.grad_R = TV_grad(epsilon=self.eps)

    def forward(self, x):
        """
        forward funciton.
        ### Parameters:
        - `x`: torch.Tensor, input tensor.
        ### Returns:
        - `xk`: torch.Tensor, result of the last iteration.
        - `xk_inter`: torch.Tensor, input + intermediate results + result of last iteration.
        - `xk_mulit_out`: torch.Tensor, intermediate results + result of last iteration.
        """
        # input + intermediate results + result of last iteration
        xk_inter = []

        # intermediate results + result of last iteration,
        # if using self-dupervised training, collect the forward projection of the reuslts
        xk_mulit_out = []

        # interpolation to the same size as the ground truth
        xk = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode="nearest-exact"
        )
        # constraint the input tensor values
        xk = self.constraint(xk)

        if self.return_inter:
            xk_inter.append(x)  # original input

        # ----------------------------------------------------------------------
        for i in range(self.num_iter):
            fp = self.FP(xk)
            dv = x / (fp + self.eps)

            dv = torch.clamp(dv, min=0.0, max=3.0)

            if self.shared_bp == True:
                bp = self.BP(dv)
            else:
                bp = self.BP[i](dv)

            xk = xk * bp
            xk = self.constraint(xk)

            if self.lam > 0:
                xk = xk / (1 + self.lam * self.grad_R(xk))
                xk = self.constraint(xk)
            # ----------------------------------------------------------------------

            if self.return_inter == True:
                xk_inter.append(xk)

            if self.multi_out == True:
                if self.self_supervised == True:
                    xk_mulit_out.append(self.FP(xk))
                else:
                    xk_mulit_out.append(xk)

        if self.return_inter == True:
            return torch.stack(xk_inter, dim=0)
        else:
            if self.multi_out == True:
                return torch.stack(xk_mulit_out, dim=0)
            else:  # only return the result of the last iteration
                if self.self_supervised == True:
                    return self.FP(xk)
                else:
                    return xk

    def constraint(self, x):
        """
        Constraint the input tensor values.
        ### Parameters:
        - `x`: torch.Tensor, input tensor.
        ### Returns:
        - `x`: torch.Tensor, output tensor.
        """
        x = torch.clamp(x, min=self.eps, max=None)
        return x


if __name__ == "__main__":
    # 2D
    # x = torch.ones(size=(1, 1, 128, 128))
    # model = KernelNet(dim=2, in_channels=1, scale_factor=1, num_iter=2,\
    #     kernel_size_fp =[25, 25], kernel_size_bp =[25, 25],\
    #     std_init=[2.0, 2.0],\
    #     init='gauss', lam=0.0, padding_mode='reflect', multi_out=False,\
    #     return_inter=False, interpolation=True, over_sampling=2,\
    #     kernel_norm=True,\
    #     shared_bp=True, conv_mode='fft')

    # ---------------------------------------------------------------------------
    # 3D
    x = torch.ones(size=(1, 1, 128, 128, 128))
    model = KernelNet(
        dim=3,
        in_channels=1,
        scale_factor=1,
        num_iter=2,
        kernel_size_fp=[25, 25, 25],
        kernel_size_bp=[31, 25, 25],
        std_init=[4.0, 2.0, 2.0],
        init="delta",
        lam=0.0,
        padding_mode="reflect",
        multi_out=False,
        return_inter=False,
        interpolation=True,
        over_sampling=2,
        kernel_norm=True,
        shared_bp=True,
        conv_mode="fft",
    )

    # ---------------------------------------------------------------------------
    st = time.time()
    o = model(x)
    print(time.time() - st)
    print(o.shape)
