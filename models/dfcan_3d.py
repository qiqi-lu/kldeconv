import torch, math, torchinfo
import torch.nn as nn
import torch.nn.functional as F


def fft3d(input, gamma=0.1):
    # input = torch.complex(real=input, imag=torch.zeros_like(input))
    fft = torch.fft.fftn(input, dim=(-3, -2, -1))  # 3D FFT over last 3 dimensions
    absfft = torch.pow(torch.abs(fft) + 1e-8, gamma)
    return absfft


def fftshift3d(input):
    _, _, d, h, w = input.shape
    # Perform fftshift manually for 3D
    # Split along depth dimension
    fs_d1 = input[:, :, -d // 2 : d, :, :]
    fs_d2 = input[:, :, 0 : d // 2, :, :]
    shifted_d = torch.cat([fs_d1, fs_d2], dim=2)

    # Split along height dimension
    fs_h1 = shifted_d[:, :, :, -h // 2 : h, :]
    fs_h2 = shifted_d[:, :, :, 0 : h // 2, :]
    shifted_h = torch.cat([fs_h1, fs_h2], dim=3)

    # Split along width dimension
    fs_w1 = shifted_h[:, :, :, :, -w // 2 : w]
    fs_w2 = shifted_h[:, :, :, :, 0 : w // 2]
    output = torch.cat([fs_w1, fs_w2], dim=4)

    return output


class FCALayer3D(nn.Module):
    def __init__(self, num_features=64, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv3d(
            in_channels=num_features,
            out_channels=num_features // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv3 = nn.Conv3d(
            in_channels=num_features // reduction,
            out_channels=num_features,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        absfft1 = fft3d(x, gamma=0.8)
        absfft1 = fftshift3d(absfft1)
        absfft2 = self.act1(self.conv1(absfft1))
        w = torch.mean(
            absfft2, dim=(-3, -2, -1), keepdim=True
        )  # Global average pooling over D, H, W
        w = self.act1(self.conv2(w))
        w = self.act2(self.conv3(w))
        mul = torch.mul(x, w)
        return mul


class FCAB3D(nn.Module):
    def __init__(self, num_features=64):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fca = FCALayer3D(num_features=num_features, reduction=16)

    def forward(self, x):
        conv = self.act(self.conv1(x))
        conv = self.act(self.conv2(conv))
        att = self.fca(conv)
        output = torch.add(att, x)
        return output


class ResidualGroup3D(nn.Module):
    def __init__(self, num_features=64, num_blocks=4):
        super().__init__()
        blocks = []
        for _ in range(num_blocks):
            blocks.append(FCAB3D(num_features=num_features))
        self.fcabs = nn.Sequential(*blocks)

    def forward(self, x):
        conv = self.fcabs(x)
        conv = torch.add(conv, x)
        return conv


class Upsampler3D(nn.Module):
    def __init__(self, scale_factor=4, num_features=64, kernel_size=3):
        super().__init__()
        self.scale_factor = scale_factor

        # Since there's no direct 3D equivalent of PixelShuffle, we'll use transposed convolutions
        if scale_factor == 2:
            self.upconv = nn.ConvTranspose3d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1,
                bias=True,
            )
        elif scale_factor == 4:
            # Two stages of 2x upsampling
            self.upconv1 = nn.ConvTranspose3d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1,
                bias=True,
            )
            self.upconv2 = nn.ConvTranspose3d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1,
                bias=True,
            )
            self.act = nn.GELU()
        elif scale_factor == 8:
            # Three stages of 2x upsampling
            self.upconv1 = nn.ConvTranspose3d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1,
                bias=True,
            )
            self.upconv2 = nn.ConvTranspose3d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1,
                bias=True,
            )
            self.upconv3 = nn.ConvTranspose3d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1,
                bias=True,
            )
            self.act = nn.GELU()
        else:
            # Fallback: use interpolation + convolution
            self.conv = nn.Conv3d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True,
            )
            self.act = nn.GELU()

    def forward(self, x):
        if self.scale_factor == 1:
            return x
        elif self.scale_factor == 2:
            return self.upconv(x)
        elif self.scale_factor == 4:
            x = self.act(self.upconv1(x))
            return self.upconv2(x)
        elif self.scale_factor == 8:
            x = self.act(self.upconv1(x))
            x = self.act(self.upconv2(x))
            return self.upconv3(x)
        else:
            # Use interpolation as fallback
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode="trilinear", align_corners=False
            )
            return self.act(self.conv(x))


class DFCAN3D(nn.Module):
    def __init__(
        self, in_channels=1, scale_factor=4, num_features=64, num_groups=4
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()

        groups = []
        for _ in range(num_groups):
            groups.append(ResidualGroup3D(num_features=num_features, num_blocks=4))
        self.residual_groups = nn.Sequential(*groups)

        self.upsampler = Upsampler3D(
            scale_factor=scale_factor, num_features=num_features, kernel_size=3
        )

        self.conv_tail = nn.Conv3d(
            in_channels=num_features,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act_tail = nn.Sigmoid()

    def forward(self, x):
        conv = self.act(self.conv1(x))
        conv = self.residual_groups(conv)
        conv_up = self.upsampler(conv)
        out = self.act_tail(self.conv_tail(conv_up))
        return out


if __name__ == "__main__":
    in_channels = 1
    input_size = (2, in_channels, 32, 64, 64)  # (B, C, D, H, W)
    x = torch.ones(size=input_size)
    bs, ch, d, h, w = x.shape
    model = DFCAN3D(
        in_channels=in_channels, scale_factor=1, num_features=64, num_groups=4
    )
    o = model(x)
    torchinfo.summary(model=model, input_size=input_size)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {o.shape}")
