"""
U-Net module, adapted from https://github.com/milesial/Pytorch-UNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
            self, in_channel, out_channel, upsample_mode=False, reduce_factor=1,
            resize_factor=1,
        ):
        super(UNet, self).__init__()

        self.resize_factor = resize_factor

        mode_factor = 2 if upsample_mode else 1
        mode_factor = reduce_factor * mode_factor

        s_64 = 64 // reduce_factor
        s_128 = 128 // reduce_factor
        s_256 = 256 // reduce_factor
        s_512 = 512 // reduce_factor
        s_1024 = 1024 // reduce_factor

        self.reduce_layer = nn.AvgPool2d(resize_factor)
        self.dereduce_layer = nn.Upsample(scale_factor=resize_factor, mode=upsample_mode)

        self.inc = _DoubleConv(in_channel, s_64)
        self.down1 = _Down(s_64, s_128)
        self.down2 = _Down(s_128, s_256)
        self.down3 = _Down(s_256, s_512)
        self.down4 = _Down(s_512, 1024 // mode_factor)
        self.up1 = _Up(s_1024, 512 // mode_factor, upsample_mode)
        self.up2 = _Up(s_512, 256 // mode_factor, upsample_mode)
        self.up3 = _Up(s_256, 128 // mode_factor, upsample_mode)
        self.up4 = _Up(s_128, s_64, upsample_mode)
        self.outc = _OutConv(s_64, out_channel)

    def forward(self, x, *args, **kwargs):
        x = self.reduce_layer(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        output = self.dereduce_layer(logits)
        return output


class _DoubleConv(nn.Module):
    """
    Apply the following model twice: Conv2d > BatchNorm2d > ReLU.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class _Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     _DoubleConv(in_channels, out_channels)
        # )
        self.a = nn.MaxPool2d(2)
        self.b = _DoubleConv(in_channels, out_channels)

    def forward(self, x):
        # return self.maxpool_conv(x)
        x = self.a(x)
        x = self.b(x)
        return x


class _OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class _Up(nn.Module):
    """Upscaling then DoubleConv."""

    def __init__(self, in_channels, out_channels, upsample_mode='bicubic'):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number
        # of channels
        if upsample_mode:
            self.up = nn.Upsample(
                scale_factor=2, mode=upsample_mode, align_corners=True,
            )
            self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2,
            )
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
