import torch
import torch.nn as nn
import torch.nn.functional as F


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class UNet(nn.Module):
    """
    4-level U-Net for spectrogram soft-mask estimation.

    Input:  mix magnitude  [B, 1, freq_bins, time_frames]
    Output: soft mask      [B, 1, freq_bins, time_frames]  (values in [0, 1])

    Multiply the output mask by the mix magnitude to get the estimated target
    stem magnitude, then reconstruct the waveform using the mix phase.
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = _double_conv(1, c)
        self.enc2 = _double_conv(c, c * 2)
        self.enc3 = _double_conv(c * 2, c * 4)
        self.enc4 = _double_conv(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = _double_conv(c * 8, c * 16)

        # Decoder (skip + upsampled features → conv block)
        self.dec4 = _double_conv(c * 16 + c * 8, c * 8)
        self.dec3 = _double_conv(c * 8 + c * 4, c * 4)
        self.dec2 = _double_conv(c * 4 + c * 2, c * 2)
        self.dec1 = _double_conv(c * 2 + c, c)

        self.output_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                   # [B, c,    F,    T]
        e2 = self.enc2(self.pool(e1))        # [B, c*2,  F/2,  T/2]
        e3 = self.enc3(self.pool(e2))        # [B, c*4,  F/4,  T/4]
        e4 = self.enc4(self.pool(e3))        # [B, c*8,  F/8,  T/8]

        # Bottleneck
        b = self.bottleneck(self.pool(e4))   # [B, c*16, F/16, T/16]

        # Decoder — interpolate to skip-connection spatial size before concat
        d4 = self.dec4(torch.cat([F.interpolate(b,  size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))

        return torch.sigmoid(self.output_conv(d1))
