import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=13, out_channels=12):
        super().__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Input: (B, 13, 48, 48)
        d1 = self.down1(x)           # (B, 64, 48, 48)
        p1 = self.pool1(d1)          # (B, 64, 24, 24)

        d2 = self.down2(p1)          # (B, 128, 24, 24)
        p2 = self.pool2(d2)          # (B, 128, 12, 12)

        bn = self.bottleneck(p2)     # (B, 256, 12, 12)

        up2 = self.up2(bn)           # (B, 128, 24, 24)
        cat2 = torch.cat([up2, d2], dim=1)
        u2 = self.upconv2(cat2)      # (B, 128, 24, 24)

        up1 = self.up1(u2)           # (B, 64, 48, 48)
        cat1 = torch.cat([up1, d1], dim=1)
        u1 = self.upconv1(cat1)      # (B, 64, 48, 48)

        out = self.final_conv(u1)    # (B, 12, 48, 48)
        return out
