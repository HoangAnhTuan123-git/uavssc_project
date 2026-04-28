from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ConvBlock3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dx = skip.shape[-3] - x.shape[-3]
        dy = skip.shape[-2] - x.shape[-2]
        dz = skip.shape[-1] - x.shape[-1]
        x = F.pad(x, [0, dz, 0, dy, 0, dx])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class DenseLidarSSCUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 19, base_ch: int = 16) -> None:
        super().__init__()
        self.inc = ConvBlock3D(in_channels, base_ch)
        self.down1 = Down3D(base_ch, base_ch * 2)
        self.down2 = Down3D(base_ch * 2, base_ch * 4)
        self.down3 = Down3D(base_ch * 4, base_ch * 8)
        self.up2 = Up3D(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up1 = Up3D(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up0 = Up3D(base_ch * 2, base_ch, base_ch)
        self.occ_head = nn.Conv3d(base_ch, 1, kernel_size=1)
        self.sem_head = nn.Conv3d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        y2 = self.up2(x3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        occ_logits = self.occ_head(y0)
        sem_logits = self.sem_head(y0)
        return {'occ_logits': occ_logits, 'sem_logits': sem_logits}
