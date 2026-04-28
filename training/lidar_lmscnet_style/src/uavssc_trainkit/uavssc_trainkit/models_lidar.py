from __future__ import annotations

import torch.nn as nn

from .model_utils import MultiPathBlock3D, Small3DDecoder, TinyBEVUNet, conv_bn_relu_3d

class LMSCNetStyleSSC(nn.Module):
    """
    Dense starter inspired by LMSCNet.
    """
    def __init__(self, num_classes: int, in_channels: int = 4, nz: int = 32, bev_base: int = 64, hidden3d: int = 64):
        super().__init__()
        self.nz = nz
        self.bev = TinyBEVUNet(in_channels * nz, base=bev_base)
        self.to3d = nn.Sequential(
            nn.Conv2d(bev_base, hidden3d * nz, 1),
            nn.BatchNorm2d(hidden3d * nz),
            nn.ReLU(inplace=True),
        )
        self.decoder = Small3DDecoder(hidden3d, hidden3d, num_classes)

    def forward(self, batch):
        x = batch["lidar_dense"]
        B, C, Z, Y, X = x.shape
        bev = x.permute(0, 1, 3, 4, 2).reshape(B, C * Z, Y, X)
        bev = self.bev(bev)
        feat = self.to3d(bev).view(B, -1, Z, Y, X)
        return self.decoder(feat)

class SCPNetStyleSSC(nn.Module):
    """
    Dense 3D starter inspired by SCPNet multi-path completion.
    Not the official sparse + KD SCPNet.
    """
    def __init__(self, num_classes: int, in_channels: int = 4, hidden: int = 48):
        super().__init__()
        self.stem = conv_bn_relu_3d(in_channels, hidden)
        self.block1 = MultiPathBlock3D(hidden, hidden)
        self.block2 = MultiPathBlock3D(hidden, hidden)
        self.block3 = MultiPathBlock3D(hidden, hidden)
        self.decoder = Small3DDecoder(hidden, hidden, num_classes)

    def forward(self, batch):
        x = batch["lidar_dense"]
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.decoder(x)
