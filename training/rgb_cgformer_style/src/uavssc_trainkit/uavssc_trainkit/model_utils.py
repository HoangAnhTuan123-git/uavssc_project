from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu_2d(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

def conv_bn_relu_3d(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv3d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm3d(cout),
        nn.ReLU(inplace=True),
    )

class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu_3d(channels, channels),
            nn.Conv3d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))

class MultiPathBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.proj = nn.Conv3d(cin, cout, 1, 1, 0, bias=False)
        self.b1 = nn.Sequential(nn.Conv3d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv3d(cout, cout, (3,1,1), 1, (1,0,0), bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv3d(cout, cout, (1,3,3), 1, (0,1,1), bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Conv3d(cout * 3, cout, 1, 1, 0, bias=False), nn.BatchNorm3d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.proj(x)
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.out(y)

class Simple2DEncoder(nn.Module):
    def __init__(self, cin=3, base=32, out_channels=96):
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu_2d(cin, base, 3, 2, 1), conv_bn_relu_2d(base, base, 3, 1, 1))
        self.stage2 = nn.Sequential(conv_bn_relu_2d(base, base * 2, 3, 2, 1), conv_bn_relu_2d(base * 2, base * 2, 3, 1, 1))
        self.stage3 = nn.Sequential(conv_bn_relu_2d(base * 2, out_channels, 3, 2, 1), conv_bn_relu_2d(out_channels, out_channels, 3, 1, 1))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

def sample_2d_features_to_voxels(feat2d: torch.Tensor, projected_pix: torch.Tensor, fov_mask: torch.Tensor, grid_size_xyz: torch.Tensor, image_hw: Tuple[int, int], mask_token: torch.Tensor | None = None):
    B, C, Hf, Wf = feat2d.shape
    img_h, img_w = image_hw
    outs = []
    for b in range(B):
        nx, ny, nz = [int(x) for x in grid_size_xyz[b].tolist()]
        uv = projected_pix[b].float().clone()
        mask = fov_mask[b].bool().view(-1)
        if uv.numel() == 0:
            outs.append(feat2d.new_zeros((C, nz, ny, nx)))
            continue
        uv[:, 0] = (uv[:, 0] / max(img_w - 1, 1)) * 2.0 - 1.0
        uv[:, 1] = (uv[:, 1] / max(img_h - 1, 1)) * 2.0 - 1.0
        grid = uv.view(1, -1, 1, 2)
        sampled = F.grid_sample(feat2d[b:b+1], grid, align_corners=True, mode="bilinear", padding_mode="zeros")
        sampled = sampled.view(C, -1).transpose(0, 1)
        if mask_token is not None:
            mt = mask_token.view(1, C).to(sampled.device)
            sampled = torch.where(mask[:, None], sampled, mt.expand_as(sampled))
        else:
            sampled = sampled * mask[:, None].float()
        vol = sampled.view(nx, ny, nz, C).permute(3, 2, 1, 0).contiguous()
        outs.append(vol)
    return torch.stack(outs, dim=0)

class Small3DDecoder(nn.Module):
    def __init__(self, cin: int, hidden: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu_3d(cin, hidden),
            ResidualBlock3D(hidden),
            conv_bn_relu_3d(hidden, hidden),
            ResidualBlock3D(hidden),
        )
        self.sem_head = nn.Conv3d(hidden, num_classes, 1)
        self.occ_head = nn.Conv3d(hidden, 1, 1)

    def forward(self, x):
        f = self.net(x)
        return {"features": f, "sem_logits": self.sem_head(f), "occ_logits": self.occ_head(f)}

class TinyBEVUNet(nn.Module):
    def __init__(self, cin: int, base: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(conv_bn_relu_2d(cin, base), conv_bn_relu_2d(base, base))
        self.enc2 = nn.Sequential(conv_bn_relu_2d(base, base * 2, 3, 2, 1), conv_bn_relu_2d(base * 2, base * 2))
        self.enc3 = nn.Sequential(conv_bn_relu_2d(base * 2, base * 4, 3, 2, 1), conv_bn_relu_2d(base * 4, base * 4))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 2, 2, 2), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True))
        self.up1 = nn.Sequential(nn.ConvTranspose2d(base * 2, base, 2, 2), nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.fuse2 = conv_bn_relu_2d(base * 4, base * 2)
        self.fuse1 = conv_bn_relu_2d(base * 2, base)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.up2(e3)
        d2 = self.fuse2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))
        return d1
