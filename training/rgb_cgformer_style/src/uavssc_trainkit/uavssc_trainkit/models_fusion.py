from __future__ import annotations

import torch
import torch.nn as nn

from .model_utils import Simple2DEncoder, Small3DDecoder, conv_bn_relu_3d, sample_2d_features_to_voxels

class FusionGate3DSSC(nn.Module):
    """
    Mid-level RGB + LiDAR voxel fusion starter.
    """
    def __init__(self, num_classes: int, image_size=(640, 640), rgb_dim: int = 96, lidar_hidden: int = 48, fuse_hidden: int = 64):
        super().__init__()
        self.image_size = tuple(image_size)
        self.rgb_encoder = Simple2DEncoder(cin=3, base=32, out_channels=rgb_dim)
        self.mask_token = nn.Parameter(torch.zeros(rgb_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.rgb_proj = nn.Sequential(
            conv_bn_relu_3d(rgb_dim, fuse_hidden),
            conv_bn_relu_3d(fuse_hidden, fuse_hidden),
        )
        self.lidar_stem = nn.Sequential(
            conv_bn_relu_3d(4, lidar_hidden),
            conv_bn_relu_3d(lidar_hidden, fuse_hidden),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(fuse_hidden * 2, fuse_hidden, 1),
            nn.BatchNorm3d(fuse_hidden),
            nn.ReLU(inplace=True),
            nn.Conv3d(fuse_hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            conv_bn_relu_3d(fuse_hidden, fuse_hidden),
            conv_bn_relu_3d(fuse_hidden, fuse_hidden),
        )
        self.decoder = Small3DDecoder(fuse_hidden, fuse_hidden, num_classes)

    def forward(self, batch):
        img = batch["image"]
        feat2d = self.rgb_encoder(img)
        uv = batch["projected_pix_1"].to(img.device)
        fov = batch["fov_mask_1"].to(img.device)
        grid = batch["grid_size_xyz"].to(img.device)
        rgb_vol = sample_2d_features_to_voxels(feat2d, uv, fov, grid, self.image_size, mask_token=self.mask_token)
        rgb_vol = self.rgb_proj(rgb_vol)
        lidar = self.lidar_stem(batch["lidar_dense"])
        gate = self.gate(torch.cat([rgb_vol, lidar], dim=1))
        fused = gate * lidar + (1.0 - gate) * rgb_vol
        fused = self.post(fused)
        out = self.decoder(fused)
        out["fusion_gate"] = gate
        return out
