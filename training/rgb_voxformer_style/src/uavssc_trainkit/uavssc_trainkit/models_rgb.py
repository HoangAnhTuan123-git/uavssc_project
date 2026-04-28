from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import Simple2DEncoder, Small3DDecoder, conv_bn_relu_3d, sample_2d_features_to_voxels

class CGFormerStyleSSC(nn.Module):
    """
    Research starter inspired by CGFormer concepts.
    This is not the official CGFormer implementation.
    """
    def __init__(self, num_classes: int, image_size=(640, 640), feat_dim: int = 96, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.image_size = tuple(image_size)
        self.encoder2d = Simple2DEncoder(cin=3, base=32, out_channels=feat_dim)
        self.mask_token = nn.Parameter(torch.zeros(feat_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.q_proj = nn.Linear(feat_dim, hidden_dim)
        self.kv_proj = nn.Linear(feat_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.refine3d = nn.Sequential(
            conv_bn_relu_3d(feat_dim + hidden_dim, hidden_dim),
            conv_bn_relu_3d(hidden_dim, hidden_dim),
        )
        self.decoder = Small3DDecoder(hidden_dim, hidden_dim, num_classes)

    def forward(self, batch):
        img = batch["image"]
        feat2d = self.encoder2d(img)
        uv = batch["projected_pix_1"].to(img.device)
        fov = batch["fov_mask_1"].to(img.device)
        grid = batch["grid_size_xyz"].to(img.device)
        vol = sample_2d_features_to_voxels(feat2d, uv, fov, grid, self.image_size, mask_token=self.mask_token)
        coarse = F.avg_pool3d(vol, kernel_size=4, stride=4)
        B, Cc, Dc, Hc, Wc = coarse.shape
        q = coarse.flatten(2).transpose(1, 2)
        q = self.q_proj(q)
        kv = feat2d.flatten(2).transpose(1, 2)
        kv = self.kv_proj(kv)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        attn_out = attn_out.transpose(1, 2).view(B, -1, Dc, Hc, Wc)
        attn_up = F.interpolate(attn_out, size=vol.shape[-3:], mode="trilinear", align_corners=False)
        fused = self.refine3d(torch.cat([vol, attn_up], dim=1))
        return self.decoder(fused)

class VoxFormerStyleSSC(nn.Module):
    """
    Research starter inspired by VoxFormer.
    This is not the official VoxFormer implementation.
    """
    def __init__(self, num_classes: int, image_size=(640, 640), feat_dim: int = 96, hidden_dim: int = 64):
        super().__init__()
        self.image_size = tuple(image_size)
        self.encoder2d = Simple2DEncoder(cin=3, base=32, out_channels=feat_dim)
        self.mask_token = nn.Parameter(torch.zeros(feat_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.lift_proj = nn.Sequential(
            conv_bn_relu_3d(feat_dim, hidden_dim),
            conv_bn_relu_3d(hidden_dim, hidden_dim),
        )
        self.context = nn.Sequential(
            conv_bn_relu_3d(hidden_dim, hidden_dim),
            conv_bn_relu_3d(hidden_dim, hidden_dim),
            conv_bn_relu_3d(hidden_dim, hidden_dim),
        )
        self.decoder = Small3DDecoder(hidden_dim, hidden_dim, num_classes)

    def forward(self, batch):
        img = batch["image"]
        feat2d = self.encoder2d(img)
        uv = batch["projected_pix_1"].to(img.device)
        fov = batch["fov_mask_1"].to(img.device)
        grid = batch["grid_size_xyz"].to(img.device)
        vol = sample_2d_features_to_voxels(feat2d, uv, fov, grid, self.image_size, mask_token=self.mask_token)
        vol = self.lift_proj(vol)
        vol = self.context(vol)
        return self.decoder(vol)
