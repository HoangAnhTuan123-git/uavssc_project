"""
Code adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py

UAVScenes adapter notes:
- Original MonoScene hard-coded EfficientNet-B7. That is too memory-heavy for UAVScenes
  even when the input RGB is resized, because the model still keeps a very large encoder
  and decoder graph.
- This version supports configurable EfficientNet backbones. For a 24GB RTX 4090, start
  with tf_efficientnet_b0_ns. Try b4 only after b0 trains successfully.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel metadata for rwightman/gen-efficientnet-pytorch geffnet models.
# The decoder uses features[4], features[5], features[6], features[8], features[11],
# which correspond to stage0, stage1, stage2, stage4, and conv_head outputs.
_EFFICIENTNET_CHANNELS = {
    "tf_efficientnet_b0_ns": {"head": 1280, "skip0": 16, "skip1": 24, "skip2": 40, "skip3": 112},
    "tf_efficientnet_b1_ns": {"head": 1280, "skip0": 16, "skip1": 24, "skip2": 40, "skip3": 112},
    "tf_efficientnet_b2_ns": {"head": 1408, "skip0": 16, "skip1": 24, "skip2": 48, "skip3": 120},
    "tf_efficientnet_b3_ns": {"head": 1536, "skip0": 24, "skip1": 32, "skip2": 48, "skip3": 136},
    "tf_efficientnet_b4_ns": {"head": 1792, "skip0": 24, "skip1": 32, "skip2": 56, "skip3": 160},
    "tf_efficientnet_b5_ns": {"head": 2048, "skip0": 24, "skip1": 40, "skip2": 64, "skip3": 176},
    "tf_efficientnet_b6_ns": {"head": 2304, "skip0": 32, "skip1": 40, "skip2": 72, "skip3": 200},
    "tf_efficientnet_b7_ns": {"head": 2560, "skip0": 32, "skip1": 48, "skip2": 80, "skip3": 224},
}


def _normalize_backbone_name(backbone_name):
    if backbone_name is None:
        return "tf_efficientnet_b0_ns"
    name = str(backbone_name).strip()
    aliases = {
        "b0": "tf_efficientnet_b0_ns",
        "b1": "tf_efficientnet_b1_ns",
        "b2": "tf_efficientnet_b2_ns",
        "b3": "tf_efficientnet_b3_ns",
        "b4": "tf_efficientnet_b4_ns",
        "b5": "tf_efficientnet_b5_ns",
        "b6": "tf_efficientnet_b6_ns",
        "b7": "tf_efficientnet_b7_ns",
        "efficientnet-b0": "tf_efficientnet_b0_ns",
        "efficientnet-b4": "tf_efficientnet_b4_ns",
        "efficientnet-b7": "tf_efficientnet_b7_ns",
    }
    return aliases.get(name.lower(), name)


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features, bottleneck_features, out_feature, use_decoder=True, skip_channels=None):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder
        skip_channels = skip_channels or {"skip0": 32, "skip1": 48, "skip2": 80, "skip3": 224}

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 16
        self.feature_1_1 = max(features // 32, 1)

        if self.use_decoder:
            self.resize_output_1_1 = nn.Conv2d(self.feature_1_1, self.out_feature_1_1, kernel_size=1)
            self.resize_output_1_2 = nn.Conv2d(self.feature_1_2, self.out_feature_1_2, kernel_size=1)
            self.resize_output_1_4 = nn.Conv2d(self.feature_1_4, self.out_feature_1_4, kernel_size=1)
            self.resize_output_1_8 = nn.Conv2d(self.feature_1_8, self.out_feature_1_8, kernel_size=1)
            self.resize_output_1_16 = nn.Conv2d(self.feature_1_16, self.out_feature_1_16, kernel_size=1)

            self.up16 = UpSampleBN(skip_input=features + skip_channels["skip3"], output_features=self.feature_1_16)
            self.up8 = UpSampleBN(skip_input=self.feature_1_16 + skip_channels["skip2"], output_features=self.feature_1_8)
            self.up4 = UpSampleBN(skip_input=self.feature_1_8 + skip_channels["skip1"], output_features=self.feature_1_4)
            self.up2 = UpSampleBN(skip_input=self.feature_1_4 + skip_channels["skip0"], output_features=self.feature_1_2)
            self.up1 = UpSampleBN(skip_input=self.feature_1_2 + 3, output_features=self.feature_1_1)
        else:
            self.resize_output_1_1 = nn.Conv2d(3, out_feature, kernel_size=1)
            self.resize_output_1_2 = nn.Conv2d(skip_channels["skip0"], out_feature * 2, kernel_size=1)
            self.resize_output_1_4 = nn.Conv2d(skip_channels["skip1"], out_feature * 4, kernel_size=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )
        bs = x_block0.shape[0]
        x_d0 = self.conv2(x_block4)

        if self.use_decoder:
            x_1_16 = self.up16(x_d0, x_block3)
            x_1_8 = self.up8(x_1_16, x_block2)
            x_1_4 = self.up4(x_1_8, x_block1)
            x_1_2 = self.up2(x_1_4, x_block0)
            x_1_1 = self.up1(x_1_2, features[0])
            return {
                "1_1": self.resize_output_1_1(x_1_1),
                "1_2": self.resize_output_1_2(x_1_2),
                "1_4": self.resize_output_1_4(x_1_4),
                "1_8": self.resize_output_1_8(x_1_8),
                "1_16": self.resize_output_1_16(x_1_16),
            }
        else:
            x_1_1 = features[0]
            x_1_2, x_1_4 = features[4], features[5]
            x_global = features[-1].reshape(bs, -1, features[-1].shape[-1] * features[-1].shape[-2]).mean(2)
            return {
                "1_1": self.resize_output_1_1(x_1_1),
                "1_2": self.resize_output_1_2(x_1_2),
                "1_4": self.resize_output_1_4(x_1_4),
                "global": x_global,
            }


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for _, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UNet2D(nn.Module):
    def __init__(self, backend, num_features, out_feature, use_decoder=True, skip_channels=None):
        super(UNet2D, self).__init__()
        self.use_decoder = use_decoder
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN(
            out_feature=out_feature,
            use_decoder=use_decoder,
            bottleneck_features=num_features,
            num_features=num_features,
            skip_channels=skip_channels,
        )

    def forward(self, x, **kwargs):
        encoded_feats = self.encoder(x)
        unet_out = self.decoder(encoded_feats, **kwargs)
        return unet_out

    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_decoder_params(self):
        return self.decoder.parameters()

    @classmethod
    def build(cls, backbone_name="tf_efficientnet_b0_ns", pretrained=True, **kwargs):
        basemodel_name = _normalize_backbone_name(backbone_name)
        if basemodel_name not in _EFFICIENTNET_CHANNELS:
            raise ValueError(
                "Unsupported backbone_name='{}'. Supported: {}".format(
                    basemodel_name, sorted(_EFFICIENTNET_CHANNELS.keys())
                )
            )
        meta = _EFFICIENTNET_CHANNELS[basemodel_name]
        num_features = int(meta["head"])
        skip_channels = {k: int(meta[k]) for k in ["skip0", "skip1", "skip2", "skip3"]}

        print("Loading base model ({})...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=bool(pretrained)
        )
        print("Done.")

        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        print(
            "Building Encoder-Decoder model with backbone={}, head_channels={}, skip_channels={}..".format(
                basemodel_name, num_features, skip_channels
            ),
            end="",
        )
        m = cls(basemodel, num_features=num_features, skip_channels=skip_channels, **kwargs)
        print("Done.")
        return m


if __name__ == "__main__":
    model = UNet2D.build(backbone_name=os.environ.get("RGB_BACKBONE", "tf_efficientnet_b0_ns"), out_feature=64, use_decoder=True)
