# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The deconvolution code is based on Simple Baseline.
# (https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    build_upsample_layer,
    constant_init,
    normal_init,
)
from models.swin_transformer_v2 import SwinTransformerV2
from utils.PQI import PSP


class BCP(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self,
        max_depth,
        min_depth,
        in_features=512,
        hidden_features=512 * 4,
        out_features=256,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def forward(self, x):
        x = torch.mean(x.flatten(start_dim=2), dim=2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = (self.max_depth - self.min_depth) * bins
        bin_widths = nn.functional.pad(
            bin_widths, (1, 0), mode="constant", value=self.min_depth
        )
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)
        return centers


class GLPDepth(nn.Module):
    def __init__(self, args=None, is_train=False):
        super().__init__()
        self.max_depth = args.max_depth

        if "tiny" in args.backbone:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
        elif "base" in args.backbone:
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif "large" in args.backbone:
            embed_dim = 192
            num_heads = [6, 12, 24, 48]
        elif "huge" in args.backbone:
            embed_dim = 352
            num_heads = [11, 22, 44, 88]
        else:
            raise ValueError(
                args.backbone
                + " is not implemented, please add it in the models/model.py."
            )

        self.encoder = SwinTransformerV2(
            embed_dim=embed_dim,
            depths=args.depths,
            num_heads=num_heads,
            window_size=args.window_size,
            pretrain_window_size=args.pretrain_window_size,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_shift=args.use_shift,
            out_indices=(0, 1, 2, 3),  # CFB
        )

        self.encoder.init_weights(pretrained=args.pretrained)

        channels_in = embed_dim * 8  # 1536
        channels_out = embed_dim  # 192

        # CFB
        # swin large in_channels
        psp_cfg = dict(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=channels_in,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=dict(type="BN", requires_grad=True),
            align_corners=False,
        )

        # additional part
        self.training = is_train

        self.psp = PSP(**psp_cfg)
        self.bcp = BCP(
            in_features=channels_in,
            out_features=128,
            max_depth=args.max_depth,
            min_depth=0.1,
        )
        self.feature_conv = nn.Conv2d(channels_in, channels_out, 3, padding=1, stride=1)

        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1),
        )

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):
        # CFB
        conv1, conv2, conv3, conv4 = self.encoder(x)

        q_psp = self.psp((conv1, conv2, conv3, conv4))  # b x channels_in x 15 x 15

        if self.training:
            bin_centers = self.bcp(q_psp)
            # q = F.interpolate(q_psp, scale_factor=2, mode='nearest')
            # q = self.feature_conv(q)                       # b x channels_in//4 x 30 x 30

        out, _ = self.decoder([conv4])  # b x channels_out x 480 x 480

        out_depth = self.last_layer_depth(out)  # b x 1 x 480 x 480
        out_depth = torch.sigmoid(out_depth) * self.max_depth

        return (
            {"pred_d": out_depth, "feature": conv4, "centers": bin_centers}
            if self.training
            else {"pred_d": out_depth}
        )


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels

        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )

        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type="Conv2d"),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        conv_layers.append(build_norm_layer(dict(type="BN"), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats[0])
        """
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out
        """

        q = self.conv_layers(out)
        out = self.up(q)
        out = self.up(out)

        return out, q

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""

        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type="deconv"),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
