# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.base_models.depth_anything import DepthAnythingCore
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import (
    Projector,
    SeedBinRegressor,
    SeedBinRegressorUnnormed,
)
from zoedepth.models.model_io import load_state_from_resource


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


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(
        x, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


class ZoeDepth(DepthModel):
    def __init__(
        self,
        core,
        n_bins=64,
        bin_centers_type="softplus",
        bin_embedding_dim=128,
        min_depth=1e-3,
        max_depth=10,
        n_attractors=[16, 8, 4, 1],
        attractor_alpha=300,
        attractor_gamma=2,
        attractor_kind="sum",
        attractor_type="exp",
        min_temp=5,
        max_temp=50,
        train_midas=True,
        midas_lr_factor=10,
        encoder_lr_factor=10,
        pos_enc_lr_factor=10,
        inverse_midas=False,
        **kwargs
    ):
        """ZoeDepth model. This is the version of ZoeDepth that has a single metric head

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.
        """
        super().__init__()

        self.core = core
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.train_midas = train_midas
        self.inverse_midas = inverse_midas

        self.sigmoid = nn.Sigmoid()

        N_MIDAS_OUT = 32

        if self.encoder_lr_factor <= 0:
            self.core.freeze_encoder(freeze_rel_pos=self.pos_enc_lr_factor <= 0)

        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]

        torch.manual_seed(1)
        self.bcp = BCP(
            in_features=btlnck_features,
            out_features=64,
            max_depth=max_depth,
            min_depth=min_depth,
        )

    def forward(
        self, x, return_final_centers=False, denorm=False, return_probs=False, **kwargs
    ):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.

        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """
        # print('input shape', x.shape)

        b, c, h, w = x.shape
        # print("input shape:", x.shape)
        self.orig_input_width = w
        self.orig_input_height = h
        rel_depth, out = self.core(
            x, denorm=denorm, return_rel_depth=True
        )  # 1 x 392 x 518

        outconv_activation = out[0]  # head output feature, 1 x 32 x 392 x 518
        btlnck = out[1]  # encoder final layer feature, 1 x 256 x 14 x 19
        # out[2, 3, 4, 5] => decoder feature r4, r3, r2, r1 (r1 is a 1/2 feature & size: 1 x 256 x 224 x 296)

        # x_d0 = self.conv2(btlnck)        # binning feature

        # pred = rel_depth.unsqueeze(dim=1)   # 뒤의 값은 scale 몇으로 upsampling할건지
        # rel_depth = self.last_conv(rel_depth)
        # rel_depth = self.sigmoid(rel_depth)
        # pred = rel_depth * self.max_depth

        # pred = self.last_conv(outconv_activation)
        # pred = self.sigmoid(pred)
        rel_depth = rel_depth.unsqueeze(dim=1)
        pred = rel_depth * self.max_depth

        if self.training:
            bin_centers = self.bcp(btlnck)  # bin_centers
            out_feature = out[2]  # out[2] and btlnck is good

            output = dict(
                metric_depth=pred,
                features=out_feature,
                mean_centers=bin_centers,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
        else:
            output = dict(metric_depth=pred)

        return output

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            if self.encoder_lr_factor > 0:
                param_conf.append(
                    {
                        "params": self.core.get_enc_params_except_rel_pos(),
                        "lr": lr / self.encoder_lr_factor,
                    }
                )

            if self.pos_enc_lr_factor > 0:
                param_conf.append(
                    {
                        "params": self.core.get_rel_pos_params(),
                        "lr": lr / self.pos_enc_lr_factor,
                    }
                )

            # midas_params = self.core.core.scratch.parameters()
            midas_params = self.core.core.depth_head.parameters()
            midas_lr_factor = self.midas_lr_factor
            param_conf.append({"params": midas_params, "lr": lr / midas_lr_factor})

        remaining_modules = []
        for name, child in self.named_children():
            if name != "core":
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules]
        )

        param_conf.append({"params": remaining_params, "lr": lr})

        return param_conf

    @staticmethod
    def build(
        midas_model_type="DPT_BEiT_L_384",
        pretrained_resource=None,
        use_pretrained_midas=False,
        train_midas=False,
        freeze_midas_bn=True,
        **kwargs
    ):
        # core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
        #                        train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)

        core = DepthAnythingCore.build(
            midas_model_type=midas_model_type,
            use_pretrained_midas=use_pretrained_midas,
            train_midas=train_midas,
            fetch_features=True,
            freeze_bn=freeze_midas_bn,
            **kwargs
        )

        model = ZoeDepth(core, **kwargs)
        if pretrained_resource:
            assert isinstance(
                pretrained_resource, str
            ), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepth.build(**config)
