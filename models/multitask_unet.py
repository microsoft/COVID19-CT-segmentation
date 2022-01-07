#!/usr/bin/env python
'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_network import BaseNetwork
from options.train_options import TrainOptions
from models.unet import ConvBlock, UpBlock


class MultitaskUnetModel(BaseNetwork):
    def __init__(self, opts):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        self.in_channels = opts.input_channels
        self.out_channels = opts.first_layer_filters
        self.net_depth = opts.net_depth

        # down transformations
        for _ in range(self.net_depth):
            conv = ConvBlock(self.in_channels, self.out_channels)
            self.downblocks.append(conv)
            self.in_channels, self.out_channels = self.out_channels, 2 * self.out_channels

        # midpoint
        self.middle_conv = ConvBlock(self.in_channels, self.out_channels)

        # up transformations
        self.in_channels, self.out_channels = self.out_channels, int(self.out_channels / 2)
        for _ in range(self.net_depth):
            upconv = UpBlock(self.in_channels, self.out_channels)
            self.upblocks.append(upconv)
            self.in_channels, self.out_channels = self.out_channels, int(self.out_channels / 2)

        self.ground_glass_shadow_seg_layer = nn.Conv2d(2 * self.out_channels, opts.num_classes, kernel_size=1)
        self.consolidation_shadow_seg_layer = nn.Conv2d(2 * self.out_channels, opts.num_classes, kernel_size=1)
        self.fibrosis_seg_layer = nn.Conv2d(2 * self.out_channels, opts.num_classes, kernel_size=1)
        self.interstitial_thickening_seg_layer = nn.Conv2d(2 * self.out_channels, opts.num_classes, kernel_size=1)


    def forward(self, x):
        decoder_outputs = []

        for op in self.downblocks:
            decoder_outputs.append(op(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for op in self.upblocks:
            x = op(x, decoder_outputs.pop())
        ground_glass = self.ground_glass_shadow_seg_layer(x)
        consolidation = self.consolidation_shadow_seg_layer(x)
        fibrosis = self.fibrosis_seg_layer(x)
        thickening = self.interstitial_thickening_seg_layer(x)

        return torch.cat([ground_glass, consolidation, fibrosis, thickening], 1)

# Test with mock data
if __name__ == "__main__":
    # A full forward pass
    opts = TrainOptions().parse()
    im = torch.randn(2, 1, 512, 512)
    model = MultitaskUnetModel(opts)
    x = model(im)
    print(x.shape)
    del model
    del x